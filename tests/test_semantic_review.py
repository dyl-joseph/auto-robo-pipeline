from copy import deepcopy
from types import SimpleNamespace

import pytest
from label_studio_sdk.types.annotation import Annotation
from label_studio_sdk.types.lse_task import LseTask

from robo.core import LABELS, atomic_json, digest, file_hash
from robo.export import validate_export
from robo.labelstudio import label_config, review_fields
from robo.native_review import NativeReviewBridge
from robo.semantic import EvaluationPlan, Predictions, episode_spec
from robo.semantic_review import bound_prediction, decoded_review, prepare
from robo.store import Conflict


@pytest.fixture
def semantic(episode, tmp_path):
    store, episode = episode
    spec = episode_spec(
        episode,
        dataset="generated",
        dataset_episode_id="generated-1",
        camera_id="front",
        split="test",
    )
    plan = EvaluationPlan(
        taxonomy_id="generated",
        normalized_actions=list(LABELS),
        annotation_guidelines_sha256=digest("generated instructions"),
        identity_audit={
            "status": "synthetic_fixture",
            "reviewer": "fixture",
            "evidence_sha256": digest("generated"),
        },
        episodes=[spec],
    )
    segments = []
    for start, end, action in [(0, 10, "motion"), (10, 20, "hold")]:
        segments.append(
            {
                "episode_id": episode["id"],
                "start_s": spec.frame_boundaries_s[start],
                "end_s": spec.frame_boundaries_s[end],
                "source_frames": {
                    "start": start,
                    "end_exclusive": end,
                    "evidence": [start, end - 1],
                },
                "subtask_text": f"Generated {action} phrase",
                "normalized_action": action,
                "object_text": "generated object" if start == 0 else None,
                "uncertainty": 0.8,
                "provenance": {
                    "kind": "synthetic_fixture",
                    "producer": "generated test",
                    "artifact_sha256": digest(action),
                },
            }
        )
    predictions = Predictions(
        plan_sha256=plan.plan_sha256,
        kind="uncorrected_predictions",
        run_id="generated",
        producer="generated test",
        code_revision="generated",
        config_sha256=digest("generated config"),
        training_episode_ids=[],
        tuning_episode_ids=[],
        reference_access="none",
        episodes=[{"episode_id": episode["id"], "segments": segments}],
    )
    plan_path, prediction_path = tmp_path / "plan.json", tmp_path / "predictions.json"
    atomic_json(plan_path, plan.model_dump())
    atomic_json(prediction_path, predictions.model_dump())
    output = tmp_path / "review"
    bundle = prepare(store, plan_path, prediction_path, episode["id"], output)
    timestamp = "2026-01-01T00:00:00Z"
    annotation = {
        "id": 3,
        "task": 2,
        "project": 1,
        "completed_by": 7,
        "created_at": timestamp,
        "updated_at": timestamp,
        "lead_time": 1.0,
        "was_cancelled": False,
        "result": deepcopy(bundle["task"]["predictions"][0]["result"]),
    }
    task = {
        "id": 2,
        "project": 1,
        "data": bundle["task"]["data"],
        "annotations": [annotation],
        "drafts": [],
        "predictions": [
            {
                "id": 4,
                "created_at": timestamp,
                "updated_at": timestamp,
                **deepcopy(bundle["task"]["predictions"][0]),
            }
        ],
    }
    sdk = SimpleNamespace(
        projects=SimpleNamespace(
            get=lambda _: SimpleNamespace(label_config=bundle["label_config"])
        ),
        tasks=SimpleNamespace(get=lambda _: LseTask.model_validate(deepcopy(task))),
        annotations=SimpleNamespace(get=lambda _: Annotation.model_validate(deepcopy(annotation))),
    )
    bridge = NativeReviewBridge(sdk, store, "http://testserver")
    return SimpleNamespace(
        store=store,
        episode=episode,
        spec=spec,
        bundle=bundle,
        bridge=bridge,
        task=task,
        annotation=annotation,
        plan_path=plan_path,
        prediction_path=prediction_path,
        output=output,
    )


def register(s):
    s.bridge.register(s.bundle, 1, 2)


def refresh(s, revision=0):
    return s.bridge.refresh(
        s.episode["id"], 3, revision, "test reviewer", "Generated test correction"
    )


def accept(s, revision=1):
    return s.bridge.decide(
        s.episode["id"], revision, "accept", "test approver", "Explicit integration test decision"
    )


def selection(s, revision=2):
    return [{"episode_id": s.episode["id"], "revision": revision, "split": "test"}]


def test_held_out_review_cannot_be_exported_as_training(semantic, tmp_path):
    s = semantic
    register(s)
    refresh(s)
    accept(s)
    changed = selection(s)
    changed[0]["split"] = "train"
    with pytest.raises(ValueError, match="frozen evaluation split"):
        s.bridge.export(changed, tmp_path / "leaked")
    assert not (tmp_path / "leaked").exists()


def test_snapshot_and_repeat_preparation_keep_uncorrected_bytes(semantic):
    s = semantic
    before = s.prediction_path.read_bytes()
    assert prepare(s.store, s.plan_path, s.prediction_path, s.episode["id"], s.output) == s.bundle
    assert s.prediction_path.read_bytes() == before
    origin = s.bundle["trusted_origin"]["semantic_prediction"]
    assert origin["files"]["predictions.json"] == file_hash(s.prediction_path)
    snapshot = s.store.root / "semantic_sources" / origin["snapshot_id"]
    assert (snapshot / "predictions.json").read_bytes() == before
    assert 'perRegion="true"' in s.bundle["label_config"]
    assert "subtask_text" not in label_config([], 10)
    register(s)
    assert s.store.latest(s.episode["id"])["status"] == "unreviewed"


def test_text_only_edits_invalidate_approval_without_mutating_predictions(semantic, tmp_path):
    s = semantic
    register(s)
    before = s.prediction_path.read_bytes()
    draft = refresh(s)
    assert draft["native"]["semantic_review"]["complete"]
    assert draft["native"]["semantic_review"]["prediction_exposure"]
    assert draft["native"]["semantic_review"]["kind"] == "model_assisted_review"
    assert draft["suggestion_run"] is None and draft["condition"] == "assisted"
    accepted = accept(s)
    s.bridge.export(selection(s), tmp_path / "accepted")
    original = validate_export(tmp_path / "accepted")[0]
    assert (
        original["annotation"]["native"]["semantic_review"]["segments"][0]["subtask_text"]
        == "Generated motion phrase"
    )
    text = next(r for r in s.annotation["result"] if r["from_name"] == "subtask_text")
    text["value"]["text"] = ["Changed review text only"]
    s.annotation["updated_at"] = "2026-01-01T00:00:01Z"
    with pytest.raises(Conflict, match="changed"):
        s.bridge.export(selection(s), tmp_path / "stale")
    changed = refresh(s, 2)
    assert changed["native"]["snapshot_hash"] != accepted["native"]["snapshot_hash"]
    assert changed["segments"] == accepted["segments"]
    accept(s, 3)
    s.bridge.export(selection(s, 4), tmp_path / "changed")
    assert validate_export(tmp_path / "accepted")[0] == original
    assert s.prediction_path.read_bytes() == before
    assert s.task["predictions"][0]["result"] == s.bundle["task"]["predictions"][0]["result"]


@pytest.mark.parametrize("cleared", [None, "", "  "])
def test_cleared_object_requires_fresh_approval_and_exports_as_absent(semantic, tmp_path, cleared):
    s = semantic
    register(s)
    before = s.prediction_path.read_bytes()
    refresh(s)
    accept(s)
    s.bridge.export(selection(s), tmp_path / "populated")
    original = validate_export(tmp_path / "populated")[0]
    text = next(r for r in s.annotation["result"] if r["from_name"] == "object_text")
    if cleared is None:
        s.annotation["result"].remove(text)
    else:
        text["value"]["text"] = [cleared]
    s.annotation["updated_at"] = "2026-01-01T00:00:01Z"
    with pytest.raises(Conflict, match="changed"):
        s.bridge.export(selection(s), tmp_path / "stale")
    changed = refresh(s, 2)
    assert changed["native"]["semantic_review"]["segments"][0]["object_text"] is None
    assert changed["native"]["semantic_review"]["complete"]
    accept(s, 3)
    s.bridge.export(selection(s, 4), tmp_path / "cleared")
    exported = validate_export(tmp_path / "cleared")[0]
    assert exported["annotation"]["native"]["semantic_review"]["segments"][0]["object_text"] is None
    assert validate_export(tmp_path / "populated")[0] == original
    assert s.prediction_path.read_bytes() == before


@pytest.mark.parametrize("problem", ["orphan", "duplicate", "range", "multiple", "nonstring"])
def test_empty_object_still_requires_valid_region_binding(semantic, problem):
    s = semantic
    results = deepcopy(s.annotation["result"])
    text = next(r for r in results if r["from_name"] == "object_text")
    text["value"]["text"] = [""]
    if problem == "orphan":
        text["id"] = "missing"
    elif problem == "duplicate":
        results.append(deepcopy(text))
    elif problem == "range":
        text["value"]["ranges"] = [{"start": 2, "end": 10}]
    elif problem == "multiple":
        text["value"]["text"] = ["", ""]
    elif problem == "nonstring":
        text["value"]["text"] = [None]
    with pytest.raises(ValueError):
        decoded_review(results, s.spec, s.bundle["proxy"])


def test_failed_prediction_can_receive_manual_segments_without_changing_failure(semantic, tmp_path):
    s = semantic
    predictions = Predictions.model_validate_json(s.prediction_path.read_text())
    predictions.episodes[0].segments = []
    predictions.episodes[0].error = "Generated inference failure"
    atomic_json(s.prediction_path, predictions.model_dump())
    failed_bytes = s.prediction_path.read_bytes()
    s.bundle = prepare(
        s.store, s.plan_path, s.prediction_path, s.episode["id"], tmp_path / "failed-review"
    )
    assert s.bundle["task"]["predictions"] == []
    assert (
        "prediction failed: Generated inference failure" in s.bundle["task"]["data"]["provenance"]
    )
    s.task["data"] = s.bundle["task"]["data"]
    s.task["predictions"] = []
    manual_results = s.annotation["result"]
    s.annotation["result"] = []
    register(s)
    empty = refresh(s)
    assert not empty["native"]["semantic_review"]["complete"]
    with pytest.raises(ValueError, match="complete semantic coverage"):
        accept(s)
    s.annotation["result"] = manual_results
    s.annotation["updated_at"] = "2026-01-01T00:00:01Z"
    completed = refresh(s, 1)
    assert completed["native"]["semantic_review"]["complete"]
    assert completed["condition"] == "manual"
    assert completed["native"]["machine_predictions"] == []
    accept(s, 2)
    s.bridge.export(selection(s, 3), tmp_path / "manual-recovery")
    record = validate_export(tmp_path / "manual-recovery")[0]
    assert record["annotation"]["native"]["semantic_review"]["kind"] == "model_assisted_review"
    assert s.prediction_path.read_bytes() == failed_bytes
    _, failed, _ = bound_prediction(s.store, s.bundle["trusted_origin"])
    assert failed.segments == [] and failed.error == "Generated inference failure"


@pytest.mark.parametrize(
    "problem",
    [
        "missing_text",
        "duplicate_text",
        "orphan",
        "wrong_type",
        "wrong_target",
        "blank",
        "oversized",
        "multiple_text",
        "multi_range",
        "duplicate_region",
        "range_mismatch",
        "bool_frame",
    ],
)
def test_ambiguous_region_text_is_rejected(semantic, problem):
    s = semantic
    results = deepcopy(s.annotation["result"])
    timeline = results[0]
    text = results[1]
    if problem == "missing_text":
        results.remove(text)
    elif problem == "duplicate_text":
        results.append(deepcopy(text))
    elif problem == "orphan":
        text["id"] = "orphan"
    elif problem == "wrong_type":
        text["type"] = "choices"
    elif problem == "wrong_target":
        text["to_name"] = "other"
    elif problem == "blank":
        text["value"]["text"] = ["  "]
    elif problem == "oversized":
        text["value"]["text"] = ["a" * 2001]
    elif problem == "multiple_text":
        text["value"]["text"].append("other")
    elif problem == "multi_range":
        timeline["value"]["ranges"].append({"start": 11, "end": 12})
    elif problem == "duplicate_region":
        results.append(deepcopy(timeline))
    elif problem == "range_mismatch":
        text["value"]["ranges"] = [{"start": 2, "end": 10}]
    elif problem == "bool_frame":
        timeline["value"]["ranges"][0]["start"] = True
    with pytest.raises(ValueError):
        decoded_review(results, s.spec, s.bundle["proxy"])


def test_semantic_text_is_not_silently_accepted_in_legacy_reviews(semantic):
    with pytest.raises(ValueError, match="unsupported"):
        review_fields(semantic.annotation["result"])


def test_missing_coverage_can_be_saved_but_not_accepted(semantic):
    s = semantic
    register(s)
    for result in s.annotation["result"]:
        if result["id"] == "segment-0":
            result["value"]["ranges"][0]["end"] = 8
    draft = refresh(s)
    assert not draft["native"]["semantic_review"]["complete"]
    with pytest.raises(ValueError, match="complete semantic coverage"):
        accept(s)
    assert s.store.latest(s.episode["id"])["status"] == "draft"


def test_snapshot_mutation_and_symlinks_are_rejected(semantic, tmp_path):
    s = semantic
    origin = s.bundle["trusted_origin"]["semantic_prediction"]
    path = s.store.root / "semantic_sources" / origin["snapshot_id"] / "predictions.json"
    original = path.read_bytes()
    path.write_text("{}")
    with pytest.raises(ValueError, match="snapshot changed"):
        bound_prediction(s.store, s.bundle["trusted_origin"])
    path.unlink()
    outside = tmp_path / "outside.json"
    outside.write_bytes(original)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        bound_prediction(s.store, s.bundle["trusted_origin"])
    assert outside.read_bytes() == original


def test_changed_projected_prediction_cannot_be_registered(semantic):
    s = semantic
    for result in (
        s.bundle["task"]["predictions"][0]["result"][1],
        s.task["predictions"][0]["result"][1],
    ):
        result["value"]["text"] = ["Changed before registration"]
    with pytest.raises(ValueError, match="uncorrected prediction snapshot"):
        register(s)


def test_coarse_frame_proxy_is_rejected(semantic):
    s = semantic
    proxy = deepcopy(s.bundle["proxy"])
    proxy["frames"].pop()
    with pytest.raises(ValueError, match="one-to-one"):
        decoded_review(s.annotation["result"], s.spec, proxy)


@pytest.mark.parametrize("field,value", [("score", 0.99), ("model_version", "unbound-model")])
def test_changed_prediction_identity_cannot_be_registered(semantic, field, value):
    s = semantic
    s.bundle["task"]["predictions"][0][field] = value
    s.task["predictions"][0][field] = value
    with pytest.raises(ValueError, match="uncorrected prediction snapshot"):
        register(s)
