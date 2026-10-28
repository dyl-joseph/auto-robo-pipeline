import json
import subprocess
import sys
from copy import deepcopy

import pytest

from robo.core import atomic_json, digest
from robo.semantic import EvaluationPlan, Predictions, References, SemanticSegment
from robo.semantic_evaluation import evaluate


def fixture_inputs():
    episodes = []
    for key, split in (("train", "train"), ("dev", "development"), ("a", "test"), ("b", "test")):
        episodes.append(
            {
                "episode_id": key,
                "dataset": "generated",
                "release": "1",
                "dataset_episode_id": key,
                "session_id": key,
                "camera_id": "front",
                "source_sha256": digest(["source", key]),
                "media_sha256": digest(["media", key]),
                "split": split,
                "frame_boundaries_s": list(range(7)),
                "timing": "inferred_nominal_fps",
            }
        )
    plan = EvaluationPlan.model_validate(
        {
            "taxonomy_id": "generated-test-v1",
            "normalized_actions": ["reach", "grasp", "release", "unknown", "not_observable"],
            "annotation_guidelines_sha256": digest("generated rules"),
            "identity_audit": {
                "status": "synthetic_fixture",
                "reviewer": "generated test",
                "evidence_sha256": digest("generated identities"),
            },
            "episodes": episodes,
        }
    )

    def segment(key, start, end, action, *, prediction=False, object_text="cup"):
        return {
            "episode_id": key,
            "start_s": start,
            "end_s": end,
            "source_frames": {"start": start, "end_exclusive": end, "evidence": [start]},
            "subtask_text": f"{action} the {object_text}",
            "normalized_action": action,
            "object_text": object_text,
            "uncertainty": 0.2 if prediction else None,
            "provenance": {
                "kind": "synthetic_fixture",
                "producer": "generated predictor" if prediction else "generated reference",
                "artifact_sha256": digest(["prediction" if prediction else "reference", key]),
            },
        }

    predicted_a = [
        segment("a", 0, 2, "reach", prediction=True),
        segment("a", 2, 5, "grasp", prediction=True, object_text="bottle"),
        segment("a", 5, 6, "release", prediction=True),
    ]
    predicted_a[1]["uncertainty"] = 0.9
    predictions = Predictions.model_validate(
        {
            "plan_sha256": plan.plan_sha256,
            "run_id": "generated-test",
            "producer": "generated predictor",
            "code_revision": "generated test revision",
            "config_sha256": digest("generated config"),
            "training_episode_ids": ["train"],
            "tuning_episode_ids": ["dev"],
            "reference_access": "none",
            "episodes": [
                {"episode_id": "a", "segments": predicted_a},
                {"episode_id": "b", "segments": [], "error": "deliberate fixture failure"},
            ],
        }
    )
    references = References.model_validate(
        {
            "plan_sha256": plan.plan_sha256,
            "evidence_kind": "synthetic_fixture",
            "episodes": [
                {
                    "episode_id": key,
                    "segments": [segment(key, 0, 3, "reach"), segment(key, 3, 6, "grasp")],
                    "audit": {
                        "status": "synthetic_fixture",
                        "method": "synthetic_fixture",
                        "annotators": ["generated test"],
                        "source_uri": "fixture://" + key,
                        "source_sha256": digest(["reference", key]),
                        "license": "generated fixture",
                        "license_url": "fixture://license",
                        "prediction_exposure": False,
                    },
                }
                for key in ("a", "b")
            ],
        }
    )
    return plan, predictions, references


def test_raw_predictions_misses_objects_boundaries_and_review_counts():
    plan, predictions, references = fixture_inputs()
    report = evaluate(plan, predictions, references)
    assert report["evidence_kind"] == "synthetic_fixture"
    assert report["failed_predictions"] == 1 and report["review_requests"] == 2
    assert report["automatic_approvals"] == 0 and report["human_active_seconds"] is None
    result = report["metrics"]["action"]["0.5"]
    assert (result["tp"], result["fp"], result["fn"]) == (2, 1, 2)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 0.5 and result["f1"] == pytest.approx(4 / 7)
    assert result["matched_boundary_median_s"] == 1 and result["matched_boundary_p90_s"] == 1
    assert result["per_action"]["release"]["fp"] == 1
    assert result["matched_object_text_exact_agreement"] == 0.5
    strict = report["metrics"]["action_object"]["0.5"]
    assert (strict["tp"], strict["fp"], strict["fn"]) == (1, 2, 3)
    assert report["metrics"]["action"]["0.7"]["tp"] == 0
    assert report["episodes"][0]["review_reasons"] == ["uncertain_segment"]
    assert report["episodes"][1]["covered_frames"] == 0
    assert report == evaluate(plan, predictions, references)


@pytest.mark.parametrize(
    "field",
    ["session_id", "dataset_episode_id", "source_sha256", "media_sha256", "near_duplicate_groups"],
)
def test_shared_sessions_views_and_duplicates_cannot_cross_splits(field):
    plan, _, _ = fixture_inputs()
    body = plan.model_dump()
    if field == "near_duplicate_groups":
        body["episodes"][0][field] = ["audit-group-7"]
        body["episodes"][2][field] = ["audit-group-7"]
    else:
        body["episodes"][2][field] = body["episodes"][0][field]
    if field == "dataset_episode_id":
        body["episodes"][2]["camera_id"] = "wrist"
        body["episodes"][2]["release"] = "repackaged-version"
    with pytest.raises(ValueError, match="split leakage"):
        EvaluationPlan.model_validate(body)


@pytest.mark.parametrize("change", ["time", "overlap", "outside", "taxonomy", "episode", "gap"])
def test_segments_must_match_source_frames_and_complete_reference(change):
    plan, predictions, references = fixture_inputs()
    body = references.model_dump()
    segment = body["episodes"][0]["segments"][1]
    if change == "time":
        segment["start_s"] = 3.1
    elif change in {"overlap", "outside", "gap"}:
        start, end = {"overlap": (2, 6), "outside": (3, 7), "gap": (4, 6)}[change]
        segment.update(
            start_s=start,
            end_s=end,
            source_frames={"start": start, "end_exclusive": end, "evidence": [start]},
        )
    elif change == "taxonomy":
        segment["normalized_action"] = "invented_action"
    else:
        segment["episode_id"] = "b"
    with pytest.raises(ValueError):
        evaluate(plan, predictions, References.model_validate(body))


@pytest.mark.parametrize(
    "change",
    ["negative", "nan", "fractional_frame", "evidence_outside", "blank_text", "uncertainty"],
)
def test_segment_contract_rejects_invalid_values(change):
    _, predictions, _ = fixture_inputs()
    body = predictions.episodes[0].segments[0].model_dump()
    if change == "negative":
        body["start_s"] = -1
    elif change == "nan":
        body["end_s"] = float("nan")
    elif change == "fractional_frame":
        body["source_frames"]["start"] = 0.5
    elif change == "evidence_outside":
        body["source_frames"]["evidence"] = [4]
    elif change == "blank_text":
        body["subtask_text"] = "  "
    else:
        body["uncertainty"] = 1.1
    with pytest.raises(ValueError):
        SemanticSegment.model_validate(body)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "duplicate",
        "corrected",
        "training",
        "tuning",
        "read_references",
        "plan",
        "human_prediction",
        "no_uncertainty",
    ],
)
def test_predictions_cannot_hide_failures_or_use_test_references(change):
    plan, predictions, references = fixture_inputs()
    body = predictions.model_dump()
    if change == "missing":
        body["episodes"].pop()
    elif change == "extra":
        body["episodes"].append({"episode_id": "other", "segments": []})
    elif change == "duplicate":
        body["episodes"].append(deepcopy(body["episodes"][0]))
    elif change == "corrected":
        body["kind"] = "human_corrected"
    elif change in {"training", "tuning"}:
        body[change + "_episode_ids"] = ["a"]
    elif change == "read_references":
        body["reference_access"] = "provided"
    elif change == "plan":
        body["plan_sha256"] = digest("different plan")
    elif change == "human_prediction":
        body["episodes"][0]["segments"][0]["provenance"]["kind"] = "independent_human"
    else:
        body["episodes"][0]["segments"][0]["uncertainty"] = None
    with pytest.raises(ValueError):
        evaluate(plan, Predictions.model_validate(body), references)


@pytest.mark.parametrize(
    "change",
    ["exposed", "model_assisted", "unverified", "identity", "self_reference", "provenance"],
)
def test_independent_reference_gate_requires_audits(change):
    plan, predictions, references = fixture_inputs()
    p = plan.model_dump()
    p["identity_audit"]["status"] = "human_verified"
    if change == "identity":
        p["identity_audit"]["status"] = "unverified"
    plan = EvaluationPlan.model_validate(p)
    pred = predictions.model_dump()
    pred["plan_sha256"] = plan.plan_sha256
    for episode in pred["episodes"]:
        for segment in episode["segments"]:
            segment["provenance"]["kind"] = "baseline"
    ref = references.model_dump()
    ref.update(plan_sha256=plan.plan_sha256, evidence_kind="independent_human_reference")
    for episode in ref["episodes"]:
        episode["audit"].update(status="human_verified", method="blind_manual")
        for segment in episode["segments"]:
            segment["provenance"]["kind"] = "independent_human"
    audit = ref["episodes"][0]["audit"]
    if change == "exposed":
        audit["prediction_exposure"] = True
    elif change == "model_assisted":
        audit["method"] = "model_assisted"
    elif change == "unverified":
        audit["status"] = "unverified"
    elif change == "self_reference":
        audit["source_sha256"] = pred["episodes"][0]["segments"][0]["provenance"]["artifact_sha256"]
    elif change == "provenance":
        ref["episodes"][0]["segments"][0]["provenance"]["artifact_sha256"] = digest("unrelated")
    with pytest.raises(ValueError):
        evaluate(plan, Predictions.model_validate(pred), References.model_validate(ref))


def test_exact_text_diagnostics_do_not_claim_paraphrase_accuracy():
    plan, predictions, references = fixture_inputs()
    p = predictions.model_dump()
    p["episodes"][0]["segments"][0].update(subtask_text=" REACH   the CUP ", object_text=" CUP ")
    result = evaluate(plan, Predictions.model_validate(p), references)
    assert result["metrics"]["action"]["0.5"]["matched_subtask_text_exact_agreement"] == 0.5
    p["episodes"][0]["segments"][0]["subtask_text"] = "Move the hand toward the cup"
    result = evaluate(plan, Predictions.model_validate(p), references)
    assert result["metrics"]["action"]["0.5"]["matched_subtask_text_exact_agreement"] == 0
    assert result["metrics"]["action"]["0.5"]["tp"] == 2


def test_cli_keeps_inputs_unchanged_and_refuses_overwrite(tmp_path):
    plan, predictions, references = fixture_inputs()
    inputs = []
    for name, body in (("plan", plan), ("predictions", predictions), ("references", references)):
        path = tmp_path / (name + ".json")
        atomic_json(path, body.model_dump())
        inputs.append(path)
    original = [p.read_bytes() for p in inputs]
    output = tmp_path / "report.json"
    command = [sys.executable, "-m", "robo.semantic_evaluation", *map(str, inputs), str(output)]
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    report = json.loads(output.read_text())
    assert report == evaluate(plan, predictions, references)
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    assert [p.read_bytes() for p in inputs] == original
    output.write_text("{}")
    failed = subprocess.run(command, capture_output=True, timeout=30)
    assert failed.returncode != 0 and b"different evaluation" in failed.stderr


def test_existing_episode_identity_and_integer_pts_survive_contract_mapping(episode):
    from robo.semantic import episode_spec

    _, source = episode
    source["video_timing"] = "inferred_nominal_fps"
    spec = episode_spec(
        source,
        dataset="generated",
        dataset_episode_id="original-episode-1",
        camera_id="front",
        split="test",
    )
    assert spec.episode_id == source["id"]
    assert spec.session_id == source["session"] and spec.source_sha256 == source["content_hash"]
    assert spec.media_sha256 == source["video_sha256"]
    assert spec.frame_boundaries_s == [i / 10 for i in range(21)]
    assert spec.timing == "inferred_nominal_fps"
