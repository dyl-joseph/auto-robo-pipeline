from copy import deepcopy
from types import SimpleNamespace

import pytest
from label_studio_sdk.types.annotation import Annotation
from label_studio_sdk.types.lse_task import LseTask

from robo.analysis import suggest
from robo.core import Edit
from robo.evaluation import study_metrics
from robo.export import export_dataset, validate_export
from robo.labelstudio import prediction_results, review_fields, task_bundle
from robo.native_review import NativeReviewBridge, local_instance
from robo.store import Conflict


@pytest.fixture
def native(episode):
    store, episode = episode
    suggest(store, episode["id"])
    bundle = task_bundle(store, episode["id"])
    timestamp = "2026-01-01T00:00:00Z"
    annotation = {
        "id": 3,
        "task": 2,
        "project": 1,
        "completed_by": 7,
        "created_at": timestamp,
        "updated_at": timestamp,
        "lead_time": 12.5,
        "was_cancelled": False,
        "result": prediction_results(
            [{"start_s": 0, "end_s": 1, "label": "reach"}], bundle["proxy"]
        )
        + [
            {
                "from_name": "outcome",
                "to_name": "video",
                "type": "choices",
                "value": {"choices": ["failure"]},
            },
            {
                "from_name": "quality",
                "to_name": "video",
                "type": "choices",
                "value": {"choices": ["recovery", "occluded"]},
            },
            {
                "from_name": "decision_note",
                "to_name": "video",
                "type": "textarea",
                "value": {"text": ["Fixture correction, not a human study."]},
            },
        ],
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
                **bundle["task"]["predictions"][0],
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
    bridge.register(bundle, 1, 2)
    return bridge, store, episode, bundle, task, annotation


def refresh(native, revision=0):
    bridge, _, episode, *_ = native
    return bridge.refresh(
        episode["id"], 3, revision, "fixture-importer", "Inspect native correction"
    )


def decide(native, revision=1, decision="accept"):
    bridge, _, episode, *_ = native
    return bridge.decide(
        episode["id"], revision, decision, "fixture-approver", "Explicit test decision"
    )


def selection(native, revision=2):
    return [{"episode_id": native[2]["id"], "revision": revision, "split": "train"}]


def test_native_fields_unknowns_and_invalid_controls():
    assert review_fields([]) == {"outcome": "unknown", "quality_flags": [], "reviewer_evidence": []}
    outcome = {
        "from_name": "outcome",
        "to_name": "video",
        "type": "choices",
        "value": {"choices": ["unknown"]},
    }
    assert review_fields([outcome])["outcome"] == "unknown"
    with pytest.raises(ValueError, match="duplicate"):
        review_fields([outcome, outcome])
    for values in (["success", "failure"], ["success_guessed"], "success", [True]):
        invalid = {**outcome, "value": {"choices": values}}
        with pytest.raises(ValueError):
            review_fields([invalid])
    with pytest.raises(ValueError, match="unsupported"):
        review_fields([{**outcome, "to_name": "another_video"}])


def test_native_draft_approval_and_idempotent_export(native, tmp_path):
    bridge, store, episode, bundle, task, annotation = native
    assert bridge.register(bundle, 1, 2) == store.native_binding(episode["id"])
    with pytest.raises(ValueError, match="draft"):
        decide(native, 0)
    draft = refresh(native)
    assert draft["status"] == "draft"
    assert draft["outcome"] == "failure"
    assert draft["quality_flags"] == ["occluded", "recovery"]
    assert draft["native"]["reviewer_evidence"] == ["Fixture correction, not a human study."]
    assert draft["native"]["native_reviewer_id"] == 7
    assert draft["active_s"] is None and draft["wall_s"] is None
    assert draft["native"]["native_lead_time_s"] == 12.5
    assert refresh(native, 1) == draft
    with pytest.raises(Conflict, match="Stale revision"):
        refresh(native, 0)
    with pytest.raises(ValueError, match="accepted"):
        bridge.export(selection(native, 1), tmp_path / "draft")
    accepted = decide(native)
    assert accepted["native"]["snapshot_hash"] == draft["native"]["snapshot_hash"]
    assert decide(native, 2) == accepted
    result = bridge.export(selection(native), tmp_path / "release")
    assert result["manifest"] == bridge.export(selection(native), tmp_path / "release")["manifest"]
    records = validate_export(tmp_path / "release")
    assert records[0]["annotation"] == accepted
    assert records[0]["episode"]["original_pts"] == episode["original_pts"]
    assert (
        records[0]["annotation"]["native"]["machine_predictions"][0]["result"]
        == task["predictions"][0]["result"]
    )
    assert len(store.history(episode["id"])) == 2
    with pytest.raises(ValueError, match="authenticated freshness"):
        export_dataset(store, selection(native), tmp_path / "bypass")
    with pytest.raises(ValueError, match="authenticated freshness"):
        store.commit(
            episode["id"],
            Edit(expected_revision=2, actor="legacy", rationale="bypass", segments=[]),
        )


def test_changed_native_content_invalidates_new_exports_not_historical_exports(native, tmp_path):
    bridge, store, episode, _, _, annotation = native
    refresh(native)
    decide(native)
    bridge.export(selection(native), tmp_path / "historical")
    original = (tmp_path / "historical" / "episodes.jsonl").read_bytes()
    annotation["result"][0]["value"]["ranges"][0]["end"] = 8
    annotation["updated_at"] = "2026-01-01T00:00:01Z"
    with pytest.raises(Conflict, match="changed"):
        bridge.export(selection(native), tmp_path / "historical")
    with pytest.raises(Conflict, match="new draft"):
        decide(native, 2)
    draft = refresh(native, 2)
    assert draft["status"] == "draft" and draft["segments"][0]["end_s"] == 0.8
    assert draft["native"]["boundary_evidence"][0]["end"]["episode_time_s"] == 0.8
    with pytest.raises(Conflict, match="stale"):
        decide(native, 2)
    decide(native, 3)
    bridge.export(selection(native, 4), tmp_path / "new")
    assert validate_export(tmp_path / "historical")[0]["annotation"]["revision"] == 2
    assert (tmp_path / "historical" / "episodes.jsonl").read_bytes() == original
    assert [r["status"] for r in store.history(episode["id"])] == [
        "draft",
        "accepted",
        "draft",
        "accepted",
    ]


def test_rejection_stays_rejected_on_unchanged_import(native, tmp_path):
    refresh(native)
    rejected = decide(native, decision="reject")
    assert rejected["status"] == "excluded"
    assert refresh(native, 2) == rejected
    with pytest.raises(ValueError, match="draft"):
        decide(native, 2)
    with pytest.raises(ValueError, match="accepted"):
        native[0].export(selection(native), tmp_path / "rejected")


@pytest.mark.parametrize("change", ["result", "version", "id", "updated_at", "deleted"])
def test_changed_machine_predictions_rejected(native, tmp_path, change):
    bridge, _, _, _, task, _ = native
    refresh(native)
    decide(native)
    if change == "deleted":
        task["predictions"] = []
    elif change == "result":
        task["predictions"][0]["result"][0]["value"]["timelinelabels"] = ["recover"]
    else:
        field = "model_version" if change == "version" else change
        task["predictions"][0][field] = (
            99
            if field == "id"
            else "2026-01-01T00:00:01Z"
            if field == "updated_at"
            else "different"
        )
    with pytest.raises(Conflict, match="predictions changed"):
        bridge.export(selection(native), tmp_path / "changed")
    with pytest.raises(Conflict, match="predictions changed"):
        refresh(native, 2)


@pytest.mark.parametrize("change", ["draft", "cancelled", "deleted", "rejected", "wrong_task"])
def test_unsubmitted_or_invalid_annotations_rejected(native, change):
    _, _, _, _, task, annotation = native
    if change == "draft":
        task["drafts"] = [{"result": []}]
    elif change == "cancelled":
        annotation["was_cancelled"] = True
    elif change == "deleted":
        task["annotations"] = []
    elif change == "rejected":
        annotation["last_action"] = "rejected"
    else:
        annotation["task"] = 44
    with pytest.raises(Conflict):
        refresh(native)


def test_native_timestamp_normalization_and_regression(native):
    annotation = native[-1]
    first = refresh(native)
    annotation["updated_at"] = "2025-12-31T19:00:00-05:00"
    assert refresh(native, 1) == first
    annotation["updated_at"] = "2025-12-31T23:59:59Z"
    with pytest.raises(Conflict, match="regressed"):
        refresh(native, 1)
    annotation["updated_at"] = "2026-01-01T00:00:00Z"
    annotation["result"][-1]["value"]["text"] = ["Changed without a new timestamp"]
    with pytest.raises(Conflict, match="without a new revision"):
        refresh(native, 1)


def test_unknown_native_timing_does_not_become_free_human_effort(native):
    draft = refresh(native)
    report = study_metrics(
        {
            "evidence_kind": "synthetic_fixture",
            "episodes": [
                {
                    "id": "native",
                    "condition": "assisted",
                    "review_history": [draft],
                }
            ],
        }
    )
    assert report["conditions"] == {}
    assert report["time_reduction_point_estimate"] is None
    assert report["episodes_with_unmeasured_time"] == ["native"]


def test_remote_or_credential_bearing_instance_rejected():
    for url in ("https://example.com", "http://user:secret@localhost", "http://localhost/?token=x"):
        with pytest.raises(ValueError, match="loopback"):
            local_instance(url)


def test_changed_project_clock_and_unbound_parent_rejected(native):
    bridge, _, episode, bundle, _, annotation = native
    original = bundle["label_config"]
    bundle["label_config"] = original.replace('frameRate="10"', 'frameRate="30"')
    with pytest.raises(Conflict, match="configuration changed"):
        refresh(native)
    bundle["label_config"] = original
    annotation["parent_prediction"] = 999
    with pytest.raises(Conflict, match="unbound parent"):
        refresh(native)


@pytest.mark.parametrize("change", ["annotation", "prediction", "draft", "task"])
def test_changes_during_sdk_observation_fail_closed(native, change):
    bridge, _, _, _, task, annotation = native
    get = bridge.sdk.annotations.get

    def change_during_read(annotation_id):
        result = get(annotation_id)
        if change == "annotation":
            annotation["updated_at"] = "2026-01-01T00:00:02Z"
        elif change == "prediction":
            task["predictions"][0]["score"] = 0.99
        elif change == "draft":
            task["drafts"] = [{"result": []}]
        else:
            task["data"]["task"] = "changed"
        return result

    bridge.sdk.annotations.get = change_during_read
    with pytest.raises(Conflict, match="during freshness"):
        refresh(native)


def test_uncertain_native_quality_cannot_be_accepted(native):
    native[-1]["result"][2]["value"]["choices"] = ["uncertain"]
    refresh(native)
    with pytest.raises(ValueError, match="resolve uncertainty"):
        decide(native)


def test_resave_invalidates_acceptance_even_without_label_changes(native, tmp_path):
    refresh(native)
    decide(native)
    native[-1]["updated_at"] = "2026-01-01T00:00:01Z"
    with pytest.raises(Conflict, match="changed"):
        native[0].export(selection(native), tmp_path / "stale-resave")
    assert refresh(native, 2)["status"] == "draft"


def test_later_active_machine_run_does_not_rebind_accepted_provenance(native, tmp_path):
    from robo.analysis import latest_run
    from robo.core import atomic_json, digest

    bridge, store, episode, *_ = native
    original = refresh(native)
    decide(native)
    run = deepcopy(latest_run(store, episode["id"]))
    run["config"]["test_new_run"] = True
    run["id"] = digest(run["config"])[:24]
    run["artifact_hash"] = digest({k: v for k, v in run.items() if k != "artifact_hash"})
    directory = store.root / "runs" / episode["id"]
    atomic_json(directory / f"{run['id']}.json", run)
    atomic_json(directory / "active.json", {"id": run["id"]})
    bridge.export(selection(native), tmp_path / "pinned")
    record = validate_export(tmp_path / "pinned")[0]
    assert record["annotation"]["suggestion_run"] == original["suggestion_run"]
    assert record["suggestion_artifact_hash"] != run["artifact_hash"]
