import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from robo.core import atomic_json, digest
from robo.ingest import import_episode
from robo.semantic import EvaluationPlan, Predictions, episode_spec
from robo.semantic_baseline import BaselineConfig, run, segments_from_evidence, state_evidence


def samples():
    times = [i / 10 for i in range(20)]
    joints = [[0.1 * max(0, min(i - 3, 4))] * 7 for i in range(20)]
    gripper = [[0.0 if i < 9 or i >= 14 else 0.8] for i in range(20)]
    return [
        {
            "name": name,
            "modality": "state",
            "units": units,
            "timestamps_s": times,
            "values": values,
            "clock_evidence": "Generated synchronized test samples",
            "tolerance_s": 0.06,
        }
        for name, units, values in (
            ("state_joint_position", "rad", joints),
            ("state_gripper_position", "normalized", gripper),
        )
    ]


def setup(fixture):
    store, path, body = fixture
    body["streams"] = samples()
    atomic_json(path, body)
    episode = import_episode(path, store)
    spec = episode_spec(
        episode,
        dataset="generated",
        dataset_episode_id="generated-1",
        camera_id="front",
        split="test",
    )
    plan = EvaluationPlan(
        taxonomy_id="coarse-test-v1",
        normalized_actions=["unknown", "not_observable", "motion", "hold", "grasp", "release"],
        annotation_guidelines_sha256=digest("generated guidelines"),
        identity_audit={
            "status": "synthetic_fixture",
            "reviewer": "generated test",
            "evidence_sha256": digest("generated identities"),
        },
        episodes=[spec],
    )
    return store, episode, spec, plan


def test_state_changes_use_named_observations_and_preserve_source_boundaries(fixture):
    _, episode, spec, _ = setup(fixture)
    config = BaselineConfig()
    raw = state_evidence(episode, spec, config)
    labels = raw["frame_labels"]
    assert labels[0] == "unknown"
    assert labels[1:4] == ["hold"] * 3
    assert labels[4:8] == ["motion"] * 4
    assert labels[9] == "grasp" and labels[14] == "release"
    segments = segments_from_evidence(raw, spec, config)
    assert segments[0].source_frames.start == 0 and segments[-1].source_frames.end_exclusive == 20
    assert all(s.object_text is None and s.uncertainty >= 0.6 for s in segments)
    assert all(s.provenance.artifact_sha256 == digest(raw) for s in segments)
    # Changing high-level instructions or command fields cannot alter a state-only baseline.
    changed = deepcopy(episode)
    changed["task"] = "Misleading generated task: pick up a different object"
    changed["streams"].append(
        {**deepcopy(changed["streams"][0]), "name": "action_joint_position", "modality": "action"}
    )
    assert state_evidence(changed, spec, config) == raw


def test_held_samples_are_unknown_instead_of_inventing_stillness(fixture):
    _, episode, spec, _ = setup(fixture)
    for stream in episode["streams"]:
        stream["timestamps_s"] = stream["timestamps_s"][::2]
        stream["values"] = stream["values"][::2]
        stream["tolerance_s"] = 0.11
    raw = state_evidence(episode, spec, BaselineConfig())
    assert all(raw["frame_labels"][i] == "unknown" for i in range(1, 20, 2))
    assert raw["frame_labels"][6] == "motion"


@pytest.mark.parametrize("problem", ["gap", "units", "missing", "shape", "gripper"])
def test_invalid_or_missing_state_is_not_filled(fixture, problem):
    _, episode, spec, _ = setup(fixture)
    if problem == "gap":
        for stream in episode["streams"]:
            del stream["timestamps_s"][4]
            del stream["values"][4]
    elif problem == "units":
        episode["streams"][0]["units"] = "rad/s"
    elif problem == "missing":
        episode["streams"].pop()
    elif problem == "shape":
        episode["streams"][0]["values"] = [[1.0] * 6 for _ in range(20)]
    else:
        episode["streams"][1]["values"][2] = [2.0]
    with pytest.raises(ValueError):
        state_evidence(episode, spec, BaselineConfig())


def test_prediction_run_never_reads_reviews_and_reruns_without_new_outputs(
    fixture, tmp_path, monkeypatch
):
    store, episode, spec, plan = setup(fixture)
    monkeypatch.setattr(store, "latest", lambda *a: pytest.fail("baseline read review labels"))
    monkeypatch.setattr(store, "history", lambda *a: pytest.fail("baseline read review labels"))
    output = tmp_path / "predictions"
    report = run(plan, store, output)
    assert report["failed_predictions"] == 0 and not report["reference_labels_read"]
    assert not report["semantic_accuracy_measured"]
    predictions = Predictions.model_validate_json((output / "predictions.json").read_bytes())
    assert predictions.episodes[0].segments[0].normalized_action == "unknown"
    assert report == run(plan, store, output)
    evidence = next(output.glob("*.evidence.json"))
    evidence.write_text("{}")
    with pytest.raises(ValueError, match="output changed"):
        run(plan, store, output)


def test_failed_episode_remains_in_predictions(fixture, tmp_path):
    store, episode, spec, plan = setup(fixture)
    (store.root / episode["media"]).write_bytes(b"changed input")
    output = tmp_path / "predictions"
    report = run(plan, store, output)
    assert report["episode_count"] == report["failed_predictions"] == 1
    predictions = json.loads((output / "predictions.json").read_text())
    assert predictions["episodes"][0]["episode_id"] == spec.episode_id
    assert predictions["episodes"][0]["segments"] == []
    assert "media hash" in predictions["episodes"][0]["error"]


@pytest.mark.parametrize("split", ["train", "test"])
def test_tuning_declarations_cannot_include_annotated_episode(fixture, tmp_path, split):
    store, _, spec, plan = setup(fixture)
    body = plan.model_dump()
    body["episodes"][0]["split"] = split
    plan = EvaluationPlan.model_validate(body)
    with pytest.raises(ValueError, match="tuning episodes"):
        run(
            plan,
            store,
            tmp_path / "predictions",
            split=split,
            config=BaselineConfig(tuning_episode_ids=[spec.episode_id]),
        )


def test_training_annotation_cli_preserves_train_split(fixture, tmp_path):
    store, _, spec, plan = setup(fixture)
    body = plan.model_dump()
    body["episodes"][0]["split"] = "train"
    plan = EvaluationPlan.model_validate(body)
    plan_path, output = tmp_path / "train-plan.json", tmp_path / "train-predictions"
    atomic_json(plan_path, plan.model_dump())
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "robo.semantic_baseline",
            str(plan_path),
            str(store.root),
            str(output),
            "--split",
            "train",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["split"] == "train"
    assert manifest["purpose"] == "training_annotations"
    assert manifest["failed_predictions"] == 0
    predictions = Predictions.model_validate_json((output / "predictions.json").read_bytes())
    assert predictions.plan_sha256 == plan.plan_sha256
    assert predictions.episodes[0].episode_id == spec.episode_id
    assert predictions.episodes[0].segments[-1].source_frames.end_exclusive == 20


def test_training_annotations_cannot_be_scored_as_a_test(fixture, tmp_path):
    from robo.semantic import References
    from robo.semantic_evaluation import validate_inputs

    store, _, _, plan = setup(fixture)
    body = plan.model_dump()
    body["episodes"][0]["split"] = "train"
    plan = EvaluationPlan.model_validate(body)
    output = tmp_path / "train-predictions"
    run(plan, store, output, split="train")
    predictions = Predictions.model_validate_json((output / "predictions.json").read_bytes())
    entry = plan.episodes[0]
    reference_hash = digest("independent generated reference fixture")
    references = References(
        plan_sha256=plan.plan_sha256,
        evidence_kind="synthetic_fixture",
        episodes=[
            {
                "episode_id": entry.episode_id,
                "segments": [
                    {
                        "episode_id": entry.episode_id,
                        "start_s": 0,
                        "end_s": 2,
                        "source_frames": {"start": 0, "end_exclusive": 20, "evidence": [0, 19]},
                        "subtask_text": "Generated reference fixture",
                        "normalized_action": "unknown",
                        "object_text": None,
                        "uncertainty": None,
                        "provenance": {
                            "kind": "synthetic_fixture",
                            "producer": "generated reference",
                            "artifact_sha256": reference_hash,
                        },
                    }
                ],
                "audit": {
                    "status": "synthetic_fixture",
                    "method": "synthetic_fixture",
                    "annotators": ["generated fixture"],
                    "source_uri": "fixture://train-reference",
                    "source_sha256": reference_hash,
                    "license": "generated fixture",
                    "license_url": "fixture://license",
                    "prediction_exposure": False,
                },
            }
        ],
    )
    with pytest.raises(ValueError, match="every episode in the chosen split"):
        validate_inputs(plan, predictions, references, "test")
    with pytest.raises(ValueError, match="evaluate only"):
        validate_inputs(plan, predictions, references, "train")


def test_failed_baseline_cli_returns_failure_and_keeps_artifacts(fixture, tmp_path):
    store, episode, _, plan = setup(fixture)
    plan_path, output = tmp_path / "plan.json", tmp_path / "failed-predictions"
    atomic_json(plan_path, plan.model_dump())
    (store.root / episode["media"]).write_bytes(b"changed source")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "robo.semantic_baseline",
            str(plan_path),
            str(store.root),
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout)["failed_predictions"] == 1
    predictions = Predictions.model_validate_json((output / "predictions.json").read_bytes())
    assert "media hash" in predictions.episodes[0].error
    assert not predictions.episodes[0].segments
