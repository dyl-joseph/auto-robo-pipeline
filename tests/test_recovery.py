import json
from copy import deepcopy

import numpy as np
import pytest

from robo.core import atomic_json, file_hash
from robo.lerobot_plan import TrainingSchema, plan_inputs
from robo.pipeline import run_batch
from robo.recovery import data_findings, normalize_source


def stream():
    return {
        "name": "joint",
        "modality": "state",
        "units": "rad",
        "timestamps_s": [0.0, 0.1, 0.2],
        "values": [[1.0], [2.0], [3.0]],
        "offset_s": 0,
        "tolerance_s": 0.04,
        "clock_evidence": "shared recording clock",
    }


def test_recovery_pairs_rows_and_never_changes_source(fixture):
    store, manifest, body = fixture
    data = stream()
    data["timestamps_s"] = [0.2, 0.1, 0, 0.1]
    data["values"] = [[3], [2], [1], [2]]
    body["streams"] = [data]
    atomic_json(manifest, body)
    original = file_hash(manifest)
    output = store.root / "normalized.json"
    report = normalize_source(manifest, output)

    normalized = json.loads(output.read_text())
    assert normalized["streams"][0]["timestamps_s"] == [0, 0.1, 0.2]
    assert normalized["streams"][0]["values"] == [[1], [2], [3]]
    assert [r["code"] for r in report["repairs"]] == [
        "sorted_timestamp_rows",
        "removed_exact_duplicate_rows",
    ]
    assert file_hash(manifest) == original
    assert report["invented_samples"] == report["interpolated_samples"] == 0
    assert normalize_source(manifest, output) == report


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda s: s.update(timestamps_s=[0, 0, 0.2]), "conflicting values"),
        (lambda s: s.pop("offset_s"), "explicit clock offset"),
        (lambda s: s.update(clock_evidence="unknown"), "clock basis is unknown"),
        (lambda s: s.update(timestamps_s=[0, float("nan"), 0.2]), "finite numbers"),
    ],
)
def test_ambiguous_rows_require_input(fixture, mutation, reason):

    store, manifest, body = fixture
    body["streams"] = [stream()]
    mutation(body["streams"][0])
    manifest.write_text(json.dumps(body))
    with pytest.raises(ValueError, match=reason):
        normalize_source(manifest, store.root / "normalized.json")
    assert not (store.root / "normalized.json").exists()


def test_declared_offset_is_used_without_rewriting_sample_times(fixture):
    store, manifest, body = fixture
    sample = stream()
    sample["timestamps_s"] = [-0.25, -0.15, -0.05]
    sample["offset_s"] = 0.25
    body["streams"] = [sample]
    atomic_json(manifest, body)
    inbox = manifest.parent / "inbox"
    inbox.mkdir()
    body["video"] = str(manifest.parent / body["video"])
    atomic_json(inbox / "source.json", body)
    batch = run_batch(inbox, store)
    episode = store.episode(batch["items"][0]["episode_id"])
    assert episode["streams"][0]["timestamps_s"] == sample["timestamps_s"]
    assert np.allclose(episode["validation"]["alignment"][0]["error_s"][:3], 0)
    assert batch["items"][0]["data_findings"][0]["code"] == "missing_aligned_samples"
    assert batch["metrics"]["data_review_requests"] == 1
    assert "missing_aligned_samples" in batch["review_queue"][0]["reasons"]
    assert batch["metrics"]["automatic_approvals"] == 0


def test_nominal_frame_gaps_are_flagged_but_vfr_is_not():
    episode = {
        "validation": {"alignment": []},
        "timestamps_s": [0, 0.1, 0.3, 0.4],
        "original_pts_s": [0.1, 0.2, 0.4, 0.5],
        "video_timing": "source_pts",
    }
    assert data_findings(episode) == []
    episode["video_timing"] = "inferred_nominal_fps"
    assert {f["code"] for f in data_findings(episode)} == {
        "irregular_nominal_video_spacing",
        "nonzero_derived_video_start",
    }
    schema = TrainingSchema.model_validate(
        {
            "robot_type": "test",
            "fps": 10,
            "width": 160,
            "height": 90,
            "max_video_age_s": 0.05,
            "state": [
                {
                    "stream": "joint",
                    "index": 0,
                    "name": "joint",
                    "units": "rad",
                    "meaning": "position",
                }
            ],
            "action": [
                {
                    "stream": "joint",
                    "index": 0,
                    "name": "joint",
                    "units": "rad",
                    "meaning": "position",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="resolve its origin"):
        plan_inputs(deepcopy(episode), schema)
