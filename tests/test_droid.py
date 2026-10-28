import hashlib
import io
import subprocess
from copy import deepcopy

import pytest

from robo.core import atomic_json
from robo.droid import CAMERAS, source_from_steps, training_schema
from robo.ingest import import_episode
from robo.labelstudio import boundary_evidence, prediction_results, task_bundle
from robo.lerobot_plan import TrainingSchema, plan_episode
from robo.store import Store
from scripts import fetch_droid


def steps():
    # Generated values deliberately disagree with the generic action vector.
    return [
        {
            "is_first": i == 0,
            "is_last": i == 2,
            "language_instruction": "Generated test instruction",
            "observation": {"joint_position": [i] * 7, "gripper_position": [0.2]},
            "action_dict": {"joint_position": [i + 1] * 7, "gripper_position": [0.3]},
            "action": [999] * 7,
        }
        for i in range(3)
    ]


METADATA = {"file_path": "/original/institution/success/day/episode/trajectory.h5"}


def test_droid_named_fields_order_clock_and_same_day_grouping():
    source = source_from_steps(steps(), METADATA, CAMERAS[0])
    schema = TrainingSchema.model_validate(training_schema())
    assert len(schema.state) == len(schema.action) == 8
    assert source["video_timing"] == "inferred_nominal_fps"
    assert source["data_kind"] == "real" and source["license"] == "CC-BY-4.0"
    assert source["streams"][2]["values"] == [[1] * 7, [2] * 7, [3] * 7]
    assert source["streams"][3]["values"] == [[0.3]] * 3
    assert source["streams"][0]["timestamps_s"] == [0, 1 / 15, 2 / 15]
    assert all("Inferred" in s["clock_evidence"] for s in source["streams"])
    other = source_from_steps(
        steps(), {"file_path": METADATA["file_path"].replace("/episode/", "/another/")}, CAMERAS[1]
    )
    assert source["session"] == other["session"]
    assert source["source_id"] != other["source_id"]


@pytest.mark.parametrize(
    "bad", ["missing", "shape", "nan", "first", "last", "instruction", "gripper"]
)
def test_droid_invalid_data_rejected_without_inventing_values(bad):
    rows = steps()
    if bad == "missing":
        del rows[0]["action_dict"]["joint_position"]
    elif bad == "shape":
        rows[0]["action_dict"]["joint_position"] = [1] * 6
    elif bad == "nan":
        rows[0]["observation"]["joint_position"][0] = float("nan")
    elif bad in {"first", "last"}:
        rows[1]["is_" + bad] = True
    elif bad == "instruction":
        rows[1]["language_instruction"] = "changed"
    else:
        rows[0]["action_dict"]["gripper_position"] = [2]
    with pytest.raises((KeyError, ValueError)):
        source_from_steps(rows, METADATA, CAMERAS[0])


def test_inferred_clock_survives_review_boundaries_and_training(fixture):
    store, manifest, body = fixture
    body.update(video_timing="inferred_nominal_fps")
    source = source_from_steps(steps(), METADATA, CAMERAS[0])
    body["streams"] = deepcopy(source["streams"])
    for stream in body["streams"]:
        stream["timestamps_s"] = [0, 0.1, 0.2]
        stream["tolerance_s"] = 1
    atomic_json(manifest, body)
    episode = import_episode(manifest, store)
    assert "inferred_video_timing" in episode["validation"]["warnings"]
    assert "not measured capture" in episode["timestamp_authority"]
    bundle = task_bundle(store, episode["id"])
    assert "inferred nominal timing" in bundle["task"]["data"]["provenance"]
    results = prediction_results(
        [{"start_s": 0, "end_s": 0.2, "label": "unknown"}], bundle["proxy"]
    )
    boundary = boundary_evidence(results, bundle["proxy"], episode)[0]["end"]
    assert boundary["is_source_frame_boundary"]
    assert not boundary["is_measured_source_boundary"]
    assert boundary["source_video_timing"] == "inferred_nominal_fps"
    episode["duration_s"] = 0.3
    schema = TrainingSchema.model_validate(training_schema()).model_copy(update={"fps": 10})
    plan = plan_episode(
        {"episode": episode, "annotation": {"status": "accepted", "segments": []}}, schema
    )
    assert plan["alignment"]["source_video_timing"] == "inferred_nominal_fps"
    assert all(
        s["clock_evidence"].startswith("Inferred") for s in plan["alignment"]["streams"].values()
    )


@pytest.mark.parametrize("response", [b"too long", b"no", b"bad"])
def test_download_budget_and_hash_fail_without_publishing(tmp_path, monkeypatch, response):
    pinned = b"yes"
    monkeypatch.setattr(
        fetch_droid,
        "FILES",
        {
            "tiny": {
                "bytes": 3,
                "sha256": hashlib.sha256(pinned).hexdigest(),
                "md5": hashlib.md5(pinned).hexdigest(),
                "generation": "123",
            }
        },
    )
    monkeypatch.setattr(fetch_droid.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(response))
    with pytest.raises(ValueError):
        fetch_droid.acquire(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_existing_corrupt_download_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fetch_droid, "FILES", {"tiny": {"bytes": 3, "sha256": "wrong", "generation": "123"}}
    )
    path = tmp_path / "tiny"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="existing"):
        fetch_droid.acquire(tmp_path)
    assert path.read_bytes() == b"bad"


def test_nominal_15hz_video_does_not_invent_an_extra_tail_frame(tmp_path):
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=15",
            "-frames:v",
            "119",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            str(tmp_path / "video.mp4"),
        ],
        check=True,
        timeout=30,
    )
    source = source_from_steps(steps(), METADATA, CAMERAS[0])
    source["streams"] = []
    atomic_json(tmp_path / "source.json", source)
    store = Store(tmp_path / "store")
    episode = import_episode(tmp_path / "source.json", store)
    assert episode["duration_s"] == 119 / 15
    bundle = task_bundle(store, episode["id"])
    assert bundle["proxy"]["fps"] == 15
    assert len(bundle["proxy"]["frames"]) == 119
    assert bundle["proxy"]["fidelity"]["skipped_source_frame_count"] == 0
