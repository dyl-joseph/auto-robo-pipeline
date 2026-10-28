import subprocess

import pytest

from robo.core import atomic_json, file_hash
from robo.ingest import import_episode
from robo.labelstudio import (
    annotation_snapshot,
    bind_task,
    decoded_segments,
    prediction_results,
    prepare_proxy,
    task_bundle,
)


def test_proxy_preserves_irregular_source_times_and_identity(fixture):
    store, manifest, body = fixture
    video = manifest.parent / body["video"]
    video.unlink()
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=10:duration=2",
            "-vf",
            "select=not(eq(n\\,5)+eq(n\\,6))",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            str(video),
        ],
        check=True,
    )
    episode = import_episode(manifest, store)
    original = file_hash(store.root / episode["media"])
    proxy = prepare_proxy(store, episode["id"], fps=10)
    assert [row["source_time_s"] for row in proxy["frames"][4:7]] == [0.4, 0.4, 0.4]
    assert proxy["frames"][7]["source_time_s"] == 0.7
    assert file_hash(store.root / episode["media"]) == original
    assert prepare_proxy(store, episode["id"], fps=10) == proxy
    segments = decoded_segments(
        [
            {
                "type": "timelinelabels",
                "from_name": "subtask",
                "to_name": "video",
                "value": {"ranges": [{"start": 6, "end": 7}], "timelinelabels": ["unknown"]},
            }
        ],
        proxy,
        episode,
    )
    assert segments == [{"start_s": 0.4, "end_s": 0.7, "label": "unknown"}]


def test_labelstudio_frame_roundtrip_and_invalid_spans(episode):
    store, e = episode
    proxy = prepare_proxy(store, e["id"], fps=10)
    segments = [{"start_s": 0, "end_s": 1, "label": "reach"}]
    predictions = prediction_results(segments, proxy)
    assert predictions[0]["value"]["ranges"] == [{"start": 1, "end": 10}]
    assert decoded_segments(predictions, proxy, e) == segments
    predictions[0]["value"]["ranges"][0]["start"] = 0
    with pytest.raises(ValueError, match="frame span"):
        decoded_segments(predictions, proxy, e)


def test_sensor_gaps_stay_blank_and_review_clock_starts_at_zero(fixture):
    store, manifest, body = fixture
    body["streams"] = [
        {
            "name": "force",
            "modality": "sensor",
            "units": "N",
            "timestamps_s": [0.5, 1],
            "values": [[10], [20]],
            "tolerance_s": 0.05,
            "clock_evidence": "fixture offset verified",
        }
    ]
    atomic_json(manifest, body)
    episode = import_episode(manifest, store)
    proxy = prepare_proxy(store, episode["id"], fps=10)
    csv = (store.root / "review_proxy" / proxy["id"] / "sensors.csv").read_text().splitlines()
    assert csv[1] == "00:00:00.000000,"
    assert csv[6] == "00:00:00.500000,10.0"
    assert store.episode(episode["id"])["streams"][0]["timestamps_s"] == [0.5, 1]


def test_trusted_binding_rejects_changed_task_and_never_auto_approves(episode):
    store, e = episode
    bundle = task_bundle(store, e["id"], fps=10)
    binding = bind_task(bundle, project_id=1, task_id=2)
    annotation = {
        "id": 3,
        "result": prediction_results(
            [{"start_s": 0, "end_s": 1, "label": "reach"}], bundle["proxy"]
        ),
    }
    task = {"id": 2, "project": 1, "data": bundle["task"]["data"], "annotations": [annotation]}
    snapshot = annotation_snapshot(store, binding, task, bundle["proxy"])
    assert snapshot["annotations"][0]["eligibility"].startswith("draft")
    assert store.latest(e["id"])["revision"] == 0
    task["data"]["episode_id"] = "different"
    with pytest.raises(ValueError, match="trusted"):
        annotation_snapshot(store, binding, task, bundle["proxy"])


def test_proxy_corruption_rejected(episode):
    store, e = episode
    proxy = prepare_proxy(store, e["id"])
    (store.root / "review_proxy" / proxy["id"] / "video.mp4").write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_proxy(store, e["id"])


def test_fractional_frame_periods_use_integer_pts_not_rounded_decimal_time(tmp_path):
    from robo.demo import make_demo
    from robo.store import Store

    episode = make_demo(tmp_path / "robotics", count=1)[0]
    store = Store(tmp_path / "robotics")
    proxy = prepare_proxy(store, episode["id"], fps=12)
    assert [row["source_frame"] for row in proxy["frames"]] == list(range(96))
    segment = {"start_s": 0, "end_s": episode["timestamps_s"][2], "label": "reach"}
    predictions = prediction_results([segment], proxy)
    assert predictions[0]["value"]["ranges"] == [{"start": 1, "end": 2}]
    assert decoded_segments(predictions, proxy, episode) == [segment]
