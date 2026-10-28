"""Robotics-specific translation around the supported Label Studio editor."""

import bisect
import csv
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from robo.analysis import latest_run
from robo.core import LABELS, atomic_json, digest, file_hash, validate_segments
from robo.store import Store


def prepare_proxy(store: Store, episode_id: str, fps: int = 12) -> dict:
    """Hold each measured source frame until the next PTS; keep an explicit frame map."""
    if not 1 <= fps <= 60:
        raise ValueError("review proxy fps must be 1–60")
    episode = store.episode(episode_id)
    source = store.root / episode["media"]
    if file_hash(source) != episode["video_sha256"]:
        raise ValueError("source media hash mismatch")
    identity = digest([episode["content_hash"], fps, "review-proxy-hold-v2-rational-pts"])[:24]
    directory = store.root / "review_proxy" / identity
    manifest = directory / "mapping.json"
    if manifest.exists():
        import json

        result = json.loads(manifest.read_text())
        if result["artifact_hash"] != digest(
            {k: v for k, v in result.items() if k != "artifact_hash"}
        ):
            raise ValueError("review proxy mapping hash mismatch")
        if any(file_hash(directory / name) != value for name, value in result["files"].items()):
            raise ValueError("review proxy artifact hash mismatch")
        return result
    directory.mkdir(parents=True, exist_ok=True)
    times = episode["timestamps_s"]
    base = Fraction(episode["time_base"])
    source_clock = [(tick - episode["original_pts"][0]) * base for tick in episode["original_pts"]]
    count = math.ceil(episode["duration_s"] * fps)
    frames = []
    capture = cv2.VideoCapture(str(source))
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{episode['width']}x{episode['height']}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(directory / "video.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    current, image = -1, None
    try:
        for index in range(count):
            clock = index / fps
            source_index = max(0, bisect.bisect_right(source_clock, Fraction(index, fps)) - 1)
            while current < source_index:
                ok, image = capture.read()
                if not ok:
                    raise ValueError("source decode ended before measured PTS")
                current += 1
            process.stdin.write(image.tobytes())
            frames.append(
                {
                    "review_frame": index + 1,
                    "review_time_s": clock,
                    "source_frame": source_index,
                    "source_pts": episode["original_pts"][source_index],
                    "source_time_s": times[source_index],
                }
            )
        process.stdin.close()
        if process.wait(timeout=60):
            raise ValueError("review proxy encoding failed")
    finally:
        capture.release()
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    columns = [
        f"{stream['name']}_{dim}"
        for stream in episode["streams"]
        for dim in range(len(stream["values"][0]))
    ]
    with (directory / "sensors.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["time", *columns])
        for frame in frames:
            timestamp = frame["review_time_s"]
            values = []
            for stream in episode["streams"]:
                samples = np.array(stream["timestamps_s"]) + stream["offset_s"]
                nearest = int(np.argmin(abs(samples - timestamp)))
                values.extend(
                    stream["values"][nearest]
                    if abs(samples[nearest] - timestamp) <= stream["tolerance_s"]
                    else [""] * len(stream["values"][0])
                )
            minutes, seconds = divmod(timestamp, 60)
            hours, minutes = divmod(int(minutes), 60)
            writer.writerow([f"{hours:02}:{minutes:02}:{seconds:09.6f}", *values])
    result = {
        "id": identity,
        "episode_id": episode_id,
        "episode_hash": episode["content_hash"],
        "fps": fps,
        "duration_s": episode["duration_s"],
        "frames": frames,
        "columns": columns,
        "policy": (
            "CFR review proxy; hold last measured source frame. Display CSV uses nearest "
            "samples with null gaps; raw streams unchanged."
        ),
        "files": {name: file_hash(directory / name) for name in ("video.mp4", "sensors.csv")},
    }
    result["artifact_hash"] = digest(result)
    atomic_json(manifest, result)
    return result


def label_config(columns: list[str], fps: int) -> str:
    import html

    labels = "".join(f'<Label value="{label}" background="#dce5e7"/>' for label in LABELS)
    channels = "".join(
        f'<Channel column="{html.escape(column, quote=True)}" strokeColor="#5f777e" height="70"/>'
        for column in columns
    )
    sensors = (
        (
            '<TimeSeries name="sensors" value="$sensors" sync="episode" timeColumn="time" '
            'timeFormat="%H:%M:%S.%f" timeDisplayFormat="%H:%M:%S.%f" overviewWidth="10%">'
            f"{channels}</TimeSeries>"
        )
        if columns
        else '<Text name="missing" value="Robot streams unavailable; no state inferred."/>'
    )
    template = Path(__file__).resolve().parent.parent / "integrations/labelstudio/review.xml"
    return (
        template.read_text()
        .replace("__FPS__", str(fps))
        .replace("__LABELS__", labels)
        .replace("__SENSORS__", sensors)
    )


def prediction_results(segments: list[dict], proxy: dict) -> list[dict]:
    clocks = [frame["source_time_s"] for frame in proxy["frames"]]
    result = []
    for index, segment in enumerate(segments):
        start = min(len(clocks), bisect.bisect_left(clocks, segment["start_s"]) + 1)
        end = min(len(clocks), max(start, bisect.bisect_left(clocks, segment["end_s"])))
        result.append(
            {
                "id": f"segment-{index}",
                "from_name": "subtask",
                "to_name": "video",
                "type": "timelinelabels",
                "value": {
                    "ranges": [{"start": start, "end": end}],
                    "timelinelabels": [segment["label"]],
                },
            }
        )
    return result


def decoded_segments(results: list[dict], proxy: dict, episode: dict) -> list[dict]:
    """Native frame spans are 1-based, inclusive; return measured source-frame intervals."""
    segments = []
    for result in results:
        if result.get("type") != "timelinelabels":
            continue
        if result.get("from_name") != "subtask" or result.get("to_name") != "video":
            raise ValueError("unexpected annotation field")
        value = result["value"]
        if len(value["timelinelabels"]) != 1:
            raise ValueError("one mutually exclusive label per region required")
        for span in value["ranges"]:
            start, end = span["start"], span["end"]
            if (
                type(start) is not int
                or type(end) is not int
                or not 1 <= start <= end <= len(proxy["frames"])
            ):
                raise ValueError("invalid native video frame span")
            a, b = (proxy["frames"][i - 1]["source_frame"] for i in (start, end))
            segments.append(
                {
                    "start_s": episode["timestamps_s"][a],
                    "end_s": episode["timestamps_s"][b + 1]
                    if b + 1 < len(episode["timestamps_s"])
                    else episode["duration_s"],
                    "label": value["timelinelabels"][0],
                }
            )
    segments.sort(key=lambda s: s["start_s"])
    validate_segments(segments, episode["duration_s"])
    return segments


def task_bundle(store: Store, episode_id: str, fps: int = 12) -> dict:
    episode = store.episode(episode_id)
    proxy = prepare_proxy(store, episode_id, fps)
    run = latest_run(store, episode_id)
    base = f"/data/local-files/?d=review_proxy/{proxy['id']}/"
    task = {
        "data": {
            "video": base + "video.mp4",
            "sensors": base + "sensors.csv",
            "task": episode["task"],
            "provenance": (
                f"{episode['data_kind'].upper()} · {episode['source_id']} · "
                "CFR display proxy; original PTS preserved."
            ),
            "episode_id": episode_id,
        },
        "predictions": [],
    }
    if run:
        task["predictions"] = [
            {
                "model_version": run["id"],
                "score": min(s["score"] for s in run["segments"]),
                "result": prediction_results(run["segments"], proxy),
            }
        ]
    return {
        "task": task,
        "proxy": proxy,
        "label_config": label_config(proxy["columns"], fps),
        "trusted_origin": {
            "episode_id": episode_id,
            "episode_hash": episode["content_hash"],
            "suggestion_run": run["id"] if run else None,
            "proxy_hash": digest(proxy),
        },
    }


def bind_task(bundle: dict, project_id: int, task_id: int) -> dict:
    return {
        **bundle["trusted_origin"],
        "project_id": project_id,
        "task_id": task_id,
        "task_data_hash": digest(bundle["task"]["data"]),
    }


def annotation_snapshot(store: Store, binding: dict, task: dict, proxy: dict) -> dict:
    """Read an SDK JSON export against trusted import provenance; never imply approval."""
    episode = store.episode(binding["episode_id"])
    if (
        task["id"] != binding["task_id"]
        or task["project"] != binding["project_id"]
        or digest(task["data"]) != binding["task_data_hash"]
        or episode["content_hash"] != binding["episode_hash"]
        or digest(proxy) != binding["proxy_hash"]
        or file_hash(store.root / episode["media"]) != episode["video_sha256"]
    ):
        raise ValueError("Label Studio export does not match trusted source/task/proxy binding")
    annotations = []
    for annotation in task.get("annotations", []):
        if annotation.get("was_cancelled"):
            continue
        annotations.append(
            {
                "annotation_id": annotation["id"],
                "content_hash": digest(annotation),
                "segments": decoded_segments(annotation["result"], proxy, episode),
                "raw_annotation": annotation,
                "eligibility": "draft; explicit bridge approval still required",
                "timing": "Label Studio lead_time is not measured active correction time",
            }
        )
    snapshot = {
        "schema_version": 1,
        "binding": binding,
        "annotations": annotations,
        "source_license": episode["license"],
        "data_kind": episode["data_kind"],
    }
    snapshot["snapshot_hash"] = digest(snapshot)
    return snapshot


def prepare_bundle(store: Store, episode_id: str, output: Path, fps: int = 12) -> dict:
    bundle = task_bundle(store, episode_id, fps)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "tasks.json", [bundle["task"]])
    atomic_json(output / "provenance.json", bundle["trusted_origin"])
    atomic_json(output / "mapping.json", bundle["proxy"])
    (output / "review.xml").write_text(bundle["label_config"])
    return {
        "output": str(output.resolve()),
        "local_storage_path": str(store.root / "review_proxy"),
        "episode_id": episode_id,
        "status": "prepared for native Label Studio import",
    }


def publish_bundle(sdk, store: Store, bundle: dict) -> dict:
    """Use the official SDK; caller supplies its own authorized client/session."""
    project = sdk.projects.create(title="Robot episode review", label_config=bundle["label_config"])
    sdk.import_storage.local.create(
        project=project.id,
        path=str(store.root / "review_proxy"),
        use_blob_urls=True,
        title="Immutable review proxies",
    )
    imported = sdk.projects.import_tasks(project.id, request=[bundle["task"]], return_task_ids=True)
    if not imported.task_ids or len(imported.task_ids) != 1:
        raise ValueError("native import did not return exactly one task identity")
    return bind_task(bundle, project.id, imported.task_ids[0])


def serve_labelstudio(store: Store, port: int) -> None:
    import importlib.util
    import os
    import sys

    if importlib.util.find_spec("label_studio") is None:
        raise ValueError("install the locked integration with uv sync --group labelstudio")
    if not 1024 <= port <= 65535:
        raise ValueError("choose a localhost port between 1024 and 65535")
    data = store.root / "labelstudio"
    data.mkdir(exist_ok=True)
    environment = {
        **os.environ,
        "LABEL_STUDIO_BASE_DATA_DIR": str(data),
        "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true",
        "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": str(store.root),
        "LABEL_STUDIO_COLLECT_ANALYTICS": "false",
        "LABEL_STUDIO_LATEST_VERSION_CHECK": "false",
        "LABEL_STUDIO_SENTRY_DSN": "",
        "LABEL_STUDIO_FRONTEND_SENTRY_DSN": "",
        "fflag_fix_front_dev_3391_interactive_view_all": "false",
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "label_studio.server",
            "start",
            "--internal-host",
            "127.0.0.1",
            "--host",
            f"http://127.0.0.1:{port}",
            "--port",
            str(port),
            "--data-dir",
            str(data),
            "--no-browser",
        ],
        env=environment,
        check=True,
    )
