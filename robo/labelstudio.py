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
from robo.core import LABELS, Edit, atomic_json, digest, file_hash, validate_segments
from robo.store import Store


def prepare_proxy(store: Store, episode_id: str, fps: int | None = None) -> dict:
    """Hold each measured source frame until the next PTS; keep an explicit frame map."""
    episode = store.episode(episode_id)
    base = Fraction(episode["time_base"])
    source_clock = [(tick - episode["original_pts"][0]) * base for tick in episode["original_pts"]]
    automatic = fps is None
    gaps = [b - a for a, b in zip(source_clock, source_clock[1:], strict=False)]
    required_fps = math.ceil(1 / min(gaps))
    fps = required_fps if automatic else fps
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError(
            f"source needs {required_fps} fps to display every source frame; pilot supports 1–60. "
            "Choose an explicit coarser --fps only if that loss is acceptable."
        )
    source = store.root / episode["media"]
    if file_hash(source) != episode["video_sha256"]:
        raise ValueError("source media hash mismatch")
    identity = digest(
        [episode["content_hash"], fps, automatic, "review-proxy-v4-continuous-clock"]
    )[:24]
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
    count = math.ceil(episode["duration_s"] * fps)
    indices = [
        max(0, bisect.bisect_right(source_clock, Fraction(i, fps)) - 1) for i in range(count)
    ]
    skipped = len(source_clock) - len(set(indices))
    if automatic and skipped:
        raise ValueError("automatic proxy cannot retain every source frame; choose an explicit fps")
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
            source_index = indices[index]
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
            writer.writerow([timestamp, *values])
    result = {
        "id": identity,
        "episode_id": episode_id,
        "episode_hash": episode["content_hash"],
        "fps": fps,
        "fidelity": {
            "automatic_fps": automatic,
            "source_frame_count": len(source_clock),
            "displayed_source_frame_count": len(set(indices)),
            "skipped_source_frame_count": skipped,
            "grid_period_s": 1 / fps,
            "max_source_gap_s": float(max(gaps)),
            "max_display_hold_s": max(
                float(Fraction(i, fps) - source_clock[j]) for i, j in enumerate(indices)
            ),
        },
        "boundary_policy": (
            "Native inclusive frames [a,b] map to episode-clock interval "
            "[(a-1)/fps,min(b/fps,duration)). Boundaries are generated grid times, "
            "not measured source PTS. Source brackets expose unobserved gaps. "
            "Machine boundaries snap to the nearest common grid edge; "
            "collapsed candidates are omitted."
        ),
        "duration_s": episode["duration_s"],
        "frames": frames,
        "columns": columns,
        "column_units": {
            f"{stream['name']}_{dim}": stream["units"]
            for stream in episode["streams"]
            for dim in range(len(stream["values"][0]))
        },
        "policy": (
            "CFR review proxy; hold last source frame. Display CSV uses nearest "
            "samples with null gaps; raw streams unchanged."
        ),
        "source_video_timing": episode.get("video_timing", "source_pts"),
        "files": {name: file_hash(directory / name) for name in ("video.mp4", "sensors.csv")},
    }
    result["artifact_hash"] = digest(result)
    atomic_json(manifest, result)
    return result


def label_config(columns: list[str], fps: int, column_units: dict[str, str] | None = None) -> str:
    import html

    labels = "".join(f'<Label value="{label}" background="#4b4b47"/>' for label in LABELS)
    channels = []
    palette = ("#000000", "#8d8b80", "#52524d", "#b2afa0")
    for index, column in enumerate(columns):
        units = (column_units or {}).get(column, "")
        channels.append(
            f'<Channel column="{html.escape(column, quote=True)}" '
            f'legend="{html.escape(column.replace("_", " "), quote=True)}" '
            f'units="{html.escape(units, quote=True)}" '
            f'height="60" strokeColor="{palette[index % len(palette)]}"/>'
        )
    channels = "".join(channels)
    sensors = (
        (
            '<TimeSeries name="sensors" value="$sensors" sync="episode" timeColumn="time" '
            'valueType="json" '
            'timeDisplayFormat=".2f" durationDisplayFormat=".2f" overviewWidth="100%">'
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
    result = []
    count = len(proxy["frames"])
    for index, segment in enumerate(segments):
        start, end = (
            min(count, max(0, math.floor(segment[key] * proxy["fps"] + 0.5)))
            for key in ("start_s", "end_s")
        )
        if start >= end:
            continue
        result.append(
            {
                "id": f"segment-{index}",
                "from_name": "subtask",
                "to_name": "video",
                "type": "timelinelabels",
                "interactive_mode": False,
                "value": {
                    "ranges": [{"start": start + 1, "end": end}],
                    "timelinelabels": [segment["label"]],
                },
            }
        )
    return result


def decoded_segments(results: list[dict], proxy: dict, episode: dict) -> list[dict]:
    """Map native inclusive frame spans to a shared continuous episode-clock grid."""
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
            segments.append(
                {
                    "start_s": (start - 1) / proxy["fps"],
                    "end_s": min(end / proxy["fps"], episode["duration_s"]),
                    "label": value["timelinelabels"][0],
                }
            )
    segments.sort(key=lambda s: s["start_s"])
    validate_segments(segments, episode["duration_s"])
    return segments


def task_bundle(store: Store, episode_id: str, fps: int | None = None) -> dict:
    episode = store.episode(episode_id)
    proxy = prepare_proxy(store, episode_id, fps)
    run = latest_run(store, episode_id)
    base = f"/data/local-files/?d=review_proxy/{proxy['id']}/"
    with (store.root / "review_proxy" / proxy["id"] / "sensors.csv").open() as source:
        rows = list(csv.DictReader(source))
    sensors = {
        column: [float(row[column]) if row[column] else None for row in rows]
        for column in ("time", *proxy["columns"])
    }
    task = {
        "data": {
            "video": base + "video.mp4",
            "sensors": sensors,
            "task": episode["task"],
            "provenance": (
                f"{episode['data_kind'].upper()} · {episode['source_id']} · "
                f"{proxy['fps']} fps review grid · "
                f"{proxy['fidelity']['skipped_source_frame_count']} "
                "source frames omitted · "
                + (
                    "inferred nominal timing; no capture timestamps"
                    if episode.get("video_timing") == "inferred_nominal_fps"
                    else "source PTS in provenance"
                )
            ),
            "episode_id": episode_id,
        },
        "predictions": [],
    }
    candidates = prediction_results(run["segments"], proxy) if run and run["segments"] else []
    projection = {
        "input_segment_count": len(run["segments"]) if run else 0,
        "display_segment_count": len(candidates),
        "omitted_collapsed_count": (len(run["segments"]) if run else 0) - len(candidates),
        "max_boundary_displacement_bound_s": 0.5 / proxy["fps"],
    }
    if candidates:
        task["predictions"] = [
            {
                "model_version": run["id"],
                "score": min(s["score"] for s in run["segments"]),
                "result": candidates,
            }
        ]
    return {
        "task": task,
        "proxy": proxy,
        "label_config": label_config(proxy["columns"], proxy["fps"], proxy["column_units"]),
        "trusted_origin": {
            "episode_id": episode_id,
            "episode_hash": episode["content_hash"],
            "suggestion_run": run["id"] if run else None,
            "proxy_hash": digest(proxy),
            "prediction_projection": projection,
        },
    }


def bind_task(bundle: dict, project_id: int, task_id: int) -> dict:
    return {
        **bundle["trusted_origin"],
        "project_id": project_id,
        "task_id": task_id,
        "task_data_hash": task_data_hash(bundle["task"]["data"], bundle["trusted_origin"]),
    }


QUEUE_FIELDS = ["review_priority", "review_reason"]


def task_data_hash(data: dict, binding: dict) -> str:
    mutable = binding.get("mutable_task_fields", [])
    if mutable and mutable != QUEUE_FIELDS:
        raise ValueError("Only queue priority and reason may change outside the source binding")
    return digest({key: value for key, value in data.items() if key not in mutable})


def annotation_snapshot(store: Store, binding: dict, task: dict, proxy: dict) -> dict:
    """Read an SDK JSON export against trusted import provenance; never imply approval."""
    episode = store.episode(binding["episode_id"])
    if (
        task["id"] != binding["task_id"]
        or task["project"] != binding["project_id"]
        or task_data_hash(task["data"], binding) != binding["task_data_hash"]
        or episode["content_hash"] != binding["episode_hash"]
        or digest(proxy) != binding["proxy_hash"]
        or file_hash(store.root / episode["media"]) != episode["video_sha256"]
    ):
        raise ValueError("Label Studio export does not match trusted source/task/proxy binding")
    directory = store.root / "review_proxy" / proxy["id"]
    if any(file_hash(directory / name) != value for name, value in proxy["files"].items()):
        raise ValueError("review proxy artifact hash mismatch")
    annotations = []
    for annotation in task.get("annotations", []):
        if annotation.get("was_cancelled"):
            continue
        annotations.append(
            {
                "annotation_id": annotation["id"],
                "content_hash": digest(annotation),
                "segments": decoded_segments(annotation["result"], proxy, episode),
                **review_fields(annotation["result"]),
                "raw_annotation": annotation,
                "boundary_evidence": boundary_evidence(annotation["result"], proxy, episode),
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
        "boundary_policy": proxy["boundary_policy"],
        "fidelity": proxy["fidelity"],
    }
    snapshot["snapshot_hash"] = digest(snapshot)
    return snapshot


def review_fields(results: list[dict]) -> dict:
    """Decode only the supported native controls; do not silently drop human decisions."""
    fields = {"outcome": "unknown", "quality_flags": [], "reviewer_evidence": []}
    seen = set()
    for result in results:
        kind, name = result.get("type"), result.get("from_name")
        if kind == "timelinelabels" and name == "subtask":
            continue
        expected = {"outcome": "choices", "quality": "choices", "decision_note": "textarea"}
        if (
            name not in expected
            or kind != expected[name]
            or result.get("to_name") != "video"
            or name in seen
        ):
            raise ValueError("unsupported or duplicate native review control")
        seen.add(name)
        value = result.get("value", {})
        values = value.get("text" if name == "decision_note" else "choices")
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError("native review values must be lists of strings")
        if name == "outcome":
            if len(values) > 1:
                raise ValueError("native outcome must be mutually exclusive")
            fields["outcome"] = values[0] if values else "unknown"
        elif name == "quality":
            if len(values) != len(set(values)):
                raise ValueError("duplicate native quality flag")
            fields["quality_flags"] = sorted(values)
        else:
            if sum(map(len, values)) > 20000:
                raise ValueError("native reviewer evidence exceeds 20000 characters")
            fields["reviewer_evidence"] = values
    Edit(
        expected_revision=0,
        actor="decoder",
        rationale="validate native review fields",
        segments=[],
        outcome=fields["outcome"],
        quality_flags=fields["quality_flags"],
    )
    return fields


def boundary_evidence(results: list[dict], proxy: dict, episode: dict) -> list[dict]:
    """Separate generated annotation times from the measured frames bracketing them."""
    ticks = episode["original_pts"]
    base = Fraction(episode["time_base"])
    clock = [(tick - ticks[0]) * base for tick in ticks]

    def context(index: int) -> dict:
        target = min(Fraction(index, proxy["fps"]), Fraction(str(episode["duration_s"])))
        right = bisect.bisect_left(clock, target)
        exact = right < len(clock) and clock[right] == target
        left = right if exact else max(0, right - 1)
        a = clock[left]
        b = clock[right] if right < len(clock) else Fraction(str(episode["duration_s"]))
        return {
            "episode_time_s": float(target),
            "is_measured_source_boundary": (
                exact and episode.get("video_timing", "source_pts") == "source_pts"
            ),
            "is_source_frame_boundary": exact,
            "source_video_timing": episode.get("video_timing", "source_pts"),
            "preceding_source_pts": ticks[left],
            "following_source_pts": ticks[right] if right < len(ticks) else None,
            "source_bracket_s": [float(a), float(b)],
            "source_bracket_width_s": float(b - a),
        }

    return [
        {
            "result_id": result.get("id"),
            "start": context(span["start"] - 1),
            "end": context(span["end"]),
        }
        for result in results
        if result.get("type") == "timelinelabels"
        for span in result["value"]["ranges"]
    ]


def prepare_bundle(store: Store, episode_id: str, output: Path, fps: int | None = None) -> dict:
    bundle = task_bundle(store, episode_id, fps)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "tasks.json", [bundle["task"]])
    atomic_json(output / "provenance.json", bundle["trusted_origin"])
    atomic_json(output / "mapping.json", bundle["proxy"])
    atomic_json(output / "bundle.json", bundle)
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
