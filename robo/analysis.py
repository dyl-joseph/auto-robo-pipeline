"""Offline OpenCV baseline: motion regions and sparse feature trajectories, not semantics."""

from pathlib import Path

import cv2
import numpy as np

from robo.core import atomic_json, digest, file_hash
from robo.store import Store


def suggest(store: Store, episode_id: str) -> dict:
    episode = store.episode(episode_id)
    video = store.root / episode["media"]
    if file_hash(video) != episode["video_sha256"]:
        raise ValueError("media hash mismatch")
    config = {
        "backend": "opencv-motion-v1",
        "implementation_sha256": file_hash(Path(__file__)),
        "opencv": cv2.__version__,
        "resize": [320, 180],
        "threshold": 18,
        "motion_fraction": 0.008,
        "min_segment_s": 0.35,
        "score_semantics": "heuristic evidence strength, not calibrated probability",
    }
    run_id = digest({"episode": episode["content_hash"], "config": config})[:24]
    path = store.root / "runs" / episode_id / f"{run_id}.json"
    if path.exists():
        import json

        cached = json.loads(path.read_text())
        if cached["artifact_hash"] != digest(
            {k: v for k, v in cached.items() if k != "artifact_hash"}
        ):
            raise ValueError("cached analysis hash mismatch")
        atomic_json(path.parent / "active.json", {"id": run_id})
        return cached
    cap = cv2.VideoCapture(str(video))
    previous = None
    points = None
    evidence = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
            box, tracks, fraction = None, [], 0.0
            if previous is not None:
                mask = cv2.threshold(cv2.absdiff(gray, previous), 18, 255, cv2.THRESH_BINARY)[1]
                fraction = float(np.mean(mask > 0))
                coords = cv2.findNonZero(mask)
                if coords is not None:
                    box = list(map(int, cv2.boundingRect(coords)))
                if points is not None:
                    nxt, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
                    if nxt is not None:
                        for a, b, valid in zip(
                            points.reshape(-1, 2), nxt.reshape(-1, 2), status.ravel(), strict=False
                        ):
                            if valid and np.isfinite(b).all():
                                tracks.append([*map(float, a), *map(float, b)])
            points = cv2.goodFeaturesToTrack(gray, maxCorners=30, qualityLevel=0.1, minDistance=10)
            evidence.append(
                {
                    "frame": len(evidence),
                    "motion_fraction": fraction,
                    "box_xywh": box,
                    "flow_vectors": tracks,
                    "brightness": float(gray.mean()),
                    "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                }
            )
            previous = gray
    finally:
        cap.release()
    if len(evidence) != len(episode["timestamps_s"]):
        raise ValueError("decoded frame count differs from measured timestamps")
    times = episode["timestamps_s"]
    segments = []
    start = 0
    labels = [
        "motion" if row["motion_fraction"] > config["motion_fraction"] else "hold"
        for row in evidence
    ]
    for index in range(1, len(labels) + 1):
        changed = index == len(labels) or labels[index] != labels[start]
        end = episode["duration_s"] if index == len(labels) else times[index]
        if changed and (end - times[start] >= config["min_segment_s"] or index == len(labels)):
            activity = float(np.mean([r["motion_fraction"] for r in evidence[start:index]]))
            segments.append(
                {
                    "start_s": times[start],
                    "end_s": end,
                    "label": labels[start],
                    "score": min(0.85, 0.35 + activity * 3),
                    "evidence_frames": [start, index - 1],
                }
            )
            start = index
    warnings = list(episode["validation"]["warnings"])
    if np.mean([e["brightness"] < 20 for e in evidence]) > 0.1:
        warnings.append("dark_frames")
    if np.mean([e["sharpness"] < 20 for e in evidence]) > 0.1:
        warnings.append("low_texture_or_blur")
    run = {
        "id": run_id,
        "episode_id": episode_id,
        "config": config,
        "segments": segments,
        "evidence": evidence,
        "warnings": warnings,
        "kind": "heuristic",
        "schema_version": 1,
    }
    run["artifact_hash"] = digest(run)
    atomic_json(path, run)
    atomic_json(path.parent / "active.json", {"id": run_id})
    return run


def latest_run(store: Store, episode_id: str, run_id: str | None = None) -> dict | None:
    import json

    directory = store.root / "runs" / episode_id
    if run_id is None:
        pointer = directory / "active.json"
        if not pointer.exists():
            return None
        run_id = json.loads(pointer.read_text())["id"]
    if len(run_id) != 24 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("invalid suggestion run identity")
    run = json.loads((directory / f"{run_id}.json").read_text())
    if run["artifact_hash"] != digest({k: v for k, v in run.items() if k != "artifact_hash"}):
        raise ValueError("analysis artifact hash mismatch")
    return run


def queue(store: Store, diverse: bool = True) -> list[dict]:
    items = []
    hashes = {}
    for episode in store.episodes():
        hashes.setdefault(episode["video_sha256"], []).append(episode["id"])
    for episode in store.episodes():
        revision = store.latest(episode["id"])
        run = latest_run(store, episode["id"])
        warnings = run["warnings"] if run else episode["validation"]["warnings"]
        low = bool(run and any(s["score"] < 0.5 for s in run["segments"]))
        reasons = list(warnings) + (["weak_motion_evidence"] if low else [])
        uncertain = revision["status"] == "needs_adjudication" or "uncertain" in revision.get(
            "quality_flags", []
        )
        if uncertain:
            reasons.append("needs_adjudication")
        if "recovery" in revision.get("quality_flags", []):
            reasons.append("preserve_recovery_coverage")
        duplicate = hashes[episode["video_sha256"]]
        if len(duplicate) > 1:
            reasons.append("exact_duplicate_group")
        score = len(warnings) * 2 + low * 2 + uncertain * 5
        items.append(
            {
                "id": episode["id"],
                "task": episode["task"],
                "session": episode["session"],
                "data_kind": episode["data_kind"],
                "duration_s": episode["duration_s"],
                "status": revision["status"],
                "priority": score,
                "reasons": reasons or ["coverage_audit"],
                "duplicate_group": duplicate,
                "revision": revision["revision"],
            }
        )
    items.sort(key=lambda item: (-item["priority"], item["id"]))
    if diverse:
        result, seen = [], set()
        for item in items:
            if item["task"] not in seen:
                result.append(item)
                seen.add(item["task"])
        result.extend(i for i in items if i not in result)
        return result
    return items
