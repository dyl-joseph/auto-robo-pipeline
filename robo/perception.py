"""Optional pinned MobileNet-SSD CPU inference; one sampled frame in memory at a time."""

import json
import time
from pathlib import Path

import cv2

from robo.core import atomic_json, digest, file_hash
from robo.store import Store

MODEL = {
    "repository": "https://github.com/chuanqi305/MobileNet-SSD",
    "commit": "bb17b6c3eef36d80be441ae8e5339be66e8e3b7a",
    "license": "MIT",
    "files": {
        "mobilenet_iter_73000.caffemodel": (
            "52eed8be80522c152a17fb56740de705b79881bde1a167e0e747310523685fc7"
        ),
        "deploy.prototxt": "2d180f723b3109e21f8287f6b3c691390d07b60eed998327cd3259ffa0e50608",
        "LICENSE": "5de433821cfe672af2fca73c3005f48af16121c95f6a11e1031b07807ea59905",
    },
}
CLASSES = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def perceive(store: Store, episode_id: str, model_dir: Path, sample_every_s=0.5, max_frames=600):
    if not 0.1 <= sample_every_s <= 10 or not 1 <= max_frames <= 2000:
        raise ValueError("sampling interval must be 0.1–10s; max_frames 1–2000")
    for name, expected in MODEL["files"].items():
        if file_hash(model_dir / name) != expected:
            raise ValueError(f"model checksum mismatch: {name}")
    episode = store.episode(episode_id)
    if file_hash(store.root / episode["media"]) != episode["video_sha256"]:
        raise ValueError("source media checksum mismatch")
    config = {
        "model": MODEL,
        "backend": "opencv-dnn-cpu",
        "opencv": cv2.__version__,
        "sample_every_s": sample_every_s,
        "max_frames": max_frames,
        "threshold": 0.35,
        "implementation_sha256": file_hash(Path(__file__)),
        "preprocess": "300x300 BGR, scale=0.007843, mean=127.5",
    }
    identity = digest([episode["content_hash"], config])[:24]
    path = store.root / "perception" / episode_id / f"{identity}.json"
    if path.exists():
        run = json.loads(path.read_text())
        if run["artifact_hash"] != digest({k: v for k, v in run.items() if k != "artifact_hash"}):
            raise ValueError("perception hash mismatch")
        atomic_json(path.parent / "active.json", {"id": identity})
        return run
    net = cv2.dnn.readNetFromCaffe(
        str(model_dir / "deploy.prototxt"), str(model_dir / "mobilenet_iter_73000.caffemodel")
    )
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    capture = cv2.VideoCapture(str(store.root / episode["media"]))
    samples, next_time, index = [], 0, 0
    start = time.perf_counter()
    try:
        while index < len(episode["timestamps_s"]):
            ok, frame = capture.read()
            if not ok:
                raise ValueError("decode ended before measured presentation times")
            timestamp = episode["timestamps_s"][index]
            if timestamp >= next_time and len(samples) < max_frames:
                net.setInput(cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5))
                output = net.forward()[0, 0]
                detections = []
                for detection in output:
                    class_id, score = int(detection[1]), float(detection[2])
                    if score >= config["threshold"] and 0 < class_id < len(CLASSES):
                        box = detection[3:7].clip(0, 1).tolist()
                        detections.append(
                            {"label": CLASSES[class_id], "score": score, "xyxy_normalized": box}
                        )
                samples.append({"frame": index, "time_s": timestamp, "detections": detections})
                next_time = timestamp + sample_every_s
            index += 1
            if len(samples) >= max_frames:
                break
    finally:
        capture.release()
    run = {
        "id": identity,
        "config": config,
        "episode_id": episode_id,
        "samples": samples,
        "sampled_frames": len(samples),
        "elapsed_s": time.perf_counter() - start,
        "coverage_end_s": samples[-1]["time_s"] if samples else 0,
        "truncated": index < len(episode["timestamps_s"]),
        "limitations": (
            "VOC object vocabulary; no robot/contact/action recognition. "
            "Uncalibrated scores. No tracking identity."
        ),
    }
    run["artifact_hash"] = digest(run)
    atomic_json(path, run)
    atomic_json(path.parent / "active.json", {"id": identity})
    return run


def active_perception(store: Store, episode_id: str, run_id: str | None = None):
    directory = store.root / "perception" / episode_id
    pointer = directory / "active.json"
    if run_id is None and not pointer.exists():
        return None
    identity = run_id or json.loads(pointer.read_text())["id"]
    if len(identity) != 24 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("invalid perception identity")
    run = json.loads((directory / f"{identity}.json").read_text())
    if run["artifact_hash"] != digest({k: v for k, v in run.items() if k != "artifact_hash"}):
        raise ValueError("perception artifact hash mismatch")
    return run
