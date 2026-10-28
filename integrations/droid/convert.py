"""Decode the pinned DROID shard locally with official TFDS and derive one review video."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

from robo.core import atomic_json, file_hash
from robo.droid import CAMERAS, CLOCK, FILES, FPS, SHARD, source_from_steps, training_schema


def verify_inputs(source: Path):
    for name, expected in FILES.items():
        path = source / name
        if path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
            raise ValueError(f"DROID input differs from pinned bytes: {name}")


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def convert(source: Path, output: Path, camera: str = CAMERAS[0]) -> dict:
    if camera not in CAMERAS:
        raise ValueError("unknown DROID camera")
    verify_inputs(source)
    if output.exists():
        raise ValueError("choose an unused output directory; derivatives are immutable")
    # No TFDS builder discovery/download. These APIs read only the pinned local files.
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    feature = tfds.features.FeatureConnector.from_config(str(source))
    iterator = iter(tf.data.TFRecordDataset([str(source / SHARD)], num_parallel_reads=1))
    record = next(iterator)
    if next(iterator, None) is not None:
        raise ValueError("pinned pilot shard must contain exactly one episode")
    episode = feature.deserialize_example(record)
    metadata = {k: v.numpy().decode("utf-8") for k, v in episode["episode_metadata"].items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".droid-", dir=output.parent))
    process = None
    try:
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                "320x180",
                "-framerate",
                str(FPS),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(temporary / "video.mp4"),
            ],
            stdin=subprocess.PIPE,
        )
        steps, image_hashes = [], []
        for step in tfds.as_numpy(episode["steps"]):
            if len(steps) >= FPS * 300:
                raise ValueError("DROID pilot exceeds 300 seconds at nominal FPS")
            hashes = {}
            for key in CAMERAS:
                frame = step["observation"][key]
                if frame.shape != (180, 320, 3) or frame.dtype != np.uint8:
                    raise ValueError(f"unexpected DROID image: {key}")
                hashes[key] = hashlib.sha256(frame.tobytes()).hexdigest()
            process.stdin.write(step["observation"][camera].tobytes())
            image_hashes.append(hashes)
            low = {k: v for k, v in step.items() if k != "observation"}
            low["observation"] = {k: v for k, v in step["observation"].items() if k not in CAMERAS}
            steps.append(plain(low))
        process.stdin.close()
        if process.wait(timeout=60):
            raise ValueError("DROID video encoding failed")
        source_manifest = source_from_steps(steps, metadata, camera)
        raw = {"episode_metadata": metadata, "steps": steps, "image_rgb_sha256": image_hashes}
        atomic_json(temporary / "rlds-episode.json", raw)
        source_manifest["provenance"] += (
            " Lossless numeric sidecar SHA256 " + file_hash(temporary / "rlds-episode.json") + "."
        )
        atomic_json(temporary / "source.json", source_manifest)
        atomic_json(temporary / "training-schema.json", training_schema())
        generic = np.asarray([s["action"] for s in steps])
        comparison = {
            key: float(
                np.max(np.abs(generic[:, :6] - np.asarray([s["action_dict"][key] for s in steps])))
            )
            for key in ("cartesian_position", "cartesian_velocity")
        }
        evidence = {
            "evidence_kind": "official_real_droid_rlds_decode",
            "source_files": FILES,
            "record_index": 0,
            "episodes": 1,
            "steps": len(steps),
            "decoded_images": {key: len(steps) for key in CAMERAS},
            "selected_camera": camera,
            "clock": CLOCK,
            "nominal_fps": FPS,
            "capture_timestamps_available": False,
            "step_fields": sorted(steps[0]),
            "observation_fields": sorted(steps[0]["observation"]),
            "action_dict_fields": sorted(steps[0]["action_dict"]),
            "generic_action_first_six_max_abs_difference": comparison,
            "generic_action_last_matches_gripper": bool(
                np.array_equal(
                    generic[:, 6:],
                    np.asarray([s["action_dict"]["gripper_position"] for s in steps]),
                )
            ),
            "generic_action_used_for_training": False,
            "training_representation": (
                "7 absolute joint positions (rad) + normalized gripper closure"
            ),
            "versions": {name: version(name) for name in ("tensorflow-datasets", "numpy")},
            "tensorflow_version": tf.__version__,
            "transform": "Official TFDS RGB -> H.264 CRF18/yuv420p, no resize, nominal 15 FPS",
            "files": {p.name: file_hash(p) for p in sorted(temporary.iterdir())},
            "human_labels_validated": False,
        }
        verify_inputs(source)
        atomic_json(temporary / "conversion.json", evidence)
        os.rename(temporary, output)
        return evidence
    finally:
        if process:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--camera", choices=CAMERAS, default=CAMERAS[0])
    args = parser.parse_args()
    result = convert(args.source, args.output, args.camera)
    print(
        json.dumps({"steps": result["steps"], "source_manifest": str(args.output / "source.json")})
    )


if __name__ == "__main__":
    main()
