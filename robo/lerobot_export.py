"""Convert an accepted export with the official LeRobot writer."""

import argparse
import json
import os
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

from robo.core import LABELS, atomic_json, canonical, digest, file_hash
from robo.export import validate_export
from robo.lerobot_plan import TrainingSchema, plan_episode

POLICY = (
    "Derived grid k/fps; latest source frame and stream sample at or before each grid point; "
    "1ns rounding allowance; reject missing/out-of-tolerance samples; no interpolation or padding. "
    "Resize RGB with area interpolation, encode H.264 CRF18; not lossless. "
    "Conversion of a historical accepted version, not a new native approval/freshness check."
)


def source_path(root: Path, episode: dict) -> Path:
    root = root.resolve()
    media = (root / episode["media"]).resolve()
    if not media.is_relative_to(root):
        raise ValueError("source media escapes the declared source store")
    if file_hash(media) != episode["video_sha256"]:
        raise ValueError("source media hash mismatch")
    if (media.parent / "episode.json").read_bytes() != canonical(episode):
        raise ValueError("source manifest differs from the accepted version")
    return media


def source_frames(media: Path, episode: dict, indices, schema: TrainingSchema):
    """Decode sequentially, retaining one source image and the bounded encoder queue."""
    import av
    import cv2

    with av.open(str(media)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "SLICE"
        stream.codec_context.thread_count = 1
        decoded = iter(container.decode(stream))
        current_index, image = -1, None
        for target in indices:
            while current_index < target:
                frame = next(decoded, None)
                current_index += 1
                if frame is None:
                    raise ValueError("source video ended before its measured frame map")
                actual = float(frame.pts * frame.time_base) if frame.pts is not None else None
                expected = episode["original_pts_s"][current_index]
                if actual is None or abs(actual - expected) > 1e-6:
                    raise ValueError("decoded video timestamps differ from source manifest")
                if current_index == target:
                    image = cv2.resize(
                        frame.to_ndarray(format="rgb24"),
                        (schema.width, schema.height),
                        interpolation=cv2.INTER_AREA,
                    )
            yield image


def validate_training_export(output: Path) -> dict:
    manifest = json.loads((output / "provenance/manifest.json").read_text())
    files = manifest["files"]
    actual = {
        p.relative_to(output).as_posix()
        for p in output.rglob("*")
        if p.is_file() and p != output / "provenance/manifest.json"
    }
    if actual != set(files):
        raise ValueError("training export file inventory changed")
    for name, expected in files.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or file_hash(path) != expected:
            raise ValueError("training export file hash mismatch")
    identity = {k: v for k, v in manifest.items() if k not in {"files", "version"}}
    if digest(identity)[:24] != manifest["version"]:
        raise ValueError("training export identity mismatch")
    validate_export(output / "provenance/accepted")
    return manifest


def convert(accepted: Path, source_root: Path, output: Path, schema: TrainingSchema, split: str):
    if split not in {"train", "development", "test"}:
        raise ValueError("choose exactly one train, development or test split")
    all_records = validate_export(accepted)
    records = [r for r in all_records if r["split"] == split]
    if not records:
        raise ValueError("accepted version contains no episodes in the requested split")
    accepted_manifest = json.loads((accepted / "manifest.json").read_text())
    # Validate all alignment/required features before starting a potentially costly encode.
    semantic_modes = set()
    for record in records:
        source_path(source_root, record["episode"])
        semantic_modes.add(plan_episode(record, schema)["semantic"] is not None)
    if len(semantic_modes) != 1:
        raise ValueError("export semantic and legacy reviews separately; text must not be dropped")
    semantic = semantic_modes == {True}
    identity = {
        "schema_version": 1,
        "adapter_version": 2 if semantic else 1,
        "lerobot_version": "0.6.1",
        "accepted_version": accepted_manifest["version"],
        "accepted_records_hash": accepted_manifest["records_hash"],
        "schema": schema.model_dump(),
        "split": split,
        "episodes": [
            {
                "episode_index": i,
                "source_episode_id": r["episode_id"],
                "accepted_revision": r["annotation"]["revision"],
                "record_hash": digest(r),
                "split_group": r["split_group"],
                "data_kind": r["episode"]["data_kind"],
            }
            for i, r in enumerate(records)
        ],
        "label_vocabulary": list(LABELS),
        "policy": POLICY,
        "video_encoding": {"codec": "h264", "crf": 18, "pixel_format": "yuv420p"},
    }
    if semantic:
        identity["semantic_policy"] = (
            "Model-assisted corrected text, not independent reference labels. "
            "Text and action labels follow source.frame_index on the causal video map. "
            "An absent object is empty text with object_present=false. "
            "The episode task remains unchanged."
        )
    export_version = digest(identity)[:24]
    if output.exists():
        existing = validate_training_export(output)
        if existing["version"] != export_version:
            raise ValueError("output already contains a different training version")
        return existing
    # Lazy imports keep PyTorch/LeRobot out of the Label Studio runtime.
    from lerobot.configs import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if version("lerobot") != identity["lerobot_version"]:
        raise ValueError("use the locked integrations/lerobot runtime")
    features = {
        key: {
            "dtype": "float32",
            "shape": (len(components),),
            "names": [c.name for c in components],
        }
        for key, components in (("observation.state", schema.state), ("action", schema.action))
    }
    features.update(
        {
            "observation.images.source": {
                "dtype": "video",
                "shape": (schema.height, schema.width, 3),
                "names": ["height", "width", "channels"],
            },
            "annotation.subtask": {"dtype": "int64", "shape": (1,), "names": ["label_id"]},
            "source.frame_index": {"dtype": "int64", "shape": (1,), "names": ["original_frame"]},
            "source.timestamp": {"dtype": "float64", "shape": (1,), "names": ["original_pts_s"]},
        }
    )
    if semantic:
        features.update(
            {
                name: {"dtype": "string", "shape": (1,), "names": ["text"]}
                for name in ("annotation.subtask_text", "annotation.object_text")
            }
        )
        features["annotation.object_present"] = {
            "dtype": "bool",
            "shape": (1,),
            "names": ["present"],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".lerobot-", dir=output.parent))
    staging = temporary / "dataset"
    writer = None
    try:
        writer = LeRobotDataset.create(
            f"local/{export_version}",
            root=staging,
            fps=schema.fps,
            robot_type=schema.robot_type,
            features=features,
            streaming_encoding=True,
            encoder_queue_maxsize=2,
            encoder_threads=1,
            rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18, preset="fast"),
        )
        for i, record in enumerate(records):
            episode = record["episode"]
            plan = plan_episode(record, schema)
            media = source_path(source_root, episode)
            frames = source_frames(media, episode, plan["video_indices"], schema)
            for j, image in enumerate(frames):
                writer.add_frame(
                    {
                        "task": episode["task"],
                        "observation.images.source": image,
                        **{key: values[j] for key, values in plan["features"].items()},
                        **{key: values[j] for key, values in (plan["semantic"] or {}).items()},
                        "annotation.subtask": np.array([plan["labels"][j]], dtype=np.int64),
                        "source.frame_index": np.array([plan["video_indices"][j]], dtype=np.int64),
                        "source.timestamp": np.array(
                            [plan["alignment"]["video_original_pts_s"][j]], dtype=np.float64
                        ),
                    }
                )
            writer.save_episode()
            source_path(source_root, episode)  # Fail a source mutation during conversion.
            atomic_json(staging / f"provenance/alignment/{i:06d}.json", plan["alignment"])
        writer.finalize()
        writer = None
        snapshot = staging / "provenance/accepted"
        snapshot.mkdir()
        for name in ("manifest.json", "episodes.jsonl", "episodes.parquet"):
            shutil.copyfile(accepted / name, snapshot / name)
        if validate_export(snapshot) != all_records:
            raise ValueError("accepted artifact changed during conversion")
        manifest = {**identity, "version": export_version}
        manifest["files"] = {
            p.relative_to(staging).as_posix(): file_hash(p)
            for p in sorted(staging.rglob("*"))
            if p.is_file()
        }
        atomic_json(staging / "provenance/manifest.json", manifest)
        validate_training_export(staging)
        os.rename(staging, output)
        return manifest
    finally:
        try:
            if writer is not None:
                writer.clear_episode_buffer()
                writer.finalize()
        finally:
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "development", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = convert(
        args.accepted,
        args.source_root,
        args.output,
        TrainingSchema.model_validate_json(args.schema.read_text()),
        args.split,
    )
    print(json.dumps({"version": result["version"], "episodes": len(result["episodes"])}))


if __name__ == "__main__":
    main()
