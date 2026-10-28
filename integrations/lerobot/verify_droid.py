"""Verify accepted real DROID against decoded RLDS values with the official LeRobot loader."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

from robo.core import LABELS, atomic_json, file_hash
from robo.droid import FILES, FPS, SHARD
from robo.export import validate_export
from robo.lerobot_export import convert, validate_training_export
from robo.lerobot_plan import TrainingSchema, plan_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("converted", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evidence", type=Path, default=Path("artifacts/droid-lerobot.json"))
    args = parser.parse_args()
    conversion = json.loads((args.converted / "conversion.json").read_text())
    for name, expected in conversion["files"].items():
        assert file_hash(args.converted / name) == expected
    assert conversion["source_files"] == FILES
    raw = json.loads((args.converted / "rlds-episode.json").read_text())
    schema = TrainingSchema.model_validate_json(
        (args.converted / "training-schema.json").read_text()
    )
    records = validate_export(args.fixture / "accepted")
    assert len(records) == 1
    record = records[0]
    episode, annotation = record["episode"], record["annotation"]
    assert episode["video_sha256"] == conversion["files"]["video.mp4"]
    assert episode["data_kind"] == "real" and episode["license"] == "CC-BY-4.0"
    assert episode["video_timing"] == "inferred_nominal_fps"
    assert annotation["status"] == "accepted" and annotation["native"]["annotation_id"] > 0
    assert annotation["outcome"] == "unknown" and annotation["quality_flags"] == ["incomplete"]
    assert annotation["active_s"] is None and annotation["wall_s"] is None
    assert annotation["segments"][0]["end_s"] == 6 / FPS
    for edge in annotation["native"]["boundary_evidence"]:
        for key in ("start", "end"):
            assert not edge[key]["is_measured_source_boundary"]
            assert edge[key]["source_video_timing"] == "inferred_nominal_fps"
    manifest = convert(
        args.fixture / "accepted", args.fixture / "source", args.output, schema, "train"
    )
    assert (
        convert(args.fixture / "accepted", args.fixture / "source", args.output, schema, "train")
        == manifest
    )
    reader = LeRobotDataset(
        f"local/{manifest['version']}",
        root=args.output,
        video_backend="torchcodec",
        delta_timestamps={"action": [0, 1 / FPS]},
    )
    plan = plan_episode(record, schema)
    assert len(reader) == len(raw["steps"]) == 119
    assert reader.num_episodes == 1
    stored_times = []
    for path in sorted(args.output.glob("data/**/*.parquet")):
        table = pq.read_table(path, columns=["source.timestamp"])
        assert str(table.schema.field("source.timestamp").type) == "double"
        stored_times.extend(table.column("source.timestamp").to_pylist())
    np.testing.assert_array_equal(stored_times, np.arange(119, dtype=np.float64) / FPS)
    for index, step in enumerate(raw["steps"]):
        row = reader[index]
        expected_state = np.asarray(
            step["observation"]["joint_position"] + step["observation"]["gripper_position"],
            dtype=np.float32,
        )
        expected_action = np.asarray(
            step["action_dict"]["joint_position"] + step["action_dict"]["gripper_position"],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(row["observation.state"].numpy(), expected_state)
        np.testing.assert_array_equal(row["action"][0].numpy(), expected_action)
        assert row["source.frame_index"].item() == index
        assert abs(row["timestamp"].item() - index / FPS) < 1e-6
        # LeRobot's Hugging Face -> torch transform casts even stored double values.
        # The Parquet column and alignment JSON above retain the original precision.
        assert row["source.timestamp"].item() == float(np.float32(index / FPS))
        image = row["observation.images.source"]
        assert image.shape == (3, 180, 320) and image.isfinite().all()
        assert 0 <= image.min().item() <= image.max().item() <= 1
        assert row["annotation.subtask"].item() == plan["labels"][index]
    assert reader[5]["annotation.subtask"].item() == LABELS.index("hold")
    assert reader[6]["annotation.subtask"].item() == LABELS.index("unknown")
    assert reader[len(reader) - 1]["action_is_pad"].tolist() == [False, True]
    batch = next(iter(DataLoader(reader, batch_size=8, num_workers=0)))
    assert batch["observation.state"].shape == (8, 8)
    assert batch["action"].shape == (8, 2, 8)
    assert plan["alignment"]["omitted_source_frames"] == 0
    assert plan["alignment"]["repeated_source_frames"] == 0
    assert plan["alignment"]["source_video_timing"] == "inferred_nominal_fps"
    assert validate_training_export(args.output) == manifest
    atomic_json(
        args.evidence,
        {
            "evidence_kind": "real_droid_native_browser_sdk_to_official_lerobot",
            "versions": {name: version(name) for name in ("lerobot", "torch", "torchcodec", "av")},
            "source_shard_sha256": FILES[SHARD]["sha256"],
            "accepted_version": manifest["accepted_version"],
            "training_version": manifest["version"],
            "frames_loaded_and_checked": len(reader),
            "episodes": reader.num_episodes,
            "image_batch_shape": list(batch["observation.images.source"].shape),
            "state_batch_shape": list(batch["observation.state"].shape),
            "action_batch_shape": list(batch["action"].shape),
            "numeric_values_match_rlds_float32": True,
            "native_outcome": annotation["outcome"],
            "native_quality_flags": annotation["quality_flags"],
            "native_corrected_boundary_s": 6 / FPS,
            "boundary_label_mapping_verified": True,
            "source_timing": "inferred_nominal_15_hz_no_capture_timestamps",
            "timestamp_storage_dtype": "float64",
            "timestamp_loader_dtype": "float32",
            "repeated_source_frames": 0,
            "source_frames_omitted": 0,
            "idempotence_verified": True,
            "action_window_padding_verified": True,
            "human_labels_validated": False,
            "human_efficiency_measured": False,
            "optimizer_step_run": False,
            "policy_evaluation_run": False,
            "apple_hardware_verified": False,
        },
    )
    print(f"Real DROID native accepted export -> official LeRobot: {len(reader)} frames verified")


if __name__ == "__main__":
    main()
