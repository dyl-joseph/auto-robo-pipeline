"""Load the real native browser/SDK smoke's accepted synthetic version with LeRobot."""

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

from robo.core import LABELS, atomic_json
from robo.export import validate_export
from robo.lerobot_export import convert, validate_training_export
from robo.lerobot_plan import TrainingSchema, plan_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evidence", type=Path, default=Path("artifacts/lerobot-native.json"))
    args = parser.parse_args()
    schema = TrainingSchema.model_validate_json(
        Path("integrations/lerobot/synthetic-schema.json").read_text()
    )
    record = validate_export(args.fixture / "accepted")[0]
    assert record["annotation"]["native"]["annotation_id"] > 0
    assert record["annotation"]["outcome"] == "failure"
    assert record["annotation"]["quality_flags"] == ["recovery"]
    assert record["annotation"]["active_s"] is None
    manifest = convert(
        args.fixture / "accepted", args.fixture / "source", args.output, schema, "train"
    )
    assert (
        convert(args.fixture / "accepted", args.fixture / "source", args.output, schema, "train")
        == manifest
    )
    reader = LeRobotDataset(
        f"local/{manifest['version']}", root=args.output, video_backend="torchcodec"
    )
    plan = plan_episode(record, schema)
    assert len(reader) == len(plan["grid"]) == 96
    for i in range(len(reader)):
        row = reader[i]
        np.testing.assert_allclose(
            row["observation.state"], plan["features"]["observation.state"][i, 0]
        )
        np.testing.assert_allclose(row["action"], plan["features"]["action"][i, 0])
        assert row["source.frame_index"].item() == plan["video_indices"][i]
        assert row["annotation.subtask"].item() == plan["labels"][i]
        assert row["observation.images.source"].shape == (3, 180, 320)
    assert reader[5]["annotation.subtask"].item() == LABELS.index("hold")
    assert reader[6]["annotation.subtask"].item() != LABELS.index("hold")
    batch = next(iter(DataLoader(reader, batch_size=8, num_workers=0)))
    assert validate_training_export(args.output) == manifest
    atomic_json(
        args.evidence,
        {
            "evidence_kind": "synthetic_native_browser_sdk_to_official_lerobot",
            "versions": {
                package: version(package) for package in ["lerobot", "torch", "torchcodec", "av"]
            },
            "accepted_version": manifest["accepted_version"],
            "training_version": manifest["version"],
            "frames_loaded_and_checked": len(reader),
            "episodes": reader.num_episodes,
            "image_batch_shape": list(batch["observation.images.source"].shape),
            "native_outcome": record["annotation"]["outcome"],
            "native_quality_flags": record["annotation"]["quality_flags"],
            "boundary_s": 0.5,
            "boundary_label_mapping_verified": True,
            "original_pts_and_alignment_retained": True,
            "idempotence_verified": True,
            "max_video_age_s": max(plan["alignment"]["video_age_s"]),
            "repeated_source_frames": plan["alignment"]["repeated_source_frames"],
            "source_frames_omitted": plan["alignment"]["omitted_source_frames"],
            "human_efficiency_measured": False,
            "real_droid_verified": False,
            "optimizer_step_run": False,
            "policy_evaluation_run": False,
        },
    )
    print(f"Native accepted export -> official LeRobot: {len(reader)} frames verified")


if __name__ == "__main__":
    main()
