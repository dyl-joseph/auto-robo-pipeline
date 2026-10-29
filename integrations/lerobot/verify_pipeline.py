"""Read the workflow's real DROID training export through the official LeRobot loader."""

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

from robo.core import LABELS, atomic_json, file_hash
from robo.export import validate_export
from robo.lerobot_export import validate_training_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.workflow / "report.json").read_text())
    assert report["status"] == "complete" and report["exit_status"] == 0
    accepted, training = (Path(report["export"][key]) for key in ("accepted", "training"))
    records = validate_export(accepted)
    assert len(records) == 1
    record = records[0]
    assert record["episode"]["data_kind"] == "real"
    assert record["episode"]["video_timing"] == "inferred_nominal_fps"
    assert record["split"] == "development"
    raw_paths = list((args.workflow / "intake/episodes").glob("*/rlds-episode.json"))
    assert len(raw_paths) == 1
    raw = json.loads(raw_paths[0].read_text())
    review = record["annotation"]["native"]["semantic_review"]
    assert review["complete"] and review["kind"] == "model_assisted_review"
    original = json.loads((args.workflow / "predictions.json").read_text())
    assert original["producer"] == "robot-state-baseline-v1"
    assert (
        original["episodes"][0]["segments"][0]["subtask_text"]
        != review["segments"][0]["subtask_text"]
    )
    manifest = validate_training_export(training)
    reader = LeRobotDataset(f"local/{manifest['version']}", root=training)
    assert len(reader) == len(raw["steps"]) == 119 and reader.num_episodes == 1
    for index, step in enumerate(raw["steps"]):
        row = reader[index]
        frame = row["source.frame_index"].item()
        assert frame == index
        segment = next(
            s for s in review["segments"] if s["source_start"] <= frame < s["source_end_exclusive"]
        )
        assert row["annotation.subtask_text"] == segment["subtask_text"]
        assert row["annotation.object_text"] == (segment["object_text"] or "")
        assert row["annotation.object_present"].item() == (segment["object_text"] is not None)
        assert row["annotation.subtask"].item() == LABELS.index(segment["normalized_action"])
        np.testing.assert_array_equal(
            row["observation.state"].numpy(),
            np.asarray(
                step["observation"]["joint_position"] + step["observation"]["gripper_position"],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            row["action"].numpy(),
            np.asarray(
                step["action_dict"]["joint_position"] + step["action_dict"]["gripper_position"],
                dtype=np.float32,
            ),
        )
        image = row["observation.images.source"]
        assert image.shape == (3, 180, 320) and image.isfinite().all()
        assert 0 <= image.min().item() <= image.max().item() <= 1
    batch = next(iter(DataLoader(reader, batch_size=8, num_workers=0)))
    assert batch["annotation.subtask_text"] == [
        reader[i]["annotation.subtask_text"] for i in range(8)
    ]
    assert batch["action"].shape == batch["observation.state"].shape == (8, 8)
    assert file_hash(args.workflow / "predictions.json") == report["prediction_sha256"]
    atomic_json(
        args.evidence,
        {
            "evidence_kind": "real_droid_baseline_orchestration_and_automated_api_review",
            "rows_checked": len(reader),
            "episodes": 1,
            "segments": len(review["segments"]),
            "split": record["split"],
            "accepted_version": manifest["accepted_version"],
            "training_version": manifest["version"],
            "prediction_sha256": report["prediction_sha256"],
            "source_frame_text_alignment": True,
            "numeric_values_match_rlds_float32": True,
            "official_lerobot_and_dataloader": True,
            "video_backend": reader._video_backend,
            "source_timing": "inferred_nominal_15_hz",
            "pretrained_qwen_run": False,
            "semantic_accuracy_measured": False,
            "human_efficiency_measured": False,
            "browser_interaction_tested_here": False,
            "source_license": "CC-BY-4.0",
            "source_attribution": "DROID, Khazatsky et al. (2024)",
        },
    )
    print(f"Loaded and compared {len(reader)} real DROID rows through official LeRobot")


if __name__ == "__main__":
    main()
