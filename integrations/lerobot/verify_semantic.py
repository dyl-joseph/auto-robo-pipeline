"""Check native semantic corrections and real DROID values through official LeRobot."""

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

from robo.core import LABELS, atomic_json, file_hash
from robo.export import validate_export
from robo.lerobot_export import convert, validate_training_export
from robo.lerobot_plan import TrainingSchema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("converted", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evidence", type=Path, default=Path("artifacts/semantic-lerobot.json"))
    args = parser.parse_args()
    conversion = json.loads((args.converted / "conversion.json").read_text())
    for name, expected in conversion["files"].items():
        assert file_hash(args.converted / name) == expected
    raw = json.loads((args.converted / "rlds-episode.json").read_text())
    records = validate_export(args.fixture / "accepted")
    assert len(records) == 1
    record = records[0]
    assert record["episode"]["video_sha256"] == conversion["files"]["video.mp4"]
    assert record["episode"]["data_kind"] == "real"
    assert record["episode"]["video_timing"] == "inferred_nominal_fps"
    native = record["annotation"]["native"]
    review = native["semantic_review"]
    assert review["kind"] == "model_assisted_review" and review["prediction_exposure"]
    assert record["split"] == review["split"] == "test"
    assert review["complete"] and len(review["segments"]) == 9
    origin = native["source_binding"]["semantic_prediction"]
    prediction_path = args.fixture / "uncorrected/predictions.json"
    before = file_hash(prediction_path)
    assert before == origin["files"]["predictions.json"]
    predictions = json.loads(prediction_path.read_text())
    assert (
        predictions["episodes"][0]["segments"][0]["subtask_text"]
        != review["segments"][0]["subtask_text"]
    )
    schema = TrainingSchema.model_validate_json(
        (args.converted / "training-schema.json").read_text()
    )
    manifest = convert(
        args.fixture / "accepted", args.fixture / "source", args.output, schema, "test"
    )
    assert (
        convert(args.fixture / "accepted", args.fixture / "source", args.output, schema, "test")
        == manifest
    )
    assert manifest["adapter_version"] == 2
    reader = LeRobotDataset(
        f"local/{manifest['version']}", root=args.output, video_backend="torchcodec"
    )
    assert len(reader) == len(raw["steps"]) == 119 and reader.num_episodes == 1
    for name in ("annotation.subtask_text", "annotation.object_text"):
        assert reader.meta.features[name]["dtype"] == "string"
    for index, step in enumerate(raw["steps"]):
        row = reader[index]
        segment = next(
            s for s in review["segments"] if s["source_start"] <= index < s["source_end_exclusive"]
        )
        assert row["annotation.subtask_text"] == segment["subtask_text"]
        assert row["annotation.object_text"] == (segment["object_text"] or "")
        assert row["annotation.object_present"].item() == (segment["object_text"] is not None)
        assert row["annotation.subtask"].item() == LABELS.index(segment["normalized_action"])
        assert row["source.frame_index"].item() == index
        assert row["task"] == record["episode"]["task"]
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
    assert reader[0]["annotation.subtask_text"] == reader[1]["annotation.subtask_text"]
    assert reader[1]["annotation.subtask_text"] != reader[2]["annotation.subtask_text"]
    assert (
        not reader[1]["annotation.object_present"].item()
        and reader[2]["annotation.object_present"].item()
    )
    assert reader[0]["annotation.object_text"] == reader[1]["annotation.object_text"] == ""
    batch = next(iter(DataLoader(reader, batch_size=8, num_workers=0)))
    assert batch["annotation.subtask_text"] == [
        reader[i]["annotation.subtask_text"] for i in range(8)
    ]
    assert batch["observation.state"].shape == batch["action"].shape == (8, 8)
    assert validate_training_export(args.output) == manifest
    assert validate_export(args.output / "provenance/accepted") == records
    assert file_hash(prediction_path) == before
    atomic_json(
        args.evidence,
        {
            "evidence_kind": "real_droid_automated_semantic_review_to_official_lerobot",
            "frames_loaded_and_checked": len(reader),
            "episodes": 1,
            "segments": 9,
            "split": "test",
            "adapter_version": manifest["adapter_version"],
            "accepted_version": manifest["accepted_version"],
            "training_version": manifest["version"],
            "source_timing": "inferred_nominal_15_hz_no_capture_timestamps",
            "text_and_object_match_corrected_source_frames": True,
            "cleared_object_absent_with_neighbor_preserved": True,
            "numeric_values_match_rlds_float32": True,
            "episode_task_unchanged": True,
            "prediction_sha256_before": before,
            "prediction_sha256_after": file_hash(prediction_path),
            "idempotence_verified": True,
            "dataloader_strings_verified": True,
            "image_batch_shape": list(batch["observation.images.source"].shape),
            "semantic_accuracy_measured": False,
            "human_review_measured": False,
            "policy_training_run": False,
            "pretrained_qwen_run": False,
            "source_license": "CC-BY-4.0",
            "source_attribution": "DROID, Khazatsky et al. (2024)",
        },
    )
    print(f"Real DROID corrected semantic text -> official LeRobot: {len(reader)} rows verified")


if __name__ == "__main__":
    main()
