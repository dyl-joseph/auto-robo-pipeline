"""Test the official LeRobot writer and loader with generated episodes."""

import json
import shutil
import subprocess

import numpy as np
import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

from robo.core import Edit, atomic_json
from robo.export import export_dataset
from robo.ingest import import_episode
from robo.lerobot_export import convert, source_frames, validate_training_export
from robo.lerobot_plan import TrainingSchema
from robo.store import Store


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    directory = tmp_path_factory.mktemp("lerobot")
    store = Store(directory / "store")
    selections, episodes = [], []
    for index, split in enumerate(["train", "train", "test"]):
        video = directory / f"source-{index}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x48:rate=10:duration=1",
                "-vf",
                f"hue=h={index * 30}",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                str(video),
            ],
            check=True,
            timeout=30,
        )
        times = (np.arange(10) / 10).tolist()
        body = {
            "source_id": f"generated-{index}",
            "release": "roundtrip-v1",
            "task": f"generated task {index}",
            "session": f"independent-fixture-{index}",
            "video": video.name,
            "license": "CC0-1.0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "provenance": "FFmpeg pattern and generated signals; not robot measurements",
            "data_kind": "synthetic",
            "streams": [
                {
                    "name": "position",
                    "modality": "state",
                    "units": "m",
                    "timestamps_s": times,
                    "values": [[i + index * 100, i / 2] for i in range(10)],
                    "tolerance_s": 0.01,
                    "clock_evidence": "generated shared clock",
                },
                {
                    "name": "command",
                    "modality": "action",
                    "units": "m/s",
                    "timestamps_s": times,
                    "values": [[-i, index] for i in range(10)],
                    "tolerance_s": 0.01,
                    "clock_evidence": "generated shared clock",
                },
            ],
        }
        manifest = directory / "source.json"
        atomic_json(manifest, body)
        episode = import_episode(manifest, store)
        edit = Edit(
            expected_revision=0,
            actor="software-test",
            rationale="Automated fixture acceptance, not a human study",
            segments=[{"start_s": 0, "end_s": 0.5, "label": "reach"}],
            outcome="failure",
            quality_flags=["recovery"],
            active_s=None,
            wall_s=None,
        )
        store.commit(episode["id"], edit)
        store.commit(episode["id"], edit.model_copy(update={"expected_revision": 1}), "accept")
        selections.append({"episode_id": episode["id"], "revision": 2, "split": split})
        episodes.append(episode)
    accepted = directory / "accepted"
    export_dataset(store, selections, accepted)
    schema = TrainingSchema.model_validate(
        {
            "robot_type": "synthetic-numeric-fixture",
            "fps": 10,
            "width": 64,
            "height": 48,
            "max_video_age_s": 0.01,
            "state": [
                {
                    "stream": "position",
                    "index": i,
                    "name": name,
                    "units": "m",
                    "meaning": "Generated test position, no robot semantics",
                }
                for i, name in enumerate(["x", "y"])
            ],
            "action": [
                {
                    "stream": "command",
                    "index": i,
                    "name": name,
                    "units": "m/s",
                    "meaning": "Generated test command, no control interface",
                }
                for i, name in enumerate(["dx", "dy"])
            ],
        }
    )
    output = directory / "training"
    manifest = convert(accepted, store.root, output, schema, "train")
    return directory, store, accepted, schema, output, manifest


def test_official_loader_numeric_images_labels_and_episode_windows(converted):
    _, store, _, schema, output, manifest = converted
    reader = LeRobotDataset(
        f"local/{manifest['version']}",
        root=output,
        delta_timestamps={"action": [0, 0.1, 0.2]},
    )
    assert len(reader) == 20 and reader.num_episodes == 2
    assert reader.meta.info.codebase_version == "v3.0"
    assert reader.meta.features["observation.state"]["names"] == ["x", "y"]
    for entry in manifest["episodes"]:
        e = entry["episode_index"]
        source = store.episode(entry["source_episode_id"])
        assert source["session"] != "independent-fixture-2"
        originals = source_frames(store.root / source["media"], source, np.arange(10), schema)
        for i, image in enumerate(originals):
            row = reader[e * 10 + i]
            np.testing.assert_allclose(row["observation.state"], source["streams"][0]["values"][i])
            np.testing.assert_allclose(row["action"][0], source["streams"][1]["values"][i])
            assert row["timestamp"].item() == pytest.approx(i / 10)
            assert row["source.timestamp"].item() == pytest.approx(source["original_pts_s"][i])
            assert row["source.frame_index"].item() == i
            assert row["annotation.subtask"].item() == (0 if i < 5 else 7)
            assert row["task"] == source["task"]
            decoded = row["observation.images.source"].permute(1, 2, 0).numpy() * 255
            assert np.abs(decoded.astype(float) - image).mean() < 12  # Lossy H.264 is declared.
        boundary = reader[e * 10 + 9]
        assert boundary["action_is_pad"].tolist() == [False, True, True]
        torch.testing.assert_close(boundary["action"][0], boundary["action"][1])
    batch = next(iter(DataLoader(reader, batch_size=4, num_workers=0)))
    assert batch["observation.state"].shape == (4, 2)
    assert batch["action"].shape == (4, 3, 2)
    assert batch["observation.images.source"].shape == (4, 3, 48, 64)
    assert validate_training_export(output) == manifest


def test_idempotence_and_historical_accepted_provenance(converted):
    _, store, accepted, schema, output, manifest = converted
    assert convert(accepted, store.root, output, schema, "train") == manifest
    records = [
        json.loads(line)
        for line in (output / "provenance/accepted/episodes.jsonl").read_text().splitlines()
    ]
    assert {r["annotation"]["outcome"] for r in records} == {"failure"}
    assert all(r["annotation"]["active_s"] is None for r in records)
    assert all(r["annotation"]["quality_flags"] == ["recovery"] for r in records)
    with pytest.raises(ValueError, match="different training version"):
        convert(accepted, store.root, output, schema.model_copy(update={"fps": 5}), "train")
    with pytest.raises(ValueError, match="no episodes"):
        convert(accepted, store.root, output.parent / "empty", schema, "development")


def test_held_out_conversion_has_only_held_out_episode(converted):
    directory, store, accepted, schema, _, _ = converted
    output = directory / "test-split"
    manifest = convert(accepted, store.root, output, schema, "test")
    assert manifest["split"] == "test" and len(manifest["episodes"]) == 1
    source = store.episode(manifest["episodes"][0]["source_episode_id"])
    assert source["session"] == "independent-fixture-2"
    reader = LeRobotDataset(f"local/{manifest['version']}", root=output)
    assert len(reader) == 10 and reader.num_episodes == 1
    assert all(reader[i]["task"] == "generated task 2" for i in range(10))


def test_export_tampering_and_media_mutation_fail(converted, tmp_path):
    _, store, accepted, schema, output, _ = converted
    copied = tmp_path / "tampered"
    shutil.copytree(output, copied)
    media = next((copied / "videos").rglob("*.mp4"))
    media.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_training_export(copied)
    source_copy = tmp_path / "store"
    shutil.copytree(store.root / "episodes", source_copy / "episodes")
    for source in source_copy.rglob("video.mp4"):
        source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source media hash mismatch"):
        convert(accepted, source_copy, tmp_path / "should-not-exist", schema, "train")
    assert not (tmp_path / "should-not-exist").exists()


def test_failed_decode_never_publishes_partial_dataset(converted, tmp_path, monkeypatch):
    _, store, accepted, schema, _, _ = converted

    def failure(*args):
        for _ in range(4):
            yield np.zeros((48, 64, 3), dtype=np.uint8)
        raise ValueError("deliberate decoder failure")

    monkeypatch.setattr("robo.lerobot_export.source_frames", failure)
    output = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="deliberate decoder failure"):
        convert(accepted, store.root, output, schema, "train")
    assert not output.exists()
    assert not list(tmp_path.glob(".lerobot-*"))
