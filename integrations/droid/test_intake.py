"""Generated RLDS records exercise intake; they are not semantic reference annotations."""

import json
import shutil

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_datasets as tfds

from integrations.droid import intake as module
from robo.core import file_hash
from robo.droid import CAMERAS


def generated_source(root, *, lengths=(2, 1), bad_record=None, step_count=3):
    root.mkdir()
    numeric = {
        "joint_position": tfds.features.Tensor(shape=(7,), dtype=np.float64),
        "gripper_position": tfds.features.Tensor(shape=(1,), dtype=np.float64),
    }
    cartesian = {
        key: tfds.features.Tensor(shape=(6,), dtype=np.float64)
        for key in ("cartesian_position", "cartesian_velocity")
    }
    features = tfds.features.FeaturesDict(
        {
            "episode_metadata": {"file_path": tfds.features.Text()},
            "steps": tfds.features.Dataset(
                {
                    "observation": {
                        **numeric,
                        **{key: tfds.features.Image(shape=(180, 320, 3)) for key in CAMERAS},
                    },
                    "action_dict": {**numeric, **cartesian},
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float64),
                    "language_instruction": tfds.features.Text(),
                    "is_first": np.bool_,
                    "is_last": np.bool_,
                    "is_terminal": np.bool_,
                }
            ),
        }
    )
    number = 0
    for shard, count in enumerate(lengths):
        path = root / f"generated-train.tfrecord-{shard:05d}-of-{len(lengths):05d}"
        with tf.io.TFRecordWriter(str(path)) as writer:
            for _ in range(count):
                rows = []
                for step in range(step_count):
                    values = {
                        "joint_position": np.full(7, number + step / 10, dtype=np.float64),
                        "gripper_position": np.array([0.3], dtype=np.float64),
                    }
                    rows.append(
                        {
                            "observation": {
                                **values,
                                **{
                                    key: np.full((180, 320, 3), 20 * number + step, dtype=np.uint8)
                                    for key in CAMERAS
                                },
                            },
                            "action_dict": {
                                **values,
                                **{key: np.zeros(6, dtype=np.float64) for key in cartesian},
                            },
                            "action": np.full(7, 999.0, dtype=np.float64),
                            "language_instruction": f"Generated task {number}",
                            "is_first": step == 0,
                            "is_last": step == step_count - 1,
                            "is_terminal": step == step_count - 1,
                        }
                    )
                if number == bad_record:
                    rows[1]["observation"]["joint_position"][0] = float("nan")
                writer.write(
                    features.serialize_example(
                        {
                            "episode_metadata": {
                                "file_path": f"/generated/day/episode-{number}/trajectory.h5"
                            },
                            "steps": rows,
                        }
                    )
                )
                number += 1
    tfds.folder_dataset.write_metadata(
        data_dir=str(root),
        features=features,
        version="1.0.0",
        check_data=False,
        split_infos=[
            tfds.core.SplitInfo(
                name="train",
                shard_lengths=list(lengths),
                num_bytes=sum(p.stat().st_size for p in root.iterdir()),
            )
        ],
    )
    return root


def run(source, output, **kwargs):
    return module.intake(
        source, output, release="generated-fixture/1.0.0", generated_fixture=True, **kwargs
    )


def tree_hashes(root):
    return {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()}


def test_multiple_records_shards_and_increased_limit_resume(tmp_path, monkeypatch):
    source = generated_source(tmp_path / "source")
    output = tmp_path / "out"
    original = tree_hashes(source)
    first = run(source, output, max_episodes=1)
    assert first["status"] == "episode_limit"
    assert first["summary"]["published_episodes"] == 1
    derivatives = tree_hashes(output / "episodes")
    second = run(source, output, max_episodes=3)
    assert second["status"] == "complete"
    assert second["summary"]["published_episodes"] == 3
    assert second["summary"]["published_steps"] == 9
    assert second["summary"]["completed_shards"] == 2
    assert all(tree_hashes(output / "episodes")[p] == sha for p, sha in derivatives.items())
    manifests = [json.loads(p.read_text()) for p in (output / "episodes").glob("*/source.json")]
    assert len({m["source_id"] for m in manifests}) == 3
    assert len({m["session"] for m in manifests}) == 1
    assert all(m["data_kind"] == "synthetic" for m in manifests)
    assert all("Generated DROID-schema fixture" in m["provenance"] for m in manifests)
    assert all(m["streams"][0]["timestamps_s"] == [0, 1 / 15, 2 / 15] for m in manifests)
    assert all(m["streams"][2]["values"][0][0] != 999 for m in manifests)
    monkeypatch.setattr(
        module, "write_episode", lambda *a, **k: pytest.fail("resume decoded again")
    )
    assert run(source, output, max_episodes=3) == second
    assert tree_hashes(source) == original
    relocated = tmp_path / "moved"
    shutil.copytree(source, relocated)
    assert run(relocated, output, max_episodes=3) == second


def test_resume_after_publication_before_checkpoint(tmp_path, monkeypatch):
    source = generated_source(tmp_path / "source", lengths=(1,))
    output = tmp_path / "out"
    real_write = module.write_episode

    def interrupted(*args, **kwargs):
        real_write(*args, **kwargs)
        raise KeyboardInterrupt("simulated process interruption")

    monkeypatch.setattr(module, "write_episode", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(source, output)
    assert len(list((output / "episodes").glob("*/source.json"))) == 1
    monkeypatch.setattr(
        module, "write_episode", lambda *a, **k: pytest.fail("published episode rewritten")
    )
    result = run(source, output)
    assert result["status"] == "complete"
    assert result["summary"]["published_episodes"] == 1


@pytest.mark.parametrize("changed", ["source", "derivative", "metadata"])
def test_changed_inputs_or_outputs_are_not_reused(tmp_path, changed):
    source = generated_source(tmp_path / "source", lengths=(1,))
    output = tmp_path / "out"
    run(source, output)
    if changed == "source":
        path = next(source.glob("*.tfrecord-*"))
        data = path.read_bytes()
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    elif changed == "derivative":
        path = next((output / "episodes").glob("*/video.mp4"))
        path.write_bytes(b"changed")
    else:
        path = source / "dataset_info.json"
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        run(source, output)


def test_missing_shards_require_explicit_partial_mode(tmp_path):
    source = generated_source(tmp_path / "source")
    next(source.glob("*-00001-of-00002")).unlink()
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="Missing 1"):
        run(source, output)
    assert not output.exists()
    result = run(source, output, allow_partial=True)
    assert result["status"] == "partial_source"
    assert result["summary"]["missing_shards"] == 1
    assert result["summary"]["published_episodes"] == 2


@pytest.mark.parametrize("case", ["shard", "selected", "steps", "invalid", "truncated", "count"])
def test_bounds_and_bad_records_report_failures_without_silent_skips(tmp_path, case):
    source = generated_source(tmp_path / "source", bad_record=1 if case == "invalid" else None)
    output = tmp_path / "out"
    kwargs = {}
    if case == "shard":
        kwargs["max_shard_bytes"] = 1
    elif case == "selected":
        size = max(p.stat().st_size for p in source.glob("*.tfrecord-*"))
        kwargs.update(max_shard_bytes=size, max_selected_bytes=size)
    elif case == "steps":
        kwargs["max_steps"] = 2
    elif case == "truncated":
        path = sorted(source.glob("*.tfrecord-*"))[0]
        path.write_bytes(path.read_bytes()[:-10])
    elif case == "count":
        path = source / "dataset_info.json"
        data = json.loads(path.read_text())
        data["splits"][0]["shardLengths"][0] = "3"
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError, tf.errors.DataLossError)):
        run(source, output, **kwargs)
    report = json.loads((output / "intake.json").read_text())
    assert report["status"] == "failed" and report["error"]["message"]
    expected = {"shard": 0, "selected": 2, "steps": 0, "invalid": 1, "truncated": 1, "count": 2}
    assert report["summary"]["published_episodes"] == expected[case]
    assert len(list((output / "episodes").glob("*/source.json"))) == expected[case]
    assert not list((output / ".staging").iterdir())


def test_input_change_during_decode_prevents_publication(tmp_path, monkeypatch):
    source = generated_source(tmp_path / "source", lengths=(1,))
    output = tmp_path / "out"
    real_write = module.write_episode

    def changing(*args, **kwargs):
        path = source / "features.json"
        path.write_text(path.read_text() + "\n")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(module, "write_episode", changing)
    with pytest.raises(ValueError, match="metadata changed"):
        run(source, output)
    assert not list((output / "episodes").iterdir())


def test_output_lock_and_disjoint_source(tmp_path):
    import fcntl

    source = generated_source(tmp_path / "source", lengths=(1,))
    with pytest.raises(ValueError, match="separate"):
        run(source, source / "out")
    output = tmp_path / "out"
    output.mkdir()
    with (output / ".intake.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another intake"):
            run(source, output)
