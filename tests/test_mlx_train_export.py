import copy

import numpy as np
import pytest

from integrations.mlx_pi.train_export import (
    camera_mapping,
    fit_quantiles,
    semantic_chunk,
    training_source,
    verify_preprocessing,
)
from robo.core import digest
from robo.droid import training_schema


@pytest.fixture
def source():
    record = {
        "episode_id": "train-episode",
        "split": "train",
        "annotation": {
            "status": "accepted",
            "revision": 2,
            "native": {
                "semantic_review": {
                    "split": "train",
                    "complete": True,
                    "segments": [{"subtask_text": "move the block"}],
                }
            },
        },
    }
    manifest = {
        "split": "train",
        "schema": training_schema(),
        "episodes": [
            {
                "source_episode_id": "train-episode",
                "accepted_revision": 2,
                "record_hash": digest(record),
            }
        ],
    }
    return manifest, [record]


def test_accepts_reviewed_train_semantics_with_explicit_schema(source):
    schema = training_source(*source, task_key="annotation.subtask_text")
    assert schema.robot_type == "droid-franka-panda-robotiq-2f85"
    assert len(schema.action) == 8
    assert schema.action[0].name == "panda_joint_1_position"
    assert schema.action[-1].name == "gripper_closed_fraction"


@pytest.mark.parametrize("split", ["development", "test"])
def test_refuses_training_on_evaluation_exports(source, split):
    manifest, records = source
    manifest["split"] = split
    with pytest.raises(ValueError, match="accepted train split"):
        training_source(manifest, records, task_key="annotation.subtask_text")


@pytest.mark.parametrize("field,value", [("split", "development"), ("status", "draft")])
def test_requires_train_records_and_explicit_approval(source, field, value):
    manifest, records = source
    if field == "split":
        records[0][field] = value
    else:
        records[0]["annotation"][field] = value
    manifest["episodes"][0]["record_hash"] = digest(records[0])
    with pytest.raises(ValueError, match="explicitly accepted train revisions"):
        training_source(manifest, records, task_key="task")


def test_cannot_relabel_a_development_semantic_review_as_training(source):
    manifest, records = source
    records[0]["annotation"]["native"]["semantic_review"]["split"] = "development"
    manifest["episodes"][0]["record_hash"] = digest(records[0])
    with pytest.raises(ValueError, match="accepted train semantics"):
        training_source(manifest, records, task_key="annotation.subtask_text")


def test_episode_instruction_can_be_selected_without_semantic_text(source):
    manifest, records = copy.deepcopy(source)
    records[0]["annotation"].pop("native")
    manifest["episodes"][0]["record_hash"] = digest(records[0])
    assert training_source(manifest, records, task_key="task").fps == 15
    with pytest.raises(ValueError, match="accepted train semantics"):
        training_source(manifest, records, task_key="annotation.subtask_text")


def test_camera_mapping_is_explicit_and_rejects_ambiguous_targets():
    assert camera_mapping(["observation.images.source=observation.images.base_0_rgb"]) == {
        "observation.images.source": "observation.images.base_0_rgb"
    }
    with pytest.raises(ValueError, match="unique SOURCE=TARGET"):
        camera_mapping(
            [
                "observation.images.source=observation.images.base_0_rgb",
                "observation.images.wrist=observation.images.base_0_rgb",
            ]
        )
    with pytest.raises(ValueError, match="unique SOURCE=TARGET"):
        camera_mapping(["observation.images.source"])
    with pytest.raises(ValueError, match="at least one"):
        camera_mapping([])


def test_quantiles_are_fitted_from_training_vectors():
    result = fit_quantiles([[0, 10], [100, 110]], [[2], [12]])
    np.testing.assert_allclose(result["observation.state"]["q01"], [1.0, 11.0])
    np.testing.assert_allclose(result["observation.state"]["q99"], [99.0, 109.0])
    np.testing.assert_allclose(result["action"]["q01"], [2.1], atol=1e-6)
    np.testing.assert_allclose(result["action"]["q99"], [11.9], atol=1e-6)


@pytest.mark.parametrize("invalid", [[], [[1, 2]], [[1, np.nan], [2, 3]], [1, 2]])
def test_quantile_fit_rejects_incomplete_or_nonfinite_vectors(invalid):
    with pytest.raises(ValueError, match="finite training rows"):
        fit_quantiles(invalid, [[1], [2]])


def test_semantic_chunk_uses_only_a_complete_unchanged_subtask():
    tasks = ["reach"] * 20 + ["move the block"] * 60
    assert semantic_chunk(tasks, 20, 50) == "move the block"
    with pytest.raises(ValueError, match="semantic subtask boundary"):
        semantic_chunk(tasks, 0, 50)
    with pytest.raises(ValueError, match="semantic subtask boundary"):
        semantic_chunk(tasks, 40, 50)


@pytest.fixture
def preprocessing():
    source = {
        "task": " move_the_block\n ",
        "state": np.array([[0, 2]], dtype=np.float32),
        "actions": np.array([[[1, 2], [2, 4]]], dtype=np.float32),
    }
    prepared = {
        "state": np.array([[0, 1]], dtype=np.float32),
        "actions": np.array([[[0, 0], [1, 1]]], dtype=np.float32),
        "padded_actions": np.array([[[0, 0, 0, 0], [1, 1, 0, 0]]], dtype=np.float32),
        "prompt": "Task: move the block, State: 128 255;\nAction: ",
        "token_ids": [2, 42, 13],
    }
    statistics = {
        "observation.state": {"q01": [-2, -2], "q99": [2, 2]},
        "action": {"q01": [0, 0], "q99": [2, 4]},
    }

    class Tokenizer:
        def encode(self, prompt, *, add_bos):
            assert prompt == "Task: move the block, State: 128 255;\nAction: "
            assert add_bos is True
            return [2, 42, 13]

    return source, prepared, statistics, Tokenizer()


def test_processed_task_state_actions_padding_and_tokens_are_verified(preprocessing):
    verify_preprocessing(*preprocessing)
    source, prepared, statistics, tokenizer = preprocessing
    prepared["prompt"] = "Task: unrelated text, State: 128 255;\nAction: "
    with pytest.raises(AssertionError, match="selected task or state prompt"):
        verify_preprocessing(source, prepared, statistics, tokenizer)


@pytest.mark.parametrize("field", ["state", "actions", "padded_actions", "token_ids"])
def test_processed_inputs_cannot_silently_change_training_targets(preprocessing, field):
    source, prepared, statistics, tokenizer = preprocessing
    if field == "token_ids":
        prepared[field] = [2, 99, 13]
    elif field == "padded_actions":
        prepared[field][0, 0, -1] = 1
    else:
        prepared[field][0] += 0.1
    with pytest.raises(AssertionError):
        verify_preprocessing(source, prepared, statistics, tokenizer)
