import copy

import mlx.core as mx
import numpy as np
import pytest
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from mlx.utils import tree_flatten
from safetensors.numpy import save_file

from robo.droid import training_schema

from .bridge import expert_from_torch, validate_droid_schema
from .reference import small_reference
from .weights import checkpoint_name, expected_shapes, load_expert


def test_checkpoint_inspection_refuses_full_download_before_reading(monkeypatch):
    from .inspect_checkpoint import byte_range

    class FullResponse:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            raise AssertionError("Must not read a full checkpoint response")

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FullResponse())
    with pytest.raises(ValueError, match="refused full checkpoint"):
        byte_range(0, 7)
    with pytest.raises(ValueError, match="budget"):
        byte_range(0, 3_000_000)


def test_official_image_resize_layout_range_and_missing_camera_mask():
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.model = small_reference()
    policy.config = PI05Config(
        device="cpu",
        image_resolution=(16, 16),
        input_features={
            "observation.images.base": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        },
    )
    images, masks = policy._preprocess_images({"observation.images.base": torch.ones(2, 3, 8, 12)})
    assert [x.shape for x in images] == [(2, 3, 16, 16)] * 2
    assert masks[0].all() and not masks[1].any()
    assert images[0].max() == 1 and images[0].min() == -1
    assert images[1].eq(-1).all()


def test_official_discrete_state_prompt_requires_state_and_task():
    step = Pi05PrepareStateTokenizerProcessorStep()
    transition = {
        TransitionKey.OBSERVATION: {"observation.state": torch.tensor([[-1.0, 0.0, 1.0]])},
        TransitionKey.COMPLEMENTARY_DATA: {"task": [" pick_up\n cup "]},
    }
    result = step(copy.deepcopy(transition))
    assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == [
        "Task: pick up  cup, State: 0 128 255;\nAction: "
    ]
    transition[TransitionKey.OBSERVATION] = {}
    with pytest.raises(ValueError, match="State is required"):
        step(transition)


def test_droid_contract_rejects_velocity_wrong_order_and_implicit_mapping():
    schema = training_schema()
    representation = "droid_absolute_joint_positions_gripper_closure"
    result = validate_droid_schema(schema, action_representation=representation)
    assert result["action_dimensions"] == 8
    assert result["checkpoint_semantics_verified"] is False
    with pytest.raises(ValueError, match="explicit"):
        validate_droid_schema(schema, action_representation="pi05_base")
    for change in ("velocity", "order", "units", "gripper"):
        wrong = copy.deepcopy(schema)
        if change == "velocity":
            wrong["action"][0]["stream"] = "action_joint_velocity"
        elif change == "order":
            wrong["action"].reverse()
        elif change == "units":
            wrong["state"][0]["units"] = "degree"
        else:
            wrong["action"][-1]["stream"] = "action_gripper_velocity"
        with pytest.raises(ValueError):
            validate_droid_schema(wrong, action_representation=representation)


def test_checkpoint_tensor_names_shapes_and_exact_conversion(tmp_path):
    expert = expert_from_torch(small_reference())
    tensors = {checkpoint_name(name): np.array(x) for name, x in tree_flatten(expert.parameters())}
    assert {k: list(v.shape) for k, v in tensors.items()} == expected_shapes(expert.config)
    path = tmp_path / "official-layout.safetensors"
    save_file(tensors, path)
    imported = load_expert(path, expert.config)
    for (_, first), (_, second) in zip(
        tree_flatten(expert.parameters()), tree_flatten(imported.parameters()), strict=True
    ):
        np.testing.assert_array_equal(np.array(first), np.array(second))
    tensors["action_out_proj.weight"] = np.zeros((1, 1), dtype=np.float32)
    save_file(tensors, path)
    with pytest.raises(ValueError, match="Expected float32"):
        load_expert(path, expert.config)


def test_masked_loss_excludes_only_padded_action_dimensions():
    expert = expert_from_torch(small_reference())
    from .bridge import as_mlx, encode_prefix
    from .reference import small_inputs

    batch = small_inputs()
    prefix = encode_prefix(
        small_reference(), batch["images"], batch["img_masks"], batch["tokens"], batch["masks"]
    )
    actions, noise, time = [as_mlx(batch[k]) for k in ("actions", "noise", "time")]
    velocity = expert(
        time[:, None, None] * noise + (1 - time[:, None, None]) * actions, time, prefix
    )
    expected = mx.mean(mx.square(velocity - (noise - actions))[..., :4])
    np.testing.assert_array_equal(
        np.array(expert.loss(actions, noise, time, prefix, action_dimensions=4)), np.array(expected)
    )
