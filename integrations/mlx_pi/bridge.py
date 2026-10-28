"""Reuse LeRobot preprocessing and frozen PI05 prefix computation."""

import mlx.core as mx
import numpy as np
import torch
from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d

from .model import ActionExpert, ExpertConfig, Prefix


def as_mlx(tensor):
    return mx.array(tensor.detach().float().cpu().numpy())


def expert_from_torch(core):
    if any(p.dtype != torch.float32 for p in core.parameters()):
        raise ValueError("This adapter currently verifies float32 PI05 models only")
    config = core.paligemma_with_expert.gemma_expert.config
    expert = ActionExpert(
        ExpertConfig(
            width=config.hidden_size,
            depth=config.num_hidden_layers,
            mlp_dim=config.intermediate_size,
            heads=config.num_attention_heads,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            action_dim=core.action_in_proj.in_features,
            horizon=core.config.chunk_size,
            min_period=core.config.min_period,
            max_period=core.config.max_period,
        )
    )
    weights = []
    for name, tensor in core.state_dict().items():
        if name.startswith("paligemma_with_expert.gemma_expert.model."):
            name = name.removeprefix("paligemma_with_expert.gemma_expert.model.")
        elif not name.startswith(
            ("action_in_proj.", "action_out_proj.", "time_mlp_in.", "time_mlp_out.")
        ):
            continue
        weights.append((name, as_mlx(tensor)))
    expert.load_weights(weights, strict=True)
    return expert


@torch.no_grad()
def encode_prefix(core, images, image_masks, tokens, token_masks):
    """Consume official preprocessed inputs; no tokenizer or image encoder is reimplemented."""
    core.eval()
    vlm = core.paligemma_with_expert.paligemma
    if any(p.dtype != torch.float32 for p in vlm.parameters()):
        raise ValueError("The frozen prefix encoder must use float32")
    vlm.requires_grad_(False)
    vlm.model.language_model.config._attn_implementation = "eager"
    embeddings, valid, block = core.embed_prefix(images, image_masks, tokens, token_masks)
    _, cache = core.paligemma_with_expert.forward(
        attention_mask=prepare_attention_masks_4d(make_att_2d_masks(valid, block)),
        position_ids=torch.cumsum(valid, dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[embeddings, None],
        use_cache=True,
    )
    return Prefix(
        tuple(as_mlx(layer.keys) for layer in cache.layers),
        tuple(as_mlx(layer.values) for layer in cache.layers),
        mx.array(valid.cpu().numpy()),
    )


def prepare_batch(policy, preprocessor, raw_batch):
    """Run the caller's saved LeRobot processor, including statistics and state tokenization."""
    batch = preprocessor(raw_batch)
    images, image_masks = policy._preprocess_images(batch)
    prefix = encode_prefix(
        policy.model,
        images,
        image_masks,
        batch["observation.language.tokens"],
        batch["observation.language.attention_mask"],
    )
    return prefix, as_mlx(policy.prepare_action(batch))


def validate_droid_schema(schema, *, action_representation):
    """Base PI05 has no robot semantics. Require the fine-tuning contract to name them."""
    if action_representation != "droid_absolute_joint_positions_gripper_closure":
        raise ValueError("DROID export requires an explicit absolute-joint/gripper contract")
    for kind in ("state", "action"):
        fields = schema[kind]
        expected = [(f"panda_joint_{i}_position", "rad", i - 1) for i in range(1, 8)]
        expected.append(("gripper_closed_fraction", "normalized", 0))
        actual = [(f["name"], f["units"], f["index"]) for f in fields]
        if actual != expected or any(f["stream"] != f"{kind}_joint_position" for f in fields[:7]):
            raise ValueError(f"DROID {kind} order, units or representation do not match")
        if fields[-1]["stream"] != f"{kind}_gripper_position":
            raise ValueError("DROID gripper representation does not match")
    if schema["robot_type"] != "droid-franka-panda-robotiq-2f85":
        raise ValueError("Robot embodiment does not match the DROID adapter")
    return {
        "action_dimensions": 8,
        "representation": action_representation,
        "checkpoint_semantics_verified": False,
    }


def numpy_prefix(prefix):
    return {
        "valid": np.array(prefix.valid),
        **{f"key_{i}": np.array(x) for i, x in enumerate(prefix.keys)},
        **{f"value_{i}": np.array(x) for i, x in enumerate(prefix.values)},
    }
