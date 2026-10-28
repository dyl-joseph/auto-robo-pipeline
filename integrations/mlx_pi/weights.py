"""Strict import of the action tensors in an official LeRobot PI05 safetensors file."""

import mlx.core as mx
from mlx.utils import tree_flatten
from safetensors import safe_open

from .model import ActionExpert, ExpertConfig


def checkpoint_name(name):
    if name.startswith(("layers.", "norm.")):
        return "paligemma_with_expert.gemma_expert.model." + name
    return name


def load_expert(path, config=None):
    config = config or ExpertConfig()
    expert = ActionExpert(config)
    weights = []
    with safe_open(path, framework="numpy") as source:
        for name, parameter in tree_flatten(expert.parameters()):
            key = checkpoint_name(name)
            tensor = source.get_slice(key)
            if tensor.get_shape() != list(parameter.shape) or tensor.get_dtype() != "F32":
                raise ValueError(f"Expected float32 {parameter.shape} for {key}")
            weights.append((name, mx.array(source.get_tensor(key))))
    expert.load_weights(weights, strict=True)
    return expert


def expected_shapes(config=None):
    c = config or ExpertConfig()
    shapes = {}
    for name, incoming, outgoing in (
        ("action_in_proj", c.action_dim, c.width),
        ("action_out_proj", c.width, c.action_dim),
        ("time_mlp_in", c.width, c.width),
        ("time_mlp_out", c.width, c.width),
    ):
        shapes[name + ".weight"] = [outgoing, incoming]
        shapes[name + ".bias"] = [outgoing]
    for i in range(c.depth):
        stem = f"layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            shapes[stem + name + ".dense.weight"] = [3 * c.width, c.width]
            shapes[stem + name + ".dense.bias"] = [3 * c.width]
        for name, incoming, outgoing in (
            ("self_attn.q_proj", c.width, c.heads * c.head_dim),
            ("self_attn.k_proj", c.width, c.kv_heads * c.head_dim),
            ("self_attn.v_proj", c.width, c.kv_heads * c.head_dim),
            ("self_attn.o_proj", c.heads * c.head_dim, c.width),
            ("mlp.gate_proj", c.width, c.mlp_dim),
            ("mlp.up_proj", c.width, c.mlp_dim),
            ("mlp.down_proj", c.mlp_dim, c.width),
        ):
            shapes[stem + name + ".weight"] = [outgoing, incoming]
    shapes["norm.dense.weight"] = [3 * c.width, c.width]
    shapes["norm.dense.bias"] = [3 * c.width]
    return {checkpoint_name(k): v for k, v in shapes.items()}
