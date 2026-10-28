# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 Big Vision Authors.
# Adapted for MLX under Apache-2.0; this file contains modifications.
# See NOTICE.md for the pinned sources and license.
import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class ExpertConfig:
    width: int = 1024
    depth: int = 18
    mlp_dim: int = 4096
    heads: int = 8
    kv_heads: int = 1
    head_dim: int = 256
    action_dim: int = 32
    horizon: int = 50
    min_period: float = 0.004
    max_period: float = 4.0

    def __post_init__(self):
        values = (
            self.width,
            self.depth,
            self.mlp_dim,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.action_dim,
            self.horizon,
        )
        if min(values) <= 0 or self.width % 2 or self.head_dim % 2:
            raise ValueError("Positive dimensions and even embedding/head widths are required")
        if self.heads % self.kv_heads or not 0 < self.min_period < self.max_period:
            raise ValueError("Invalid grouped attention or timestep periods")


@dataclass(frozen=True)
class Prefix:
    """Frozen, post-RoPE keys and values: one [batch, kv_heads, tokens, head_dim] per layer."""

    keys: tuple[mx.array, ...]
    values: tuple[mx.array, ...]
    valid: mx.array

    def validate(self, config: ExpertConfig, batch: int):
        if self.valid.ndim != 2 or self.valid.shape[0] != batch or self.valid.dtype != mx.bool_:
            raise ValueError("Prefix mask must be boolean [batch, tokens]")
        if len(self.keys) != config.depth or len(self.values) != config.depth:
            raise ValueError("Prefix depth does not match action expert")
        shape = (batch, config.kv_heads, self.valid.shape[1], config.head_dim)
        if any(x.shape != shape for x in (*self.keys, *self.values)):
            raise ValueError("Prefix KV layout does not match action expert")


def rope(x, positions):
    frequencies = 10000 ** (-mx.arange(x.shape[-1] // 2, dtype=mx.float32) * 2 / x.shape[-1])
    angle = positions[:, None, :, None] * frequencies
    first, second = mx.split(x, 2, axis=-1)
    return mx.concatenate(
        (
            first * mx.cos(angle) - second * mx.sin(angle),
            second * mx.cos(angle) + first * mx.sin(angle),
        ),
        axis=-1,
    )


def time_embedding(time, config):
    fraction = mx.linspace(0, 1, config.width // 2)
    period = config.min_period * (config.max_period / config.min_period) ** fraction
    angle = time[:, None] * (2 * math.pi / period)
    return mx.concatenate((mx.sin(angle), mx.cos(angle)), axis=-1)


class AdaptiveNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.dense = nn.Linear(width, 3 * width)
        self.dense.weight = mx.zeros_like(self.dense.weight)
        self.dense.bias = mx.zeros_like(self.dense.bias)

    def __call__(self, x, condition):
        normalized = x * mx.rsqrt(
            mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True) + 1e-6
        )
        scale, shift, gate = mx.split(self.dense(condition)[:, None, :], 3, axis=-1)
        return normalized * (1 + scale) + shift, gate


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.width, config.heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.width, config.kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.width, config.kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.heads * config.head_dim, config.width, bias=False)

    def __call__(self, x, positions, prefix_key, prefix_value, valid):
        b, s, _ = x.shape
        c = self.config
        q = rope(self.q_proj(x).reshape(b, s, c.heads, c.head_dim).transpose(0, 2, 1, 3), positions)
        k = rope(
            self.k_proj(x).reshape(b, s, c.kv_heads, c.head_dim).transpose(0, 2, 1, 3), positions
        )
        v = self.v_proj(x).reshape(b, s, c.kv_heads, c.head_dim).transpose(0, 2, 1, 3)
        k = mx.repeat(mx.concatenate((prefix_key, k), axis=2), c.heads // c.kv_heads, axis=1)
        v = mx.repeat(mx.concatenate((prefix_value, v), axis=2), c.heads // c.kv_heads, axis=1)
        logits = (q @ k.transpose(0, 1, 3, 2)) * (c.head_dim**-0.5)
        # The action chunk is one bidirectional block; only padded prefix tokens are hidden.
        mask = mx.concatenate((valid, mx.ones((b, s), dtype=mx.bool_)), axis=1)
        probabilities = mx.softmax(mx.where(mask[:, None, None, :], logits, -2.3819763e38), axis=-1)
        return self.o_proj((probabilities @ v).transpose(0, 2, 1, 3).reshape(b, s, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.width, config.mlp_dim, bias=False)
        self.up_proj = nn.Linear(config.width, config.mlp_dim, bias=False)
        self.down_proj = nn.Linear(config.mlp_dim, config.width, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = AdaptiveNorm(config.width)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = AdaptiveNorm(config.width)
        self.mlp = MLP(config)

    def __call__(self, x, condition, positions, key, value, valid):
        normalized, gate = self.input_layernorm(x, condition)
        x = x + gate * self.self_attn(normalized, positions, key, value, valid)
        normalized, gate = self.post_attention_layernorm(x, condition)
        return x + gate * self.mlp(normalized)


class ActionExpert(nn.Module):
    """PI05 suffix with the official encoder held fixed. Parameters use PyTorch layouts."""

    def __init__(self, config: ExpertConfig):
        super().__init__()
        self.config = config
        self.action_in_proj = nn.Linear(config.action_dim, config.width)
        self.action_out_proj = nn.Linear(config.width, config.action_dim)
        self.time_mlp_in = nn.Linear(config.width, config.width)
        self.time_mlp_out = nn.Linear(config.width, config.width)
        self.layers = [Layer(config) for _ in range(config.depth)]
        self.norm = AdaptiveNorm(config.width)

    def __call__(self, noisy_actions, time, prefix: Prefix):
        c = self.config
        if noisy_actions.shape[1:] != (c.horizon, c.action_dim):
            raise ValueError("Actions must match configured [batch, horizon, action_dim]")
        b = noisy_actions.shape[0]
        if time.shape != (b,):
            raise ValueError("Time must have shape [batch]")
        prefix.validate(c, b)
        condition = nn.silu(self.time_mlp_out(nn.silu(self.time_mlp_in(time_embedding(time, c)))))
        x = self.action_in_proj(noisy_actions)
        positions = mx.sum(prefix.valid, axis=1)[:, None] + mx.arange(c.horizon)[None, :]
        for layer, key, value in zip(self.layers, prefix.keys, prefix.values, strict=True):
            x = layer(x, condition, positions, key, value, prefix.valid)
        x, _ = self.norm(x, condition)
        return self.action_out_proj(x)

    def loss(self, actions, noise, time, prefix, *, action_dimensions=None):
        if actions.shape != noise.shape:
            raise ValueError("Noise and action shapes must match")
        dimensions = self.config.action_dim if action_dimensions is None else action_dimensions
        if not 0 < dimensions <= self.config.action_dim:
            raise ValueError("Invalid number of supervised action dimensions")
        t = time[:, None, None]
        velocity = self(t * noise + (1 - t) * actions, time, prefix)
        return mx.mean(mx.square(velocity - (noise - actions))[..., :dimensions])

    def sample(self, noise, prefix, *, steps=10):
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("Euler steps must be a positive integer")
        x = noise
        for step in range(steps):
            time = mx.full((x.shape[0],), 1 - step / steps)
            x = x - self(x, time, prefix) / steps
        return x
