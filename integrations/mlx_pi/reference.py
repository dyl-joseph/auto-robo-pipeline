"""Small configurations of the installed official model, used only for numerical verification."""

import os
import platform
from types import SimpleNamespace

import mlx.core as mx
import torch
from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel, PI05Pytorch
from lerobot.policies.pi_gemma import (
    PaliGemmaForConditionalGenerationWithPiGemma,
    PiGemmaForCausalLM,
)
from transformers import GemmaConfig, PaliGemmaConfig, SiglipVisionConfig


def configure_runtime(device):
    if device == "gpu":
        if not mx.metal.is_available() or not torch.backends.mps.is_available():
            raise RuntimeError("Verification requires MLX Metal and PyTorch MPS")
        if os.environ.get("MLX_ENABLE_TF32") != "0":
            raise RuntimeError("Launch with MLX_ENABLE_TF32=0 for float32 verification")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
            raise RuntimeError("PyTorch MPS CPU fallback must be disabled")
        mx.set_default_device(mx.gpu)
        return torch.device("mps")
    if platform.system() == "Darwin":
        raise RuntimeError("Local verification requires Metal; CPU verification is for Linux CI")
    mx.set_default_device(mx.cpu)
    torch.set_num_threads(2)
    return torch.device("cpu")


def reference_device():
    if platform.system() == "Darwin":
        if not torch.backends.mps.is_available():
            raise RuntimeError("The local PI05 reference requires PyTorch MPS")
        return torch.device("mps")
    return torch.device("cpu")


def small_reference(seed=7, device=None):
    with torch.device(device or reference_device()):
        return _small_reference(seed)


def _small_reference(seed):
    torch.manual_seed(seed)
    common = dict(
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        hidden_activation="gelu_pytorch_tanh",
        attention_dropout=0.0,
        attn_implementation="eager",
    )
    text = GemmaConfig(hidden_size=64, intermediate_size=96, **common)
    vision = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=16,
        patch_size=8,
    )
    config = PaliGemmaConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        projection_dim=64,
        image_token_index=63,
    )
    expert_config = GemmaConfig(
        hidden_size=32, intermediate_size=64, use_adarms=True, adarms_cond_dim=32, **common
    )
    # Build the official classes with reduced dimensions, without allocating their default 3B VLM.
    wrapper = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config)
    wrapper.gemma_expert = PiGemmaForCausalLM(expert_config)
    wrapper.gemma_expert.model.embed_tokens = None
    wrapper.freeze_vision_encoder = True
    wrapper.train_expert_only = True
    core = PI05Pytorch.__new__(PI05Pytorch)
    torch.nn.Module.__init__(core)
    core.config = SimpleNamespace(
        chunk_size=3, min_period=0.004, max_period=4.0, num_inference_steps=3, rtc_config=None
    )
    core.gradient_checkpointing_enabled = False
    core.rtc_processor = None
    core.paligemma_with_expert = wrapper
    core.action_in_proj = torch.nn.Linear(8, 32)
    core.action_out_proj = torch.nn.Linear(32, 8)
    core.time_mlp_in = torch.nn.Linear(32, 32)
    core.time_mlp_out = torch.nn.Linear(32, 32)
    return core.eval()


def small_inputs(device=None):
    with torch.device(device or reference_device()):
        return _small_inputs()


def _small_inputs():
    torch.manual_seed(11)
    return dict(
        images=[
            torch.rand(2, 3, 16, 16) * 2 - 1,
            torch.rand(2, 3, 16, 16) * 2 - 1,
        ],
        img_masks=[torch.tensor([True, True]), torch.tensor([False, True])],
        tokens=torch.tensor([[2, 3, 0, 0], [4, 5, 6, 0]]),
        masks=torch.tensor([[True, True, False, False], [True, True, True, False]]),
        actions=torch.randn(2, 3, 8),
        noise=torch.randn(2, 3, 8),
        time=torch.tensor([0.17, 0.81]),
    )
