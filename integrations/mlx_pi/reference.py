"""Small configurations of the installed official model, used only for numerical verification."""

from types import SimpleNamespace

import torch
from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel, PI05Pytorch
from lerobot.policies.pi_gemma import (
    PaliGemmaForConditionalGenerationWithPiGemma,
    PiGemmaForCausalLM,
)
from transformers import GemmaConfig, PaliGemmaConfig, SiglipVisionConfig


def small_reference(seed=7):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
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


def small_inputs():
    generator = torch.Generator().manual_seed(11)
    return dict(
        images=[
            torch.rand(2, 3, 16, 16, generator=generator) * 2 - 1,
            torch.rand(2, 3, 16, 16, generator=generator) * 2 - 1,
        ],
        img_masks=[torch.tensor([True, True]), torch.tensor([False, True])],
        tokens=torch.tensor([[2, 3, 0, 0], [4, 5, 6, 0]]),
        masks=torch.tensor([[True, True, False, False], [True, True, True, False]]),
        actions=torch.randn(2, 3, 8, generator=generator),
        noise=torch.randn(2, 3, 8, generator=generator),
        time=torch.tensor([0.17, 0.81]),
    )
