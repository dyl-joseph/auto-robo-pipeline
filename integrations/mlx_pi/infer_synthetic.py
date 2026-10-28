"""Compare full local PI05 inference on generated tensors using MPS and MLX Metal."""

import argparse
import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import chain
from pathlib import Path

import mlx.core as mx
import numpy as np
import sentencepiece as spm
import torch
from lerobot.policies.pi05 import processor_pi05
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
from lerobot.processor import (
    PolicyProcessorPipeline,
    policy_action_to_transition,
    transition_to_policy_action,
)
from safetensors.torch import load_file
from transformers import GemmaTokenizer

from .bridge import as_mlx, encode_prefix, expert_from_torch

STEPS = 10
ATOL = 5e-5
RTOL = 5e-5


@dataclass(frozen=True)
class SyntheticInputs:
    observations: dict[str, torch.Tensor | str]
    statistics: dict[str, dict[str, list[float]]]
    noise: torch.Tensor


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _load_core(checkpoint: Path) -> tuple[PI05Pytorch, dict]:
    config = PI05Config.from_pretrained(checkpoint, local_files_only=True)
    config.device = "mps"
    config.dtype = "float32"
    config.compile_model = False
    core = PI05Pytorch(config)
    state = load_file(checkpoint / "model.safetensors", device="cpu")
    source_tensor_count = len(state)
    if any(tensor.dtype != torch.float32 for tensor in state.values()):
        raise ValueError("The local checkpoint must contain only float32 tensors")
    embedding = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    head = "paligemma_with_expert.paligemma.lm_head.weight"
    aliases = {}
    if embedding not in state and head in state:
        state[embedding] = state[head]
        aliases[embedding] = head
    core.load_state_dict(state, strict=True, assign=True)
    loaded_tensor_count = len(core.state_dict())
    del state
    core.requires_grad_(False).eval().to("mps")
    if any(tensor.device.type != "mps" for tensor in chain(core.parameters(), core.buffers())):
        raise ValueError("Every model parameter and buffer must be on MPS")
    return core, {
        "source_tensor_count": source_tensor_count,
        "loaded_tensor_count": loaded_tensor_count,
        "strict_load": True,
        "embedding_aliases": aliases,
    }


def _synthetic_inputs(config: PI05Config, seed: int) -> SyntheticInputs:
    torch.mps.manual_seed(seed)
    state_dim = config.input_features["observation.state"].shape[0]
    action_dim = config.output_features["action"].shape[0]
    observations = {key: torch.rand((3, 180, 320), device="mps") for key in config.image_features}
    observations["observation.state"] = torch.linspace(-1.0, 1.0, state_dim, device="mps")
    observations["task"] = "move_the_red_block to the blue area"
    return SyntheticInputs(
        observations=observations,
        statistics={
            "observation.state": {"q01": [-2.0] * state_dim, "q99": [2.0] * state_dim},
            "action": {"q01": [-0.5] * action_dim, "q99": [0.5] * action_dim},
        },
        noise=torch.randn((1, config.chunk_size, config.max_action_dim), device="mps"),
    )


def _processors(checkpoint: Path, config: PI05Config, statistics: dict):
    tokenizer = GemmaTokenizer.from_pretrained(
        checkpoint,
        vocab_file=str(checkpoint / "tokenizer.model"),
        local_files_only=True,
        add_bos_token=True,
        add_eos_token=False,
        padding_side="right",
        fix_mistral_regex=False,
    )
    features = config.input_features | config.output_features
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename="policy_preprocessor.json",
        local_files_only=True,
        overrides={
            "tokenizer_processor": {"tokenizer": tokenizer, "tokenizer_name": None},
            "normalizer_processor": {
                "features": features,
                "stats": statistics,
                "device": "mps",
            },
            "device_processor": {"device": "mps"},
        },
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename="policy_postprocessor.json",
        local_files_only=True,
        overrides={
            "unnormalizer_processor": {
                "features": features,
                "stats": statistics,
                "device": "mps",
            },
            "device_processor": {"device": "mps"},
        },
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


@torch.no_grad()
def infer(checkpoint: Path, output: Path, *, seed: int = 7) -> dict:
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        raise ValueError("Set MLX_ENABLE_TF32=0 before starting float32 numerical verification")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("MPS CPU fallback must be disabled with PYTORCH_ENABLE_MPS_FALLBACK=0")
    if not mx.metal.is_available() or not torch.backends.mps.is_available():
        raise ValueError("This inference check requires both MLX Metal and PyTorch MPS")
    checkpoint, output = Path(checkpoint), Path(output)
    assets = [
        checkpoint / name
        for name in (
            "config.json",
            "model.safetensors",
            "tokenizer.model",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        )
    ]
    for path in assets:
        if not path.is_file():
            raise FileNotFoundError(path)
    if (output / "report.json").exists():
        raise FileExistsError("Choose an output directory without an existing success report")
    output.mkdir(parents=True, exist_ok=True)
    asset_report = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)} for path in assets
    }
    mx.set_default_device(mx.gpu)
    torch.set_float32_matmul_precision("highest")
    core, loading = _load_core(checkpoint)
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = core.config
    policy.model = core
    inputs = _synthetic_inputs(core.config, seed)
    preprocessor, postprocessor = _processors(checkpoint, core.config, inputs.statistics)
    batch = preprocessor(inputs.observations)
    normalized_state = batch["observation.state"]
    torch.testing.assert_close(
        normalized_state[0], inputs.observations["observation.state"] / 2, atol=1e-6, rtol=1e-6
    )
    prompt = batch["task"][0]
    tokens = batch["observation.language.tokens"]
    token_mask = batch["observation.language.attention_mask"].bool()
    bins = np.linspace(-1, 1, 257)[:-1]
    state_bins = np.digitize(normalized_state[0].cpu().numpy(), bins) - 1
    state_text = " ".join(map(str, state_bins))
    task = inputs.observations["task"].strip().replace("_", " ").replace("\n", " ")
    if prompt != f"Task: {task}, State: {state_text};\nAction: ":
        raise AssertionError("The saved processor must include the generated task and state")
    sentencepiece = spm.SentencePieceProcessor(model_file=str(checkpoint / "tokenizer.model"))
    expected_tokens = sentencepiece.encode(prompt, add_bos=True)
    if tokens[0][token_mask[0]].cpu().tolist() != expected_tokens:
        raise AssertionError("Saved preprocessing tokens differ from direct SentencePiece encoding")
    images, image_masks = policy._preprocess_images(batch)
    image_shape = (1, 3, *core.config.image_resolution)
    if len(images) != 3 or any(image.shape != image_shape for image in images):
        raise AssertionError("All three cameras must use the official image preprocessing")
    expert = expert_from_torch(core)
    prefix = encode_prefix(core, images, image_masks, tokens, token_mask)
    sampled = expert.sample(as_mlx(inputs.noise), prefix, steps=STEPS)
    mx.eval(sampled)
    actual = np.array(sampled)
    reference = policy.predict_action_chunk(batch, noise=inputs.noise, num_steps=STEPS)
    expected = reference.cpu().numpy()
    shape = (1, core.config.chunk_size, core.config.output_features["action"].shape[0])
    actual = actual[:, :, : shape[-1]]
    if actual.shape != shape or expected.shape != shape:
        raise AssertionError(f"Expected both action chunks to have shape {shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError("Both action chunks must contain only finite values")
    final = postprocessor(torch.from_numpy(actual).to("mps")).cpu().numpy()
    final_reference = postprocessor(reference).cpu().numpy()
    if final.shape != shape or final_reference.shape != shape:
        raise AssertionError("The saved postprocessor must return the entire action chunk")
    if not np.isfinite(final).all() or not np.isfinite(final_reference).all():
        raise AssertionError("Postprocessed actions must contain only finite values")
    input_arrays = {
        **{
            f"raw_{name}": (
                value.cpu().numpy() if isinstance(value, torch.Tensor) else np.array(value)
            )
            for name, value in inputs.observations.items()
        },
        **{f"image_{i}": image.cpu().numpy() for i, image in enumerate(images)},
        **{f"image_mask_{i}": mask.cpu().numpy() for i, mask in enumerate(image_masks)},
        "normalized_state": normalized_state.cpu().numpy(),
        "full_prompt": np.array(prompt),
        "tokens": tokens.cpu().numpy(),
        "token_mask": token_mask.cpu().numpy(),
        "noise": inputs.noise.cpu().numpy(),
    }
    output_arrays = {
        "mlx_actions": actual,
        "torch_actions": expected,
        "postprocessed_mlx_actions": final,
        "postprocessed_torch_actions": final_reference,
    }
    np.savez_compressed(output / "inputs.npz", **input_arrays)
    np.savez_compressed(output / "actions.npz", **output_arrays)
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(final, final_reference, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(final, actual * 0.5, atol=1e-6, rtol=1e-6)
    error = np.abs(actual - expected)
    report = {
        "kind": "local_pi05_full_pipeline_synthetic_inference",
        "assets": asset_report,
        "loading": loading,
        "versions": {
            name: version(name)
            for name in (
                "mlx",
                "lerobot",
                "torch",
                "transformers",
                "numpy",
                "sentencepiece",
                "protobuf",
            )
        },
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "mlx_device": str(mx.default_device()),
        "torch_device": str(next(core.parameters()).device),
        "metal_available": mx.metal.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "precision_environment": {
            "MLX_ENABLE_TF32": os.environ["MLX_ENABLE_TF32"],
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
        },
        "model_compute": {"encoder": "mps", "reference": "mps", "action_expert": "mlx.gpu"},
        "host_operations": [
            "local checkpoint loading",
            "official state discretization and SentencePiece tokenization",
            "PyTorch-to-MLX tensor transfer",
            "array checks and artifact serialization",
        ],
        "config": asdict(expert.config),
        "seed": seed,
        "euler_steps": STEPS,
        "inputs": {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
            for name, array in input_arrays.items()
        },
        "camera_valid_count": sum(int(mask.sum().item()) for mask in image_masks),
        "token_valid_count": int(token_mask.sum().item()),
        "task": inputs.observations["task"],
        "full_prompt": prompt,
        "state_prompt_processor": processor_pi05.Pi05PrepareStateTokenizerProcessorStep.__name__,
        "normalization_statistics": {"kind": "synthetic_quantiles", "values": inputs.statistics},
        "outputs": {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
            for name, array in output_arrays.items()
        },
        "artifacts": {
            name: {"bytes": (output / name).stat().st_size, "sha256": _sha256(output / name)}
            for name in ("inputs.npz", "actions.npz")
        },
        "output_shape": list(shape),
        "all_actions_finite": True,
        "parity_passed": True,
        "parity_atol": ATOL,
        "parity_rtol": RTOL,
        "sample_max_absolute_error": float(error.max()),
        "sample_mean_absolute_error": float(error.mean()),
        "postprocessed_max_absolute_error": float(np.abs(final - final_reference).max()),
        "pretrained_full_model_run": True,
        "full_inference_pipeline_run": True,
        "synthetic_inputs": True,
        "input_boundary": "raw synthetic cameras, robot state and task text",
        "preprocessing_tested": True,
        "tokenizer_tested": True,
        "sentencepiece_token_parity": True,
        "synthetic_normalization_tested": True,
        "postprocessing_tested": True,
        "dataset_normalization_tested": False,
        "robot_policy_evaluation": False,
        "performance_benchmark": False,
        "cpu_model_inference": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/pi05-synthetic"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(infer(args.checkpoint, args.output, seed=args.seed), indent=2))
