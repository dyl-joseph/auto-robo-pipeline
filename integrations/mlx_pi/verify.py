"""Measure the MLX adapter against small official PI05 models, without pretrained downloads."""

import argparse
import dataclasses
import json
import platform
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .bridge import as_mlx, encode_prefix, expert_from_torch
from .reference import small_inputs, small_reference
from .training import Trainer


def verify(output):
    start = time.perf_counter()
    core, batch = small_reference(), small_inputs()
    expert = expert_from_torch(core)
    prefix = encode_prefix(
        core, batch["images"], batch["img_masks"], batch["tokens"], batch["masks"]
    )
    actions, noise, timestep = [as_mlx(batch[key]) for key in ("actions", "noise", "time")]
    reference_loss = core(**batch).mean()
    reference_loss.backward()
    loss, grads = nn.value_and_grad(expert, lambda: expert.loss(actions, noise, timestep, prefix))()
    parameters = dict(core.named_parameters())
    gradient_error = 0.0
    for name, grad in tree_flatten(grads):
        reference_name = (
            name
            if name.startswith(("action_", "time_"))
            else ("paligemma_with_expert.gemma_expert.model." + name)
        )
        expected = parameters[reference_name].grad.numpy()
        np.testing.assert_allclose(np.array(grad), expected, atol=4e-5, rtol=2e-3)
        gradient_error = max(gradient_error, float(np.max(np.abs(np.array(grad) - expected))))
    loss_error = abs(float(loss) - reference_loss.item())
    assert loss_error < 3e-5
    sampled = expert.sample(noise, prefix, steps=3)
    expected_sample = core.sample_actions(
        batch["images"],
        batch["img_masks"],
        batch["tokens"],
        batch["masks"],
        noise=batch["noise"],
        num_steps=3,
    ).numpy()
    np.testing.assert_allclose(np.array(sampled), expected_sample, atol=5e-5, rtol=5e-5)
    trainer = Trainer(expert, seed=42)
    contract = {"reference": "random-small-official-pi05", "pretrained": False}
    step1 = trainer.step(prefix, actions)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mlx-verify-", dir=output.parent) as directory:
        checkpoint = Path(directory) / "step1"
        trainer.save(checkpoint, contract=contract)
        step2 = trainer.step(prefix, actions)
        resumed = Trainer.load(checkpoint, contract=contract)
        assert resumed.step(prefix, actions) == step2
        for (_, left), (_, right) in zip(
            tree_flatten(trainer.model.parameters()),
            tree_flatten(resumed.model.parameters()),
            strict=True,
        ):
            np.testing.assert_array_equal(np.array(left), np.array(right))
    report = {
        "kind": "small_official_pi05_frozen_encoder_mlx_action_expert",
        "versions": {name: version(name) for name in ("mlx", "lerobot", "torch", "transformers")},
        "platform": platform.system(),
        "architecture": platform.machine(),
        "mlx_device": str(mx.default_device()),
        "metal_available": mx.metal.is_available(),
        "config": dataclasses.asdict(expert.config),
        "trainable_parameters": sum(x.size for _, x in tree_flatten(expert.parameters())),
        "gradient_tensors_compared": len(tree_flatten(grads)),
        "reference_loss": reference_loss.item(),
        "mlx_loss": float(loss),
        "loss_absolute_error": loss_error,
        "gradient_max_absolute_error": gradient_error,
        "sample_max_absolute_error": float(np.max(np.abs(np.array(sampled) - expected_sample))),
        "training_steps": [step1, step2],
        "checkpoint_next_update_exact": True,
        "elapsed_seconds": time.perf_counter() - start,
        "pretrained_full_model_run": False,
        "robot_policy_evaluation": False,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/mlx-pi05.json"))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    if args.device == "gpu" and not mx.metal.is_available():
        parser.error("The Metal verification command requires an Apple GPU")
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    print(json.dumps(verify(args.output), indent=2))
