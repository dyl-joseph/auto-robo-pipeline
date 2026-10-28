import dataclasses
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from .model import ActionExpert, ExpertConfig


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Trainer:
    def __init__(
        self,
        model,
        *,
        seed=0,
        learning_rate=2.5e-5,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        clip_norm=1.0,
    ):
        if learning_rate <= 0 or clip_norm <= 0:
            raise ValueError("Learning rate and gradient clip must be positive")
        self.model = model
        self.settings = dict(
            learning_rate=learning_rate,
            betas=list(betas),
            weight_decay=weight_decay,
            clip_norm=clip_norm,
        )
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            bias_correction=True,
        )
        self.key = mx.random.key(seed)

    def step(self, prefix, actions, *, action_dimensions=None):
        next_key, noise_key, time_key = mx.random.split(self.key, 3)
        noise = mx.random.normal(actions.shape, key=noise_key)
        # Beta(1.5, 1) has inverse CDF u**(1/1.5), matching PI05's time distribution.
        time = mx.random.uniform(shape=(actions.shape[0],), key=time_key) ** (1 / 1.5)
        time = time * 0.999 + 0.001
        loss, grads = nn.value_and_grad(
            self.model,
            lambda: self.model.loss(
                actions, noise, time, prefix, action_dimensions=action_dimensions
            ),
        )()
        norm = mx.sqrt(sum(mx.sum(mx.square(x)) for _, x in tree_flatten(grads)))
        mx.eval(loss, norm)
        if not bool(mx.isfinite(loss)) or not bool(mx.isfinite(norm)):
            raise ValueError("Non-finite loss or gradient; no optimizer update was applied")
        scale = mx.minimum(1.0, self.settings["clip_norm"] / (norm + 1e-6))
        self.optimizer.update(self.model, tree_map(lambda x: x * scale, grads))
        self.key = next_key
        mx.eval(self.model.parameters(), self.optimizer.state, self.key)
        return {
            "loss": float(loss),
            "gradient_norm": float(norm),
            "step": int(self.optimizer.state["step"]),
        }

    def save(self, destination, *, contract):
        """Publish a new checkpoint directory only after all files and hashes are complete."""
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".mlx-checkpoint-", dir=destination.parent))
        try:
            self.optimizer.init(self.model.trainable_parameters())
            self.model.save_weights(str(temporary / "model.safetensors"))
            state = dict(tree_flatten(self.optimizer.state))
            mx.savez(str(temporary / "optimizer.npz"), **state, random_key=self.key)
            manifest = {
                "version": 1,
                "config": dataclasses.asdict(self.model.config),
                "optimizer": self.settings,
                "contract": contract,
                "files": {
                    name: digest(temporary / name)
                    for name in ("model.safetensors", "optimizer.npz")
                },
            }
            (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            temporary.rename(destination)
        except BaseException:
            shutil.rmtree(temporary)
            raise

    @classmethod
    def load(cls, source, *, contract):
        source = Path(source)
        manifest = json.loads((source / "manifest.json").read_text())
        if manifest["version"] != 1 or manifest["contract"] != contract:
            raise ValueError("Checkpoint version or data/model contract does not match")
        for name in ("model.safetensors", "optimizer.npz"):
            if manifest["files"].get(name) != digest(source / name):
                raise ValueError(f"Checkpoint checksum failed: {name}")
        model = ActionExpert(ExpertConfig(**manifest["config"]))
        model.load_weights(str(source / "model.safetensors"), strict=True)
        trainer = cls(model, **manifest["optimizer"])
        state = mx.load(str(source / "optimizer.npz"))
        trainer.key = state.pop("random_key")
        trainer.optimizer.state = tree_unflatten(list(state.items()))
        return trainer
