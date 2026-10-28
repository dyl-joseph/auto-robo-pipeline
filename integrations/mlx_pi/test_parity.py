import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten

from .bridge import as_mlx, encode_prefix, expert_from_torch
from .model import Prefix
from .reference import reference_device, small_inputs, small_reference
from .training import Trainer


@pytest.fixture
def case():
    core = small_reference()
    batch = small_inputs()
    device = reference_device().type
    assert {p.device.type for p in core.parameters()} == {device}
    assert all(x.device.type == device for x in (*batch["images"], *batch["img_masks"]))
    assert all(
        batch[key].device.type == device for key in ("tokens", "masks", "actions", "noise", "time")
    )
    prefix = encode_prefix(
        core, batch["images"], batch["img_masks"], batch["tokens"], batch["masks"]
    )
    return core, batch, prefix, expert_from_torch(core)


def test_official_joint_forward_loss_and_all_expert_gradients(case):
    core, batch, prefix, expert = case
    reference = core(**batch).mean()
    reference.backward()
    arguments = [as_mlx(batch[name]) for name in ("actions", "noise", "time")]
    loss, grads = nn.value_and_grad(expert, lambda: expert.loss(*arguments, prefix))()
    np.testing.assert_allclose(
        np.array(loss), reference.detach().cpu().numpy(), atol=3e-5, rtol=3e-5
    )
    expected = dict(core.named_parameters())
    for name, gradient in tree_flatten(grads):
        original = (
            name
            if name.startswith(("action_", "time_"))
            else ("paligemma_with_expert.gemma_expert.model." + name)
        )
        assert expected[original].grad is not None, original
        np.testing.assert_allclose(
            np.array(gradient),
            expected[original].grad.cpu().numpy(),
            atol=4e-5,
            rtol=2e-3,
            err_msg=name,
        )
    assert all(p.grad is None for p in core.paligemma_with_expert.paligemma.parameters())
    assert np.linalg.norm(np.array(grads["layers"][0]["self_attn"]["q_proj"]["weight"])) > 0


def test_padding_and_masked_camera_cannot_influence_actions(case):
    core, batch, prefix, expert = case
    original = expert(as_mlx(batch["noise"]), as_mlx(batch["time"]), prefix)
    batch["images"][1][0] = torch.randn_like(batch["images"][1][0]) * 100
    batch["tokens"][~batch["masks"]] = 55
    changed = encode_prefix(
        core, batch["images"], batch["img_masks"], batch["tokens"], batch["masks"]
    )
    actual = expert(as_mlx(batch["noise"]), as_mlx(batch["time"]), changed)
    np.testing.assert_allclose(np.array(actual), np.array(original), atol=2e-6)


def test_visible_prefix_changes_action_prediction(case):
    core, batch, prefix, expert = case
    original = expert(as_mlx(batch["noise"]), as_mlx(batch["time"]), prefix)
    batch["images"][0] = -batch["images"][0]
    batch["tokens"][0, 0] = 42
    changed = encode_prefix(
        core, batch["images"], batch["img_masks"], batch["tokens"], batch["masks"]
    )
    actual = expert(as_mlx(batch["noise"]), as_mlx(batch["time"]), changed)
    assert float(mx.max(mx.abs(actual - original))) > 1e-6


def test_official_euler_sampling(case):
    core, batch, prefix, expert = case
    actual = expert.sample(as_mlx(batch["noise"]), prefix, steps=3)
    expected = core.sample_actions(
        batch["images"],
        batch["img_masks"],
        batch["tokens"],
        batch["masks"],
        noise=batch["noise"],
        num_steps=3,
    )
    np.testing.assert_allclose(np.array(actual), expected.cpu().numpy(), atol=5e-5, rtol=5e-5)


def test_reject_wrong_cache_layout_and_chunk(case):
    _, batch, prefix, expert = case
    with pytest.raises(ValueError, match="KV layout"):
        expert(
            as_mlx(batch["noise"]),
            as_mlx(batch["time"]),
            Prefix(
                tuple(x.transpose(0, 2, 1, 3) for x in prefix.keys), prefix.values, prefix.valid
            ),
        )
    with pytest.raises(ValueError, match="horizon"):
        expert(as_mlx(batch["noise"][:, :1]), as_mlx(batch["time"]), prefix)


def test_optimizer_update_matches_official_loss(case):
    core, batch, prefix, expert = case
    trainer = Trainer(expert, seed=42, learning_rate=1e-3)
    _, noise_key, time_key = mx.random.split(trainer.key, 3)
    batch["noise"] = torch.from_numpy(
        np.array(mx.random.normal(batch["actions"].shape, key=noise_key))
    ).to(batch["actions"].device)
    time = mx.random.uniform(shape=(2,), key=time_key) ** (1 / 1.5) * 0.999 + 0.001
    batch["time"] = torch.from_numpy(np.array(time)).to(batch["actions"].device)
    optimizer = torch.optim.AdamW(
        [p for p in core.parameters() if p.requires_grad],
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    loss = core(**batch).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
    optimizer.step()
    before = np.array(expert.action_out_proj.weight)
    trainer.step(prefix, as_mlx(batch["actions"]))
    expected = dict(tree_flatten(expert_from_torch(core).parameters()))
    for name, weight in tree_flatten(expert.parameters()):
        np.testing.assert_allclose(
            np.array(weight), np.array(expected[name]), atol=4e-5, rtol=3e-4, err_msg=name
        )
    assert not np.array_equal(before, np.array(expert.action_out_proj.weight))


def test_checkpoint_resume_preserves_next_update_rng_and_optimizer(case, tmp_path):
    _, batch, prefix, expert = case
    trainer = Trainer(expert, seed=42)
    actions = as_mlx(batch["actions"])
    contract = {"source": "random-small-official-model", "action_dimensions": 8}
    trainer.step(prefix, actions)
    checkpoint = tmp_path / "step1"
    trainer.save(checkpoint, contract=contract)
    expected = trainer.step(prefix, actions)
    resumed = Trainer.load(checkpoint, contract=contract)
    assert resumed.step(prefix, actions) == expected
    for (_, left), (_, right) in zip(
        tree_flatten(trainer.model.parameters()),
        tree_flatten(resumed.model.parameters()),
        strict=True,
    ):
        np.testing.assert_array_equal(np.array(left), np.array(right))
    with pytest.raises(ValueError, match="contract"):
        Trainer.load(checkpoint, contract={"source": "different-dataset"})
    with pytest.raises(FileExistsError):
        trainer.save(checkpoint, contract=contract)
    with (checkpoint / "model.safetensors").open("ab") as file:
        file.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        Trainer.load(checkpoint, contract=contract)


def test_invalid_loss_does_not_advance_training(case):
    _, batch, prefix, expert = case
    trainer = Trainer(expert)
    key = np.array(trainer.key)
    weight = np.array(expert.action_out_proj.weight)
    with pytest.raises(ValueError, match="Non-finite"):
        trainer.step(prefix, as_mlx(batch["actions"]) * float("nan"))
    np.testing.assert_array_equal(np.array(trainer.key), key)
    np.testing.assert_array_equal(np.array(expert.action_out_proj.weight), weight)
