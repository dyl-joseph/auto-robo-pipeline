"""Train and resume a pretrained PI05 action expert from an accepted train export."""

import argparse
import gc
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np

from robo.core import digest, file_hash
from robo.export import validate_export
from robo.lerobot_export import validate_training_export
from robo.lerobot_plan import TrainingSchema


def camera_mapping(values):
    mapping = {}
    for value in values:
        source, separator, target = value.partition("=")
        if (
            not separator
            or not source.startswith("observation.images.")
            or not target.startswith("observation.images.")
            or source in mapping
            or target in mapping.values()
        ):
            raise ValueError("Camera mappings require unique SOURCE=TARGET image feature names")
        mapping[source] = target
    if not mapping:
        raise ValueError("Supply at least one explicit camera mapping")
    return mapping


def training_source(manifest, records, *, task_key):
    if manifest["split"] != "train":
        raise ValueError(
            "Policy training requires an accepted train split, not development or test"
        )
    if task_key not in {"task", "annotation.subtask_text"}:
        raise ValueError("Choose task or annotation.subtask_text explicitly for conditioning")
    accepted = {record["episode_id"]: record for record in records}
    if not manifest["episodes"]:
        raise ValueError("The training export contains no episodes")
    for episode in manifest["episodes"]:
        record = accepted.get(episode["source_episode_id"])
        if (
            record is None
            or record["split"] != "train"
            or record["annotation"]["status"] != "accepted"
            or record["annotation"]["revision"] != episode["accepted_revision"]
            or digest(record) != episode["record_hash"]
        ):
            raise ValueError("Training episodes must match explicitly accepted train revisions")
        if task_key == "annotation.subtask_text":
            review = (record["annotation"].get("native") or {}).get("semantic_review")
            if (
                review is None
                or review.get("split") != "train"
                or review.get("complete") is not True
                or not review.get("segments")
                or any(not segment["subtask_text"].strip() for segment in review["segments"])
            ):
                raise ValueError("Subtask conditioning requires complete accepted train semantics")
    return TrainingSchema.model_validate(manifest["schema"])


def fit_quantiles(states, actions):
    statistics = {}
    for name, values in (("observation.state", states), ("action", actions)):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(
                "Fit normalization only from finite training rows with vector features"
            )
        statistics[name] = {
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    return statistics


def semantic_chunk(tasks, start, horizon):
    selected = list(tasks[start : start + horizon])
    if (
        len(selected) != horizon
        or not selected[0].strip()
        or any(task != selected[0] for task in selected)
    ):
        raise ValueError(
            "The selected action chunk crosses a semantic subtask boundary; "
            "choose --sample-index inside one complete subtask"
        )
    return selected[0]


def verify_preprocessing(source, prepared, statistics, tokenizer):
    for field, name in (("state", "observation.state"), ("actions", "action")):
        quantiles = statistics[name]
        low = np.asarray(quantiles["q01"], dtype=np.float32)
        high = np.asarray(quantiles["q99"], dtype=np.float32)
        denominator = np.where(high == low, np.float32(1e-8), high - low)
        expected = 2 * (source[field] - low) / denominator - 1
        np.testing.assert_allclose(prepared[field], expected, atol=1e-6, rtol=1e-6)
    dimensions = source["actions"].shape[-1]
    np.testing.assert_array_equal(prepared["padded_actions"][..., :dimensions], prepared["actions"])
    np.testing.assert_array_equal(
        prepared["padded_actions"][..., dimensions:],
        np.zeros_like(prepared["padded_actions"][..., dimensions:]),
    )
    state_bins = np.digitize(prepared["state"][0], np.linspace(-1, 1, 257)[:-1]) - 1
    task = source["task"].strip().replace("_", " ").replace("\n", " ")
    expected_prompt = f"Task: {task}, State: {' '.join(map(str, state_bins))};\nAction: "
    if prepared["prompt"] != expected_prompt:
        raise AssertionError("The official processor changed the selected task or state prompt")
    if prepared["token_ids"] != tokenizer.encode(expected_prompt, add_bos=True):
        raise AssertionError("Processed tokens differ from direct SentencePiece encoding")


def parameter_hashes(model):
    from mlx.utils import tree_flatten

    return {
        name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
        for name, value in tree_flatten(model.parameters())
    }


def train_export(
    dataset,
    checkpoint,
    output,
    *,
    camera_map,
    task_key,
    action_representation,
    sample_index=0,
    seed=42,
):
    dataset, checkpoint, output = [Path(path).resolve() for path in (dataset, checkpoint, output)]
    if any(
        output.is_relative_to(path) or path.is_relative_to(output) for path in (dataset, checkpoint)
    ):
        raise ValueError("Keep training output separate from the dataset and pretrained assets")
    if output.exists():
        raise FileExistsError("Choose a fresh training output directory")
    if not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError("The training sample index must be a nonnegative integer")
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        raise ValueError("Set MLX_ENABLE_TF32=0 before starting training verification")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("Set PYTORCH_ENABLE_MPS_FALLBACK=0; CPU model fallback is forbidden")
    manifest = validate_training_export(dataset)
    records = validate_export(dataset / "provenance/accepted")
    schema = training_source(manifest, records, task_key=task_key)
    camera_map = camera_mapping([source + "=" + target for source, target in camera_map.items()])

    import mlx.core as mx
    import sentencepiece as spm
    import torch
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from mlx.utils import tree_flatten
    from torch.utils.data import DataLoader, Subset

    from .bridge import as_mlx, encode_prefix, expert_from_torch, validate_droid_schema
    from .infer_synthetic import _load_core, _processors
    from .training import Trainer

    action_contract = validate_droid_schema(
        schema.model_dump(), action_representation=action_representation
    )
    if not mx.metal.is_available() or not torch.backends.mps.is_available():
        raise ValueError("Training verification requires both MLX Metal and PyTorch MPS")
    mx.set_default_device(mx.gpu)
    torch.set_float32_matmul_precision("highest")
    output.mkdir(parents=True, exist_ok=False)
    assets = {
        name: {"bytes": (checkpoint / name).stat().st_size, "sha256": file_hash(checkpoint / name)}
        for name in (
            "model.safetensors",
            "config.json",
            "tokenizer.model",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        )
    }
    core, loading = _load_core(checkpoint)
    core.config.input_features["observation.state"] = PolicyFeature(
        type=FeatureType.STATE, shape=(len(schema.state),)
    )
    core.config.output_features["action"] = PolicyFeature(
        type=FeatureType.ACTION, shape=(len(schema.action),)
    )
    if not set(camera_map.values()) <= set(core.config.image_features):
        raise ValueError("A mapped camera is absent from the pretrained policy configuration")
    reader = LeRobotDataset(
        "local/" + manifest["version"],
        root=dataset,
        delta_timestamps={"action": [i / schema.fps for i in range(core.config.chunk_size)]},
    )
    if not set(camera_map) <= set(reader.meta.features):
        raise ValueError("A mapped camera is absent from the training export")
    if (
        task_key == "annotation.subtask_text"
        and reader.meta.features.get(task_key, {}).get("dtype") != "string"
    ):
        raise ValueError("The training export has no frame-aligned semantic text")
    states = np.stack([np.asarray(value) for value in reader.hf_dataset["observation.state"]])
    actions = np.stack([np.asarray(value) for value in reader.hf_dataset["action"]])
    if states.shape[1] != len(schema.state) or actions.shape[1] != len(schema.action):
        raise ValueError("Training feature dimensions differ from the accepted action schema")
    statistics = fit_quantiles(states, actions)
    if sample_index >= len(reader):
        raise ValueError("The training sample index is outside the export")
    batch = next(iter(DataLoader(Subset(reader, [sample_index]), batch_size=1, num_workers=0)))
    if batch["action_is_pad"].any():
        raise ValueError("Choose a training frame with a complete action chunk in its episode")
    task = batch[task_key][0]
    if not isinstance(task, str) or not task.strip():
        raise ValueError("The selected training frame has no task conditioning text")
    if task_key == "annotation.subtask_text":
        if (
            semantic_chunk(reader.hf_dataset[task_key], sample_index, core.config.chunk_size)
            != task
        ):
            raise AssertionError("The training batch task differs from its exported source frame")
    raw_batch = {
        "observation.state": batch["observation.state"],
        "action": batch["action"],
        "task": [task],
        **{target: batch[source] for source, target in camera_map.items()},
    }
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.config, policy.model = core.config, core
    preprocessor, postprocessor = _processors(checkpoint, core.config, statistics)
    processed = preprocessor(raw_batch)
    images, image_masks = policy._preprocess_images(processed)
    tokens = processed["observation.language.tokens"]
    token_mask = processed["observation.language.attention_mask"]
    training_actions = as_mlx(policy.prepare_action(processed))
    if training_actions.shape != (1, core.config.chunk_size, core.config.max_action_dim):
        raise ValueError("The official preprocessor did not produce the full action chunk")
    verify_preprocessing(
        {
            "task": task,
            "state": raw_batch["observation.state"].detach().cpu().numpy(),
            "actions": raw_batch["action"].detach().cpu().numpy(),
        },
        {
            "state": processed["observation.state"].detach().cpu().numpy(),
            "actions": processed["action"].detach().cpu().numpy(),
            "padded_actions": np.asarray(training_actions),
            "prompt": processed["task"][0],
            "token_ids": tokens[0][token_mask[0].bool()].detach().cpu().tolist(),
        },
        statistics,
        spm.SentencePieceProcessor(model_file=str(checkpoint / "tokenizer.model")),
    )
    prefix = encode_prefix(core, images, image_masks, tokens, token_mask)
    expert = expert_from_torch(core)
    trainable_parameters = sum(value.size for _, value in tree_flatten(expert.parameters()))
    initial_hashes = parameter_hashes(expert)
    contract = {
        "training_version": manifest["version"],
        "accepted_version": manifest["accepted_version"],
        "schema": schema.model_dump(),
        "action_contract": action_contract,
        "asset_sha256": {name: asset["sha256"] for name, asset in assets.items()},
        "statistics": statistics,
        "statistics_split": "train",
        "camera_mapping": camera_map,
        "task_key": task_key,
        "sample_index": sample_index,
        "semantic_chunk_consistent": task_key == "annotation.subtask_text",
    }
    input_report = {
        "task": task,
        "full_prompt": processed["task"][0],
        "valid_tokens": int(token_mask.sum().item()),
        "valid_cameras": sum(int(mask.sum().item()) for mask in image_masks),
        "action_shape": list(training_actions.shape),
        "source_frame": int(batch["source.frame_index"][0].item()),
    }
    del core, policy, reader, batch, images, processed, preprocessor
    gc.collect()
    torch.mps.empty_cache()
    trainer = Trainer(expert, seed=seed)
    dimensions = action_contract["action_dimensions"]
    step1 = trainer.step(prefix, training_actions, action_dimensions=dimensions)
    if step1["gradient_norm"] <= 0 or parameter_hashes(trainer.model) == initial_hashes:
        raise AssertionError(
            "The pretrained action expert did not receive a nonzero training update"
        )
    checkpoint_path = output / "step1"
    trainer.save(checkpoint_path, contract=contract)
    step2 = trainer.step(prefix, training_actions, action_dimensions=dimensions)
    expected_hashes = parameter_hashes(trainer.model)
    noise = mx.random.normal(training_actions.shape, key=mx.random.key(seed + 1))
    expected_actions = np.asarray(trainer.model.sample(noise, prefix, steps=10))
    del trainer, expert
    gc.collect()
    mx.clear_cache()
    resumed = Trainer.load(checkpoint_path, contract=contract)
    resumed_step2 = resumed.step(prefix, training_actions, action_dimensions=dimensions)
    if resumed_step2 != step2 or parameter_hashes(resumed.model) != expected_hashes:
        raise AssertionError("Checkpoint resume changed the next optimizer update")
    actual_actions = np.asarray(resumed.model.sample(noise, prefix, steps=10))
    np.testing.assert_array_equal(actual_actions, expected_actions)
    final = postprocessor(torch.from_numpy(actual_actions[:, :, :dimensions]).to("mps"))
    final_actions = final.cpu().numpy()
    if final_actions.shape != (1, 50, dimensions) or not np.isfinite(final_actions).all():
        raise AssertionError("The trained policy must return a finite complete postprocessed chunk")
    if not np.isfinite(actual_actions).all():
        raise AssertionError("The trained action expert returned non-finite actions")
    np.savez_compressed(
        output / "actions.npz",
        normalized_actions=actual_actions,
        normalized_actions_before_resume=expected_actions,
        postprocessed_actions=final_actions,
    )
    if validate_training_export(dataset) != manifest:
        raise ValueError("The source training export changed during training")
    report = {
        "kind": "accepted_train_export_pretrained_pi05_training_resume",
        "contract": contract,
        "assets": assets,
        "loading": loading,
        "versions": {name: version(name) for name in ("mlx", "lerobot", "torch", "transformers")},
        "inputs": input_report,
        "trainable_parameters": trainable_parameters,
        "training_steps": [step1, step2],
        "resumed_next_step": resumed_step2,
        "parameter_tensors_verified_after_resume": len(expected_hashes),
        "checkpoint_next_update_exact": True,
        "checkpoint_sampled_actions_exact": True,
        "expert_parameters_changed": True,
        "normalization_and_padding_verified": True,
        "selected_task_prompt_verified": True,
        "sentencepiece_token_parity": True,
        "all_actions_finite": True,
        "output_shape": list(final_actions.shape),
        "model_compute": {"frozen_encoder": "mps", "action_expert_training": "mlx.gpu"},
        "cpu_model_inference": False,
        "mps_cpu_fallback": False,
        "robot_policy_quality_measured": False,
        "files": {"actions.npz": file_hash(output / "actions.npz")},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-map", action="append", required=True, metavar="SOURCE=TARGET")
    parser.add_argument("--task-key", choices=("task", "annotation.subtask_text"), required=True)
    parser.add_argument("--action-representation", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            train_export(
                args.dataset,
                args.checkpoint,
                args.output,
                camera_map=camera_mapping(args.camera_map),
                task_key=args.task_key,
                action_representation=args.action_representation,
                sample_index=args.sample_index,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
