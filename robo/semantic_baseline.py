"""Propose coarse temporal labels from declared robot state, without reference labels."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from pydantic import Field

from robo.core import Contract, Stream, atomic_json, canonical, digest, file_hash
from robo.lerobot_plan import previous_samples
from robo.semantic import (
    EvaluationPlan,
    Identifier,
    PredictionEpisode,
    Predictions,
    SemanticSegment,
    episode_spec,
    validate_episode_segments,
)
from robo.store import Store

PRODUCER = "robot-state-baseline-v1"
TEXT = {
    "unknown": "Insufficient fresh robot state to estimate motion.",
    "motion": "The robot joints move; the manipulated object is unknown.",
    "hold": "The robot joints and gripper change little between state samples.",
    "grasp": "The gripper closes; grasping is a candidate, with contact unverified.",
    "release": "The gripper opens; releasing is a candidate, with contact unverified.",
}


class BaselineConfig(Contract):
    joint_stream: str = "state_joint_position"
    gripper_stream: str = "state_gripper_position"
    joint_speed_rad_s: float = Field(gt=0, default=0.12)
    gripper_speed_per_s: float = Field(gt=0, default=0.2)
    uncertainty: float = Field(ge=0.6, le=1, default=0.85)
    tuning_episode_ids: list[Identifier] = Field(default_factory=list)


def source_episode(store, spec):
    episode = store.episode(spec.episode_id)
    directory = store.root / "episodes" / episode["id"]
    if (directory / "episode.json").read_bytes() != canonical(episode):
        raise ValueError("stored source manifest differs from its immutable snapshot")
    if file_hash(store.root / episode["media"]) != spec.media_sha256:
        raise ValueError("source media hash differs from the evaluation plan")
    actual = episode_spec(
        episode,
        dataset=spec.dataset,
        dataset_episode_id=spec.dataset_episode_id,
        camera_id=spec.camera_id,
        split=spec.split,
        near_duplicate_groups=spec.near_duplicate_groups,
    )
    if actual != spec:
        raise ValueError("stored episode identity or frame clock differs from the evaluation plan")
    return episode


def state_evidence(episode, spec, config):
    grid = np.asarray(spec.frame_boundaries_s[:-1], dtype=np.float64)
    streams = {s["name"]: Stream.model_validate(s) for s in episode["streams"]}
    aligned = {}
    for name, units, width in (
        (config.joint_stream, "rad", 7),
        (config.gripper_stream, "normalized", 1),
    ):
        if name not in streams:
            raise ValueError(f"missing required robot-state stream: {name}")
        stream = streams[name]
        if stream.modality != "state" or stream.units != units:
            raise ValueError(f"{name}: robot-state units or modality differ from baseline schema")
        values = np.asarray(stream.values)
        if values.shape != (len(stream.timestamps_s), width):
            raise ValueError(f"{name}: unexpected state width")
        if units == "normalized" and ((values < 0).any() or (values > 1).any()):
            raise ValueError("gripper closure must be in [0,1]")
        times = np.asarray(stream.timestamps_s) + stream.offset_s
        indices, ages = previous_samples(times.tolist(), grid, stream.tolerance_s, name)
        # Reusing one sample at a later video frame does not establish zero motion.
        fresh = np.r_[False, np.diff(indices) > 0]
        rates = np.zeros((len(grid), width))
        changed = np.flatnonzero(fresh)
        with np.errstate(over="ignore", invalid="ignore"):
            rates[changed] = (values[indices[changed]] - values[indices[changed - 1]]) / (
                times[indices[changed]] - times[indices[changed - 1]]
            )[:, None]
        if not np.isfinite(rates).all():
            raise ValueError("state differences produce non-finite rates")
        aligned[name] = {
            "indices": indices.tolist(),
            "ages_s": ages.tolist(),
            "fresh": fresh,
            "rates": rates,
        }
    joints, gripper = aligned[config.joint_stream], aligned[config.gripper_stream]
    labels = []
    for i in range(len(grid)):
        if not (joints["fresh"][i] and gripper["fresh"][i]):
            label = "unknown"
        elif gripper["rates"][i, 0] >= config.gripper_speed_per_s:
            label = "grasp"
        elif gripper["rates"][i, 0] <= -config.gripper_speed_per_s:
            label = "release"
        elif np.max(np.abs(joints["rates"][i])) >= config.joint_speed_rad_s:
            label = "motion"
        else:
            label = "hold"
        labels.append(label)
    return {
        "episode_id": spec.episode_id,
        "source_sha256": spec.source_sha256,
        "timing": spec.timing,
        "config": config.model_dump(),
        "frame_labels": labels,
        "state_alignment": {
            name: {
                "indices": row["indices"],
                "ages_s": row["ages_s"],
                "fresh": row["fresh"].tolist(),
                "rates": row["rates"].tolist(),
            }
            for name, row in aligned.items()
        },
        "rate_semantics": (
            "Changes between distinct observed state samples divided by their source-time interval."
        ),
        "uncertainty_semantics": "Fixed heuristic review score; not a calibrated probability.",
        "object_observed": False,
        "contact_observed": False,
    }


def segments_from_evidence(evidence, spec, config):
    labels = evidence["frame_labels"]
    starts = [0] + [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
    ends = starts[1:] + [len(labels)]
    if len(starts) > 500:
        raise ValueError("state baseline exceeds 500 segments; inspect unstable state or sampling")
    artifact_sha = digest(evidence)
    return [
        SemanticSegment(
            episode_id=spec.episode_id,
            start_s=spec.frame_boundaries_s[start],
            end_s=spec.frame_boundaries_s[end],
            source_frames={
                "start": start,
                "end_exclusive": end,
                "evidence": list(range(start, end)),
            },
            subtask_text=TEXT[labels[start]],
            normalized_action=labels[start],
            object_text=None,
            uncertainty=1 if labels[start] == "unknown" else config.uncertainty,
            provenance={"kind": "baseline", "producer": PRODUCER, "artifact_sha256": artifact_sha},
        )
        for start, end in zip(starts, ends, strict=True)
    ]


def run(plan, store, output, *, split="test", config=None):
    config = config or BaselineConfig()
    if not set(TEXT).issubset(plan.normalized_actions):
        raise ValueError("baseline taxonomy requires unknown, motion, hold, grasp and release")
    if split not in {"development", "test"}:
        raise ValueError("choose development or test for baseline prediction")
    selected = [e for e in plan.episodes if e.split == split]
    if not selected:
        raise ValueError("no episodes in chosen split")
    population = {e.episode_id: e for e in plan.episodes}
    if len(config.tuning_episode_ids) != len(set(config.tuning_episode_ids)) or any(
        key not in population or population[key].split == "test" or population[key].split == split
        for key in config.tuning_episode_ids
    ):
        raise ValueError("tuning episodes must be declared outside the evaluated split")
    output = Path(output).resolve()
    if output.is_relative_to(store.root) or store.root.is_relative_to(output):
        raise ValueError("Keep prediction output separate from the source store")
    implementation = file_hash(Path(__file__))
    configuration = {
        "baseline": config.model_dump(),
        "implementation_sha256": implementation,
        "numpy": np.__version__,
    }
    run_id = digest({"plan": plan.plan_sha256, "split": split, "configuration": configuration})[:24]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".semantic-state-", dir=output.parent))
    results = []
    try:
        for spec in selected:
            key = digest(spec.episode_id)[:24]
            try:
                episode = source_episode(store, spec)
                evidence = state_evidence(episode, spec, config)
                segments = segments_from_evidence(evidence, spec, config)
                validate_episode_segments(segments, spec, plan.normalized_actions)
                atomic_json(temporary / (key + ".evidence.json"), evidence)
                results.append(PredictionEpisode(episode_id=spec.episode_id, segments=segments))
            except (KeyError, ValueError, OSError) as exc:
                results.append(
                    PredictionEpisode(
                        episode_id=spec.episode_id, segments=[], error=str(exc)[:1000]
                    )
                )
        predictions = Predictions(
            plan_sha256=plan.plan_sha256,
            run_id=run_id,
            producer=PRODUCER,
            code_revision="sha256:" + implementation,
            config_sha256=digest(configuration),
            training_episode_ids=[],
            tuning_episode_ids=config.tuning_episode_ids,
            reference_access="none",
            episodes=results,
        )
        atomic_json(temporary / "predictions.json", predictions.model_dump())
        manifest = {
            "run_id": run_id,
            "configuration": configuration,
            "files": {p.name: file_hash(p) for p in temporary.iterdir()},
            "episode_count": len(results),
            "failed_predictions": sum(e.error is not None for e in results),
            "reference_labels_read": False,
            "semantic_accuracy_measured": False,
        }
        atomic_json(temporary / "run.json", manifest)
        if output.exists():
            if json.loads((output / "run.json").read_text()) != manifest:
                raise ValueError("Output differs from this prediction run; choose a new path")
            for name, sha256 in manifest["files"].items():
                if file_hash(output / name) != sha256:
                    raise ValueError("Existing prediction output changed")
        else:
            os.rename(temporary, output)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("store", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("development", "test"), default="test")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if not (args.store / "review.sqlite").is_file():
        raise ValueError("Use an existing imported episode store")
    plan = EvaluationPlan.model_validate_json(args.plan.read_bytes())
    config = (
        BaselineConfig.model_validate_json(args.config.read_bytes())
        if args.config
        else BaselineConfig()
    )
    print(
        json.dumps(
            run(plan, Store(args.store), args.output, split=args.split, config=config), indent=2
        )
    )


if __name__ == "__main__":
    main()
