"""Create a label-free, one-episode plan from an existing imported source store."""

import argparse
import json
from pathlib import Path

from robo.core import LABELS, atomic_json, canonical, digest, file_hash
from robo.pipeline import pipeline_lock
from robo.semantic import EvaluationPlan, episode_spec
from robo.semantic_baseline import source_episode
from robo.store import Store


def build_plan(store, episodes, guidelines, *, dataset, camera, split):
    """Inventory only the supplied episodes; labels and review history are never read."""
    if not guidelines.is_file() or not 0 < guidelines.stat().st_size <= 64 * 1024:
        raise ValueError("Supply a nonempty guidelines file of at most 64 KiB")
    specs = [
        episode_spec(
            episode,
            dataset=dataset,
            dataset_episode_id=episode["source_id"],
            camera_id=camera,
            split=split,
        )
        for episode in episodes
    ]
    for spec in specs:
        source_episode(store, spec)
    return EvaluationPlan(
        taxonomy_id="robot-coarse-v1",
        normalized_actions=list(LABELS),
        annotation_guidelines_sha256=file_hash(guidelines),
        identity_audit={
            "status": "unverified",
            "reviewer": "automatic source inventory; not independently audited",
            "evidence_sha256": digest(
                specs[0].model_dump() if len(specs) == 1 else [s.model_dump() for s in specs]
            ),
        },
        episodes=specs,
    )


def prepare(store, output, guidelines, *, dataset, camera, split, episode_id=None):
    episodes = store.episodes()
    selected = [e for e in episodes if e["id"] == episode_id] if episode_id else episodes
    if len(selected) != 1:
        raise ValueError(
            "Select exactly one imported episode with --episode, or use a one-episode store"
        )
    plan = build_plan(store, selected, guidelines, dataset=dataset, camera=camera, split=split)
    output = Path(output)
    with pipeline_lock(store):
        if output.is_symlink():
            raise ValueError("Plan output must not be a symlink")
        if output.exists():
            if output.read_bytes() != canonical(plan.model_dump()):
                raise ValueError("Plan output already contains a different frozen plan")
        else:
            atomic_json(output, plan.model_dump())
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guidelines", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--split", choices=("train", "development", "test"), required=True)
    parser.add_argument("--episode")
    args = parser.parse_args()
    if not (args.store / "review.sqlite").is_file():
        parser.error("Import an episode into the source store first")
    plan = prepare(
        Store(args.store),
        args.output,
        args.guidelines,
        dataset=args.dataset,
        camera=args.camera,
        split=args.split,
        episode_id=args.episode,
    )
    print(
        json.dumps(
            {
                "plan": str(args.output),
                "plan_sha256": plan.plan_sha256,
                "episode_id": plan.episodes[0].episode_id,
                "split": plan.episodes[0].split,
                "identity_audit": plan.identity_audit.status,
                "reference_labels_created": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
