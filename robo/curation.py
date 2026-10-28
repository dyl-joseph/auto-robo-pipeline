"""Deterministic review selections with coverage and duplicate accounting."""

import random
from collections import Counter

from robo.analysis import queue
from robo.core import digest
from robo.store import Store


def curate(store: Store, count: int, policy: str = "prioritized", seed: int = 7) -> dict:
    if count < 1 or policy not in {"prioritized", "random"}:
        raise ValueError("positive count and prioritized or random policy required")
    pool = [i for i in queue(store) if i["status"] != "excluded"]
    if policy == "random":
        pool.sort(key=lambda i: i["id"])
        random.Random(seed).shuffle(pool)
    chosen, duplicates = [], set()
    for item in pool:
        group = tuple(item["duplicate_group"])
        if group not in duplicates and len(chosen) < count:
            chosen.append(item)
            duplicates.add(group)

    def coverage(items):
        return {
            "tasks": dict(Counter(i["task"] for i in items)),
            "sessions": dict(Counter(i["session"] for i in items)),
            "statuses": dict(Counter(i["status"] for i in items)),
            "recovery": sum(
                "recovery" in store.latest(i["id"]).get("quality_flags", []) for i in items
            ),
        }

    report = {
        "schema_version": 1,
        "policy": policy,
        "seed": seed,
        "requested_count": count,
        "actual_count": len(chosen),
        "selected": chosen,
        "pool_hash": digest(pool),
        "before": coverage(pool),
        "after": coverage(chosen),
        "limitations": (
            "Exact duplicates only; no near-duplicate embeddings. "
            "Missing tasks are collection gaps, not inferred errors."
        ),
    }
    report["selection_id"] = digest(report)[:24]
    return report
