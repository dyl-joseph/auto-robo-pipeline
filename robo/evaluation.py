"""Temporal matching and fixed-budget evaluation from explicit reference artifacts."""

import random

import numpy as np

from robo.core import validate_segments


def iou(a: dict, b: dict) -> float:
    intersection = max(0, min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"]))
    union = max(a["end_s"], b["end_s"]) - min(a["start_s"], b["start_s"])
    return intersection / union if union else 0


def match(predicted: list[dict], reference: list[dict], threshold: float) -> list[tuple[int, int]]:
    # Successive shortest augmenting paths: maximum cardinality, then maximum total IoU.
    n, m = len(predicted), len(reference)
    size = n + m + 2
    source, sink = size - 2, size - 1
    graph = [[] for _ in range(size)]

    def edge(a, b, cost):
        graph[a].append([b, 1, cost, len(graph[b])])
        graph[b].append([a, 0, -cost, len(graph[a]) - 1])

    for a in range(n):
        edge(source, a, 0)
    for b in range(m):
        edge(n + b, sink, 0)
    for a, prediction in enumerate(predicted):
        for b, truth in enumerate(reference):
            overlap = iou(prediction, truth)
            if prediction["label"] == truth["label"] and overlap >= threshold:
                edge(a, n + b, -overlap)
    while True:
        distance, parent = [float("inf")] * size, [None] * size
        distance[source] = 0
        for _ in range(size - 1):
            changed = False
            for a in range(size):
                for index, (b, capacity, cost, _) in enumerate(graph[a]):
                    if capacity and distance[a] + cost < distance[b] - 1e-12:
                        distance[b] = distance[a] + cost
                        parent[b] = (a, index)
                        changed = True
            if not changed:
                break
        if parent[sink] is None:
            break
        node = sink
        while node != source:
            a, index = parent[node]
            entry = graph[a][index]
            entry[1] = 0
            graph[node][entry[3]][1] = 1
            node = a
    return [
        (a, b - n)
        for a in range(n)
        for b, capacity, _, _ in graph[a]
        if n <= b < n + m and capacity == 0
    ]


def temporal_metrics(predicted: list[dict], reference: list[dict], threshold=0.5) -> dict:
    pairs = match(predicted, reference, threshold)
    tp, fp, fn = len(pairs), len(predicted) - len(pairs), len(reference) - len(pairs)
    errors = [
        abs(predicted[a][key] - reference[b][key]) for a, b in pairs for key in ("start_s", "end_s")
    ]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        "boundary_median_s": float(np.median(errors)) if errors else None,
        "boundary_p90_s": float(np.percentile(errors, 90)) if errors else None,
    }


def evaluate(protocol: dict) -> dict:
    if protocol.get("evidence_kind") not in {"synthetic_fixture", "human_reference"}:
        raise ValueError("declare evidence_kind: synthetic_fixture or human_reference")
    episodes = protocol["episodes"]
    if not episodes or len({row["id"] for row in episodes}) != len(episodes):
        raise ValueError("evaluation needs unique episodes")
    for row in episodes:
        for field in ("predicted", "reference"):
            validate_segments(row[field], row["duration_s"])
    result = {
        "evidence_kind": protocol["evidence_kind"],
        "episode_count": len(episodes),
        "human_efficiency": None,
        "inter_annotator_agreement": None,
        "limitations": [
            "No human study inferred from system tests or synthetic references.",
            "Boundary metrics cover matched segments; false negatives remain explicit.",
        ],
        "thresholds": {},
    }
    for threshold in (0.3, 0.5, 0.7):
        per_episode = [
            temporal_metrics(row["predicted"], row["reference"], threshold) for row in episodes
        ]
        counts = {key: sum(row[key] for row in per_episode) for key in ("tp", "fp", "fn")}
        denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
        labels = sorted(
            {
                s["label"]
                for row in episodes
                for name in ("predicted", "reference")
                for s in row[name]
            }
        )
        by_class = {}
        for label in labels:
            rows = [
                temporal_metrics(
                    [s for s in row["predicted"] if s["label"] == label],
                    [s for s in row["reference"] if s["label"] == label],
                    threshold,
                )
                for row in episodes
            ]
            tp, fp, fn = (sum(r[key] for r in rows) for key in ("tp", "fp", "fn"))
            by_class[label] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
        result["thresholds"][str(threshold)] = {
            **counts,
            "micro_f1": 2 * counts["tp"] / denominator if denominator else None,
            "macro_f1": float(np.mean([v for v in by_class.values() if v is not None]))
            if by_class
            else None,
            "class_f1": by_class,
            "episodes": per_episode,
        }
    study = study_metrics(protocol)
    result["study"] = study
    if protocol["evidence_kind"] == "human_reference":
        result["human_efficiency"] = study["conditions"] or None
        result["inter_annotator_agreement"] = study["agreement_at_iou_0.5"]
    if "error_pool" in protocol:
        result["review_budget"] = budget_metrics(
            protocol["error_pool"], protocol["budget_s"], protocol.get("seed", 7)
        )
    return result


def budget_metrics(items: list[dict], budget_s: float, seed: int = 7) -> dict:
    if (
        budget_s <= 0
        or not items
        or any(i["review_s"] <= 0 or not isinstance(i["erroneous"], bool) for i in items)
    ):
        raise ValueError(
            "budget evaluation needs positive measured costs and adjudicated boolean "
            "errors for the full pool"
        )
    ranked = sorted(items, key=lambda i: (-i["priority"], i["id"]))
    shuffled = sorted(items, key=lambda i: i["id"])
    random.Random(seed).shuffle(shuffled)
    total_errors = sum(i["erroneous"] for i in items)
    result = {}
    for name, order in (("prioritized", ranked), ("random", shuffled)):
        spent, selected = 0, []
        for item in order:
            if spent + item["review_s"] <= budget_s:
                selected.append(item)
                spent += item["review_s"]
        found = sum(i["erroneous"] for i in selected)
        result[name] = {
            "selected": [i["id"] for i in selected],
            "spent_s": spent,
            "precision": found / len(selected) if selected else None,
            "recall": found / total_errors if total_errors else None,
            "missed_errors": total_errors - found,
        }
    return result


def study_metrics(protocol: dict) -> dict:
    """Analyze reviewer artifacts when actually supplied; no simulated default timings."""
    episodes = protocol["episodes"]
    evidence = protocol["evidence_kind"]
    agreements = []
    for row in episodes:
        if "reviewer_a" in row and "reviewer_b" in row:
            for field in ("reviewer_a", "reviewer_b"):
                validate_segments(row[field], row["duration_s"])
            agreements.append(
                {"episode_id": row["id"], **temporal_metrics(row["reviewer_a"], row["reviewer_b"])}
            )
    conditions = {}
    rng = np.random.default_rng(protocol.get("seed", 7))
    for condition in ("manual", "assisted"):
        rows = [r for r in episodes if r.get("condition") == condition and "review_history" in r]
        if not rows:
            continue
        for row in rows:
            if not row.get("group") or not row.get("reviewer"):
                raise ValueError("timing analysis needs session group and reviewer assignment")
            for event in row["review_history"]:
                if (
                    not np.isfinite(event["active_s"])
                    or not np.isfinite(event["wall_s"])
                    or not 0 <= event["active_s"] <= event["wall_s"]
                ):
                    raise ValueError("invalid review timing")
        groups = sorted({r["group"] for r in rows})

        def aggregate(sample):
            active = sum(e["active_s"] for r in sample for e in r["review_history"])
            wall = sum(e["wall_s"] for r in sample for e in r["review_history"])
            accepted = sum(len(r["predicted"]) for r in sample if r.get("accepted"))
            return {
                "active_s": active,
                "wall_s": wall,
                "accepted_segments": accepted,
                "seconds_per_accepted_segment": active / accepted if accepted else None,
                "accepted_segments_per_hour": accepted * 3600 / active if active else None,
            }

        values = aggregate(rows)
        interval = None
        if len(groups) >= 2:
            samples = []
            for _ in range(1000):
                chosen = rng.choice(groups, len(groups), replace=True)
                sample = [r for group in chosen for r in rows if r["group"] == group]
                value = aggregate(sample)["seconds_per_accepted_segment"]
                if value is not None:
                    samples.append(value)
            if samples:
                interval = np.percentile(samples, [2.5, 97.5]).tolist()
        conditions[condition] = {
            **values,
            "episode_count": len(rows),
            "independent_groups": len(groups),
            "reviewers": sorted({r["reviewer"] for r in rows}),
            "seconds_per_segment_ci95": interval,
        }
    reduction = None
    if all(c in conditions for c in ("manual", "assisted")):
        manual = conditions["manual"]["seconds_per_accepted_segment"]
        assisted = conditions["assisted"]["seconds_per_accepted_segment"]
        if manual and assisted is not None:
            reduction = 1 - assisted / manual
    return {
        "evidence_kind": evidence,
        "conditions": conditions,
        "time_reduction_point_estimate": reduction,
        "agreement_at_iou_0.5": agreements or None,
        "timing_basis": "Sum of ALL review events, including corrections, rejection and acceptance",
        "inference": "Descriptive pilot only. No noninferiority or population claim inferred.",
    }
