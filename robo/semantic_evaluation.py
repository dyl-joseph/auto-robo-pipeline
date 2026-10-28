"""Score uncorrected semantic predictions against separately supplied references."""

import argparse
import json
from pathlib import Path

import numpy as np

from robo.core import atomic_json, digest
from robo.evaluation import match
from robo.semantic import EvaluationPlan, Predictions, References, validate_episode_segments


def normalized_text(value):
    return " ".join(value.casefold().split()) if value is not None else None


def indexed(rows):
    result = {row.episode_id: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate episode records")
    return result


def validate_inputs(plan, predictions, references, split):
    if split not in {"development", "test"}:
        raise ValueError("evaluate only development or held-out test episodes")
    if predictions.plan_sha256 != plan.plan_sha256 or references.plan_sha256 != plan.plan_sha256:
        raise ValueError("prediction/reference plan hash differs from the frozen manifest")
    population = {e.episode_id: e for e in plan.episodes}
    selected = {key: row for key, row in population.items() if row.split == split}
    predicted, reference = indexed(predictions.episodes), indexed(references.episodes)
    if not selected or set(predicted) != set(selected) or set(reference) != set(selected):
        raise ValueError(
            "predictions and references must include every episode in the chosen split"
        )
    for name, allowed in (
        ("training_episode_ids", {"train"}),
        ("tuning_episode_ids", {"train", "development"}),
    ):
        ids = getattr(predictions, name)
        if len(ids) != len(set(ids)) or any(
            key not in population or population[key].split not in allowed for key in ids
        ):
            raise ValueError(
                "training/tuning declarations use unknown or disallowed split episodes"
            )
        if set(ids) & set(selected):
            raise ValueError("evaluation episodes were declared as training/tuning inputs")
    real = references.evidence_kind == "independent_human_reference"
    if real and plan.identity_audit.status != "human_verified":
        raise ValueError("real reference evaluation requires a verified identity/split audit")
    prediction_hashes = {
        s.provenance.artifact_sha256 for e in predictions.episodes for s in e.segments
    }
    for key, episode in selected.items():
        p, r = predicted[key], reference[key]
        validate_episode_segments(p.segments, episode, plan.normalized_actions)
        validate_episode_segments(r.segments, episode, plan.normalized_actions, complete=True)
        if r.audit.prediction_exposure:
            raise ValueError("reference annotators must not have seen the evaluated predictions")
        if r.audit.source_sha256 in prediction_hashes:
            raise ValueError("reference and prediction artifacts must be independent")
        if real and (
            r.audit.status != "human_verified"
            or r.audit.method not in {"blind_manual", "publisher_annotation"}
        ):
            raise ValueError(
                "real references must be independently human-verified, not model-assisted"
            )
        if not real and (
            r.audit.status != "synthetic_fixture" or r.audit.method != "synthetic_fixture"
        ):
            raise ValueError("fixture references require explicit synthetic audit metadata")
        for segment in p.segments:
            kinds = {"baseline", "model"} if real else {"baseline", "model", "synthetic_fixture"}
            if (
                segment.provenance.kind not in kinds
                or segment.uncertainty is None
                or segment.provenance.producer != predictions.producer
            ):
                raise ValueError(
                    "predictions need machine provenance, matching producer and uncertainty"
                )
        for segment in r.segments:
            kind = "independent_human" if real else "synthetic_fixture"
            if (
                segment.provenance.kind != kind
                or segment.provenance.artifact_sha256 != r.audit.source_sha256
            ):
                raise ValueError("reference segment provenance differs from its audit")
    return selected, predicted, reference


def counts(tp, fp, fn):
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
    }


def score_segments(predicted, reference, threshold, *, include_object=False):
    def rows(segments):
        return [
            {
                "start_s": s.start_s,
                "end_s": s.end_s,
                "label": (
                    (s.normalized_action, normalized_text(s.object_text))
                    if include_object
                    else s.normalized_action
                ),
            }
            for s in segments
        ]

    pairs = match(rows(predicted), rows(reference), threshold)
    errors = [
        abs(getattr(predicted[a], name) - getattr(reference[b], name))
        for a, b in pairs
        for name in ("start_s", "end_s")
    ]
    per_action = {}
    for label in sorted({s.normalized_action for s in [*predicted, *reference]}):
        tp = sum(predicted[a].normalized_action == label for a, _ in pairs)
        per_action[label] = counts(
            tp,
            sum(s.normalized_action == label for s in predicted) - tp,
            sum(s.normalized_action == label for s in reference) - tp,
        )
    return pairs, errors, per_action


def evaluate(
    plan: EvaluationPlan, predictions: Predictions, references: References, *, split="test"
):
    selected, predicted, reference = validate_inputs(plan, predictions, references, split)
    episodes = []
    for key, spec in selected.items():
        row = predicted[key]
        reasons = []
        if row.error:
            reasons.append("prediction_failed")
        covered = sum(s.source_frames.end_exclusive - s.source_frames.start for s in row.segments)
        frames = len(spec.frame_boundaries_s) - 1
        if covered < frames:
            reasons.append("unlabeled_frames")
        if any(s.normalized_action in {"unknown", "not_observable"} for s in row.segments):
            reasons.append("unknown_or_not_observable")
        if any(s.uncertainty >= plan.review_uncertainty_threshold for s in row.segments):
            reasons.append("uncertain_segment")
        episodes.append(
            {
                "episode_id": key,
                "prediction_failed": row.error is not None,
                "prediction_error": row.error,
                "predicted_segments": len(row.segments),
                "reference_segments": len(reference[key].segments),
                "covered_frames": covered,
                "source_frames": frames,
                "review_reasons": reasons,
            }
        )
    metrics = {}
    for mode in ("action", "action_object"):
        thresholds = {}
        for threshold in (0.3, 0.5, 0.7):
            total_tp, total_fp, total_fn = 0, 0, 0
            errors, text_matches, object_matches = [], [], []
            classes = {}
            for key in selected:
                p, r = predicted[key].segments, reference[key].segments
                pairs, boundary_errors, actions = score_segments(
                    p, r, threshold, include_object=mode == "action_object"
                )
                total_tp += len(pairs)
                total_fp += len(p) - len(pairs)
                total_fn += len(r) - len(pairs)
                errors.extend(boundary_errors)
                text_matches.extend(
                    normalized_text(p[a].subtask_text) == normalized_text(r[b].subtask_text)
                    for a, b in pairs
                )
                object_matches.extend(
                    normalized_text(p[a].object_text) == normalized_text(r[b].object_text)
                    for a, b in pairs
                )
                for action, result in actions.items():
                    values = classes.setdefault(action, {"tp": 0, "fp": 0, "fn": 0})
                    for name in values:
                        values[name] += result[name]
            per_action = {key: counts(**values) for key, values in sorted(classes.items())}
            f1s = [v["f1"] for v in per_action.values() if v["f1"] is not None]
            thresholds[str(threshold)] = {
                **counts(total_tp, total_fp, total_fn),
                "per_action": per_action,
                "macro_f1": float(np.mean(f1s)) if f1s else None,
                "matched_boundary_median_s": float(np.median(errors)) if errors else None,
                "matched_boundary_p90_s": float(np.percentile(errors, 90)) if errors else None,
                "matched_subtask_text_exact_agreement": float(np.mean(text_matches))
                if text_matches
                else None,
                "matched_object_text_exact_agreement": float(np.mean(object_matches))
                if object_matches
                else None,
            }
        metrics[mode] = thresholds
    return {
        "schema_version": 1,
        "evidence_kind": references.evidence_kind,
        "split": split,
        "plan_sha256": plan.plan_sha256,
        "predictions_sha256": digest(predictions.model_dump()),
        "references_sha256": digest(references.model_dump()),
        "run_id": predictions.run_id,
        "producer": predictions.producer,
        "episode_count": len(selected),
        "failed_predictions": sum(e["prediction_failed"] for e in episodes),
        "review_requests": sum(bool(e["review_reasons"]) for e in episodes),
        "review_fraction": sum(bool(e["review_reasons"]) for e in episodes) / len(episodes),
        "episodes": episodes,
        "metrics": metrics,
        "automatic_approvals": 0,
        "human_active_seconds": None,
        "limitations": [
            "Reference independence and identity audits are supplied declarations, "
            "not automatically verified facts.",
            "All raw predictions are scored; uncertainty and failed episodes are not filtered out.",
            "Boundary and exact-text diagnostics cover matched segments only; "
            "misses remain in precision/recall/F1.",
            "Object and subtask text use case/whitespace-normalized exact comparison, "
            "not semantic equivalence.",
            "Split checks cover the declared catalog and duplicate groups, "
            "not unknown pretrained-model contamination.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("references", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("development", "test"), default="test")
    args = parser.parse_args()
    paths = [args.plan, args.predictions, args.references]
    if any(path.suffix != ".json" or path.stat().st_size > 64 * 1024 * 1024 for path in paths):
        raise ValueError("Use JSON input files no larger than 64 MiB")
    if args.output.resolve() in [path.resolve() for path in paths]:
        raise ValueError("Keep evaluation output separate from inputs")
    plan, predictions, references = [
        model.model_validate_json(path.read_bytes())
        for model, path in zip((EvaluationPlan, Predictions, References), paths, strict=True)
    ]
    result = evaluate(plan, predictions, references, split=args.split)
    if args.output.exists() and json.loads(args.output.read_text()) != result:
        raise ValueError("Output contains a different evaluation; choose a new path")
    atomic_json(args.output, result)
    print(
        json.dumps(
            {key: result[key] for key in ("evidence_kind", "episode_count", "review_requests")}
        )
    )


if __name__ == "__main__":
    main()
