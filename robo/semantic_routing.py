"""Prioritize machine predictions without consulting reference labels."""

from robo.semantic import validate_episode_segments


def review_reasons(plan, spec, prediction):
    validate_episode_segments(prediction.segments, spec, plan.normalized_actions)
    reasons = []
    if prediction.error:
        reasons.append("prediction_failed")
    covered = sum(
        s.source_frames.end_exclusive - s.source_frames.start for s in prediction.segments
    )
    if covered < len(spec.frame_boundaries_s) - 1:
        reasons.append("unlabeled_frames")
    if any(s.normalized_action in {"unknown", "not_observable"} for s in prediction.segments):
        reasons.append("unknown_or_not_observable")
    if any(
        s.uncertainty is None or s.uncertainty >= plan.review_uncertainty_threshold
        for s in prediction.segments
    ):
        reasons.append("uncertain_segment")
    return reasons
