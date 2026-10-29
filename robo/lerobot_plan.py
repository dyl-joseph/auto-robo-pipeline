"""Explicit, causal resampling of an accepted source version for LeRobot."""

import math
from fractions import Fraction
from typing import Literal

import numpy as np
from pydantic import Field, StrictInt, model_validator

from robo.core import LABELS, Contract, Stream, validate_segments


class Component(Contract):
    stream: str
    index: StrictInt = Field(ge=0)
    name: str = Field(min_length=1)
    units: str = Field(min_length=1)
    meaning: str = Field(min_length=3)


class TrainingSchema(Contract):
    schema_version: Literal[1] = 1
    robot_type: str = Field(min_length=3)
    fps: StrictInt = Field(ge=1, le=60)
    width: StrictInt = Field(ge=16, le=1920)
    height: StrictInt = Field(ge=16, le=1080)
    max_video_age_s: float = Field(gt=0, le=1)
    state: list[Component] = Field(min_length=1, max_length=256)
    action: list[Component] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def valid_schema(self):
        if self.width % 2 or self.height % 2:
            raise ValueError("H.264 output dimensions must be even")
        for components in (self.state, self.action):
            if len({c.name for c in components}) != len(components):
                raise ValueError("feature component names must be unique")
            if len({(c.stream, c.index) for c in components}) != len(components):
                raise ValueError("duplicate source component")
        return self


def previous_samples(times: list[float], grid: np.ndarray, tolerance: float, name: str):
    """Latest sample no later than grid time, allowing only 1 ns rounding noise."""
    samples = np.asarray(times, dtype=np.float64)
    if not len(samples) or not np.isfinite(samples).all() or np.any(np.diff(samples) <= 0):
        raise ValueError(f"{name}: timestamps must be finite and strictly increasing")
    indices = np.searchsorted(samples, grid + 1e-9, side="right") - 1
    ages = grid - samples[indices.clip(0)]
    invalid = (indices < 0) | (ages > tolerance + 1e-9)
    if invalid.any():
        first = int(np.flatnonzero(invalid)[0])
        raise ValueError(f"{name}: no causal sample within tolerance at {grid[first]:.9f}s")
    return indices, ages


def plan_inputs(episode: dict, schema: TrainingSchema) -> dict:
    """Check causal alignment and feature mapping without approving or exporting labels."""
    if (
        episode.get("video_timing") == "inferred_nominal_fps"
        and abs(episode["original_pts_s"][0]) > 1e-6
    ):
        raise ValueError(
            "derived video starts after clock zero; resolve its origin before training"
        )
    duration = episode["duration_s"]
    if not math.isfinite(duration) or not 0 < duration <= 300:
        raise ValueError("training conversion supports episodes up to 300 seconds")
    grid = np.arange(math.ceil(duration * schema.fps - 1e-9), dtype=np.float64) / schema.fps
    # ffprobe's decimal pts_time is rounded to six places. Use integer PTS/time_base
    # so a 1/12s frame is not spuriously treated as future at its own grid point.
    time_base = Fraction(episode["time_base"])
    pts = episode["original_pts"]
    if time_base <= 0 or len(pts) != len(episode["timestamps_s"]):
        raise ValueError("invalid original PTS/time-base mapping")
    exact_seconds = [float(p * time_base) for p in pts]
    normalized = [float((p - pts[0]) * time_base) for p in pts]
    video_indices, video_ages = previous_samples(normalized, grid, schema.max_video_age_s, "video")
    streams = {s["name"]: Stream.model_validate(s) for s in episode["streams"]}
    features, alignment = {}, {}
    for modality, key in (("state", "observation.state"), ("action", "action")):
        columns = []
        for component in getattr(schema, modality):
            if component.stream not in streams:
                raise ValueError(f"missing required {modality} stream: {component.stream}")
            stream = streams[component.stream]
            if stream.modality != modality or stream.units != component.units:
                raise ValueError(f"{stream.name}: modality or units differ from declared schema")
            if component.index >= len(stream.values[0]):
                raise ValueError(f"{stream.name}: component index out of range")
            if stream.name not in alignment:
                mapped = [t + stream.offset_s for t in stream.timestamps_s]
                indices, ages = previous_samples(mapped, grid, stream.tolerance_s, stream.name)
                alignment[stream.name] = {
                    "indices": indices.tolist(),
                    "age_s": ages.tolist(),
                    "source_timestamps_s": [stream.timestamps_s[i] for i in indices],
                    "offset_s": stream.offset_s,
                    "tolerance_s": stream.tolerance_s,
                    "clock_evidence": stream.clock_evidence,
                }
            indices = alignment[stream.name]["indices"]
            columns.append([stream.values[i][component.index] for i in indices])
        values = np.asarray(columns, dtype=np.float64).T
        with np.errstate(over="ignore"):
            features[key] = values.astype(np.float32)
        if not np.isfinite(features[key]).all():
            raise ValueError("source values cannot be represented as finite float32")
    return {
        "grid": grid,
        "video_indices": video_indices,
        "features": features,
        "alignment": {
            "source_video_timing": episode.get("video_timing", "source_pts"),
            "timestamp_authority": episode.get("timestamp_authority", "source video PTS"),
            "timestamps_s": grid.tolist(),
            "video_frame_indices": video_indices.tolist(),
            "video_original_pts": [episode["original_pts"][i] for i in video_indices],
            "video_original_pts_s": [exact_seconds[i] for i in video_indices],
            "video_age_s": video_ages.tolist(),
            "omitted_source_frames": len(episode["timestamps_s"]) - len(set(video_indices)),
            "repeated_source_frames": len(grid) - len(set(video_indices)),
            "duration_rounding_s": len(grid) / schema.fps - duration,
            "streams": alignment,
        },
    }


def plan_episode(record: dict, schema: TrainingSchema) -> dict:
    episode, annotation = record["episode"], record["annotation"]
    if annotation["status"] != "accepted":
        raise ValueError("only an explicitly accepted version can be converted")
    validate_segments(annotation["segments"], episode["duration_s"])
    plan = plan_inputs(episode, schema)
    grid = plan["grid"]
    # A grid point in an unlabeled interval is explicitly unknown, never success.
    labels = np.full(len(grid), LABELS.index("unknown"), dtype=np.int64)
    for segment in annotation["segments"]:
        mask = (grid >= segment["start_s"]) & (grid < segment["end_s"])
        labels[mask] = LABELS.index(segment["label"])
    semantic = semantic_frames(record, plan["video_indices"])
    if semantic is not None:
        labels = semantic.pop("labels")
    return {**plan, "labels": labels, "semantic": semantic}


def semantic_frames(record: dict, indices):
    """Attach reviewed text to the selected source frame, including deliberate repeats."""
    from robo.semantic_review import ReviewedSegment

    episode, annotation = record["episode"], record["annotation"]
    review = (annotation.get("native") or {}).get("semantic_review")
    if review is None:
        return None
    if (
        review.get("schema_version") != 1
        or review.get("kind") != "model_assisted_review"
        or review.get("prediction_exposure") is not True
        or review.get("complete") is not True
        or review.get("split") != record["split"]
    ):
        raise ValueError("training export requires complete model-assisted semantic review")
    segments = [ReviewedSegment.model_validate(s) for s in review["segments"]]
    boundaries = [*episode["timestamps_s"], episode["duration_s"]]
    end, source_rows = 0, []
    if len(segments) != len(annotation["segments"]):
        raise ValueError("semantic and timeline segments disagree")
    for segment, timeline in zip(segments, annotation["segments"], strict=True):
        if (
            segment.source_start != end
            or not end < segment.source_end_exclusive < len(boundaries)
            or segment.normalized_action not in LABELS
            or not segment.subtask_text.strip()
            or (segment.object_text is not None and not segment.object_text.strip())
            or abs(segment.start_s - boundaries[end]) > 1e-6
            or abs(segment.end_s - boundaries[segment.source_end_exclusive]) > 1e-6
            or abs(segment.start_s - timeline["start_s"]) > 1e-6
            or abs(segment.end_s - timeline["end_s"]) > 1e-6
            or segment.normalized_action != timeline["label"]
        ):
            raise ValueError("semantic coverage, source frames or timeline labels disagree")
        source_rows.extend([segment] * (segment.source_end_exclusive - end))
        end = segment.source_end_exclusive
    if not source_rows or end != len(boundaries) - 1:
        raise ValueError("semantic review does not cover all source frames")
    selected = [source_rows[i] for i in indices]
    return {
        "labels": np.asarray([LABELS.index(s.normalized_action) for s in selected], dtype=np.int64),
        "annotation.subtask_text": [s.subtask_text for s in selected],
        "annotation.object_text": [s.object_text or "" for s in selected],
        "annotation.object_present": np.asarray(
            [[s.object_text is not None] for s in selected], dtype=np.bool_
        ),
    }
