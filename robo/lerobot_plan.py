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


def plan_episode(record: dict, schema: TrainingSchema) -> dict:
    episode, annotation = record["episode"], record["annotation"]
    if annotation["status"] != "accepted":
        raise ValueError("only an explicitly accepted version can be converted")
    duration = episode["duration_s"]
    validate_segments(annotation["segments"], duration)
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
    # A grid point in an unlabeled interval is explicitly unknown, never success.
    labels = np.full(len(grid), LABELS.index("unknown"), dtype=np.int64)
    for segment in annotation["segments"]:
        mask = (grid >= segment["start_s"]) & (grid < segment["end_s"])
        labels[mask] = LABELS.index(segment["label"])
    return {
        "grid": grid,
        "video_indices": video_indices,
        "features": features,
        "labels": labels,
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
