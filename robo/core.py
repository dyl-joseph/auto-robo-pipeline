"""Validated input contracts and canonical artifact operations."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

LABELS = (
    "reach",
    "grasp",
    "transport",
    "release",
    "recover",
    "motion",
    "hold",
    "unknown",
    "not_observable",
)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Stream(Contract):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    modality: Literal["state", "action", "sensor"]
    units: Literal["rad", "rad/s", "m", "m/s", "normalized", "N", "bool"]
    timestamps_s: list[float] = Field(min_length=1)
    values: list[list[float]] = Field(min_length=1)
    offset_s: float = 0
    tolerance_s: float = Field(gt=0, le=1, default=0.08)
    clock_evidence: str = Field(min_length=3)

    @model_validator(mode="after")
    def valid_samples(self):
        if len(self.timestamps_s) != len(self.values):
            raise ValueError("stream timestamps and values have different lengths")
        if any(b <= a for a, b in zip(self.timestamps_s, self.timestamps_s[1:], strict=False)):
            raise ValueError("stream timestamps must be strictly increasing")
        sizes = {len(row) for row in self.values}
        if len(sizes) != 1 or 0 in sizes:
            raise ValueError("stream value dimensions must be constant and nonempty")
        return self


class Source(Contract):
    schema_version: Literal[1] = 1
    source_id: str = Field(min_length=1, max_length=200)
    release: str = Field(min_length=1)
    task: str = Field(min_length=1)
    session: str = Field(min_length=1)
    video: str
    license: str = Field(min_length=1)
    license_url: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    data_kind: Literal["synthetic", "real"]
    streams: list[Stream] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_streams(self):
        if len({s.name for s in self.streams}) != len(self.streams):
            raise ValueError("duplicate stream name")
        return self


class Segment(Contract):
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    label: Literal[
        "reach",
        "grasp",
        "transport",
        "release",
        "recover",
        "motion",
        "hold",
        "unknown",
        "not_observable",
    ]

    @model_validator(mode="after")
    def interval(self):
        if self.end_s <= self.start_s:
            raise ValueError("segments use half-open intervals with start < end")
        return self


class Edit(Contract):
    expected_revision: int = Field(ge=0)
    actor: str = Field(min_length=1, max_length=100)
    rationale: str = Field(min_length=1, max_length=2000)
    segments: list[Segment] = Field(max_length=500)
    outcome: Literal["unknown", "success", "failure"] = "unknown"
    quality_flags: list[Literal["recovery", "occluded", "corrupt", "uncertain", "incomplete"]] = (
        Field(default_factory=list)
    )
    active_s: float | None = Field(ge=0, le=86400, default=0)
    wall_s: float | None = Field(ge=0, le=86400, default=0)
    condition: Literal["manual", "assisted"] = "manual"
    suggestion_run: str | None = Field(default=None, pattern=r"^[a-f0-9]{24}$")
    perception_run: str | None = Field(default=None, pattern=r"^[a-f0-9]{24}$")

    @model_validator(mode="after")
    def timing(self):
        if self.condition == "manual" and (self.suggestion_run or self.perception_run):
            raise ValueError("manual reviews cannot claim machine evidence origins")
        if (self.active_s is None) != (self.wall_s is None):
            raise ValueError("active and wall time must both be measured or both unknown")
        if self.active_s is not None and self.active_s > self.wall_s + 0.01:
            raise ValueError("active time cannot exceed wall time")
        if not self.actor.strip() or not self.rationale.strip():
            raise ValueError("actor and rationale cannot be blank")
        return self


def canonical(value) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(canonical(value))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_segments(segments: list[dict], duration: float) -> None:
    previous = 0.0
    for raw in segments:
        segment = Segment.model_validate(raw)
        if segment.start_s < previous or segment.end_s > duration + 1e-6:
            raise ValueError("segments must be ordered, nonoverlapping and within video duration")
        previous = segment.end_s
    if not math.isfinite(duration):
        raise ValueError("invalid duration")


class ExportItem(Contract):
    episode_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    revision: StrictInt = Field(ge=1)
    split: Literal["train", "development", "test"]


class ExportRequest(Contract):
    selection: list[ExportItem] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_episodes(self):
        if len({item.episode_id for item in self.selection}) != len(self.selection):
            raise ValueError("duplicate selected episode")
        return self
