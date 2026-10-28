"""Typed semantic annotations and a frozen, label-free evaluation split manifest."""

from fractions import Fraction
from typing import Annotated, Literal

from pydantic import Field, StrictInt, StringConstraints, model_validator

from robo.core import Contract, digest

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Action = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]


class FrameSpan(Contract):
    start: StrictInt = Field(ge=0)
    end_exclusive: StrictInt = Field(gt=0)
    evidence: list[StrictInt] = Field(min_length=1, max_length=20000)

    @model_validator(mode="after")
    def valid_span(self):
        if self.end_exclusive <= self.start:
            raise ValueError("source frame ranges must have start < end_exclusive")
        if self.evidence != sorted(set(self.evidence)) or any(
            not self.start <= i < self.end_exclusive for i in self.evidence
        ):
            raise ValueError("evidence frames must be sorted, unique and inside the segment")
        return self


class SegmentProvenance(Contract):
    kind: Literal["baseline", "model", "independent_human", "synthetic_fixture"]
    producer: Identifier
    artifact_sha256: Sha256


class SemanticSegment(Contract):
    episode_id: Identifier
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    source_frames: FrameSpan
    subtask_text: Text
    normalized_action: Action
    object_text: Text | None
    uncertainty: float | None = Field(ge=0, le=1)
    provenance: SegmentProvenance

    @model_validator(mode="after")
    def interval(self):
        if self.end_s <= self.start_s:
            raise ValueError("semantic segments use half-open start/end intervals")
        return self


class EpisodeSpec(Contract):
    episode_id: Identifier
    dataset: Identifier
    release: Identifier
    dataset_episode_id: Identifier
    session_id: Identifier
    camera_id: Identifier
    source_sha256: Sha256
    media_sha256: Sha256
    near_duplicate_groups: list[Identifier] = Field(default_factory=list, max_length=100)
    split: Literal["train", "development", "test"]
    # Includes the exclusive end of the final frame, so no nominal tail is guessed here.
    frame_boundaries_s: list[float] = Field(min_length=3, max_length=20001)
    timing: Literal["source_pts", "inferred_nominal_fps"]

    @model_validator(mode="after")
    def increasing_frames(self):
        times = self.frame_boundaries_s
        if times[0] != 0 or any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("source frame boundaries must start at zero and strictly increase")
        return self

    def leakage_keys(self):
        return [
            ("episode", self.dataset, self.dataset_episode_id),
            ("session", self.dataset, self.session_id),
            ("source", self.source_sha256),
            ("media", self.media_sha256),
            *(("near_duplicate", group) for group in self.near_duplicate_groups),
        ]


class IdentityAudit(Contract):
    status: Literal["unverified", "human_verified", "synthetic_fixture"]
    reviewer: Identifier
    evidence_sha256: Sha256


def episode_spec(
    episode, *, dataset, dataset_episode_id, camera_id, split, near_duplicate_groups=()
):
    """Carry an imported episode's identity and source clock into a split manifest."""
    base = Fraction(episode["time_base"])
    ticks = episode["original_pts"]
    return EpisodeSpec(
        episode_id=episode["id"],
        dataset=dataset,
        release=episode["release"],
        dataset_episode_id=dataset_episode_id,
        session_id=episode["session"],
        camera_id=camera_id,
        source_sha256=episode["content_hash"],
        media_sha256=episode["video_sha256"],
        near_duplicate_groups=list(near_duplicate_groups),
        split=split,
        frame_boundaries_s=[float((t - ticks[0]) * base) for t in ticks] + [episode["duration_s"]],
        timing=episode.get("video_timing", "source_pts"),
    )


class EvaluationPlan(Contract):
    schema_version: Literal[1] = 1
    taxonomy_id: Identifier
    normalized_actions: list[Action] = Field(min_length=1, max_length=100)
    annotation_guidelines_sha256: Sha256
    identity_audit: IdentityAudit
    episodes: list[EpisodeSpec] = Field(min_length=1, max_length=200)
    review_uncertainty_threshold: float = Field(ge=0, le=1, default=0.6)

    @model_validator(mode="after")
    def isolated_splits(self):
        if len(set(self.normalized_actions)) != len(self.normalized_actions):
            raise ValueError("taxonomy actions must be unique")
        if not {"unknown", "not_observable"}.issubset(self.normalized_actions):
            raise ValueError("taxonomy must include unknown and not_observable")
        if len({e.episode_id for e in self.episodes}) != len(self.episodes):
            raise ValueError("episode IDs must be unique")
        views, groups = set(), {}
        for episode in self.episodes:
            view = (episode.dataset, episode.dataset_episode_id, episode.camera_id)
            if view in views:
                raise ValueError("duplicate dataset episode/camera entry")
            views.add(view)
            for key in episode.leakage_keys():
                if key in groups and groups[key] != episode.split:
                    raise ValueError(
                        "split leakage: shared episode, session, source or duplicate media"
                    )
                groups[key] = episode.split
        return self

    @property
    def plan_sha256(self):
        return digest(self.model_dump())


class PredictionEpisode(Contract):
    episode_id: Identifier
    segments: list[SemanticSegment] = Field(max_length=500)
    error: Text | None = None

    @model_validator(mode="after")
    def failed_prediction(self):
        if self.error is not None and self.segments:
            raise ValueError("failed predictions must contain no scored segments")
        return self


class Predictions(Contract):
    schema_version: Literal[1] = 1
    plan_sha256: Sha256
    kind: Literal["uncorrected_predictions"] = "uncorrected_predictions"
    run_id: Identifier
    producer: Identifier
    code_revision: Identifier
    config_sha256: Sha256
    training_episode_ids: list[Identifier]
    tuning_episode_ids: list[Identifier]
    reference_access: Literal["none"]
    episodes: list[PredictionEpisode] = Field(min_length=1, max_length=200)


class ReferenceAudit(Contract):
    status: Literal["unverified", "human_verified", "synthetic_fixture"]
    method: Literal["blind_manual", "publisher_annotation", "model_assisted", "synthetic_fixture"]
    annotators: list[Identifier] = Field(min_length=1)
    source_uri: Text
    source_sha256: Sha256
    license: Identifier
    license_url: Text
    prediction_exposure: bool


class ReferenceEpisode(Contract):
    episode_id: Identifier
    segments: list[SemanticSegment] = Field(min_length=1, max_length=500)
    audit: ReferenceAudit


class References(Contract):
    schema_version: Literal[1] = 1
    plan_sha256: Sha256
    evidence_kind: Literal["synthetic_fixture", "independent_human_reference"]
    episodes: list[ReferenceEpisode] = Field(min_length=1, max_length=200)


def validate_episode_segments(segments, episode, actions, *, complete=False):
    end = 0
    for segment in segments:
        span = segment.source_frames
        if segment.episode_id != episode.episode_id:
            raise ValueError("segment points to a different episode")
        if span.end_exclusive >= len(episode.frame_boundaries_s) or span.start < end:
            raise ValueError("segments must be ordered, non-overlapping and inside source frames")
        if complete and span.start != end:
            raise ValueError(
                "reference segments must cover every source frame; mark unknown explicitly"
            )
        for time, index in ((segment.start_s, span.start), (segment.end_s, span.end_exclusive)):
            if abs(time - episode.frame_boundaries_s[index]) > 1e-6:
                raise ValueError("segment times disagree with source frame boundaries")
        if segment.normalized_action not in actions:
            raise ValueError("segment action is outside the frozen taxonomy")
        end = span.end_exclusive
    if complete and end != len(episode.frame_boundaries_s) - 1:
        raise ValueError("reference segments must cover the complete episode")
