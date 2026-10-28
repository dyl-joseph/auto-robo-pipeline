from copy import deepcopy

import numpy as np
import pytest

from robo.core import LABELS
from robo.lerobot_plan import TrainingSchema, plan_episode, previous_samples


@pytest.fixture
def training_case():
    schema = TrainingSchema.model_validate(
        {
            "robot_type": "synthetic-test-signals",
            "fps": 10,
            "width": 64,
            "height": 48,
            "max_video_age_s": 0.11,
            "state": [
                {
                    "stream": "position",
                    "index": 1,
                    "name": "y",
                    "units": "m",
                    "meaning": "generated y",
                }
            ],
            "action": [
                {
                    "stream": "command",
                    "index": 0,
                    "name": "dx",
                    "units": "m/s",
                    "meaning": "generated command",
                }
            ],
        }
    )
    record = {
        "episode": {
            "duration_s": 0.4,
            "time_base": "1/10",
            "timestamps_s": [0, 0.1, 0.2, 0.3],
            "original_pts": [100, 101, 102, 103],
            "original_pts_s": [10, 10.1, 10.2, 10.3],
            "streams": [
                {
                    "name": "position",
                    "modality": "state",
                    "units": "m",
                    "timestamps_s": [-0.01, 0.09, 0.19, 0.29],
                    "offset_s": 0.01,
                    "values": [[1, 2], [3, 4], [5, 6], [7, 8]],
                    "tolerance_s": 0.03,
                    "clock_evidence": "fixture offset 10ms",
                },
                {
                    "name": "command",
                    "modality": "action",
                    "units": "m/s",
                    "timestamps_s": [0, 0.15, 0.3],
                    "values": [[10], [20], [30]],
                    "tolerance_s": 0.11,
                    "clock_evidence": "fixture shared clock",
                },
            ],
        },
        "annotation": {
            "status": "accepted",
            "segments": [{"start_s": 0.1, "end_s": 0.3, "label": "reach"}],
        },
    }
    return record, schema


def test_causal_alignment_offsets_units_and_half_open_labels(training_case):
    record, schema = training_case
    original = deepcopy(record)
    plan = plan_episode(record, schema)
    np.testing.assert_array_equal(plan["features"]["observation.state"][:, 0], [2, 4, 6, 8])
    np.testing.assert_array_equal(plan["features"]["action"][:, 0], [10, 10, 20, 30])
    assert plan["labels"].tolist() == [
        LABELS.index(k) for k in ["unknown", "reach", "reach", "unknown"]
    ]
    assert plan["alignment"]["video_original_pts_s"] == [10, 10.1, 10.2, 10.3]
    assert plan["alignment"]["streams"]["command"]["indices"] == [0, 0, 1, 2]
    assert record == original


@pytest.mark.parametrize(
    "problem", ["missing", "units", "modality", "dimension", "gap", "future", "overflow", "draft"]
)
def test_required_data_never_guessed_or_padded(training_case, problem):
    record, schema = training_case
    stream = record["episode"]["streams"][0]
    if problem == "missing":
        record["episode"]["streams"] = []
    elif problem in {"units", "modality"}:
        stream[problem] = "normalized" if problem == "units" else "action"
    elif problem == "dimension":
        stream["values"] = [[1]] * 4
    elif problem == "gap":
        stream["timestamps_s"] = [-0.01, 0.05, 0.15, 0.25]
    elif problem == "future":
        stream["offset_s"] = 0.02
    elif problem == "overflow":
        stream["values"][0][1] = 1e100
    else:
        record["annotation"]["status"] = "draft"
    with pytest.raises(ValueError):
        plan_episode(record, schema)


def test_vfr_held_frames_gaps_and_explicit_coarse_grid(training_case):
    record, schema = training_case
    record["episode"].update(
        time_base="1/20",
        timestamps_s=[0, 0.05, 0.1, 0.3],
        original_pts=[200, 201, 202, 206],
        original_pts_s=[10, 10.05, 10.1, 10.3],
    )
    plan = plan_episode(record, schema)
    assert plan["video_indices"].tolist() == [0, 2, 2, 3]
    assert plan["alignment"]["omitted_source_frames"] == 1
    assert plan["alignment"]["repeated_source_frames"] == 1
    assert plan["alignment"]["video_age_s"][2] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="video: no causal sample"):
        plan_episode(record, schema.model_copy(update={"max_video_age_s": 0.09}))


def test_grid_tail_and_no_floating_point_future_leakage(training_case):
    record, schema = training_case
    record["episode"]["duration_s"] = 0.35
    plan = plan_episode(record, schema)
    assert len(plan["grid"]) == 4
    assert plan["alignment"]["duration_rounding_s"] == pytest.approx(0.05)
    indices, _ = previous_samples([0, 0.100001], np.array([0.1]), 0.11, "action")
    assert indices.tolist() == [0]


def test_integer_pts_prevents_ffprobe_rounding_frame_loss(training_case):
    record, schema = training_case
    record["episode"].update(
        duration_s=1 / 3,
        time_base="1/12288",
        original_pts=[0, 1024, 2048, 3072],
        original_pts_s=[0, 0.083333, 0.166667, 0.25],
        timestamps_s=[0, 0.083333, 0.166667, 0.25],
    )
    for stream in record["episode"]["streams"]:
        stream["tolerance_s"] = 1
    plan = plan_episode(record, schema.model_copy(update={"fps": 12}))
    assert plan["video_indices"].tolist() == [0, 1, 2, 3]
    assert plan["alignment"]["video_age_s"] == [0, 0, 0, 0]
    assert plan["alignment"]["omitted_source_frames"] == 0


@pytest.mark.parametrize(
    "change",
    [{"fps": 0}, {"fps": 10.5}, {"width": 65}, {"max_video_age_s": float("nan")}, {"state": []}],
)
def test_invalid_training_schema(training_case, change):
    _, schema = training_case
    with pytest.raises(ValueError):
        TrainingSchema.model_validate({**schema.model_dump(), **change})


@pytest.fixture
def semantic_case(training_case):
    record, schema = training_case
    record["split"] = "test"
    reviewed = []
    for start, end, action, text, obj in [
        (0, 2, "reach", "Reach toward cup", "cup"),
        (2, 4, "hold", "Hold position; contact unknown", None),
    ]:
        reviewed.append(
            dict(
                region_id=f"segment-{start}",
                source_start=start,
                source_end_exclusive=end,
                start_s=start / 10,
                end_s=end / 10,
                normalized_action=action,
                subtask_text=text,
                object_text=obj,
            )
        )
    record["annotation"]["segments"] = [
        dict(start_s=s["start_s"], end_s=s["end_s"], label=s["normalized_action"]) for s in reviewed
    ]
    record["annotation"]["native"] = {
        "semantic_review": dict(
            schema_version=1,
            kind="model_assisted_review",
            prediction_exposure=True,
            complete=True,
            split="test",
            segments=reviewed,
        )
    }
    return record, schema


def test_semantic_text_follows_source_frames_and_object_absence(semantic_case):
    record, schema = semantic_case
    plan = plan_episode(record, schema)
    assert plan["semantic"]["annotation.subtask_text"] == [
        "Reach toward cup",
        "Reach toward cup",
        "Hold position; contact unknown",
        "Hold position; contact unknown",
    ]
    assert plan["semantic"]["annotation.object_text"] == ["cup", "cup", "", ""]
    assert plan["semantic"]["annotation.object_present"].tolist() == [
        [True],
        [True],
        [False],
        [False],
    ]
    assert plan["labels"].tolist() == [LABELS.index(x) for x in ["reach", "reach", "hold", "hold"]]


def test_repeated_video_frame_keeps_its_semantic_text(semantic_case):
    record, schema = semantic_case
    record["episode"].update(
        time_base="1/20",
        original_pts=[200, 201, 202, 206],
        original_pts_s=[10, 10.05, 10.1, 10.3],
        timestamps_s=[0, 0.05, 0.1, 0.3],
    )
    segments = record["annotation"]["native"]["semantic_review"]["segments"]
    segments[0]["end_s"] = segments[1]["start_s"] = 0.1
    record["annotation"]["segments"][0]["end_s"] = record["annotation"]["segments"][1][
        "start_s"
    ] = 0.1
    plan = plan_episode(record, schema)
    assert plan["video_indices"].tolist() == [0, 2, 2, 3]
    assert plan["semantic"]["annotation.subtask_text"] == [
        "Reach toward cup",
        *["Hold position; contact unknown"] * 3,
    ]
    assert plan["labels"].tolist() == [LABELS.index("reach"), *[LABELS.index("hold")] * 3]


@pytest.mark.parametrize(
    "problem", ["gap", "overlap", "tail", "time", "action", "blank", "incomplete", "split"]
)
def test_inconsistent_semantic_training_labels_fail(semantic_case, problem):
    record, schema = semantic_case
    review = record["annotation"]["native"]["semantic_review"]
    segment = review["segments"][1]
    if problem == "gap":
        segment["source_start"] = 3
    elif problem == "overlap":
        segment["source_start"] = 1
    elif problem == "tail":
        segment["source_end_exclusive"] = 3
    elif problem == "time":
        segment["start_s"] = 0.21
    elif problem == "action":
        segment["normalized_action"] = "reach"
    elif problem == "blank":
        segment["subtask_text"] = " "
    elif problem == "incomplete":
        review["complete"] = False
    else:
        record["split"] = "train"
    with pytest.raises(ValueError, match="semantic"):
        plan_episode(record, schema)
