"""Validate the pinned DROID sample and its explicit feature mapping."""

import numpy as np
from pydantic import Field

from robo.core import Contract, Source, digest

RELEASE = "droid_100/1.0.0"
PREFIX = "robotics/droid_100/1.0.0/"
SHARD = "r2d2_faceblur-train.tfrecord-00009-of-00031"
DOCS = "https://droid-dataset.github.io/droid/the-droid-dataset.html"
PAPER = "https://arxiv.org/html/2403.12945v2"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
ATTRIBUTION = "Khazatsky et al., DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset (2024)"
CAMERAS = ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")
FPS = 15
CLOCK = (
    "Inferred RLDS step index / nominal 15 Hz (DROID paper, Appendix B). "
    "Images and numeric values share a step; no measured capture timestamps or clock offsets."
)
# Generation-pinned official objects, selected from all 31 shards on 2026-10-28.
FILES = {
    "dataset_info.json": {
        "bytes": 760,
        "generation": "1710527613369658",
        "sha256": "58646aa75fd054a0d8670edd676620267adb32550dc228545d70a3418abb1834",
        "md5": "24e71319dc1e3f0fa2ca678b86ba9abe",
    },
    "features.json": {
        "bytes": 18665,
        "generation": "1710517501027287",
        "sha256": "4c556e2e38e179e41089e07cf54273b1540d67d3c78a420898370329813c9a25",
        "md5": "0146b25aed881d4fc6332eb56f2690a6",
    },
    SHARD: {
        "bytes": 9541347,
        "generation": "1710528094394407",
        "sha256": "645d500e23e7359c21121b12fe1a1395157ba421b1b326eee7df829eb608a03d",
        "md5": "f7a5b4a2b498a101adbef69f8f9a0bb7",
    },
}


class LocalOrigin(Contract):
    release: str = Field(min_length=1, max_length=200)
    shard: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    shard_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    features_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_info_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    record_index: int = Field(ge=0)
    generated_fixture: bool = False


def source_from_steps(
    steps: list[dict], metadata: dict, camera: str, origin: LocalOrigin | None = None
) -> dict:
    """Map explicitly named position fields; never guess generic action semantics."""
    if camera not in CAMERAS:
        raise ValueError("unknown DROID camera")
    if not 2 <= len(steps) <= FPS * 300:
        raise ValueError("DROID episode requires 2–4500 steps")
    for key, expected in (("is_first", 0), ("is_last", len(steps) - 1)):
        if [i for i, s in enumerate(steps) if s[key]] != [expected]:
            raise ValueError(f"invalid RLDS {key} boundaries")
    instructions = {s["language_instruction"] for s in steps}
    if len(instructions) != 1 or not next(iter(instructions)).strip():
        raise ValueError("requires one nonempty primary instruction per episode")
    streams = []
    for group, modality in (("observation", "state"), ("action_dict", "action")):
        for field, width, units in (
            ("joint_position", 7, "rad"),
            ("gripper_position", 1, "normalized"),
        ):
            values = np.asarray([s[group][field] for s in steps], dtype=np.float64)
            if values.shape != (len(steps), width) or not np.isfinite(values).all():
                raise ValueError(f"invalid DROID {group}.{field}")
            if field == "gripper_position" and ((values < 0).any() or (values > 1).any()):
                raise ValueError("DROID gripper position outside normalized [0,1]")
            streams.append(
                {
                    "name": f"{modality}_{field}",
                    "modality": modality,
                    "units": units,
                    "timestamps_s": [i / FPS for i in range(len(steps))],
                    "values": values.tolist(),
                    "clock_evidence": CLOCK,
                    "tolerance_s": 0.5 / FPS,
                }
            )
    # Group the original institution / outcome / day conservatively for split isolation.
    # Source paths are opaque provenance: never read a source filesystem path.
    original = metadata["file_path"]
    if not isinstance(original, str) or not original.endswith("/trajectory.h5"):
        raise ValueError("unexpected DROID original path metadata")
    session = original.rsplit("/", 2)[0]
    source = {
        "source_id": f"{SHARD}:record-0:{camera}",
        "release": RELEASE,
        "task": next(iter(instructions)),
        "session": "droid-day-" + digest(session)[:24],
        "video": "video.mp4",
        "video_timing": "inferred_nominal_fps",
        "license": "CC-BY-4.0",
        "license_url": LICENSE,
        "data_kind": "real",
        "provenance": (
            f"{ATTRIBUTION}; {DOCS}; gs://gresearch/{PREFIX}{SHARD}; "
            f"generation {FILES[SHARD]['generation']}; SHA256 {FILES[SHARD]['sha256']}; "
            f"record 0, {len(steps)} steps, selected {camera}. "
            "RGB decoded by official TFDS then H.264 CRF18/yuv420p: lossy derivative. "
            "All numeric fields, instructions, metadata and per-camera RGB hashes retained "
            "in the local rlds-episode.json. Generic action is not used for training. " + CLOCK
        ),
        "streams": streams,
    }
    if origin is not None:
        source.update(
            source_id="local-rlds-" + digest([origin.model_dump(), camera])[:40],
            release=origin.release,
            data_kind="synthetic" if origin.generated_fixture else "real",
            provenance=(
                (
                    "Generated DROID-schema fixture. "
                    if origin.generated_fixture
                    else f"{ATTRIBUTION}; {DOCS}. "
                    "User-supplied local DROID RLDS; origin not authenticated. "
                )
                + f"Shard {origin.shard}; SHA256 {origin.shard_sha256}; "
                f"features.json SHA256 {origin.features_sha256}; "
                f"dataset_info.json SHA256 {origin.dataset_info_sha256}; "
                f"record {origin.record_index}, {len(steps)} steps, selected {camera}. "
                "RGB decoded by official TFDS then H.264 CRF18/yuv420p: lossy derivative. "
                "All non-image fields and per-camera RGB hashes retained in rlds-episode.json. "
                "Generic action is not used for training. " + CLOCK
            ),
        )
    return Source.model_validate(source).model_dump()


def training_schema() -> dict:
    def components(modality):
        return [
            {
                "stream": f"{modality}_joint_position",
                "index": i,
                "name": f"panda_joint_{i + 1}_position",
                "units": "rad",
                "meaning": (
                    f"{'Observed' if modality == 'state' else 'Commanded absolute'} "
                    f"Franka Panda joint {i + 1} position, radians; DROID named field order"
                ),
            }
            for i in range(7)
        ] + [
            {
                "stream": f"{modality}_gripper_position",
                "index": 0,
                "name": "gripper_closed_fraction",
                "units": "normalized",
                "meaning": (
                    f"{'Observed' if modality == 'state' else 'Commanded'} "
                    "gripper closure fraction: 0 open, 1 closed; DROID robot.py convention"
                ),
            }
        ]

    return {
        "schema_version": 1,
        "robot_type": "droid-franka-panda-robotiq-2f85",
        "fps": FPS,
        "width": 320,
        "height": 180,
        "max_video_age_s": 0.5 / FPS,
        "state": components("state"),
        "action": components("action"),
    }
