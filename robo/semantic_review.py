"""Prepare immutable semantic predictions for correction in native Label Studio."""

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from pydantic import Field

from robo.core import LABELS, Contract, atomic_json, canonical, digest, file_hash
from robo.semantic import EvaluationPlan, Predictions, validate_episode_segments
from robo.semantic_baseline import source_episode

TEXT_CONTROLS = {"subtask_text", "object_text"}


class ReviewedSegment(Contract):
    region_id: str = Field(min_length=1, max_length=200)
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    source_start: int = Field(ge=0, strict=True)
    source_end_exclusive: int = Field(gt=0, strict=True)
    normalized_action: str
    subtask_text: str = Field(min_length=1, max_length=2000)
    object_text: str | None = Field(max_length=2000)


def read_inputs(plan_bytes, prediction_bytes):
    plan = EvaluationPlan.model_validate_json(plan_bytes)
    predictions = Predictions.model_validate_json(prediction_bytes)
    if predictions.plan_sha256 != plan.plan_sha256:
        raise ValueError("semantic predictions do not match the frozen plan")
    population = {row.episode_id: row for row in plan.episodes}
    seen = set()
    for row in predictions.episodes:
        if row.episode_id not in population or row.episode_id in seen:
            raise ValueError("unknown or duplicate prediction episode")
        seen.add(row.episode_id)
        validate_episode_segments(row.segments, population[row.episode_id], plan.normalized_actions)
        if any(
            s.provenance.kind not in {"baseline", "model", "synthetic_fixture"}
            for s in row.segments
        ):
            raise ValueError("review intake requires uncorrected machine predictions")
    return plan, predictions


def snapshot(store, plan_path, prediction_path):
    from robo.pipeline import pipeline_lock

    paths = {"plan.json": plan_path, "predictions.json": prediction_path}
    payloads = {}
    for name, path in paths.items():
        with Path(path).open("rb") as stream:
            payloads[name] = stream.read(32 * 1024 * 1024 + 1)
        if len(payloads[name]) > 32 * 1024 * 1024:
            raise ValueError("split semantic inputs into files of at most 32 MiB")
    read_inputs(payloads["plan.json"], payloads["predictions.json"])
    import hashlib

    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    identity = digest(hashes)
    parent = store.root / "semantic_sources"
    directory = parent / identity
    with pipeline_lock(store):
        if parent.is_symlink():
            raise ValueError("semantic snapshot parent must not be a symlink")
        parent.mkdir(exist_ok=True)
        if not directory.exists():
            temporary = Path(tempfile.mkdtemp(prefix=".semantic-", dir=parent))
            try:
                for name, data in payloads.items():
                    (temporary / name).write_bytes(data)
                atomic_json(temporary / "files.json", hashes)
                os.rename(temporary, directory)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        load_snapshot(store, identity)
    return identity


def load_snapshot(store, identity):
    if not re.fullmatch(r"[a-f0-9]{64}", identity):
        raise ValueError("invalid semantic snapshot identity")
    parent = store.root / "semantic_sources"
    directory = parent / identity
    paths = [directory / name for name in ("files.json", "plan.json", "predictions.json")]
    if parent.is_symlink() or directory.is_symlink() or any(p.is_symlink() for p in paths):
        raise ValueError("semantic snapshot paths must not be symlinks")
    hashes = json.loads(paths[0].read_text())
    if set(hashes) != {"plan.json", "predictions.json"} or digest(hashes) != identity:
        raise ValueError("semantic snapshot identity changed")
    if any(file_hash(directory / name) != sha for name, sha in hashes.items()):
        raise ValueError("immutable semantic input snapshot changed")
    return (*read_inputs(paths[1].read_bytes(), paths[2].read_bytes()), hashes)


def bound_prediction(store, binding):
    origin = binding["semantic_prediction"]
    plan, predictions, hashes = load_snapshot(store, origin["snapshot_id"])
    if hashes != origin["files"] or plan.plan_sha256 != origin["plan_sha256"]:
        raise ValueError("semantic prediction binding differs from its snapshot")
    episode_id = binding["episode_id"]
    specs = [s for s in plan.episodes if s.episode_id == episode_id]
    rows = [r for r in predictions.episodes if r.episode_id == episode_id]
    if len(specs) != 1 or len(rows) != 1:
        raise ValueError("semantic snapshot lacks the bound episode")
    source_episode(store, specs[0])
    return specs[0], rows[0], predictions


def exact_grid(spec, proxy):
    if len(proxy["frames"]) != len(spec.frame_boundaries_s) - 1 or any(
        frame["source_frame"] != index
        or abs(frame["review_time_s"] - spec.frame_boundaries_s[index]) > 1e-6
        for index, frame in enumerate(proxy["frames"])
    ):
        raise ValueError("semantic review requires a one-to-one source/review frame grid")


def prediction_results(row):
    results = []
    for index, segment in enumerate(row.segments):
        if segment.normalized_action not in LABELS:
            raise ValueError("semantic action is not supported by the native review taxonomy")
        region_id = f"segment-{index}"
        ranges = [
            {"start": segment.source_frames.start + 1, "end": segment.source_frames.end_exclusive}
        ]
        fields = [("subtask", "timelinelabels", {"timelinelabels": [segment.normalized_action]})]
        fields.append(("subtask_text", "textarea", {"text": [segment.subtask_text]}))
        if segment.object_text is not None:
            fields.append(("object_text", "textarea", {"text": [segment.object_text]}))
        for name, kind, value in fields:
            results.append(
                {
                    "id": region_id,
                    "from_name": name,
                    "to_name": "video",
                    "type": kind,
                    "interactive_mode": False,
                    "value": {"ranges": ranges, **value},
                }
            )
    return results


def prepare(store, plan_path, prediction_path, episode_id, output):
    from robo.labelstudio import label_config, task_bundle

    identity = snapshot(store, plan_path, prediction_path)
    plan, predictions, hashes = load_snapshot(store, identity)
    binding = {
        "episode_id": episode_id,
        "semantic_prediction": {
            "snapshot_id": identity,
            "files": hashes,
            "plan_sha256": plan.plan_sha256,
        },
    }
    spec, row, predictions = bound_prediction(store, binding)
    bundle = task_bundle(store, episode_id)
    proxy = bundle["proxy"]
    exact_grid(spec, proxy)
    results = prediction_results(row)
    bundle["task"]["predictions"] = (
        [
            {
                "model_version": f"semantic-{identity}",
                "result": results,
                "score": min(
                    1 - (s.uncertainty if s.uncertainty is not None else 1) for s in row.segments
                ),
            }
        ]
        if results
        else []
    )
    bundle["task"]["data"]["provenance"] += (
        f" · {predictions.producer} · uncorrected semantic suggestions; review required"
        + (f" · prediction failed: {row.error}" if row.error else "")
    )
    bundle["label_config"] = label_config(
        proxy["columns"], proxy["fps"], proxy["column_units"], semantic=True
    )
    bundle["trusted_origin"].update(binding)
    bundle["trusted_origin"]["suggestion_run"] = None
    bundle["trusted_origin"]["prediction_projection"] = {
        "input_segment_count": len(row.segments),
        "display_segment_count": len(row.segments),
        "omitted_collapsed_count": 0,
        "max_boundary_displacement_bound_s": 0,
    }
    output = Path(output)
    documents = {
        "bundle.json": bundle,
        "tasks.json": [bundle["task"]],
        "mapping.json": proxy,
        "provenance.json": bundle["trusted_origin"],
    }
    payloads = {name: canonical(value) for name, value in documents.items()}
    payloads["review.xml"] = bundle["label_config"].encode()
    if output.is_symlink():
        raise ValueError("semantic review output must not be a symlink")
    if output.exists():
        if set(p.name for p in output.iterdir()) != set(payloads) or any(
            (output / name).is_symlink() or (output / name).read_bytes() != data
            for name, data in payloads.items()
        ):
            raise ValueError("review output already contains a different semantic preparation")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".semantic-review-", dir=output.parent))
        try:
            for name, data in payloads.items():
                (temporary / name).write_bytes(data)
            os.rename(temporary, output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return bundle


def decoded_review(results, spec, proxy):
    exact_grid(spec, proxy)
    timelines, texts = {}, {}
    for result in results:
        name = result.get("from_name")
        if name not in {"subtask", *TEXT_CONTROLS}:
            continue
        region_id = result.get("id")
        if not isinstance(region_id, str) or not region_id or len(region_id) > 200:
            raise ValueError("semantic controls require a region identity")
        expected_type = "timelinelabels" if name == "subtask" else "textarea"
        if result.get("type") != expected_type or result.get("to_name") != "video":
            raise ValueError("invalid semantic review control")
        if name == "subtask":
            if region_id in timelines or len(result["value"]["ranges"]) != 1:
                raise ValueError("one unique timeline range per semantic region is required")
            timelines[region_id] = result["value"]
        else:
            key = (region_id, name)
            values = result.get("value", {}).get("text")
            if key in texts or not isinstance(values, list) or len(values) != 1:
                raise ValueError("one text submission per semantic region/control is required")
            if not isinstance(values[0], str) or not values[0].strip() or len(values[0]) > 2000:
                raise ValueError("semantic text must contain 1–2000 nonblank characters")
            texts[key] = result["value"]
    if any(region not in timelines for region, _ in texts):
        raise ValueError("semantic text has no matching timeline region")
    segments = []
    for region_id, value in timelines.items():
        span = value["ranges"][0]
        start, end = span["start"], span["end"]
        if (
            type(start) is not int
            or type(end) is not int
            or not 1 <= start <= end < len(spec.frame_boundaries_s)
        ):
            raise ValueError("invalid semantic source frame range")
        labels = value["timelinelabels"]
        if len(labels) != 1 or labels[0] not in LABELS:
            raise ValueError("semantic region needs one supported action")
        if (region_id, "subtask_text") not in texts:
            raise ValueError("every semantic region needs subtask text")
        content = {}
        for name in TEXT_CONTROLS:
            text = texts.get((region_id, name))
            if text and "ranges" in text and text["ranges"] != value["ranges"]:
                raise ValueError("semantic text and timeline ranges disagree")
            content[name] = text["text"][0].strip() if text else None
        segments.append(
            ReviewedSegment(
                region_id=region_id,
                start_s=spec.frame_boundaries_s[start - 1],
                end_s=spec.frame_boundaries_s[end],
                source_start=start - 1,
                source_end_exclusive=end,
                normalized_action=labels[0],
                **content,
            ).model_dump()
        )
    segments.sort(key=lambda s: s["source_start"])
    end, complete = 0, bool(segments)
    for segment in segments:
        if segment["source_start"] < end:
            raise ValueError("semantic review regions overlap")
        complete = complete and segment["source_start"] == end
        end = segment["source_end_exclusive"]
    return {
        "schema_version": 1,
        "kind": "model_assisted_review",
        "prediction_exposure": True,
        "segments": segments,
        "complete": complete and end == len(spec.frame_boundaries_s) - 1,
    }


def main():
    from robo.store import Store

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "predictions", "store", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--episode", required=True)
    args = parser.parse_args()
    bundle = prepare(Store(args.store), args.plan, args.predictions, args.episode, args.output)
    print(
        json.dumps(
            {
                "episode": args.episode,
                "bundle": str(args.output / "bundle.json"),
                "snapshot_id": bundle["trusted_origin"]["semantic_prediction"]["snapshot_id"],
                "automatic_approvals": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
