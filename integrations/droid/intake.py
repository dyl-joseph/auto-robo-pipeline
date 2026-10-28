"""Convert a bounded set of local DROID TFDS episodes into a resumable batch inbox."""

import argparse
import fcntl
import json
import shutil
from functools import partial
from pathlib import Path

from integrations.droid.convert import bounded_options, write_episode
from robo.core import atomic_json, digest, file_hash
from robo.droid import CAMERAS, LocalOrigin

METADATA = ("dataset_info.json", "features.json")
DERIVATIVES = {"source.json", "video.mp4", "training-schema.json", "rlds-episode.json"}
MIB = 1024 * 1024


def inventory(source, split, allow_partial):
    import tensorflow_datasets as tfds

    for name in METADATA:
        path = source / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MIB:
            raise ValueError(f"Requires local {name}, at most 1 MiB, without symlinks")
    hashes = {name: file_hash(source / name) for name in METADATA}
    builder = tfds.builder_from_directory(str(source))
    if builder.info.file_format.value != "tfrecord" or split not in builder.info.splits:
        raise ValueError("Requires a declared split in a TFRecord TFDS directory")
    if not isinstance(builder.info.features["steps"], tfds.features.Dataset):
        raise ValueError("Requires nested RLDS steps")
    available, missing = [], []
    instructions = builder.info.splits[split].file_instructions
    for item in instructions:
        path = Path(item.filename)
        if path.parent != source or path.is_symlink():
            raise ValueError(
                "Shard paths must be regular files directly inside the source directory"
            )
        if not path.exists():
            missing.append(path.name)
            continue
        if not path.is_file() or item.skip != 0:
            raise ValueError("Requires complete local TFRecord shards")
        available.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "declared_episodes": item.examples_in_shard,
            }
        )
    if missing and not allow_partial:
        raise ValueError(f"Missing {len(missing)} declared shards; use --allow-partial explicitly")
    if not available:
        raise ValueError("No declared shards are available locally")
    return builder, {
        "metadata": hashes,
        "available": available,
        "missing": missing,
        "declared_episodes": builder.info.splits[split].num_examples,
    }


def verify_snapshot(source, snapshot, shard=None, sha256=None):
    for name, expected in snapshot["metadata"].items():
        if file_hash(source / name) != expected:
            raise ValueError(f"Input metadata changed: {name}")
    if shard is not None and file_hash(source / shard) != sha256:
        raise ValueError(f"Input shard changed: {shard}")


def verify_derivative(directory, origin, camera, expected_sha256=None):
    if directory.name != digest([origin.model_dump(), camera])[:32] or directory.is_symlink():
        raise ValueError("Published episode directory differs from its identity")
    evidence_path = directory / "conversion.json"
    if expected_sha256 and file_hash(evidence_path) != expected_sha256:
        raise ValueError(f"Published evidence changed: {directory.name}")
    evidence = json.loads(evidence_path.read_text())
    if (
        evidence.get("origin") != origin.model_dump()
        or evidence.get("selected_camera") != camera
        or set(evidence["files"]) != DERIVATIVES
    ):
        raise ValueError(f"Published episode origin or file set differs: {directory.name}")
    for name, sha256 in evidence["files"].items():
        path = directory / name
        if path.is_symlink() or file_hash(path) != sha256:
            raise ValueError(f"Published derivative changed: {directory.name}/{name}")
    return {
        "directory": directory.name,
        "conversion_sha256": file_hash(evidence_path),
        "steps": evidence["steps"],
    }


def intake(
    source,
    output,
    *,
    release,
    split="train",
    camera=CAMERAS[0],
    max_episodes=10,
    max_steps=4500,
    max_shard_bytes=128 * MIB,
    max_selected_bytes=512 * MIB,
    allow_partial=False,
    generated_fixture=False,
):
    import tensorflow as tf

    source, output = Path(source).resolve(), Path(output).resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Keep output separate from the source directory")
    if not 1 <= max_episodes <= 200 or not 2 <= max_steps <= 4500:
        raise ValueError("Use 1–200 episodes and 2–4500 steps per episode")
    if max_shard_bytes <= 0 or max_selected_bytes < max_shard_bytes:
        raise ValueError("Selected-byte budget must be at least the positive shard-byte bound")
    if camera not in CAMERAS or not release.strip():
        raise ValueError("Requires a DROID camera and an explicit release name")
    builder, snapshot = inventory(source, split, allow_partial)
    contract = {
        "schema_version": 1,
        "release": release,
        "split": split,
        "camera": camera,
        "max_steps": max_steps,
        "generated_fixture": generated_fixture,
        "inventory": snapshot,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".intake.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another intake is using this output") from exc
        checkpoint = output / "intake.json"
        if checkpoint.exists():
            report = json.loads(checkpoint.read_text())
            if report["contract"] != contract:
                raise ValueError("Input inventory or conversion settings changed; use a new output")
        else:
            if set(p.name for p in output.iterdir()) != {".intake.lock"}:
                raise ValueError("Output is not an intake directory; choose an unused output")
            report = {"contract": contract, "shards": {}, "status": "running"}
            atomic_json(checkpoint, report)
        episodes = output / "episodes"
        episodes.mkdir(exist_ok=True)
        staging = output / ".staging"
        # Only this locked job owns staging; interrupted, unpublished work can be discarded.
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        report.update(status="running", error=None, requested_max_episodes=max_episodes)
        converted = sum(len(s["records"]) for s in report["shards"].values())
        if max_episodes < converted:
            raise ValueError("Episode limit is below the already published count")
        selected_bytes = 0
        limited = False
        current_shard, record_index = None, None
        try:
            for entry in snapshot["available"]:
                current_shard, record_index = entry["name"], None
                prior = report["shards"].get(current_shard)
                if converted >= max_episodes and prior is None:
                    limited = True
                    break
                if entry["bytes"] > max_shard_bytes:
                    raise ValueError(f"Shard exceeds --max-shard-bytes: {current_shard}")
                if selected_bytes + entry["bytes"] > max_selected_bytes:
                    raise ValueError(
                        "Selected shards exceed --max-selected-bytes; raise explicitly"
                    )
                selected_bytes += entry["bytes"]
                path = source / current_shard
                if path.stat().st_size != entry["bytes"]:
                    raise ValueError(f"Input shard size changed: {current_shard}")
                sha256 = file_hash(path)
                if prior and prior["sha256"] != sha256:
                    raise ValueError(f"Input shard changed: {current_shard}")
                shard = report["shards"].setdefault(
                    current_shard, {"sha256": sha256, "complete": False, "records": []}
                )

                def origin_at(index, shard_name=current_shard, shard_sha=sha256):
                    return LocalOrigin(
                        release=release,
                        shard=shard_name,
                        shard_sha256=shard_sha,
                        features_sha256=snapshot["metadata"]["features.json"],
                        dataset_info_sha256=snapshot["metadata"]["dataset_info.json"],
                        record_index=index,
                        generated_fixture=generated_fixture,
                    )

                for index, saved in enumerate(shard["records"]):
                    verify_derivative(
                        episodes / saved["directory"],
                        origin_at(index),
                        camera,
                        saved["conversion_sha256"],
                    )
                if shard["complete"]:
                    verify_snapshot(source, snapshot, current_shard, sha256)
                    continue
                dataset = tf.data.TFRecordDataset(
                    [str(path)], num_parallel_reads=1, buffer_size=1024 * 1024
                ).with_options(bounded_options())
                for record_index, record in enumerate(dataset):
                    if record_index < len(shard["records"]):
                        continue
                    if converted >= max_episodes:
                        limited = True
                        break
                    origin = origin_at(record_index)
                    directory = episodes / digest([origin.model_dump(), camera])[:32]
                    if not directory.exists():
                        episode = builder.info.features.deserialize_example(record)
                        write_episode(
                            episode,
                            directory,
                            camera,
                            origin=origin,
                            max_steps=max_steps,
                            staging_parent=staging,
                            verify=partial(
                                verify_snapshot, source, snapshot, current_shard, sha256
                            ),
                        )
                    else:
                        # Publication may have succeeded just before an interrupted checkpoint.
                        verify_snapshot(source, snapshot, current_shard, sha256)
                    shard["records"].append(verify_derivative(directory, origin, camera))
                    converted += 1
                    atomic_json(checkpoint, report)
                else:
                    if len(shard["records"]) != entry["declared_episodes"]:
                        raise ValueError(
                            f"Record count differs from dataset_info.json: {current_shard}"
                        )
                    shard["complete"] = True
                verify_snapshot(source, snapshot, current_shard, sha256)
                atomic_json(checkpoint, report)
                if limited:
                    break
            verify_snapshot(source, snapshot)
            report["status"] = (
                "episode_limit"
                if limited
                else ("partial_source" if snapshot["missing"] else "complete")
            )
        except Exception as exc:
            report.update(
                status="failed",
                error={
                    "shard": current_shard,
                    "record": record_index,
                    "type": type(exc).__name__,
                    "message": str(exc)[:500],
                },
            )
            raise
        finally:
            report["summary"] = {
                "published_episodes": sum(len(s["records"]) for s in report["shards"].values()),
                "published_steps": sum(
                    r["steps"] for s in report["shards"].values() for r in s["records"]
                ),
                "declared_shards": len(snapshot["available"]) + len(snapshot["missing"]),
                "available_shards": len(snapshot["available"]),
                "completed_shards": sum(s["complete"] for s in report["shards"].values()),
                "selected_shard_bytes": selected_bytes,
                "missing_shards": len(snapshot["missing"]),
                "declared_episodes": snapshot["declared_episodes"],
                "available_declared_episodes": sum(
                    s["declared_episodes"] for s in snapshot["available"]
                ),
                "human_labels_validated": False,
            }
            atomic_json(checkpoint, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--release", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--camera", choices=CAMERAS, default=CAMERAS[0])
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=4500)
    parser.add_argument("--max-shard-bytes", type=int, default=128 * MIB)
    parser.add_argument("--max-selected-bytes", type=int, default=512 * MIB)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--generated-fixture", action="store_true")
    args = vars(parser.parse_args())
    result = intake(**args)
    print(json.dumps({"status": result["status"], **result["summary"]}))


if __name__ == "__main__":
    main()
