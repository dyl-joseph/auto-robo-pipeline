"""Write accepted revisions and their provenance as JSONL and Parquet."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from robo import __version__
from robo.analysis import latest_run
from robo.core import atomic_json, canonical, digest, file_hash, validate_segments
from robo.perception import active_perception
from robo.store import Store


def export_dataset(
    store: Store, selection: list[dict], output: Path, *, native_validator=None
) -> dict:
    # Freeze review eligibility while collecting a consistent export snapshot.
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        return _export_dataset(store, selection, output, native_validator)


def _export_dataset(
    store: Store, selection: list[dict], output: Path, native_validator=None
) -> dict:
    if not selection:
        raise ValueError("select at least one accepted revision")
    if len({row["episode_id"] for row in selection}) != len(selection):
        raise ValueError("duplicate selected episode")
    episodes = [store.episode(item["episode_id"]) for item in selection]
    # Include unselected bridge episodes when forming leakage groups.
    population = store.episodes()
    groups = {e["id"]: {e["id"]} for e in population}
    for a in population:
        for b in population:
            if a["session"] == b["session"] or a["video_sha256"] == b["video_sha256"]:
                merged = groups[a["id"]] | groups[b["id"]]
                for member in merged:
                    groups[member] = merged
    group_splits = {}
    records = []
    for item, episode in zip(selection, episodes, strict=False):
        if item.get("split") not in {"train", "development", "test"}:
            raise ValueError("each selection needs train, development or test split")
        group = digest(sorted(groups[episode["id"]]))[:16]
        if group in group_splits and group_splits[group] != item["split"]:
            raise ValueError("split leakage: shared session or duplicate media crosses splits")
        group_splits[group] = item["split"]
        revision = store.latest(episode["id"])
        if revision["revision"] != item["revision"] or revision["status"] != "accepted":
            raise ValueError("export requires the latest explicitly accepted revision")
        if store.native_binding(episode["id"]) or revision.get("native"):
            if not revision.get("native") or native_validator is None:
                raise ValueError("native export requires an authenticated freshness check")
            native_validator(revision)
        video = store.root / episode["media"]
        if file_hash(video) != episode["video_sha256"]:
            raise ValueError("source hash mismatch")
        if (video.parent / "episode.json").read_bytes() != canonical(episode):
            raise ValueError("source manifest hash mismatch")
        validate_segments(revision["segments"], episode["duration_s"])
        run = (
            latest_run(store, episode["id"], revision["suggestion_run"])
            if revision["suggestion_run"]
            else None
        )
        records.append(
            {
                "schema_version": 1,
                "episode_id": episode["id"],
                "split": item["split"],
                "split_group": group,
                "episode": episode,
                "annotation": revision,
                "audit_trail": store.history(episode["id"]),
                "suggestion_artifact_hash": run["artifact_hash"] if run else None,
                "suggestion_config": run["config"] if run else None,
                "perception": (
                    active_perception(store, episode["id"], revision["perception_run"])
                    if revision.get("perception_run")
                    else None
                ),
            }
        )
    records.sort(key=lambda r: r["episode_id"])
    rows = b"".join(canonical(row) for row in records)
    manifest = {
        "schema_version": 1,
        "software_version": __version__,
        "records_hash": digest(records),
        "selection": sorted(selection, key=lambda r: r["episode_id"]),
        "episode_count": len(records),
        "group_count": len(group_splits),
        "policy": "latest accepted revision; grouped by session and exact media duplicates",
        "media_policy": "media referenced by SHA256, not bundled; resolve in local source store",
    }
    version = digest(manifest)[:24]
    manifest["version"] = version
    if output.exists():
        existing = json.loads((output / "manifest.json").read_text())
        if {k: v for k, v in existing.items() if k != "files"} != manifest:
            raise ValueError("output exists with a different dataset version")
        validate_export(output)
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=".export-"))
    try:
        (temporary / "episodes.jsonl").write_bytes(rows)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "episode_id": r["episode_id"],
                        "split": r["split"],
                        "record_json": canonical(r).decode(),
                    }
                    for r in records
                ]
            ),
            temporary / "episodes.parquet",
        )
        manifest["files"] = {
            name: file_hash(temporary / name) for name in ("episodes.jsonl", "episodes.parquet")
        }
        # File hashes are derived, not part of the semantic version.
        atomic_json(temporary / "manifest.json", manifest)
        validate_export(temporary)
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def validate_export(path: Path) -> list[dict]:
    manifest = json.loads((path / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if name not in {"episodes.jsonl", "episodes.parquet"} or file_hash(path / name) != expected:
            raise ValueError("export file hash mismatch")
    records = [json.loads(line) for line in (path / "episodes.jsonl").read_text().splitlines()]
    if digest(records) != manifest["records_hash"]:
        raise ValueError("semantic record hash mismatch")
    table = pq.read_table(path / "episodes.parquet").to_pylist()
    if [json.loads(row["record_json"]) for row in table] != records:
        raise ValueError("Parquet differs from JSONL")
    semantic = {k: v for k, v in manifest.items() if k not in {"files", "version"}}
    if digest(semantic)[:24] != manifest["version"]:
        raise ValueError("dataset version mismatch")
    return records
