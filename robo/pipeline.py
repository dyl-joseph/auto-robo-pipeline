"""Run a folder of episode manifests through import, suggestions and native review preparation."""

import fcntl
import json
import subprocess
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from robo.analysis import queue, suggest
from robo.core import atomic_json, digest, file_hash
from robo.ingest import import_episode
from robo.labelstudio import task_bundle
from robo.recovery import data_findings, normalize_source
from robo.store import Store


@contextmanager
def pipeline_lock(store):
    with (store.root / ".pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "Another batch operation is using this store; retry when it finishes"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def discover(inbox):
    inbox = Path(inbox).resolve()
    if not inbox.is_dir():
        raise ValueError("Batch inbox must be a directory")
    manifests = sorted({*inbox.rglob("source.json"), *inbox.rglob("*.source.json")})
    if not manifests:
        raise ValueError("No source.json or *.source.json manifests found in the inbox")
    if len(manifests) > 200:
        raise ValueError("Split this inbox into batches of at most 200 manifests")
    return inbox, manifests


def summarize(store, report):
    by_episode = {row["id"]: row for row in queue(store, diverse=False)}
    selected, seen = [], set()
    for item in report["items"]:
        episode_id = item.get("episode_id")
        if item["status"] != "prepared" or episode_id in seen:
            continue
        seen.add(episode_id)
        row = by_episode[episode_id]
        if row["status"] in {"accepted", "excluded"}:
            continue
        extra_reasons = sorted({finding["code"] for finding in item.get("data_findings", [])})
        selected.append(
            {
                **row,
                "priority": row["priority"] + 2 * len(extra_reasons),
                "reasons": list(dict.fromkeys([*row["reasons"], *extra_reasons])),
                "bundle": item["bundle"],
                "stage_file": item["stage_file"],
            }
        )
    selected.sort(key=lambda row: (-row["priority"], row["id"]))
    report["review_queue"] = selected
    completed = [item["episode_id"] for item in report["items"] if item["status"] == "prepared"]
    report["metrics"] = {
        "input_records": len(report["items"]),
        "prepared_records": len(completed),
        "unique_prepared_episodes": len(set(completed)),
        "duplicate_input_records": len(completed) - len(set(completed)),
        "failed_records": sum(item["status"] == "failed" for item in report["items"]),
        "failures_by_stage": dict(
            Counter(item["stage"] for item in report["items"] if item["status"] == "failed")
        ),
        "review_requests": len(selected),
        "priority_review_requests": sum(row["priority"] > 0 for row in selected),
        "accepted_episodes": sum(
            store.latest(episode_id)["status"] == "accepted" for episode_id in set(completed)
        ),
        "automatic_approvals": 0,
        "human_active_seconds": None,
        "data_review_requests": len(
            {
                item["episode_id"]
                for item in report["items"]
                if item["status"] == "prepared" and item.get("data_findings")
            }
        )
        + sum(item["status"] == "failed" for item in report["items"]),
    }
    report["status"] = (
        "needs_input"
        if report["metrics"]["failed_records"]
        else ("review_pending" if selected else "review_complete")
    )
    return report


def run_batch(inbox, store: Store):
    inbox, manifests = discover(inbox)
    if store.root.is_relative_to(inbox):
        raise ValueError("Keep the store outside the inbox so generated files cannot become inputs")
    inventory = [
        {"path": str(path.relative_to(inbox)), "manifest_sha256": file_hash(path)}
        for path in manifests
    ]
    batch_id = digest({"inbox": str(inbox), "inventory": inventory})[:24]
    directory = store.root / "batches" / batch_id
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "batch_id": batch_id,
        "inbox": str(inbox),
        "items": [],
        "report_path": str(directory / "report.json"),
    }
    with pipeline_lock(store):
        for path, entry in zip(manifests, inventory, strict=True):
            item_id = digest(entry)[:24]
            stage_file = directory / "items" / f"{item_id}.json"
            item = {
                **entry,
                "stage_file": str(stage_file),
                "status": "running",
                "stage": "validate",
                "completed_stages": [],
            }
            previous = json.loads(stage_file.read_text()) if stage_file.exists() else {}
            item["completed_stages"] = previous.get("completed_stages", [])
            try:
                normalized = directory / "normalized" / item_id / "source.json"
                item["recovery"] = normalize_source(path, normalized)
                _complete(stage_file, item, "normalize")
                item["stage"] = "import"
                episode = import_episode(normalized, store)
                if file_hash(path) != entry["manifest_sha256"]:
                    raise ValueError("Source manifest changed during import; rerun the inbox")
                item["episode_id"] = episode["id"]
                item["source_hash"] = episode["content_hash"]
                item["data_findings"] = data_findings(episode)
                _complete(stage_file, item, "import")
                item["stage"] = "suggest"
                run = suggest(store, episode["id"])
                item["suggestion_run"] = run["id"]
                _complete(stage_file, item, "suggest")
                item["stage"] = "prepare"
                bundle = task_bundle(store, episode["id"])
                bundle_path = directory / "bundles" / f"{episode['id']}.json"
                atomic_json(bundle_path, bundle)
                item["bundle"] = str(bundle_path)
                _complete(stage_file, item, "prepare")
                item["status"] = "prepared"
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                item["status"] = "failed"
                if isinstance(exc, ValidationError):
                    item["errors"] = [
                        {"field": ".".join(map(str, e["loc"])), "message": e["msg"]}
                        for e in exc.errors(include_input=False, include_url=False)
                    ]
                else:
                    item["errors"] = [{"field": item["stage"], "message": str(exc)[:500]}]
                item["next_action"] = (
                    "Correct the source metadata and rerun the inbox"
                    if item["stage"] == "validate"
                    else "Inspect the source or stage error, then rerun; prior imports are retained"
                )
            atomic_json(stage_file, item)
            report["items"].append(item)
            persist_report(summarize(store, report))
            atomic_json(store.root / "batches" / "latest.json", {"report": report["report_path"]})
    return report


def _complete(path, item, stage):
    if stage not in item["completed_stages"]:
        item["completed_stages"].append(stage)
    atomic_json(path, item)


def read_report(path, store):
    if path is None:
        pointer = store.root / "batches" / "latest.json"
        if not pointer.exists():
            raise ValueError("No batch has run in this store yet")
        path = json.loads(pointer.read_text())["report"]
    path = Path(path).resolve()
    if not path.is_relative_to(store.root / "batches"):
        raise ValueError("Choose a batch report from this store")
    report = json.loads(path.read_text())
    if report["schema_version"] != 1 or report["report_path"] != str(path):
        raise ValueError("Batch report identity does not match its path")
    native = path.parent / "native-queue.json"
    if native.exists():
        report["native_tasks"] = json.loads(native.read_text())["tasks"]
    return summarize(store, report)


def persist_report(report):
    path = Path(report["report_path"])
    atomic_json(path, report)
    metrics = report["metrics"]
    lines = [
        f"Batch {report['batch_id']}: {report['status']}",
        f"{metrics['input_records']} inputs; "
        f"{metrics['unique_prepared_episodes']} unique episodes; "
        f"{metrics['failed_records']} failed; {metrics['duplicate_input_records']} duplicates.",
        f"{metrics['review_requests']} reviews pending; {metrics['accepted_episodes']} approved.",
        "",
        "Input failures:",
    ]
    for item in report["items"]:
        if item["status"] == "failed":
            lines.append(
                f"- {item['path']} ({item['stage']}): "
                + "; ".join(f"{error['field']}: {error['message']}" for error in item["errors"])
            )
            lines.append(f"  {item['next_action']}")
    lines.extend(["", "Review queue (highest priority first):"])
    for row in report["review_queue"]:
        lines.append(f"- {row['id']}: {row['task']} — " + "; ".join(row["reasons"]))
        native = report.get("native_tasks", {}).get(row["id"])
        if native:
            lines.append("  " + native["url"])
    lines.extend(
        ["", "Approvals require an explicit decision. Human review time has not been measured."]
    )
    path.with_name("summary.txt").write_text("\n".join(lines) + "\n")


def accepted_selection(report, store, split):
    if split not in {"train", "development", "test"}:
        raise ValueError("Choose an explicit train, development or test split")
    episode_ids = {item["episode_id"] for item in report["items"] if item["status"] == "prepared"}
    selection = []
    for episode_id in sorted(episode_ids):
        revision = store.latest(episode_id)
        if revision["status"] == "accepted":
            selection.append(
                {"episode_id": episode_id, "revision": revision["revision"], "split": split}
            )
    if not selection:
        raise ValueError(
            "No approved episodes in this batch; submit and approve a native review first"
        )
    return selection
