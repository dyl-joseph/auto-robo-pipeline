"""Exercise batch recovery, native task ordering and approval with a disposable SDK session."""

import json
import os
import shutil
import subprocess
import threading
from copy import deepcopy
from pathlib import Path
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIServer, make_server

from robo.core import Edit, atomic_json
from robo.export import validate_export
from robo.native_queue import sync_queue
from robo.native_review import NativeReviewBridge
from robo.pipeline import accepted_selection, read_report, run_batch


def verify_batch(sdk, store, directory, cookies, real_source=None):
    source_path = real_source or store.root / "demo_sources" / "source-0.json"
    source = json.loads(source_path.read_text())
    inbox = directory / "inbox"
    inbox.mkdir()
    shutil.copyfile(source_path.parent / source["video"], inbox / "video.mp4")
    source["video"] = "video.mp4"
    source["source_id"] += "-batch-control"
    source["provenance"] += "; automated batch integration test, not human review"
    atomic_json(inbox / "source.json", source)
    atomic_json(inbox / "duplicate.source.json", source)
    attention = deepcopy(source)
    attention["source_id"] += "-attention"
    attention["video_timing"] = "inferred_nominal_fps"
    # Keep the same channel layout so both tasks share a native project.
    if attention.get("streams"):
        stream = attention["streams"][0]
        stream["timestamps_s"] = stream["timestamps_s"][:2] + stream["timestamps_s"][6:]
        stream["values"] = stream["values"][:2] + stream["values"][6:]
    atomic_json(inbox / "attention.source.json", attention)
    invalid = deepcopy(source)
    del invalid["license"]
    atomic_json(inbox / "invalid.source.json", invalid)
    report = run_batch(inbox, store)
    assert report["metrics"]["input_records"] == 4
    assert report["metrics"]["failed_records"] == 1
    assert report["metrics"]["duplicate_input_records"] == 1
    assert report["metrics"]["review_requests"] == 2
    assert run_batch(inbox, store) == report
    report_path = Path(report["report_path"])
    attention_id = next(
        e["id"] for e in store.episodes() if e["source_id"] == attention["source_id"]
    )
    store.commit(
        attention_id,
        Edit(
            expected_revision=0,
            actor="fixture-triage",
            rationale="Synthetic uncertainty fixture",
            quality_flags=["uncertain"],
            segments=[],
        ),
        "uncertain",
    )
    report = read_report(report_path, store)
    for row in report["review_queue"]:
        path = Path(row["bundle"])
        bundle = json.loads(path.read_text())
        bundle["task"]["data"]["provenance"] += " · automated integration test; not human review"
        atomic_json(path, bundle)
    original_import = sdk.projects.import_tasks

    def interrupted(*args, **kwargs):
        original_import(*args, **kwargs)
        raise OSError("Injected interruption after native task creation")

    sdk.projects.import_tasks = interrupted
    try:
        try:
            sync_queue(sdk, store, report_path, "http://testserver")
        except OSError:
            pass
        else:
            raise AssertionError("Interruption was not exercised")
    finally:
        sdk.projects.import_tasks = original_import
    synced = sync_queue(sdk, store, report_path, "http://testserver")
    assert len(synced["native_tasks"]) == 2
    assert sync_queue(sdk, store, report_path, "http://testserver") == synced
    project_ids = {task["project_id"] for task in synced["native_tasks"].values()}
    assert len(project_ids) == 1
    project_id = next(iter(project_ids))
    tasks = list(sdk.tasks.list(project=project_id))
    assert len(tasks) == 2
    assert len(sdk.import_storage.local.list(project=project_id)) == 1
    views = [
        view
        for view in sdk.views.list(project=project_id)
        if (view.data or {}).get("title") == "Review priority"
    ]
    assert len(views) == 1
    ordered = list(sdk.tasks.list(project=project_id, view=views[0].id))
    priorities = [task.data["review_priority"] for task in ordered]
    assert priorities == sorted(priorities, reverse=True)
    assert priorities[0] > priorities[1]
    assert priorities[0] >= 10 and priorities[1] < 10

    from django.core.wsgi import get_wsgi_application

    class Server(ThreadingMixIn, WSGIServer):
        daemon_threads = True

    server = make_server("127.0.0.1", 8134, get_wsgi_application(), server_class=Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        subprocess.run(
            ["node", "scripts/labelstudio_queue.mjs"],
            check=True,
            timeout=90,
            env={
                **os.environ,
                "LS_FIXTURE_COOKIES": json.dumps(cookies),
                "LS_FIXTURE_PROJECT": str(project_id),
                "LS_FIXTURE_TASK": str(ordered[0].id),
                "LS_FIXTURE_VIEW": str(views[0].id),
            },
        )
    finally:
        server.shutdown()
        server.server_close()

    control = sdk.tasks.get(str(ordered[1].id))
    control_result = deepcopy(control.model_dump(mode="json")["predictions"][0]["result"])
    control_result.append(
        {
            "from_name": "quality",
            "to_name": "video",
            "type": "choices",
            "value": {"choices": ["uncertain"]},
        }
    )
    uncertain = sdk.annotations.create(control.id, result=control_result, lead_time=1.0)
    bridge = NativeReviewBridge(sdk, store, "http://testserver")
    bridge.refresh(
        control.data["episode_id"], uncertain.id, 0, "fixture-importer", "Uncertain test label"
    )
    sync_queue(sdk, store, report_path, "http://testserver")
    updated = sdk.tasks.get(str(control.id))
    assert updated.data["review_priority"] == priorities[1] + 5
    assert "needs_adjudication" in updated.data["review_reason"]

    episode_id = ordered[0].data["episode_id"]
    task = sdk.tasks.get(str(ordered[0].id))
    result = deepcopy(task.model_dump(mode="json")["predictions"][0]["result"])
    result[0]["value"]["timelinelabels"] = ["unknown"]
    result.extend(
        [
            {
                "from_name": "outcome",
                "to_name": "video",
                "type": "choices",
                "value": {"choices": ["unknown"]},
            },
            {
                "from_name": "quality",
                "to_name": "video",
                "type": "choices",
                "value": {"choices": ["incomplete"]},
            },
            {
                "from_name": "decision_note",
                "to_name": "video",
                "type": "textarea",
                "value": {
                    "text": ["Automated batch correction and approval test; not human review."]
                },
            },
        ]
    )
    annotation = sdk.annotations.create(task.id, result=result, lead_time=1.0)
    bridge = NativeReviewBridge(sdk, store, "http://testserver")
    draft = bridge.refresh(
        episode_id,
        annotation.id,
        store.latest(episode_id)["revision"],
        "fixture-importer",
        "Automated correction",
    )
    bridge.decide(
        episode_id,
        draft["revision"],
        "accept",
        "fixture-approver",
        "Explicit integration test decision",
    )
    selection = accepted_selection(read_report(report_path, store), store, "train")
    export = directory / "accepted-batch"
    bridge.export(selection, export)
    records = validate_export(export)
    assert len(records) == 1
    assert records[0]["annotation"]["active_s"] is None
    refreshed = read_report(report_path, store)
    assert refreshed["metrics"]["accepted_episodes"] == 1
    assert refreshed["metrics"]["review_requests"] == 1
    evidence = {
        "kind": "automated_native_batch_workflow",
        "real_droid": real_source is not None,
        "input_metrics": report["metrics"],
        "after_review_metrics": refreshed["metrics"],
        "native_projects": len(project_ids),
        "native_tasks": len(tasks),
        "priority_order": priorities,
        "interrupted_import_reconciled": True,
        "rerun_idempotent": True,
        "uncertain_annotation_updates_native_priority": True,
        "native_export_records": len(records),
        "human_efficiency_measured": False,
    }
    atomic_json(Path("artifacts/batch-workflow.json"), evidence)
    print(json.dumps(evidence, indent=2))
