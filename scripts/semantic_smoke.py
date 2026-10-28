"""Exercise text corrections on real DROID through the native application and bridge."""

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIServer, make_server

from robo.core import LABELS, atomic_json, digest, file_hash
from robo.export import validate_export
from robo.labelstudio import publish_bundle
from robo.native_review import NativeReviewBridge
from robo.semantic import EvaluationPlan, episode_spec
from robo.semantic_baseline import run as predict
from robo.semantic_review import prepare
from robo.store import Conflict


def verify_semantic(sdk, store, episode, directory, cookies):
    from django.core.wsgi import get_wsgi_application

    spec = episode_spec(
        episode,
        dataset="DROID",
        dataset_episode_id=episode["source_id"],
        camera_id="exterior1",
        split="test",
    )
    plan = EvaluationPlan(
        taxonomy_id="robot-coarse-v1",
        normalized_actions=list(LABELS),
        annotation_guidelines_sha256=digest(
            "Automated integration fixture, not a semantic benchmark"
        ),
        identity_audit={
            "status": "unverified",
            "reviewer": "automated integration fixture",
            "evidence_sha256": digest(episode["content_hash"]),
        },
        episodes=[spec],
    )
    plan_path = directory / "semantic-plan.json"
    atomic_json(plan_path, plan.model_dump())
    prediction_dir = directory / "machine-predictions"
    report = predict(plan, store, prediction_dir)
    assert report["failed_predictions"] == 0
    prediction_path = prediction_dir / "predictions.json"
    before = file_hash(prediction_path)
    bundle = prepare(
        store, plan_path, prediction_path, episode["id"], directory / "semantic-bundle"
    )
    bundle["task"]["data"]["video"] = bundle["task"]["data"]["video"].replace("?d=", "?d=robotics/")
    bundle["task"]["data"]["provenance"] += (
        " · automated integration check; labels unvalidated, not human review"
    )
    binding = publish_bundle(sdk, store, bundle)
    bridge = NativeReviewBridge(sdk, store, "http://127.0.0.1:8134")
    bridge.register(bundle, binding["project_id"], binding["task_id"])
    count = sum(r["type"] == "timelinelabels" for r in bundle["task"]["predictions"][0]["result"])

    class ThreadedServer(ThreadingMixIn, WSGIServer):
        daemon_threads = True

    server = make_server("127.0.0.1", 8134, get_wsgi_application(), server_class=ThreadedServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        subprocess.run(
            ["node", "scripts/semantic_review_browser.mjs"],
            check=True,
            timeout=240,
            env={
                **os.environ,
                "LS_FIXTURE_COOKIES": json.dumps(cookies),
                "LS_FIXTURE_PROJECT": str(binding["project_id"]),
                "LS_FIXTURE_TASK": str(binding["task_id"]),
                "LS_FIXTURE_TITLE": episode["task"],
                "LS_FIXTURE_SEGMENTS": str(count),
                "LS_FIXTURE_FRAMES": str(len(spec.frame_boundaries_s) - 1),
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    task = sdk.tasks.get(str(binding["task_id"])).model_dump(mode="json")
    assert len(task["annotations"]) == 1
    annotation_id = task["annotations"][0]["id"]
    draft = bridge.refresh(
        episode["id"],
        annotation_id,
        0,
        "automated-semantic-fixture",
        "Browser correction integration test, not human review",
    )
    semantic = draft["native"]["semantic_review"]
    assert semantic["complete"] and len(semantic["segments"]) == count
    first = semantic["segments"][0]
    assert (
        first["subtask_text"] == "Automated test correction: initial state has no earlier sample."
    )
    assert first["object_text"] is None
    assert first["source_start"] == 0 and first["source_end_exclusive"] == 2
    assert semantic["segments"][1]["source_start"] == 2
    assert semantic["segments"][1]["object_text"] == "Automated fixture object; unvalidated"
    accepted = bridge.decide(
        episode["id"],
        1,
        "accept",
        "automated-test-approver",
        "Explicit automated integration decision; labels unvalidated",
    )
    output = directory / "accepted"
    selection = [{"episode_id": episode["id"], "revision": 2, "split": "test"}]
    bridge.export(selection, output)
    records = validate_export(output)
    assert records[0]["annotation"] == accepted
    assert file_hash(prediction_path) == before
    fixture = os.environ.get("ROBO_SEMANTIC_EXPORT_FIXTURE")
    if fixture:
        target = Path(fixture)
        if target.exists():
            raise ValueError("choose a new semantic export fixture path")
        target.mkdir(parents=True)
        shutil.copytree(output, target / "accepted")
        source = store.root / episode["media"]
        media = target / "source" / episode["media"]
        media.parent.mkdir(parents=True)
        shutil.copyfile(source, media)
        shutil.copyfile(source.parent / "episode.json", media.parent / "episode.json")
        shutil.copytree(store.root / "semantic_sources", target / "source/semantic_sources")
        shutil.copytree(prediction_dir, target / "uncorrected")
        shutil.copyfile(plan_path, target / "plan.json")
    changed = task["annotations"][0]["result"]
    next(r for r in changed if r["from_name"] == "subtask_text")["value"]["text"] = [
        "Changed after approval"
    ]
    sdk.annotations.update(annotation_id, result=changed)
    try:
        bridge.export(selection, directory / "stale")
    except Conflict:
        pass
    else:
        raise AssertionError("text-only edit did not invalidate semantic approval")
    assert file_hash(prediction_path) == before
    atomic_json(
        Path("artifacts/semantic-native.json"),
        {
            "evidence_kind": "real_droid_automated_semantic_integration",
            "episodes": 1,
            "source_frames": 119,
            "candidate_segments": count,
            "corrected_segments": len(semantic["segments"]),
            "source_timing": spec.timing,
            "prediction_sha256_before": before,
            "prediction_sha256_after": file_hash(prediction_path),
            "native_text_and_object_edit_verified": True,
            "cleared_object_saved_reloaded_and_exported_as_absent": True,
            "boundary_edit_preserves_coverage": True,
            "saved_reload_verified": True,
            "text_only_edit_invalidates_approval": True,
            "approved_json_export_verified": True,
            "human_review_measured": False,
            "semantic_accuracy_measured": False,
            "automatic_approvals_in_product": 0,
            "approval_actor": "automated-test-approver",
            "source_license": "CC-BY-4.0",
            "source_attribution": "DROID, Khazatsky et al. (2024)",
        },
    )
    print(
        "Real DROID semantic prediction -> native text correction -> "
        "approval -> immutable export passed"
    )
