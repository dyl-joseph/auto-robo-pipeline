"""Verify supplied model predictions through native Label Studio APIs and accepted export.

Run in the root Label Studio environment with a fresh output directory. The source
is imported into a disposable store. Corrections and explicit approvals are test
inputs, not human review or independent reference annotations. No browser is used.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from robo.core import atomic_json, file_hash
from robo.export import validate_export
from robo.ingest import import_episode
from robo.labelstudio import publish_bundle
from robo.native_review import NativeReviewBridge, prediction_identity
from robo.semantic import validate_episode_segments
from robo.semantic_review import decoded_review, load_snapshot, prepare, read_inputs
from robo.store import Conflict, Store

INSTANCE = "http://testserver"
CORRECTED_TEXT = "Automated API correction. This text is an unvalidated integration input."
LATER_TEXT = "Later automated API correction. Fresh approval is required."
OBJECT_TEXT = "Automated integration object. This description is unvalidated."


@contextmanager
def authenticated_backend(directory, store):
    data = directory / "labelstudio"
    data.mkdir()
    values = {
        "BASE_DATA_DIR": str(data),
        "DATABASE_NAME": str(data / "labelstudio.sqlite3"),
        "LOCAL_FILES_SERVING_ENABLED": "true",
        "LOCAL_FILES_DOCUMENT_ROOT": str(store.root),
        "COLLECT_ANALYTICS": "false",
        "LATEST_VERSION_CHECK": "false",
        "SENTRY_DSN": "",
        "FRONTEND_SENTRY_DSN": "",
        "FEATURE_FLAGS_OFFLINE": "true",
        "ALLOWED_HOSTS": "localhost,127.0.0.1,testserver",
        "DJANGO_DB": "sqlite",
        "HOST": INSTANCE,
        "LOG_LEVEL": "ERROR",
    }
    for key, value in values.items():
        os.environ[key] = value
        os.environ[f"LABEL_STUDIO_{key}"] = value
    os.environ["fflag_fix_front_dev_3391_interactive_view_all"] = "false"
    import label_studio

    sys.path.insert(0, str(Path(label_studio.__file__).parent))
    os.environ["DJANGO_SETTINGS_MODULE"] = "label_studio.core.settings.label_studio"
    import django

    django.setup()
    import httpx
    from django.conf import settings
    from django.contrib.auth import get_user_model
    from django.core.management import call_command
    from django.core.wsgi import get_wsgi_application
    from django.db import connections
    from django.test import Client
    from label_studio_sdk import LabelStudio
    from organizations.models import Organization
    from rest_framework.authtoken.models import Token

    call_command("migrate", verbosity=0, interactive=False)
    user = get_user_model().objects.create(email="semantic-api-fixture@localhost")
    user.set_unusable_password()
    user.save()
    Token.objects.filter(user=user).delete()
    user.active_organization = Organization.create_organization(
        created_by=user, title="Disposable semantic API verification"
    )
    user.save()
    session_client = Client()
    session_client.force_login(user, backend="django.contrib.auth.backends.ModelBackend")
    session = session_client.session
    session["last_login"] = time.time()
    session.save()
    session_client.cookies[settings.SESSION_COOKIE_NAME] = session.session_key
    session_client.get("/")
    assert "_auth_user_id" in session_client.session
    cookies = {name: value.value for name, value in session_client.cookies.items()}

    class SessionAuth(httpx.Auth):
        def auth_flow(self, request):
            request.headers.pop("Authorization", None)
            yield request

    client = httpx.Client(
        auth=SessionAuth(),
        transport=httpx.WSGITransport(app=get_wsgi_application()),
        base_url=INSTANCE,
        cookies=cookies,
        headers={"X-CSRFToken": cookies.get("csrftoken", "")},
        timeout=30,
    )
    sdk = LabelStudio(base_url=INSTANCE, api_key="local-session-transport", httpx_client=client)
    try:
        yield sdk, client
    finally:
        client.close()
        session_client.logout()
        Token.objects.filter(user=user).delete()
        assert not Token.objects.filter(user=user).exists()
        connections.close_all()


def fetch_task(sdk, task_id):
    task = sdk.tasks.get(str(task_id)).model_dump(mode="json")
    task["predictions"] = [
        sdk.predictions.get(item).model_dump(mode="json") if type(item) is int else item
        for item in task.get("predictions") or []
    ]
    return task


def groups(results):
    grouped = {}
    for result in results:
        if result.get("from_name") in {"subtask", "subtask_text", "object_text"}:
            grouped.setdefault(result["id"], {})[result["from_name"]] = result
    return sorted(grouped.values(), key=lambda item: item["subtask"]["value"]["ranges"][0]["start"])


def saved_annotation(sdk, task_id, annotation_id, expected, machine_before, output):
    for _ in range(2):
        assert sdk.annotations.get(annotation_id).model_dump(mode="json")["result"] == expected
    task = fetch_task(sdk, task_id)
    assert len(task["annotations"]) == 1
    assert task["annotations"][0]["id"] == annotation_id
    assert task["annotations"][0]["result"] == expected
    assert not task.get("drafts") and not task.get("draft_exists")
    assert [prediction_identity(item) for item in task["predictions"]] == machine_before
    atomic_json(
        output,
        {"task_id": task_id, "predictions": machine_before, "annotations": task["annotations"]},
    )
    return task


def edit_boundary(results):
    ordered = groups(results)
    for left, right in zip(ordered, ordered[1:], strict=False):
        a = left["subtask"]["value"]["ranges"][0]
        b = right["subtask"]["value"]["ranges"][0]
        if a["end"] + 1 != b["start"]:
            continue
        for boundary in (a["end"] + 1, a["end"] - 1):
            if a["start"] <= boundary < b["end"]:
                old = a["end"]
                for controls, start, end in (
                    (left, a["start"], boundary),
                    (right, boundary + 1, b["end"]),
                ):
                    for control in controls.values():
                        control["value"]["ranges"] = [{"start": start, "end": end}]
                return {"old_left_end": old, "new_left_end": boundary}
    return None


def run(args):
    source = args.source.resolve(strict=True)
    plan_path, prediction_path = (
        args.plan.resolve(strict=True),
        args.predictions.resolve(strict=True),
    )
    directory = args.output.resolve()
    if directory.exists() or args.output.is_symlink():
        raise ValueError("choose a fresh verification output directory")
    before = {str(path): file_hash(path) for path in (source, plan_path, prediction_path)}
    plan, predictions = read_inputs(plan_path.read_bytes(), prediction_path.read_bytes())
    directory.mkdir(mode=0o700, parents=True)
    store = Store(directory / "source")
    episode = import_episode(source, store)
    specs = [spec for spec in plan.episodes if spec.episode_id == episode["id"]]
    rows = [row for row in predictions.episodes if row.episode_id == episode["id"]]
    if len(specs) != 1 or len(rows) != 1:
        raise ValueError("source must match one episode in the supplied plan and predictions")
    spec, row = specs[0], rows[0]
    if row.error or not row.segments:
        raise ValueError("verification requires successful supplied predictions")
    allowed = {"synthetic_fixture"} if args.synthetic_fixture else {"model"}
    if any(segment.provenance.kind not in allowed for segment in row.segments):
        raise ValueError("supply model predictions, or explicitly select a synthetic fixture")
    validate_episode_segments(row.segments, spec, plan.normalized_actions, complete=True)
    bundle = prepare(store, plan_path, prediction_path, episode["id"], directory / "review")
    snapshot_id = bundle["trusted_origin"]["semantic_prediction"]["snapshot_id"]
    report = {
        "evidence_kind": "synthetic_fixture"
        if args.synthetic_fixture
        else "supplied_model_api_verification",
        "producer": predictions.producer,
        "prediction_run_id": predictions.run_id,
        "episode_id": episode["id"],
        "data_kind": episode["data_kind"],
        "source_store": str(store.root),
        "split": spec.split,
        "machine_segments": len(row.segments),
        "source_frames": len(spec.frame_boundaries_s) - 1,
        "versions": {name: version(name) for name in ["label-studio", "label-studio-sdk"]},
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "implementation_sha256": file_hash(Path(__file__)),
        "input_sha256": before,
        "browser_used": False,
        "semantic_accuracy_measured": False,
        "human_review_measured": False,
        "corrections_and_approvals_are_automated_fixtures": True,
    }
    snapshots = directory / "snapshots"
    snapshots.mkdir()
    with authenticated_backend(directory, store) as (sdk, client):
        binding = publish_bundle(sdk, store, bundle)
        bridge = NativeReviewBridge(sdk, store, INSTANCE)
        bridge.register(bundle, binding["project_id"], binding["task_id"])
        task_id = binding["task_id"]
        task = fetch_task(sdk, task_id)
        machine = [prediction_identity(item) for item in task["predictions"]]
        response = client.get(bundle["task"]["data"]["video"])
        assert response.status_code == 200
        import hashlib

        assert hashlib.sha256(response.content).hexdigest() == bundle["proxy"]["files"]["video.mp4"]
        results = deepcopy(bundle["task"]["predictions"][0]["result"])
        annotation_id = sdk.annotations.create(task_id, result=results, lead_time=0).id
        saved_annotation(
            sdk, task_id, annotation_id, results, machine, snapshots / "01-model-copy.json"
        )
        missing = groups(results)[-1]["subtask"]["id"]
        incomplete = [item for item in results if item["id"] != missing]
        sdk.annotations.update(annotation_id, result=incomplete)
        draft = bridge.refresh(
            episode["id"], annotation_id, 0, "automated-fixture", "Test incomplete coverage"
        )
        try:
            bridge.decide(
                episode["id"],
                draft["revision"],
                "accept",
                "automated-fixture",
                "Must refuse incomplete coverage",
            )
        except ValueError as exc:
            assert "complete semantic coverage" in str(exc)
            report["incomplete_coverage_rejected"] = True
        else:
            raise AssertionError("incomplete semantic annotation was approved")
        target = groups(results)[0]
        target["subtask_text"]["value"]["text"] = [CORRECTED_TEXT]
        if "object_text" not in target:
            results.append(
                {
                    **deepcopy(target["subtask_text"]),
                    "from_name": "object_text",
                    "value": {
                        "ranges": deepcopy(target["subtask"]["value"]["ranges"]),
                        "text": [OBJECT_TEXT],
                    },
                }
            )
        else:
            target["object_text"]["value"]["text"] = [OBJECT_TEXT]
        report["boundary_edit"] = edit_boundary(results)
        sdk.annotations.update(annotation_id, result=results)
        saved_annotation(
            sdk,
            task_id,
            annotation_id,
            results,
            machine,
            snapshots / "02-text-object-boundary.json",
        )
        populated = deepcopy(results)
        groups(results)[0]["object_text"]["value"]["text"] = [""]
        sdk.annotations.update(annotation_id, result=results)
        saved_annotation(
            sdk, task_id, annotation_id, results, machine, snapshots / "03-cleared-object.json"
        )
        assert groups(populated)[1:] == groups(results)[1:]
        reviewed = decoded_review(results, spec, bundle["proxy"])
        assert reviewed["complete"]
        assert reviewed["segments"][0]["subtask_text"] == CORRECTED_TEXT
        assert reviewed["segments"][0]["object_text"] is None
        refreshed = bridge.refresh(
            episode["id"],
            annotation_id,
            draft["revision"],
            "automated-fixture",
            "Test saved corrections",
        )
        accepted = bridge.decide(
            episode["id"],
            refreshed["revision"],
            "accept",
            "automated-fixture",
            "Explicit integration approval; labels unvalidated",
        )
        selection = [
            {"episode_id": episode["id"], "revision": accepted["revision"], "split": spec.split}
        ]
        first = directory / "accepted-first"
        bridge.export(selection, first)
        first_records = validate_export(first)
        assert first_records[0]["annotation"] == accepted
        historical = {
            str(path.relative_to(first)): file_hash(path)
            for path in first.iterdir()
            if path.is_file()
        }
        if spec.split != "train":
            try:
                bridge.export(
                    [{**selection[0], "split": "train"}], directory / "wrong-split-must-not-exist"
                )
            except ValueError as exc:
                assert "frozen evaluation split" in str(exc)
                assert not (directory / "wrong-split-must-not-exist").exists()
                report["heldout_split_change_rejected"] = True
            else:
                raise AssertionError("held-out semantic annotation exported as training data")
        groups(results)[0]["subtask_text"]["value"]["text"] = [LATER_TEXT]
        sdk.annotations.update(annotation_id, result=results)
        saved_annotation(
            sdk, task_id, annotation_id, results, machine, snapshots / "04-later-text.json"
        )
        for action in (
            lambda: bridge.export(selection, directory / "stale-must-not-exist"),
            lambda: bridge.decide(
                episode["id"],
                accepted["revision"],
                "accept",
                "automated-fixture",
                "Must require fresh draft",
            ),
        ):
            try:
                action()
            except Conflict as exc:
                assert "refresh" in str(exc)
            else:
                raise AssertionError("changed annotation reused stale approval")
        assert not (directory / "stale-must-not-exist").exists()
        changed = bridge.refresh(
            episode["id"],
            annotation_id,
            accepted["revision"],
            "automated-fixture",
            "Refresh later text",
        )
        second = bridge.decide(
            episode["id"],
            changed["revision"],
            "accept",
            "automated-fixture",
            "Fresh explicit integration approval",
        )
        final_output = directory / "accepted-refreshed"
        bridge.export([{**selection[0], "revision": second["revision"]}], final_output)
        final = validate_export(final_output)
        assert final[0]["annotation"] == second
        assert (
            final[0]["annotation"]["native"]["semantic_review"]["segments"][0]["subtask_text"]
            == LATER_TEXT
        )
        assert validate_export(first) == first_records
        assert all(file_hash(first / name) == expected for name, expected in historical.items())
        saved_annotation(sdk, task_id, annotation_id, results, machine, snapshots / "05-final.json")
        report.update(
            text_save_reload=True,
            object_clear_save_reload=True,
            stale_approval_rejected=True,
            explicit_approval_export=True,
            historical_export_unchanged=True,
            machine_predictions_unchanged=True,
            first_accepted_revision=accepted["revision"],
            final_accepted_revision=second["revision"],
            accepted_first=str(first),
            accepted_refreshed=str(final_output),
        )
    assert all(file_hash(Path(path)) == expected for path, expected in before.items())
    _, _, hashes = load_snapshot(store, snapshot_id)
    assert hashes == {
        "plan.json": before[str(plan_path)],
        "predictions.json": before[str(prediction_path)],
    }
    report.update(status="passed", original_inputs_unchanged=True, session_cleanup=True)
    atomic_json(directory / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "plan", "predictions", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--synthetic-fixture",
        action="store_true",
        help="Require explicitly generated fixture prediction provenance",
    )
    args = parser.parse_args()
    previous = logging.root.manager.disable
    logging.disable(sys.maxsize)
    try:
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            report = run(args)
    finally:
        logging.disable(previous)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(args.output / "report.json"),
                "accepted": report["accepted_refreshed"],
                "source_store": report["source_store"],
            }
        )
    )


if __name__ == "__main__":
    main()
