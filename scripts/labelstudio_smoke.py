"""Exercise the real Label Studio application and official SDK in a disposable local database.

Uses an unusable-password fixture user and session. Native-export fixture tokens are revoked.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def run(directory: Path, real_source: Path | None = None):
    real = real_source is not None
    evidence_prefix = "droid" if real else "labelstudio"
    evidence_kind = "real_droid_automated_integration" if real else "synthetic_fixture"
    expected_outcome = "unknown" if real else "failure"
    expected_quality = ["incomplete"] if real else ["recovery"]
    data = directory / "labelstudio"
    data.mkdir()
    values = {
        "BASE_DATA_DIR": str(data),
        "DATABASE_NAME": str(data / "labelstudio.sqlite3"),
        "LOCAL_FILES_SERVING_ENABLED": "true",
        "LOCAL_FILES_DOCUMENT_ROOT": str(directory),
        "COLLECT_ANALYTICS": "false",
        "LATEST_VERSION_CHECK": "false",
        "SENTRY_DSN": "",
        "FRONTEND_SENTRY_DSN": "",
        "FEATURE_FLAGS_OFFLINE": "true",
        "ALLOWED_HOSTS": "localhost,127.0.0.1,testserver",
        "DJANGO_DB": "sqlite",
        "HOST": "http://127.0.0.1:8134",
        "LOG_LEVEL": "ERROR",
    }
    for key, value in values.items():
        os.environ[f"LABEL_STUDIO_{key}"] = value
        os.environ[key] = value
    # Compare All in 1.23.2 leaves the annotation sync manager uninitialized.
    # Use the supported single-annotation mode for these tests.
    os.environ["fflag_fix_front_dev_3391_interactive_view_all"] = "false"
    import label_studio

    sys.path.insert(0, str(Path(label_studio.__file__).parent))
    os.environ["DJANGO_SETTINGS_MODULE"] = "label_studio.core.settings.label_studio"
    import django

    django.setup()
    from django.core.management import call_command

    call_command("migrate", verbosity=0, interactive=False)
    from django.contrib.auth import get_user_model
    from django.test import Client
    from organizations.models import Organization
    from rest_framework.authtoken.models import Token

    user = get_user_model().objects.create(email="fixture@localhost")
    user.set_unusable_password()
    user.save()
    Token.objects.filter(user=user).delete()
    organization = Organization.create_organization(
        created_by=user, title="Disposable robotics integration test"
    )
    user.active_organization = organization
    user.save()
    browser = Client()
    browser.force_login(user, backend="django.contrib.auth.backends.ModelBackend")
    import time

    from django.conf import settings

    session = browser.session
    session["last_login"] = time.time()
    session.save()
    browser.cookies[settings.SESSION_COOKIE_NAME] = session.session_key
    browser.get("/")
    assert "_auth_user_id" in browser.session, (
        "fixture login did not survive Label Studio session policy"
    )
    import httpx
    from django.core.wsgi import get_wsgi_application
    from label_studio_sdk import LabelStudio

    from robo.analysis import suggest
    from robo.core import atomic_json
    from robo.demo import make_demo
    from robo.ingest import import_episode
    from robo.labelstudio import (
        annotation_snapshot,
        bind_task,
        decoded_segments,
        publish_bundle,
        task_bundle,
    )
    from robo.store import Store

    store = Store(directory / "robotics")
    # Deliberately missing source frames: review boundaries remain on a known grid,
    # and exported evidence must expose the gap in the measured source clock.
    import subprocess

    if real:
        from robo.droid import RELEASE

        source = json.loads(real_source.read_text())
        assert source["data_kind"] == "real" and source["release"] == RELEASE
        assert source["video_timing"] == "inferred_nominal_fps"
        episodes = [import_episode(real_source, store)]
    else:
        make_demo(directory / "robotics", count=1)
        sources = store.root / "demo_sources"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(sources / "tabletop-0.mp4"),
                "-vf",
                "select=not(eq(n\\,5)+eq(n\\,6))",
                "-fps_mode",
                "vfr",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                str(sources / "irregular.mp4"),
            ],
            check=True,
            timeout=30,
        )
        source = json.loads((sources / "source-0.json").read_text())
        source["video"] = "irregular.mp4"
        source["source_id"] += "-vfr"
        source["provenance"] += "; deliberately dropped source frames 5/6 for VFR verification"
        atomic_json(sources / "irregular.json", source)
        episodes = [import_episode(sources / "irregular.json", store)]
    suggest(store, episodes[0]["id"])
    bundle = task_bundle(store, episodes[0]["id"])
    if real:
        bundle["task"]["data"]["provenance"] += " · automated integration check; not human review"
    for field in ("video",):
        bundle["task"]["data"][field] = bundle["task"]["data"][field].replace("?d=", "?d=robotics/")
    cookies = {name: value.value for name, value in browser.cookies.items()}
    headers = {"X-CSRFToken": cookies.get("csrftoken", "")}

    class SessionAuth(httpx.Auth):
        def auth_flow(self, request):
            # SDK 2.x requires a constructor key; actual requests use the real Django
            # session and CSRF cookie. The sentinel is never sent as a credential.
            request.headers.pop("Authorization", None)
            yield request

    client = httpx.Client(
        auth=SessionAuth(),
        transport=httpx.WSGITransport(app=get_wsgi_application()),
        base_url="http://testserver",
        cookies=cookies,
        headers=headers,
    )
    sdk = LabelStudio(
        base_url="http://testserver", api_key="local-session-transport", httpx_client=client
    )
    if "--semantic-pipeline" in sys.argv:
        from django.test import override_settings

        from scripts.semantic_pipeline_smoke import verify_pipeline

        with override_settings(
            LOCAL_FILES_DOCUMENT_ROOT=str(directory / "semantic-pipeline/store")
        ):
            return verify_pipeline(sdk, directory, args.semantic_pipeline, browser)
    if "--semantic" in sys.argv:
        from scripts.semantic_smoke import verify_semantic

        return verify_semantic(sdk, store, episodes[0], directory, cookies)
    if "--batch" in sys.argv:
        from django.test import override_settings

        from scripts.batch_smoke import verify_batch

        with override_settings(LOCAL_FILES_DOCUMENT_ROOT=str(store.root)):
            return verify_batch(sdk, store, directory, cookies, real_source)
    binding = publish_bundle(sdk, store, bundle)
    project_id = binding["project_id"]
    task_id = binding["task_id"]
    task = sdk.tasks.get(task_id)
    print("SDK task", task.id, "predictions", len(task.predictions or []))
    proposed = bundle["task"]["predictions"][0]["result"]
    corrected = json.loads(json.dumps(proposed))
    corrected[0]["value"]["timelinelabels"] = ["reach"]
    annotation = sdk.annotations.create(task_id, result=corrected, lead_time=1.5)

    # Label Studio's native export implementation dereferences user.auth_token
    # even for JSON without resource downloads. Keep it only in the disposable
    # fixture database for this call; SDK requests still authenticate by session.
    def native_export():
        fixture_token = Token.objects.create(user=user)
        try:
            return json.loads(
                b"".join(
                    sdk.projects.exports.download_sync(
                        project_id,
                        export_type="JSON",
                        download_all_tasks=True,
                        download_resources=False,
                    )
                )
            )
        finally:
            fixture_token.delete()

    exported = native_export()
    assert exported[0]["annotations"][0]["id"] == annotation.id
    fetched = sdk.tasks.get(task_id)
    segments = decoded_segments(annotation.result, bundle["proxy"], episodes[0])
    assert segments[0]["label"] == "reach"
    assert len(fetched.annotations) == 1
    snapshot = annotation_snapshot(store, binding, exported[0], bundle["proxy"])
    assert snapshot["annotations"][0]["eligibility"].startswith("draft")
    assert not Token.objects.filter(user=user).exists()
    Path("artifacts").mkdir(exist_ok=True)
    atomic_json(
        Path(f"artifacts/{evidence_prefix}-smoke.json"),
        {
            "evidence_kind": evidence_kind,
            "label_studio": "1.23.2",
            "sdk": "2.1.2",
            "project_id": project_id,
            "task_id": task_id,
            "annotation_id": annotation.id,
            "prediction_count": len(task.predictions or []),
            "segments": segments,
            "api_credentials": (
                "no persistent credential; SDK session; export fixture token revoked"
            ),
            "real_droid": real,
            "frame_semantics_browser_verified": False,
        },
    )
    print("Label Studio SDK import/correction/export smoke passed")
    if "--browser" in sys.argv or "--sync-contract" in sys.argv:
        fresh = sdk.projects.import_tasks(
            project_id, request=[bundle["task"]], return_task_ids=True
        )
        browser_task_id = fresh.task_ids[0]
        browser_task = sdk.tasks.get(browser_task_id)
        print(
            "Fresh browser task",
            browser_task.id,
            "predictions",
            len(browser_task.predictions or []),
        )
        assert len(browser_task.predictions or []) == 1
        import threading
        from socketserver import ThreadingMixIn
        from wsgiref.simple_server import WSGIServer, make_server

        class ThreadedServer(ThreadingMixIn, WSGIServer):
            daemon_threads = True

        server = make_server("127.0.0.1", 8134, get_wsgi_application(), server_class=ThreadedServer)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            environment = {
                **os.environ,
                "LS_FIXTURE_COOKIES": json.dumps(cookies),
                "LS_FIXTURE_PROJECT": str(project_id),
                "LS_FIXTURE_TASK": str(browser_task_id),
                "LS_FIXTURE_SEGMENTS": str(len(proposed)),
                "LS_FIXTURE_REAL_DROID": "1" if real else "0",
                "LS_FIXTURE_TITLE": episodes[0]["task"],
                "LS_FIXTURE_CHANNELS": str(len(bundle["proxy"]["columns"])),
                "LS_FIXTURE_FPS": str(bundle["proxy"]["fps"]),
                "PLAYWRIGHT_BROWSERS_PATH": os.environ.get(
                    "PLAYWRIGHT_BROWSERS_PATH", str(Path(".cache/playwright").resolve())
                ),
            }
            if os.environ.get("LS_TASK_LOADING_CONTRACT") == "1":
                environment["LS_LOADING_CASES"] = json.dumps(
                    [publish_bundle(sdk, store, bundle) for _ in range(2)]
                )
            if "--sync-contract" in sys.argv:
                from copy import deepcopy

                cases = []
                for variant in ("official-time", "numeric", "collapsed", "open-resized", "themed"):
                    data = deepcopy(bundle["task"]["data"])
                    sensor = (
                        '<TimeSeries name="sensors" value="$sensors" valueType="json" '
                        'sync="episode" timeColumn="time" overviewWidth="100%" __FORMAT__>'
                        '<Channel column="gripper_0" height="100"/></TimeSeries>'
                    )
                    if variant == "official-time":
                        import csv

                        data["sensors"]["time"] = [
                            f"00:00:{t:09.6f}" for t in data["sensors"]["time"]
                        ]
                        example = store.root / "review_proxy" / "sync-contract.csv"
                        with example.open("w", newline="") as output:
                            writer = csv.writer(output)
                            writer.writerow(data["sensors"])
                            writer.writerows(zip(*data["sensors"].values(), strict=True))
                        data["sensors"] = (
                            "/data/local-files/?d=robotics/review_proxy/sync-contract.csv"
                        )
                        sensor = sensor.replace('valueType="json"', 'valueType="url"')
                        sensor = sensor.replace(
                            "__FORMAT__", 'timeFormat="%H:%M:%S.%f" timeDisplayFormat="%H:%M:%S.%f"'
                        )
                    else:
                        sensor = sensor.replace("__FORMAT__", 'timeDisplayFormat=".2f"')
                    if variant in ("collapsed", "open-resized"):
                        sensor = f'<Collapse><Panel value="Signals">{sensor}</Panel></Collapse>'
                    if variant == "open-resized":
                        sensor = sensor.replace("<Collapse>", '<Collapse open="true">')
                    config = (
                        '<View><Video name="video" value="$video" sync="episode" frameRate="12"/>'
                        f'{sensor}<TimeSeriesLabels name="labels" toName="sensors">'
                        '<Label value="A"/><Label value="B"/></TimeSeriesLabels></View>'
                    )
                    if variant == "themed":
                        config = bundle["label_config"]
                    project = sdk.projects.create(title=variant, label_config=config)
                    sdk.import_storage.local.create(
                        project=project.id,
                        path=str(store.root / "review_proxy"),
                        use_blob_urls=True,
                    )
                    task = sdk.projects.import_tasks(
                        project.id, request=[{"data": data}], return_task_ids=True
                    )
                    cases.append({"name": variant, "project": project.id, "task": task.task_ids[0]})
                environment["LS_SYNC_CASES"] = json.dumps(cases)
                subprocess.run(
                    ["node", "scripts/labelstudio_sync.mjs"],
                    env=environment,
                    check=True,
                    timeout=90,
                )
                return
            subprocess.run(
                ["node", "scripts/labelstudio_browser.mjs"],
                env=environment,
                check=True,
                timeout=150,
            )
            exported_task = next(task for task in native_export() if task["id"] == browser_task_id)
            prediction = sdk.predictions.get(exported_task["predictions"][0])
            assert prediction.result == proposed
            assert prediction.model_version == bundle["trusted_origin"]["suggestion_run"]
            snapshot = annotation_snapshot(
                store,
                bind_task(bundle, project_id, browser_task_id),
                exported_task,
                bundle["proxy"],
            )
            first = snapshot["annotations"][0]["segments"][0]
            assert first == {
                "start_s": 0,
                "end_s": 6 / bundle["proxy"]["fps"],
                "label": proposed[0]["value"]["timelinelabels"][0],
            }
            boundary = snapshot["annotations"][0]["boundary_evidence"][0]["end"]
            assert not boundary["is_measured_source_boundary"]
            if real:
                assert boundary["source_video_timing"] == "inferred_nominal_fps"
                assert boundary["is_source_frame_boundary"]
                assert boundary["source_bracket_width_s"] == 0
            else:
                assert boundary["source_bracket_s"] == [4 / 12, 7 / 12]
                assert boundary["source_bracket_width_s"] == 3 / 12
            assert not Token.objects.filter(user=user).exists()
            atomic_json(Path(f"artifacts/{evidence_prefix}-native-snapshot.json"), snapshot)
            evidence = json.loads(Path(f"artifacts/{evidence_prefix}-smoke.json").read_text())
            evidence.update(frame_semantics_browser_verified=True, native_export_boundary=first)
            atomic_json(Path(f"artifacts/{evidence_prefix}-smoke.json"), evidence)
            print(
                "Native temporal edit -> reload -> SDK export -> "
                "review grid and measured source brackets passed"
            )
            from robo.export import validate_export
            from robo.native_review import NativeReviewBridge
            from robo.store import Conflict

            bridge = NativeReviewBridge(sdk, store, "http://testserver")
            bridge.register(bundle, project_id, browser_task_id)
            native_id = exported_task["annotations"][0]["id"]
            episode_id = episodes[0]["id"]
            draft = bridge.refresh(
                episode_id, native_id, 0, "fixture-importer", "Native GUI correction"
            )
            assert draft["outcome"] == expected_outcome
            assert draft["quality_flags"] == expected_quality
            assert draft["native"]["reviewer_evidence"]
            assert draft["active_s"] is None and draft["wall_s"] is None
            assert bridge.refresh(episode_id, native_id, 1, "fixture-importer", "Repeat") == draft
            accepted = bridge.decide(
                episode_id, 1, "accept", "fixture-approver", "Explicit fixture approval"
            )
            selection = [{"episode_id": episode_id, "revision": 2, "split": "train"}]
            release = directory / "accepted-export"
            result = bridge.export(selection, release)
            assert bridge.export(selection, release)["manifest"] == result["manifest"]
            record = validate_export(release)[0]
            assert record["annotation"] == accepted
            assert record["annotation"]["native"]["boundary_evidence"][0]["end"] == boundary
            # Transfer only source media/manifests and the accepted artifact to the
            # isolated LeRobot runtime. Never copy the application's database/session.
            if fixture_path := os.environ.get("ROBO_NATIVE_EXPORT_FIXTURE"):
                fixture_path = Path(fixture_path)
                fixture_path.mkdir(parents=True, exist_ok=False)
                shutil.copytree(release, fixture_path / "accepted")
                source = store.root / record["episode"]["media"]
                target = fixture_path / "source" / record["episode"]["media"]
                target.parent.mkdir(parents=True)
                shutil.copyfile(source, target)
                shutil.copyfile(source.parent / "episode.json", target.parent / "episode.json")
            changed = json.loads(json.dumps(exported_task["annotations"][0]["result"]))
            changed[0]["value"]["ranges"][0]["end"] = 5
            sdk.annotations.update(native_id, result=changed)
            try:
                bridge.export(selection, release)
            except Conflict:
                pass
            else:
                raise AssertionError("new export accepted a stale native revision")
            assert validate_export(release)[0] == record
            revised = bridge.refresh(
                episode_id, native_id, 2, "fixture-importer", "Changed native content"
            )
            assert revised["status"] == "draft" and revised["revision"] == 3
            rejected = bridge.decide(
                episode_id, 3, "reject", "fixture-approver", "Fixture rejection"
            )
            assert (
                bridge.refresh(episode_id, native_id, 4, "fixture-importer", "Repeat") == rejected
            )
            atomic_json(
                Path(f"artifacts/{evidence_prefix}-approval.json"),
                {
                    "evidence_kind": evidence_kind,
                    "native_draft": draft,
                    "accepted_revision": accepted,
                    "export": result,
                    "changed_native_content_blocks_new_export": True,
                    "historical_export_remains_valid": True,
                    "rejected_refresh_is_idempotent": True,
                    "active_human_time_measured": False,
                },
            )
            print(
                "Native SDK refresh -> explicit approval -> immutable export -> "
                "edit invalidation -> rejection passed"
            )
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", action="store_true")
    parser.add_argument("--sync-contract", action="store_true")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--semantic-pipeline", type=Path, help="Local DROID TFDS pilot directory")
    parser.add_argument("--source", type=Path, help="Converted local real DROID source.json")
    args = parser.parse_args()
    if args.semantic and not args.source:
        parser.error("--semantic requires the converted real DROID --source")
    if args.source and (
        not (args.browser or args.batch or args.semantic or args.semantic_pipeline)
        or args.sync_contract
    ):
        parser.error(
            "real DROID --source requires --browser, --batch or --semantic; "
            "excludes --sync-contract"
        )
    Path("artifacts").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="labelstudio-", dir=Path("artifacts").resolve()
    ) as temporary:
        try:
            run(Path(temporary), args.source)
        except Exception as exc:
            # API exceptions can contain session cookies in their headers.
            if hasattr(exc, "status_code"):
                print(type(exc).__name__, exc.status_code, getattr(exc, "body", None))
                raise SystemExit(1) from None
            raise
