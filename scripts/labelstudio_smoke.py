"""Exercise the real Label Studio application and official SDK in a disposable local database.

Uses an unusable-password fixture user and session. Native-export fixture tokens are revoked.
"""

import json
import os
import sys
import tempfile
from pathlib import Path


def run(directory: Path):
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
    # The supported feature-flag surface disables Compare All in this single-task
    # harness: 1.23.2 leaves its per-annotation sync manager uninitialized on mount.
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

    episodes = make_demo(directory / "robotics", count=1)
    store = Store(directory / "robotics")
    # Deliberately missing source frames: review boundaries remain on a known grid,
    # and exported evidence must expose the gap in the measured source clock.
    import subprocess

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
    # Local files are beneath the fixture's document root, not the filesystem root.
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
        Path("artifacts/labelstudio-smoke.json"),
        {
            "evidence_kind": "synthetic_fixture",
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
            "real_droid": False,
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
                "PLAYWRIGHT_BROWSERS_PATH": os.environ.get(
                    "PLAYWRIGHT_BROWSERS_PATH", str(Path(".cache/playwright").resolve())
                ),
            }
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
                ["node", "scripts/labelstudio_browser.mjs"], env=environment, check=True, timeout=90
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
            assert first == {"start_s": 0, "end_s": 0.5, "label": "hold"}
            boundary = snapshot["annotations"][0]["boundary_evidence"][0]["end"]
            assert not boundary["is_measured_source_boundary"]
            assert boundary["source_bracket_s"] == [4 / 12, 7 / 12]
            assert boundary["source_bracket_width_s"] == 3 / 12
            assert not Token.objects.filter(user=user).exists()
            atomic_json(Path("artifacts/labelstudio-native-snapshot.json"), snapshot)
            evidence = json.loads(Path("artifacts/labelstudio-smoke.json").read_text())
            evidence.update(frame_semantics_browser_verified=True, native_export_boundary=first)
            atomic_json(Path("artifacts/labelstudio-smoke.json"), evidence)
            print(
                "Native temporal edit -> reload -> SDK export -> "
                "review grid and measured source brackets passed"
            )
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    Path("artifacts").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="labelstudio-", dir=Path("artifacts").resolve()
    ) as temporary:
        try:
            run(Path(temporary))
        except Exception as exc:
            # API exceptions can contain session cookies in their headers.
            if hasattr(exc, "status_code"):
                print(type(exc).__name__, exc.status_code, getattr(exc, "body", None))
                raise SystemExit(1) from None
            raise
