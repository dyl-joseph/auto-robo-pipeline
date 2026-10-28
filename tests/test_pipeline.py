import asyncio
import json
import shutil
import subprocess

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from robo.analysis import latest_run, queue, suggest
from robo.api import create_app
from robo.core import Edit, Source, atomic_json
from robo.evaluation import budget_metrics, evaluate, temporal_metrics
from robo.export import export_dataset, validate_export
from robo.ingest import import_episode
from robo.store import Conflict


def edit(revision=0, **changes):
    return Edit.model_validate(
        {
            "expected_revision": revision,
            "actor": "test-reviewer",
            "rationale": "Automated fixture verification; not human review evidence",
            "segments": [{"start_s": 0, "end_s": 1, "label": "reach"}],
            **changes,
        }
    )


def accepted(store, episode_id):
    store.commit(episode_id, edit())
    return store.commit(episode_id, edit(1), "accept")


def test_import_identity_snapshot_missing_and_relocation(fixture, tmp_path):
    store, manifest, body = fixture
    first = import_episode(manifest, store)
    assert import_episode(manifest, store) == first
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy(manifest, moved / manifest.name)
    shutil.copy(manifest.parent / body["video"], moved / body["video"])
    assert import_episode(moved / manifest.name, store)["id"] == first["id"]
    assert first["validation"]["warnings"] == ["missing_state", "missing_action", "missing_sensor"]
    assert len(first["original_pts"]) == 20
    assert len(store.episodes()) == 1
    body["task"] = "conflicting task"
    atomic_json(manifest, body)
    with pytest.raises(Conflict, match="different content"):
        import_episode(manifest, store)


def test_variable_pts_and_partial_publication_recovery(fixture):
    store, manifest, body = fixture
    video = manifest.parent / body["video"]
    video.unlink()
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=10:duration=2",
            "-vf",
            "select=not(eq(n\\,5)+eq(n\\,6))",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            str(video),
        ],
        check=True,
    )
    first = import_episode(manifest, store)
    gaps = [b - a for a, b in zip(first["timestamps_s"], first["timestamps_s"][1:], strict=False)]
    assert max(gaps) > 0.25
    assert "video_timestamp_gap" in first["validation"]["warnings"]
    with store.connect() as db:
        db.execute("DELETE FROM episodes")
    assert import_episode(manifest, store) == first  # rename completed before interrupted DB commit


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "regression", "units", "offset", "nan"])
def test_invalid_sources_fail(fixture, mutation):
    store, manifest, body = fixture
    if mutation == "missing":
        body["video"] = "absent.mp4"
    elif mutation == "corrupt":
        (manifest.parent / body["video"]).write_bytes(b"not a movie")
    else:
        body["streams"] = [
            {
                "name": "joint",
                "modality": "state",
                "units": "rad",
                "timestamps_s": [0, 1],
                "values": [[0], [1]],
                "clock_evidence": "fixture clock",
            }
        ]
        stream = body["streams"][0]
        if mutation == "regression":
            stream["timestamps_s"] = [1, 0]
        elif mutation == "units":
            stream["units"] = "degrees_guessed"
        elif mutation == "offset":
            stream["offset_s"] = 20
        else:
            stream["values"][0] = [float("nan")]
    manifest.write_text(json.dumps(body))
    with pytest.raises((ValueError, ValidationError)):
        import_episode(manifest, store)
    assert store.episodes() == []


def test_alignment_masks_preserve_original_samples(fixture):
    store, manifest, body = fixture
    body["streams"] = [
        {
            "name": "force",
            "modality": "sensor",
            "units": "N",
            "timestamps_s": [0, 1],
            "values": [[1], [2]],
            "offset_s": 0.03,
            "tolerance_s": 0.05,
            "clock_evidence": "fixture shared clock plus 30ms offset",
        }
    ]
    atomic_json(manifest, body)
    episode = import_episode(manifest, store)
    alignment = episode["validation"]["alignment"][0]
    assert alignment["nearest_indices"][0] == 0
    assert alignment["nearest_indices"][3] is None
    assert episode["streams"] == Source.model_validate(body).model_dump()["streams"]


def test_stale_edit_acceptance_invalidation_and_audit(episode):
    store, e = episode
    accepted(store, e["id"])
    with pytest.raises(Conflict):
        store.commit(e["id"], edit())
    latest = store.commit(
        e["id"], edit(2, segments=[{"start_s": 0.1, "end_s": 1, "label": "grasp"}])
    )
    assert latest["status"] == "draft"
    assert [r["status"] for r in store.history(e["id"])] == ["draft", "accepted", "draft"]
    with pytest.raises(ValueError, match="save changed labels"):
        store.commit(e["id"], edit(3), "accept")


@pytest.mark.parametrize(
    "changes",
    [
        {"segments": [{"start_s": 1, "end_s": 3, "label": "grasp"}]},
        {
            "segments": [
                {"start_s": 0, "end_s": 1, "label": "grasp"},
                {"start_s": 0.5, "end_s": 1.5, "label": "hold"},
            ]
        },
    ],
)
def test_interval_validation(episode, changes):
    store, e = episode
    with pytest.raises(ValueError):
        store.commit(e["id"], edit(**changes))


def test_analysis_cache_immutable_and_manual_override(episode):
    store, e = episode
    run = suggest(store, e["id"])
    path = store.root / "runs" / e["id"] / f"{run['id']}.json"
    original = path.read_bytes()
    assert suggest(store, e["id"]) == run
    assert len(run["evidence"]) == len(e["timestamps_s"])
    store.commit(e["id"], edit(suggestion_run=run["id"], condition="assisted"))
    assert path.read_bytes() == original
    run["segments"][0]["label"] = "recover"
    atomic_json(path, run)
    with pytest.raises(ValueError, match="hash mismatch"):
        latest_run(store, e["id"])


def test_export_gates_hashes_idempotence_and_consumer(episode, tmp_path):
    store, e = episode
    selection = [{"episode_id": e["id"], "revision": 0, "split": "train"}]
    out = tmp_path / "export"
    with pytest.raises(ValueError, match="accepted"):
        export_dataset(store, selection, out)
    accepted(store, e["id"])
    selection[0]["revision"] = 2
    manifest = export_dataset(store, selection, out)
    assert export_dataset(store, selection, out)["version"] == manifest["version"]
    rows = validate_export(out)
    assert rows[0]["annotation"]["segments"][0]["label"] == "reach"
    assert len(rows[0]["audit_trail"]) == 2
    other = export_dataset(store, selection, tmp_path / "other")
    assert manifest == other
    store.commit(e["id"], edit(2))
    with pytest.raises(ValueError, match="accepted"):
        export_dataset(store, selection, tmp_path / "stale")
    (out / "episodes.jsonl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_export(out)


def test_duplicate_and_session_leakage(fixture, tmp_path):
    store, manifest, body = fixture
    a = import_episode(manifest, store)
    body.update(source_id="another", session="separate")
    atomic_json(manifest, body)
    b = import_episode(manifest, store)
    for e in (a, b):
        accepted(store, e["id"])
    selection = [
        {"episode_id": e["id"], "revision": 2, "split": split}
        for e, split in ((a, "train"), (b, "test"))
    ]
    with pytest.raises(ValueError, match="leakage"):
        export_dataset(store, selection, tmp_path / "leaking")
    assert all("exact_duplicate_group" in row["reasons"] for row in queue(store))


def test_matching_and_error_budget():
    truth = [{"start_s": 0, "end_s": 1, "label": "reach"}]
    predicted = truth + [{"start_s": 1, "end_s": 2, "label": "hold"}]
    result = temporal_metrics(predicted, truth)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert result["boundary_median_s"] == 0
    report = evaluate(
        {
            "evidence_kind": "synthetic_fixture",
            "episodes": [{"id": "a", "duration_s": 2, "predicted": predicted, "reference": truth}],
        }
    )
    assert report["human_efficiency"] is None
    assert report["thresholds"]["0.5"]["micro_f1"] == pytest.approx(2 / 3)
    budget = budget_metrics(
        [{"id": str(i), "priority": 10 - i, "review_s": 2, "erroneous": i < 2} for i in range(4)], 4
    )
    assert budget["prioritized"]["recall"] == 1
    assert budget["prioritized"]["missed_errors"] == 0


def test_api_complete_review_export_range_and_conflict(episode):
    store, e = episode

    async def scenario():
        async with AsyncClient(
            transport=ASGITransport(app=create_app(store.root)), base_url="http://testserver"
        ) as client:
            assert (await client.get("/")).status_code == 200
            assert (await client.get("/api/queue")).json()["items"][0]["id"] == e["id"]
            response = await client.get(f"/api/media/{e['id']}", headers={"Range": "bytes=0-99"})
            assert response.status_code == 206 and len(response.content) == 100
            url = f"/api/reviews/{e['id']}/save"
            body = edit().model_dump()
            assert (await client.post(url, json=body)).status_code == 200
            assert (await client.post(url, json=body)).status_code == 409
            body["expected_revision"] = 1
            assert (
                await client.post(f"/api/reviews/{e['id']}/accept", json=body)
            ).status_code == 200
            selection = [{"episode_id": e["id"], "revision": 2, "split": "train"}]
            response = await client.post("/api/export", json={"selection": selection})
            assert response.status_code == 200, response.text
            assert (await client.get(response.json()["download"])).status_code == 200
            assert (
                await client.post(url, json=body, headers={"Origin": "http://evil.example"})
            ).status_code == 403
            assert (
                await client.get("/api/queue", headers={"Host": "evil.example"})
            ).status_code == 403

    asyncio.run(scenario())


def test_uncertainty_exclusion_and_restoring_never_accepts(episode):
    store, e = episode
    store.commit(e["id"], edit(quality_flags=["uncertain"]), "uncertain")
    with pytest.raises(ValueError, match="uncertainty"):
        store.commit(e["id"], edit(1, quality_flags=["uncertain"]), "accept")
    assert store.commit(e["id"], edit(1), "exclude")["status"] == "excluded"
    assert store.commit(e["id"], edit(2), "restore")["status"] == "draft"


def test_model_smoke_and_bounded_sampling(episode):
    from pathlib import Path

    from robo.perception import active_perception, perceive

    model = Path("models/local/mobilenet-ssd")
    if not (model / "mobilenet_iter_73000.caffemodel").is_file():
        pytest.skip("optional pinned pretrained weights not downloaded")
    store, e = episode
    run = perceive(store, e["id"], model, max_frames=2)
    assert run["sampled_frames"] == 2 and run["truncated"]
    assert run == perceive(store, e["id"], model, max_frames=2)
    assert active_perception(store, e["id"]) == run
    assert run["config"]["model"]["commit"] == "bb17b6c3eef36d80be441ae8e5339be66e8e3b7a"


def test_study_timing_corrections_and_agreement():
    from robo.evaluation import study_metrics

    segment = {"start_s": 0, "end_s": 1, "label": "reach"}
    protocol = {
        "evidence_kind": "synthetic_fixture",
        "episodes": [
            {
                "id": "a",
                "duration_s": 1,
                "group": "a",
                "reviewer": "fixture-a",
                "condition": "manual",
                "accepted": True,
                "predicted": [segment],
                "reference": [segment],
                "reviewer_a": [segment],
                "reviewer_b": [segment],
                "review_history": [{"active_s": 10, "wall_s": 12}, {"active_s": 5, "wall_s": 5}],
            },
            {
                "id": "b",
                "duration_s": 1,
                "group": "b",
                "reviewer": "fixture-b",
                "condition": "assisted",
                "accepted": True,
                "predicted": [segment],
                "reference": [segment],
                "review_history": [{"active_s": 20, "wall_s": 25}],
            },
        ],
    }
    report = study_metrics(protocol)
    assert report["conditions"]["manual"]["active_s"] == 15
    assert report["time_reduction_point_estimate"] < 0  # correction cost can erase any gain
    assert report["agreement_at_iou_0.5"][0]["f1"] == 1
    assert report["conditions"]["manual"]["seconds_per_segment_ci95"] is None


def test_source_and_export_tamper_rejected(episode, tmp_path):
    store, e = episode
    accepted(store, e["id"])
    source = store.root / e["media"]
    source.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        export_dataset(
            store, [{"episode_id": e["id"], "revision": 2, "split": "train"}], tmp_path / "bad"
        )


def test_concurrent_review_has_one_winner(episode):
    from concurrent.futures import ThreadPoolExecutor

    store, e = episode

    def attempt():
        try:
            return store.commit(e["id"], edit())["revision"]
        except Conflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(lambda _: attempt(), range(2)))
    assert sorted(map(str, values)) == ["1", "conflict"]
    assert len(store.history(e["id"])) == 1


def test_curation_reproducible_and_duplicate_aware(fixture):
    from robo.curation import curate

    store, manifest, body = fixture
    import_episode(manifest, store)
    body["source_id"] = "duplicate"
    atomic_json(manifest, body)
    import_episode(manifest, store)
    first = curate(store, 2)
    assert first["actual_count"] == 1
    assert first == curate(store, 2)
    assert curate(store, 2, "random", 123) == curate(store, 2, "random", 123)


def test_unselected_episode_cannot_bridge_leaking_splits(fixture, tmp_path):
    store, manifest, body = fixture
    a = import_episode(manifest, store)
    body.update(source_id="bridge", video="second.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:size=160x90:rate=10:duration=2",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            str(manifest.parent / "second.mp4"),
        ],
        check=True,
    )
    atomic_json(manifest, body)
    import_episode(manifest, store)
    body.update(source_id="far-end", session="session-b")
    atomic_json(manifest, body)
    c = import_episode(manifest, store)
    for e in (a, c):
        accepted(store, e["id"])
    selection = [
        {"episode_id": a["id"], "revision": 2, "split": "train"},
        {"episode_id": c["id"], "revision": 2, "split": "test"},
    ]
    with pytest.raises(ValueError, match="leakage"):
        export_dataset(store, selection, tmp_path / "indirect")


def test_reject_browser_incompatible_codec(fixture):
    store, manifest, body = fixture
    video = manifest.parent / "other.avi"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=blue:size=160x90:rate=10:duration=1",
            "-c:v",
            "mpeg4",
            "-threads",
            "1",
            str(video),
        ],
        check=True,
    )
    body["video"] = video.name
    atomic_json(manifest, body)
    with pytest.raises(ValueError, match="H.264 in MP4"):
        import_episode(manifest, store)


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        "bad",
        42,
        {"selection": None},
        {"selection": [None]},
        {"selection": [{"episode_id": [], "revision": 2, "split": "train"}]},
        {"selection": [{"episode_id": "a" * 24, "revision": "2", "split": "train"}]},
        {"selection": [{"episode_id": "a" * 24, "revision": True, "split": "train"}]},
    ],
)
def test_export_api_rejects_malformed_bodies(episode, body):
    store, _ = episode

    async def scenario():
        async with AsyncClient(
            transport=ASGITransport(app=create_app(store.root)), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/api/export",
                content=json.dumps(body),
                headers={"content-type": "application/json"},
            )
            assert response.status_code == 422, response.text

    asyncio.run(scenario())


def test_review_pins_evidence_and_reexport_ignores_active_changes(episode, tmp_path):
    from pathlib import Path

    from robo.core import digest
    from robo.perception import perceive

    model = Path("models/local/mobilenet-ssd")
    if not (model / "mobilenet_iter_73000.caffemodel").exists():
        pytest.skip("optional pretrained model weights required")
    store, e = episode
    original = suggest(store, e["id"])
    perception = perceive(store, e["id"], model, max_frames=1)
    origins = {
        "suggestion_run": original["id"],
        "perception_run": perception["id"],
        "condition": "assisted",
    }
    store.commit(e["id"], edit(**origins))
    store.commit(e["id"], edit(1, **origins), "accept")
    selection = [{"episode_id": e["id"], "revision": 2, "split": "train"}]
    before = export_dataset(store, selection, tmp_path / "before")
    newer = json.loads(json.dumps(original))
    newer["id"] = "f" * 24
    newer["config"]["fixture_version"] = 2
    newer["artifact_hash"] = digest({k: v for k, v in newer.items() if k != "artifact_hash"})
    atomic_json(store.root / "runs" / e["id"] / f"{newer['id']}.json", newer)
    atomic_json(store.root / "runs" / e["id"] / "active.json", {"id": newer["id"]})
    perceive(store, e["id"], model, max_frames=2)
    after = export_dataset(store, selection, tmp_path / "after")
    assert before == after
    record = validate_export(tmp_path / "after")[0]
    assert record["annotation"]["suggestion_run"] == original["id"]
    assert record["perception"]["id"] == perception["id"]
    assert record["suggestion_artifact_hash"] == original["artifact_hash"]
    restored = store.commit(e["id"], edit(2, **origins), "restore")
    assert restored["status"] == "draft" and restored["perception_run"] == perception["id"]
