import json
import shutil
from copy import deepcopy

import pytest

from robo.core import atomic_json, file_hash
from robo.pipeline import accepted_selection, pipeline_lock, read_report, run_batch
from robo.preflight import preflight


def inbox_for(fixture):
    store, manifest, body = fixture
    inbox = manifest.parent / "inbox"
    inbox.mkdir()
    shutil.copyfile(manifest.parent / body["video"], inbox / body["video"])
    atomic_json(inbox / "source.json", body)
    return store, inbox, body


def test_batch_isolates_bad_inputs_deduplicates_and_repeats(fixture):
    store, inbox, body = inbox_for(fixture)
    atomic_json(inbox / "duplicate.source.json", body)
    missing = deepcopy(body)
    del missing["license"]
    atomic_json(inbox / "missing.source.json", missing)
    corrupt = {**body, "source_id": "corrupt", "video": "corrupt.mp4"}
    (inbox / "corrupt.mp4").write_bytes(b"not a video")
    atomic_json(inbox / "corrupt.source.json", corrupt)
    before = {p.name: file_hash(p) for p in inbox.iterdir()}
    first = run_batch(inbox, store)
    assert first["metrics"]["input_records"] == 4
    assert first["metrics"]["prepared_records"] == 2
    assert first["metrics"]["duplicate_input_records"] == 1
    assert first["metrics"]["review_requests"] == 1
    assert first["metrics"]["failures_by_stage"] == {"import": 1, "validate": 1}
    assert first["metrics"]["automatic_approvals"] == 0
    assert first["metrics"]["human_active_seconds"] is None
    assert len(store.episodes()) == 1
    assert all(store.history(e["id"]) == [] for e in store.episodes())
    assert run_batch(inbox, store) == first
    assert {p.name: file_hash(p) for p in inbox.iterdir()} == before
    assert read_report(first["report_path"], store) == first
    assert first["status"] == "needs_input"
    with pytest.raises(ValueError, match="No approved episodes"):
        accepted_selection(first, store, "train")


def test_failed_stage_resumes_without_reimport_or_duplicate_history(fixture, monkeypatch):
    import robo.pipeline as pipeline

    store, inbox, _ = inbox_for(fixture)
    original = pipeline.suggest
    monkeypatch.setattr(
        pipeline, "suggest", lambda *args: (_ for _ in ()).throw(OSError("interrupted"))
    )
    failed = run_batch(inbox, store)
    assert failed["items"][0]["completed_stages"] == ["normalize", "import"]
    assert failed["items"][0]["stage"] == "suggest"
    assert len(store.episodes()) == 1
    monkeypatch.setattr(pipeline, "suggest", original)
    resumed = run_batch(inbox, store)
    assert resumed["batch_id"] == failed["batch_id"]
    assert resumed["items"][0]["completed_stages"] == ["normalize", "import", "suggest", "prepare"]
    assert resumed["metrics"]["failed_records"] == 0
    assert len(store.episodes()) == 1
    assert store.history(store.episodes()[0]["id"]) == []


def test_changed_media_is_not_hidden_by_a_completed_stage(fixture):
    store, inbox, body = inbox_for(fixture)
    report = run_batch(inbox, store)
    assert report["metrics"]["prepared_records"] == 1
    with (inbox / body["video"]).open("ab") as stream:
        stream.write(b"changed source")
    changed = run_batch(inbox, store)
    assert changed["metrics"]["failed_records"] == 1
    assert changed["review_queue"] == []
    assert "different content" in changed["items"][0]["errors"][0]["message"]


def test_conflicting_pipeline_operations_and_nested_store_are_rejected(fixture):
    store, inbox, _ = inbox_for(fixture)
    with pipeline_lock(store):
        with pytest.raises(ValueError, match="Another batch"):
            run_batch(inbox, store)
    with pytest.raises(ValueError, match="outside the inbox"):
        run_batch(inbox.parent, store)


def test_preflight_keeps_missing_tokenizer_and_mac_gates_explicit(tmp_path):
    report = preflight()
    assert len(report["pretrained_gates"]) == 3
    assert report["full_pretrained_run_verified"] is False
    truncated = tmp_path / "model.safetensors"
    truncated.write_bytes(b"x")
    cpu_evidence = tmp_path / "cpu.json"
    cpu_evidence.write_text(json.dumps({"platform": "Linux", "metal_available": False}))
    report = preflight(truncated, tmp_path, cpu_evidence)
    assert report["local_weights"] is None
    assert report["small_model_metal_evidence"] is False
    assert any("HTTP 401" in gate for gate in report["pretrained_gates"])
