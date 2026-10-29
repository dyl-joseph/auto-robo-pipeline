import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_semantic_pipeline import workflow as workflow

from robo.core import atomic_json, file_hash
from robo.semantic_runtime import StageFailure
from scripts import semantic_workflow_acceptance as acceptance


@pytest.fixture
def acceptance_fixture(workflow, monkeypatch):
    engine, runtime, sdk, body = workflow
    calls = []

    def create(task_id, *, result):
        row = sdk.save_annotation(task_id)
        row["result"] = deepcopy(result)
        return sdk.annotation(row["id"])

    def load(project, arguments, logs):
        calls.append((project, arguments))
        atomic_json(
            arguments[-1],
            {
                "official_lerobot_and_dataloader": True,
                "prediction_sha256": file_hash(engine.root / "predictions.json"),
                "test_fixture_only": True,
            },
        )

    sdk.annotations.create = create
    monkeypatch.setattr(runtime, "call", load)
    client = SimpleNamespace(
        get=lambda url: SimpleNamespace(
            status_code=200, headers={"Content-Type": "video/mp4"}, close=lambda: None
        )
    )
    path = engine.root.parent / "acceptance.json"
    return engine, runtime, sdk, body, client, path, calls


def test_no_fixture_approval_means_no_provider_or_native_mutations(acceptance_fixture):
    engine, runtime, sdk, _, client, path, calls = acceptance_fixture
    report = acceptance.exercise(engine, client, path, approve_fixture=False)
    assert report["status"] == "skipped" and report["exit_status"] == 3
    assert report["fixture_edits"] == report["fixture_approvals"] == 0
    assert runtime.calls == calls == [] and not sdk.task_rows


@pytest.mark.parametrize("split", ["train", "development"])
def test_fixture_run_reuses_predictions_and_exports_without_model_claim(acceptance_fixture, split):
    engine, runtime, sdk, _, client, path, calls = acceptance_fixture
    engine.config = engine.config.model_copy(update={"split": split})
    report = acceptance.exercise(engine, client, path, approve_fixture=True)
    assert report["status"] == "passed", report
    assert report["exit_status"] == 0
    assert report["fixture_edits"] == report["fixture_approvals"] == 1
    assert report["production_automatic_approvals"] == 0
    assert report["model_execution"]["status"] == "skipped"
    assert report["pretrained_qwen_run"] is False
    assert report["semantic_accuracy_measured"] is False
    assert report["evaluation_status"] == "blocked"
    assert len(runtime.calls) == len(calls) == sdk.import_calls == 1
    assert (path.parent / "saved-annotation.json").is_file()
    assert json.loads(path.read_text()) == report


@pytest.mark.parametrize("transport", [False, True])
def test_failed_qwen_skips_edits_approval_and_loader(acceptance_fixture, transport):
    engine, runtime, sdk, body, client, path, calls = acceptance_fixture
    engine.config = engine.config.model_copy(
        update={"provider": "qwen", "model_dir": path.parent / "models", "device": "mps"}
    )
    runtime.fail = [body["source_id"]]
    runtime.crash = transport
    report = acceptance.exercise(engine, client, path, approve_fixture=True)
    assert report["status"] == "failed" and report["exit_status"] == 2, report
    assert report["model_execution"]["status"] == "failed"
    assert report["pretrained_qwen_run"] is False
    assert report["fixture_edits"] == report["fixture_approvals"] == 0
    assert not calls and all(not t["annotations"] for t in sdk.task_rows.values())
    assert all(
        report["stages"][stage]["status"] == "skipped"
        for stage in ("correction", "fixture_approval", "export", "official_loader")
    )
    predictions = json.loads((engine.root / "predictions.json").read_text())
    assert predictions["episodes"][0]["error"]
    assert "Qwen/" in predictions["producer"]


@pytest.mark.parametrize("kind", ["model", "synthetic_fixture"])
def test_qwen_evidence_requires_model_provenance(acceptance_fixture, monkeypatch, kind):
    engine, runtime, sdk, _, client, path, calls = acceptance_fixture
    engine.config = engine.config.model_copy(
        update={"provider": "qwen", "model_dir": path.parent / "models", "device": "mps"}
    )
    predict = runtime.predict

    def with_asset_metadata(config, plan, store, output):
        predict(config, plan, store, output)
        run = json.loads((output / "run.json").read_text())
        predictions_path = output / "predictions.json"
        predictions = json.loads(predictions_path.read_text())
        predictions["episodes"][0]["segments"][0]["provenance"]["kind"] = kind
        atomic_json(predictions_path, predictions)
        run["files"]["predictions.json"] = file_hash(predictions_path)
        pin = acceptance.PIN
        run["configuration"] = {
            "assets": {
                "model_id": pin["model_id"],
                "revision": pin["revision"],
                "weight_bytes": pin["files"]["model.safetensors"]["bytes"],
                "files_sha256": {"model.safetensors": pin["files"]["model.safetensors"]["sha256"]},
                "device": "mps:0",
            }
        }
        atomic_json(output / "run.json", run)

    monkeypatch.setattr(runtime, "predict", with_asset_metadata)
    report = acceptance.exercise(engine, client, path, approve_fixture=True)
    if kind == "model":
        # Controlled metadata tests report propagation, not actual pretrained execution.
        assert report["test_fixture_only"]
        assert report["status"] == "passed" and report["pretrained_qwen_run"] is True
        assert report["model_execution"]["status"] == "passed"
        assert report["stages"]["official_loader"]["status"] == "passed"
        return
    assert report["status"] == "failed" and report["exit_status"] == 2
    assert "execution evidence" in report["stages"]["workflow"]["reason"]
    assert report["model_execution"]["status"] == "failed"
    assert not report["pretrained_qwen_run"]
    assert report["fixture_approvals"] == 0 and not calls
    assert all(not task["annotations"] for task in sdk.task_rows.values())


@pytest.mark.parametrize("failure", ["save", "prediction_mutation", "loader"])
def test_native_or_loader_failure_cannot_pass(acceptance_fixture, monkeypatch, failure):
    engine, runtime, sdk, _, client, path, calls = acceptance_fixture
    create = sdk.annotations.create

    def broken_save(task_id, *, result):
        if failure == "save":
            raise ValueError("Controlled native save failure")
        annotation = create(task_id, result=result)
        sdk.task_rows[task_id]["predictions"][0]["score"] = 0.999
        return annotation

    def broken_loader(*args):
        raise StageFailure("Controlled official loader failure")

    if failure == "loader":
        monkeypatch.setattr(runtime, "call", broken_loader)
    else:
        sdk.annotations.create = broken_save
    report = acceptance.exercise(engine, client, path, approve_fixture=True)
    assert report["status"] == "failed" and report["exit_status"] == 2
    stage = "official_loader" if failure == "loader" else "correction"
    assert report["stages"][stage]["status"] == "failed"
    assert report["fixture_approvals"] == (1 if failure == "loader" else 0)
    if failure == "loader":
        assert report["stages"]["export"]["status"] == "passed"
        assert report["export"]["status"] == "validated"
    else:
        assert not calls and report["stages"]["export"]["status"] == "skipped"


def test_missing_environment_has_report_and_no_backend(workflow, monkeypatch):
    engine, _, _, _ = workflow
    output = engine.root.parent / "unavailable"
    engine.config = engine.config.model_copy(
        update={"provider": "qwen", "model_dir": output.parent / "models", "device": "mps"}
    )

    def missing(*args):
        raise acceptance.Unavailable("Pinned Qwen assets are missing")

    def forbidden(*args):
        pytest.fail("Backend must not start when prerequisites are unavailable")

    monkeypatch.setattr(acceptance, "check_environment", missing)
    monkeypatch.setattr(acceptance, "authenticated_backend", forbidden)
    report = acceptance.run(engine.config, output, approve_fixture=True)
    assert report["status"] == "unavailable" and report["exit_status"] == 3
    assert report["stages"]["workflow"]["status"] == "skipped"
    assert report["model_execution"]["status"] == "skipped"
    assert json.loads((output / "report.json").read_text()) == report
    with pytest.raises(ValueError, match="fresh"):
        acceptance.run(engine.config, output, approve_fixture=True)


@pytest.mark.parametrize("missing", ["runtime", "assets", "mps"])
def test_preflight_unavailable_never_loads_model(workflow, tmp_path, monkeypatch, missing):
    engine, runtime, _, _ = workflow
    root, source, model = tmp_path / "repo", tmp_path / "rlds", tmp_path / "models"
    for project in (".", "integrations/droid", "integrations/lerobot", "integrations/semantic"):
        if missing != "runtime":
            python = root / project / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
    source.mkdir()
    for name in ("dataset_info.json", "features.json"):
        (source / name).touch()
    model.mkdir()
    if missing != "assets":
        for name in acceptance.PIN["files"]:
            (model / name).touch()
    config = engine.config.model_copy(
        update={"provider": "qwen", "model_dir": model, "device": "mps"}
    )
    monkeypatch.setattr(acceptance, "REPO", root)
    monkeypatch.setattr(acceptance, "which", lambda name: "/fixture/" + name)

    def device_probe(project, arguments, logs):
        assert missing == "mps" and arguments[-1] == "mps"
        atomic_json(logs / "stdout.log", {"available": False})

    def no_model(*args):
        pytest.fail("Model assets must not be loaded on an unavailable device")

    monkeypatch.setattr(runtime, "call", device_probe)
    monkeypatch.setattr(acceptance, "model_preflight", no_model)
    with pytest.raises(acceptance.Unavailable):
        acceptance.check_environment(config, tmp_path, runtime)


def test_run_guard_precedes_runtime_setup(workflow, monkeypatch):
    engine, _, _, _ = workflow

    def forbidden(*args):
        pytest.fail("No fixture permission must stop before runtime setup")

    monkeypatch.setattr(acceptance, "check_environment", forbidden)
    report = acceptance.run(engine.config, engine.root.parent / "guarded", approve_fixture=False)
    assert report["exit_status"] == 3 and report["fixture_approvals"] == 0


@pytest.mark.parametrize("status,code", [("passed", 0), ("failed", 2), ("unavailable", 3)])
def test_cli_propagates_acceptance_status(workflow, monkeypatch, capsys, status, code):
    engine, _, _, _ = workflow
    monkeypatch.setattr(
        "sys.argv",
        [
            "acceptance",
            "--provider",
            "baseline",
            "--source",
            str(engine.config.source),
            "--output",
            str(engine.root.parent / "cli"),
            "--approve-fixture",
        ],
    )
    monkeypatch.setattr(
        acceptance,
        "run",
        lambda *a, **kw: {
            "status": status,
            "exit_status": code,
            "model_execution": {"status": "skipped"},
        },
    )
    with pytest.raises(SystemExit) as result:
        acceptance.main()
    assert result.value.code == code
    assert json.loads(capsys.readouterr().out)["status"] == status
