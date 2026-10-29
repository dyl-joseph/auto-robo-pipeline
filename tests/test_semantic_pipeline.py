import json
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from label_studio_sdk.types.annotation import Annotation
from label_studio_sdk.types.lse_task import LseTask

from robo.core import atomic_json, digest, file_hash
from robo.export import validate_export
from robo.native_review import NativeReviewBridge
from robo.semantic import EvaluationPlan, Predictions
from robo.semantic_pipeline import Configuration, Workflow, exit_status
from robo.semantic_runtime import Runtime, StageFailure


class FixtureRuntime(Runtime):
    def __init__(self, manifest, bodies, *, fail=(), crash=False):
        self.manifest, self.bodies, self.fail = manifest, bodies, fail
        self.crash = crash
        self.calls = []

    def intake(self, config, output):
        if (output / "intake.json").exists():
            return
        records = []
        for index, body in enumerate(self.bodies):
            directory = output / "episodes" / str(index)
            directory.mkdir(parents=True)
            shutil.copyfile(self.manifest.parent / body["video"], directory / body["video"])
            atomic_json(directory / "source.json", body)
            atomic_json(directory / "training-schema.json", {"fixture": True})
            files = {p.name: file_hash(p) for p in directory.iterdir()}
            atomic_json(directory / "conversion.json", {"files": files})
            records.append(
                {
                    "directory": str(index),
                    "steps": 20,
                    "conversion_sha256": file_hash(directory / "conversion.json"),
                }
            )
        atomic_json(
            output / "intake.json",
            {
                "status": "complete",
                "summary": {"published_episodes": len(records)},
                "shards": {"generated": {"records": records}},
            },
        )

    def predict(self, config, plan_path, store, output):
        plan = EvaluationPlan.model_validate_json(plan_path.read_bytes())
        spec = plan.episodes[0]
        self.calls.append(spec.episode_id)
        if self.crash:
            raise StageFailure("Controlled transport failure")
        producer = self.provider_identity(config)["producer"]
        error = "Controlled invalid model reply" if spec.dataset_episode_id in self.fail else None
        segments = (
            []
            if error
            else [
                {
                    "episode_id": spec.episode_id,
                    "start_s": 0,
                    "end_s": spec.frame_boundaries_s[-1],
                    "source_frames": {"start": 0, "end_exclusive": 20, "evidence": [0, 19]},
                    "subtask_text": "Generated motion fixture",
                    "normalized_action": "motion",
                    "object_text": None,
                    "uncertainty": 0.8,
                    "provenance": {
                        "kind": "synthetic_fixture",
                        "producer": producer,
                        "artifact_sha256": digest("controlled provider fixture"),
                    },
                }
            ]
        )
        predictions = Predictions(
            plan_sha256=plan.plan_sha256,
            run_id="controlled-provider",
            producer=producer,
            code_revision="generated",
            config_sha256=digest("fixture"),
            training_episode_ids=[],
            tuning_episode_ids=[],
            reference_access="none",
            episodes=[{"episode_id": spec.episode_id, "segments": segments, "error": error}],
        )
        atomic_json(output / "predictions.json", predictions.model_dump())
        atomic_json(
            output / "run.json",
            {
                "files": {"predictions.json": file_hash(output / "predictions.json")},
                "failed_predictions": int(error is not None),
            },
        )

    def training(self, accepted, store, schema, split, output):
        records = validate_export(accepted)
        atomic_json(
            output / "fixture.json",
            {
                "version": digest(records)[:24],
                "records": records,
                "synthetic_training_fixture": True,
            },
        )


class FixtureSDK:
    def __init__(self):
        self.project_rows, self.task_rows, self.storage_rows, self.view_rows = {}, {}, [], []
        self.import_calls = 0
        self.interrupt_import = False
        self.projects = SimpleNamespace(
            list=lambda: list(self.project_rows.values()),
            get=lambda key: self.project_rows[key],
            create=self.project,
            import_tasks=self.import_tasks,
        )
        self.tasks = SimpleNamespace(
            list=lambda project: [
                self.task(t) for t in self.task_rows if self.task_rows[t]["project"] == project
            ],
            get=self.task,
            update=lambda key, data: self.task_rows[int(key)].update(data=data),
        )
        self.annotations = SimpleNamespace(get=self.annotation)
        self.import_storage = SimpleNamespace(
            local=SimpleNamespace(
                list=lambda project: [s for s in self.storage_rows if s.project == project],
                create=lambda **kw: self.storage_rows.append(SimpleNamespace(**kw)),
            )
        )
        self.views = SimpleNamespace(
            list=lambda project: [v for v in self.view_rows if v.project == project],
            create=self.view,
        )

    def project(self, **kw):
        row = SimpleNamespace(id=len(self.project_rows) + 1, **kw)
        self.project_rows[row.id] = row
        return row

    def task(self, key):
        return LseTask.model_validate(deepcopy(self.task_rows[int(key)]))

    def annotation(self, key):
        return Annotation.model_validate(
            deepcopy(
                next(
                    a
                    for task in self.task_rows.values()
                    for a in task["annotations"]
                    if a["id"] == key
                )
            )
        )

    def import_tasks(self, project, request, return_task_ids):
        self.import_calls += 1
        key = len(self.task_rows) + 1
        task = deepcopy(request[0])
        timestamp = "2026-01-01T00:00:00Z"
        task.update(id=key, project=project, annotations=[], drafts=[])
        task["predictions"] = [
            {
                **p,
                "id": key,
                "created_at": timestamp,
                "updated_at": timestamp,
            }
            for p in task["predictions"]
        ]
        self.task_rows[key] = task
        if self.interrupt_import:
            self.interrupt_import = False
            raise RuntimeError("Response interrupted after native import")
        return SimpleNamespace(task_ids=[key])

    def view(self, **kw):
        row = SimpleNamespace(id=len(self.view_rows) + 1, **kw)
        self.view_rows.append(row)
        return row

    def save_annotation(self, task_id):
        task = self.task_rows[task_id]
        annotation = {
            "id": task_id + 100,
            "task": task_id,
            "project": task["project"],
            "completed_by": 1,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "lead_time": 0,
            "was_cancelled": False,
            "result": deepcopy(task["predictions"][0]["result"]),
        }
        task["annotations"].append(annotation)
        return annotation


@pytest.fixture
def workflow(fixture, tmp_path, monkeypatch):
    _, manifest, body = fixture
    guidelines = tmp_path / "guidelines.txt"
    guidelines.write_text("Generated workflow fixture, not gold truth")
    config = Configuration(
        source=tmp_path / "rlds",
        release="generated",
        guidelines=guidelines,
        provider="baseline",
        instance="http://testserver",
    )
    runtime = FixtureRuntime(manifest, [body])
    sdk = FixtureSDK()
    engine = Workflow(tmp_path / "workflow", config, runtime=runtime, sdk=sdk)
    monkeypatch.setattr(
        "robo.semantic_pipeline.validate_training_export",
        lambda path: json.loads((path / "fixture.json").read_text()),
    )
    return engine, runtime, sdk, body


def accept(engine, item):
    return NativeReviewBridge(engine.sdk, engine.store, engine.config.instance).decide(
        item["episode_id"],
        item["revision"],
        "accept",
        "fixture-approver",
        "Explicit automated test decision; not a human accuracy result",
    )


def test_resumes_prediction_publication_review_and_versioned_export(workflow):
    engine, runtime, sdk, _ = workflow
    sdk.interrupt_import = True
    first = engine.run()
    assert first["status"] == "blocked" and exit_status(first) == 2
    assert len(runtime.calls) == sdk.import_calls == 1
    second = engine.run()
    assert second["status"] == "awaiting_review" and exit_status(second) == 3
    assert len(runtime.calls) == sdk.import_calls == 1
    assert len(engine.store.episodes()) == 1
    assert second["evaluation"]["status"] == "blocked"
    assert second["metrics"]["automatic_approvals"] == 0
    original_predictions = (engine.root / "predictions.json").read_bytes()
    annotation = sdk.save_annotation(1)
    draft = engine.run()
    assert draft["status"] == "awaiting_approval" and draft["items"][0]["revision"] == 1
    assert engine.run()["items"][0]["revision"] == 1
    accept(engine, draft["items"][0])
    completed = engine.run()
    assert completed["status"] == "complete" and exit_status(completed) == 0
    accepted = Path(completed["export"]["accepted"])
    records = validate_export(accepted)
    assert records[0]["annotation"]["native"]["semantic_review"]["complete"]
    assert records[0]["split"] == "development"
    assert engine.run()["export"] == completed["export"]
    assert len(runtime.calls) == sdk.import_calls == 1
    text = next(r for r in annotation["result"] if r["from_name"] == "subtask_text")
    text["value"]["text"] = ["Changed saved wording"]
    annotation["updated_at"] = "2026-01-01T00:00:01Z"
    stale = engine.run()
    assert stale["status"] == "awaiting_approval" and stale["export"]["status"] == "blocked"
    assert stale["items"][0]["revision"] == 3
    assert validate_export(accepted) == records
    accept(engine, stale["items"][0])
    updated = engine.run()
    assert updated["export"]["accepted_version"] != completed["export"]["accepted_version"]
    assert (engine.root / "predictions.json").read_bytes() == original_predictions


def test_partial_failure_deduplicates_and_preserves_failed_model_output(workflow):
    engine, runtime, sdk, body = workflow
    runtime.bodies = [body, body, {**body, "source_id": "bad"}]
    runtime.fail = {"bad"}
    report = engine.run()
    assert report["metrics"]["imported_episodes"] == 2
    assert report["metrics"]["duplicate_input_records"] == 1
    assert report["metrics"]["failed_predictions"] == 1 and exit_status(report) == 2
    assert len(runtime.calls) == sdk.import_calls == 2
    failed = next(i for i in report["items"] if i["prediction_failed"])
    assert failed["prediction_error"] == "Controlled invalid model reply"
    task = next(
        t for t in sdk.task_rows.values() if t["data"]["episode_id"] == failed["episode_id"]
    )
    assert task["predictions"] == []
    good = next(t for t in sdk.task_rows.values() if t["predictions"])
    sdk.save_annotation(good["id"])
    pending = engine.run()
    item = next(i for i in pending["items"] if not i["prediction_failed"])
    accept(engine, item)
    partial = engine.run()
    assert partial["status"] == "partial_failure" and exit_status(partial) == 2
    assert partial["export"]["episodes"] == 1
    assert len(runtime.calls) == sdk.import_calls == 2


@pytest.mark.parametrize("transport", [False, True])
def test_all_failed_is_never_success_and_never_silently_retried(workflow, transport):
    engine, runtime, sdk, body = workflow
    runtime.fail = {body["source_id"]}
    runtime.crash = transport
    report = engine.run()
    assert report["status"] == "failed" and exit_status(report) == 2
    assert report["metrics"]["failed_predictions"] == 1
    assert engine.run()["metrics"]["failed_predictions"] == 1
    assert len(runtime.calls) == sdk.import_calls == 1
    assert not list((engine.root / "exports").glob("*"))


@pytest.mark.parametrize("changed", ["input", "prediction", "guidelines", "native_prediction"])
def test_changed_artifacts_block_resume(workflow, changed):
    engine, runtime, sdk, body = workflow
    first = engine.run()
    key = first["items"][0]["episode_id"]
    if changed == "native_prediction":
        sdk.task_rows[1]["predictions"][0]["result"] = []
    else:
        path = {
            "input": engine.root / "intake/episodes/0" / body["video"],
            "prediction": engine.root / f"jobs/{key}/provider/predictions.json",
            "guidelines": engine.config.guidelines,
        }[changed]
        with path.open("ab") as stream:
            stream.write(b"changed")
    result = engine.run()
    assert result["errors"] and exit_status(result) == 2
    assert result["export"]["status"] == "blocked"
    assert len(runtime.calls) == 1


def test_multiple_annotations_require_explicit_selection(workflow):
    engine, _, sdk, _ = workflow
    engine.run()
    sdk.save_annotation(1)
    duplicate = deepcopy(sdk.task_rows[1]["annotations"][0])
    duplicate["id"] = 102
    sdk.task_rows[1]["annotations"].append(duplicate)
    report = engine.run()
    assert report["items"][0]["state"] == "annotation_selection_required"
    assert engine.store.history(report["items"][0]["episode_id"]) == []
    assert exit_status(report) == 3


def test_failed_intake_revalidation_blocks_even_previously_accepted_export(workflow, monkeypatch):
    engine, runtime, sdk, _ = workflow
    engine.run()
    sdk.save_annotation(1)
    accept(engine, engine.run()["items"][0])

    def changed_source(*args):
        raise StageFailure("Input shard changed")

    monkeypatch.setattr(runtime, "intake", changed_source)
    report = engine.run()
    assert report["status"] == "blocked" and exit_status(report) == 2
    assert report["errors"][0]["stage"] == "intake"
    assert report["export"]["status"] == "blocked"
    assert not (engine.root / "exports").exists()


def test_missing_sdk_does_not_claim_tasks_were_published(workflow):
    engine, _, _, _ = workflow
    engine.sdk = None
    report = engine.run()
    assert report["status"] == "awaiting_label_studio" and exit_status(report) == 3
    assert "native_tasks" not in report


def test_runtime_qwen_is_selected_explicitly_without_baseline_fallback(workflow, monkeypatch):
    engine, _, _, _ = workflow
    config = engine.config.model_copy(
        update={"provider": "qwen", "model_dir": Path("weights"), "device": "mps"}
    )
    runtime = Runtime()
    seen = []
    monkeypatch.setattr(runtime, "call", lambda *args: seen.append(args))
    runtime.predict(config, Path("plan.json"), engine.store, Path("result"))
    assert len(seen) == 1 and seen[0][0] == "integrations/semantic"
    assert "integrations.semantic.qwen" in seen[0][1] and "mps" in seen[0][1]


def test_aggregate_keeps_provider_tuning_declarations(workflow, monkeypatch):
    engine, runtime, _, _ = workflow
    predict = runtime.predict

    def tuned(*args):
        predict(*args)
        output = args[-1]
        path = output / "predictions.json"
        predictions = json.loads(path.read_text())
        predictions["tuning_episode_ids"] = [predictions["episodes"][0]["episode_id"]]
        atomic_json(path, predictions)
        manifest = json.loads((output / "run.json").read_text())
        manifest["files"]["predictions.json"] = file_hash(path)
        atomic_json(output / "run.json", manifest)

    monkeypatch.setattr(runtime, "predict", tuned)
    report = engine.run()
    predictions = Predictions.model_validate_json((engine.root / "predictions.json").read_bytes())
    assert predictions.tuning_episode_ids == [report["items"][0]["episode_id"]]


@pytest.mark.parametrize("target", ["episodes", "conversion.json", "receipt.json"])
def test_symlinked_owned_artifacts_block_resume(workflow, tmp_path, target):
    engine, runtime, _, _ = workflow
    first = engine.run()
    key = first["items"][0]["episode_id"]
    path = {
        "episodes": engine.root / "intake/episodes",
        "conversion.json": engine.root / "intake/episodes/0/conversion.json",
        "receipt.json": engine.root / f"jobs/{key}/receipt.json",
    }[target]
    outside = tmp_path / "outside"
    path.rename(outside)
    path.symlink_to(outside, target_is_directory=outside.is_dir())
    report = engine.run()
    assert report["errors"] and exit_status(report) == 2
    assert "symlink" in report["errors"][0]["message"]
    assert len(runtime.calls) == 1


@pytest.mark.parametrize("state,expected", [("waiting", 3), ("failed", 2), ("accepted", 0)])
def test_cli_status_matches_actual_workflow(workflow, monkeypatch, capsys, state, expected):
    from robo import semantic_pipeline

    engine, runtime, sdk, body = workflow
    if state == "failed":
        runtime.fail = {body["source_id"]}
    if state == "accepted":
        engine.run()
        sdk.save_annotation(1)
        accept(engine, engine.run()["items"][0])
    path = engine.root / "cli-config.json"
    atomic_json(path, engine.config.model_dump(mode="json"))
    monkeypatch.setenv("LABEL_STUDIO_API_KEY", "")
    monkeypatch.setattr(semantic_pipeline, "Workflow", lambda *args, **kwargs: engine)
    monkeypatch.setattr("sys.argv", ["semantic_pipeline", str(path), str(engine.root)])
    with pytest.raises(SystemExit) as result:
        semantic_pipeline.main()
    assert result.value.code == expected
    assert json.loads(capsys.readouterr().out)["exit_status"] == expected
