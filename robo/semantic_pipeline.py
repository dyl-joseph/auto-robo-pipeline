"""Resume local DROID intake, semantic review and explicitly approved training export."""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from robo.core import Contract, atomic_json, canonical, digest, file_hash
from robo.export import validate_export
from robo.ingest import import_episode
from robo.lerobot_export import validate_training_export
from robo.native_queue import publish_queue
from robo.native_review import NativeReviewBridge, local_instance
from robo.pipeline import pipeline_lock
from robo.semantic import EvaluationPlan, PredictionEpisode, Predictions, References
from robo.semantic_baseline import source_episode
from robo.semantic_evaluation import evaluate
from robo.semantic_plan import build_plan
from robo.semantic_review import prepare, read_inputs
from robo.semantic_routing import review_reasons
from robo.semantic_runtime import Runtime
from robo.store import Store


class Configuration(Contract):
    source: Path
    release: str = Field(min_length=1, max_length=200)
    guidelines: Path
    provider: Literal["baseline", "qwen"]
    model_dir: Path | None = None
    device: Literal["cpu", "mps", "cuda"] | None = None
    camera: Literal["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"] = (
        "exterior_image_1_left"
    )
    source_split: str = Field(default="train", min_length=1, max_length=50)
    split: Literal["train", "development", "test"] = "development"
    max_episodes: int = Field(default=1, ge=1, le=200, strict=True)
    allow_partial: bool = False
    instance: str = "http://127.0.0.1:8000"
    audited_plan: Path | None = None
    references: Path | None = None

    @model_validator(mode="after")
    def local_paths(self):
        self.instance = local_instance(self.instance)
        for name in ("source", "guidelines", "model_dir", "audited_plan", "references"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, value.expanduser().resolve())
        if self.provider == "qwen" and (self.model_dir is None or self.device is None):
            raise ValueError("Qwen requires existing model_dir and an explicit device")
        if self.provider == "baseline" and (self.model_dir is not None or self.device is not None):
            raise ValueError("Baseline does not use model_dir or device; choose one provider")
        return self


def freeze(path, value):
    if path.is_symlink():
        raise ValueError(f"Owned workflow file became a symlink: {path}")
    if path.exists():
        if path.read_bytes() != canonical(value):
            raise ValueError(f"Frozen workflow artifact changed: {path}")
    else:
        atomic_json(path, value)


def exit_status(report):
    if report["errors"] or report["metrics"].get("failed_predictions", 0):
        return 2
    return 0 if report["status"] == "complete" else 3


class Workflow:
    def __init__(self, root, config, *, runtime=None, sdk=None):
        root = Path(root).absolute()
        if root.is_symlink():
            raise ValueError("Workflow root must not be a symlink")
        self.root, self.config = root.resolve(), config
        if self.root.is_relative_to(config.source) or config.source.is_relative_to(self.root):
            raise ValueError("Keep the workflow outside the source directory")
        self.root.mkdir(parents=True, exist_ok=True)
        self.runtime, self.sdk = runtime or Runtime(), sdk
        self.store = Store(self.owned("store"))

    def owned(self, relative):
        path = self.root / relative
        if not path.is_relative_to(self.root) or ".." in path.parts:
            raise ValueError("Workflow output must stay inside its root")
        cursor = self.root
        for part in path.relative_to(self.root).parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError(f"Owned workflow path became a symlink: {cursor}")
        return path

    def persist(self, report):
        report["exit_status"] = exit_status(report)
        atomic_json(self.owned("report.json"), report)

    def error(self, report, stage, error, episode_id=None):
        message = (
            f"{type(error).__name__} (HTTP {error.status_code})"
            if hasattr(error, "status_code")
            else str(error)[:1000]
        )
        report["errors"].append(
            {
                "stage": stage,
                "episode_id": episode_id,
                "message": message,
            }
        )
        report["status"] = "blocked"
        self.persist(report)

    def run(self):
        report = {
            "schema_version": 1,
            "status": "running",
            "errors": [],
            "items": [],
            "report_path": str(self.owned("report.json")),
            "review_queue": [],
            "metrics": {"automatic_approvals": 0, "human_active_seconds": None},
            "evaluation": {"status": "blocked", "reason": "References were not supplied"},
            "export": {"status": "blocked", "reason": "No fresh explicitly accepted revisions"},
        }
        with pipeline_lock(self):
            try:
                return self._run(report)
            except Exception as exc:
                self.error(report, report.get("stage", "configuration"), exc)
                return report

    def _run(self, report):
        config = self.config
        identity = self.runtime.provider_identity(config)
        frozen = {
            "configuration": config.model_dump(mode="json"),
            "guidelines_sha256": file_hash(config.guidelines),
            "provider": identity,
            "audited_plan_sha256": file_hash(config.audited_plan) if config.audited_plan else None,
        }
        freeze(self.owned("configuration.json"), frozen)
        report["batch_id"] = digest(frozen)[:24]
        report["provider"] = identity
        report["stage"] = "intake"
        intake_path = self.owned("intake")
        try:
            self.runtime.intake(config, intake_path)
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
            self.error(report, "intake", exc)
            return report
        checkpoint = self.owned("intake/intake.json")
        if not checkpoint.exists():
            if not report["errors"]:
                self.error(report, "intake", "Intake did not publish a checkpoint")
            return report
        intake = json.loads(checkpoint.read_text())
        report["intake"] = {"status": intake["status"], **intake["summary"]}
        if intake["status"] == "failed" and not report["errors"]:
            self.error(report, "intake", intake.get("error", "Intake failed"))
            return report
        records = [r for shard in intake["shards"].values() for r in shard["records"]]
        episodes, schemas = {}, {}
        for record in records:
            report["stage"] = "import"
            try:
                name = record["directory"]
                if Path(name).name != name:
                    raise ValueError("Invalid intake episode directory")
                directory = self.owned(f"intake/episodes/{name}")
                conversion = self.owned(f"intake/episodes/{name}/conversion.json")
                if file_hash(conversion) != record["conversion_sha256"]:
                    raise ValueError("Intake conversion receipt changed")
                for filename, expected in json.loads(conversion.read_text())["files"].items():
                    path = self.owned(str((directory / filename).relative_to(self.root)))
                    if file_hash(path) != expected:
                        raise ValueError(f"Intake derivative changed: {filename}")
                episode = import_episode(directory / "source.json", self.store)
                episodes[episode["id"]] = episode
                schemas[episode["id"]] = directory / "training-schema.json"
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                self.error(report, "import", exc)
        report["metrics"].update(
            published_inputs=len(records),
            imported_episodes=len(episodes),
            duplicate_input_records=len(records)
            - len(episodes)
            - sum(e["stage"] == "import" for e in report["errors"]),
        )
        if not episodes:
            self.error(report, "import", "No valid episodes are available")
            return report
        report["stage"] = "plan"
        plan = build_plan(
            self.store,
            [episodes[key] for key in sorted(episodes)],
            config.guidelines,
            dataset="DROID",
            camera=config.camera,
            split=config.split,
        )
        if config.audited_plan:
            plan = EvaluationPlan.model_validate_json(config.audited_plan.read_bytes())
            if (
                {s.episode_id for s in plan.episodes} != set(episodes)
                or any(s.split != config.split for s in plan.episodes)
                or plan.annotation_guidelines_sha256 != frozen["guidelines_sha256"]
            ):
                raise ValueError("Audited plan must match this population, split and guidelines")
            for spec in plan.episodes:
                source_episode(self.store, spec)
        plan_path = self.owned("plan.json")
        freeze(plan_path, plan.model_dump())
        rows, receipts = [], {}
        report["stage"] = "prediction"
        for spec in plan.episodes:
            source_episode(self.store, spec)
            row, receipt = self.prediction(plan, spec, identity)
            rows.append(row)
            receipts[spec.episode_id] = receipt
        predictions = Predictions(
            plan_sha256=plan.plan_sha256,
            run_id=digest(receipts)[:24],
            producer=identity["producer"],
            code_revision="semantic-workflow-v1",
            config_sha256=digest(frozen),
            training_episode_ids=sorted(
                {key for r in receipts.values() for key in r.get("training_episode_ids", [])}
            ),
            tuning_episode_ids=sorted(
                {key for r in receipts.values() for key in r.get("tuning_episode_ids", [])}
            ),
            reference_access="none",
            episodes=rows,
        )
        prediction_path = self.owned("predictions.json")
        freeze(prediction_path, predictions.model_dump())
        report["metrics"].update(
            predicted_episodes=len(rows),
            failed_predictions=sum(r.error is not None for r in rows),
            review_requests=len(rows),
        )
        report["prediction_artifact"] = str(prediction_path)
        report["prediction_sha256"] = file_hash(prediction_path)
        report["stage"] = "review_preparation"
        for spec, row in zip(plan.episodes, rows, strict=True):
            reasons = review_reasons(plan, spec, row)
            output = self.owned(f"review/{spec.episode_id}")
            prepare(self.store, plan_path, prediction_path, spec.episode_id, output)
            report["review_queue"].append(
                {
                    "id": spec.episode_id,
                    "priority": len(reasons),
                    "reasons": reasons + ["explicit_approval_required"],
                    "bundle": str(output / "bundle.json"),
                }
            )
            report["items"].append(
                {
                    "episode_id": spec.episode_id,
                    "prediction_failed": row.error is not None,
                    "prediction_error": row.error,
                    "state": "awaiting_review",
                    "reasons": reasons,
                }
            )
        report["review_queue"].sort(key=lambda row: (-row["priority"], row["id"]))
        if config.references:
            try:
                references = References.model_validate_json(config.references.read_bytes())
                evaluation = evaluate(plan, predictions, references, split=config.split)
                path = self.owned(f"evaluations/{digest(evaluation)[:24]}.json")
                freeze(path, evaluation)
                report["evaluation"] = {"status": "scored", "path": str(path)}
            except (ValueError, OSError) as exc:
                report["evaluation"] = {"status": "blocked", "reason": str(exc)[:1000]}
        if self.sdk is None:
            report["status"] = (
                "failed"
                if report["metrics"]["failed_predictions"] or report["errors"]
                else "awaiting_label_studio"
            )
            self.persist(report)
            return report
        report["stage"] = "native_publish"
        publish_queue(self.sdk, self.store, report, config.instance, persist=self.persist)
        report["stage"] = "native_refresh"
        bridge = NativeReviewBridge(self.sdk, self.store, config.instance)
        selection = []
        for item in report["items"]:
            try:
                self.refresh(item, bridge)
                if item["state"] == "accepted":
                    selection.append(
                        {
                            "episode_id": item["episode_id"],
                            "revision": item["revision"],
                            "split": config.split,
                        }
                    )
            except (ValueError, OSError) as exc:
                item["state"] = "blocked"
                self.error(report, "native_refresh", exc, item["episode_id"])
        if selection:
            report["stage"] = "export"
            accepted = self.owned(f"exports/{digest(selection)[:24]}")
            result = bridge.export(selection, accepted)
            validate_export(accepted)
            schema_paths = [schemas[s["episode_id"]] for s in selection]
            if len({file_hash(path) for path in schema_paths}) != 1:
                raise ValueError("Selected episodes need one compatible training schema")
            training = self.owned("training/" + result["manifest"]["version"])
            self.runtime.training(accepted, self.store, schema_paths[0], config.split, training)
            training_manifest = validate_training_export(training)
            report["export"] = {
                "status": "validated",
                "accepted": str(accepted),
                "training": str(training),
                "accepted_version": result["manifest"]["version"],
                "training_version": training_manifest["version"],
                "episodes": len(selection),
            }
        states = {item["state"] for item in report["items"]}
        report["metrics"].update(
            accepted_episodes=sum(item["state"] == "accepted" for item in report["items"]),
            excluded_episodes=sum(item["state"] == "excluded" for item in report["items"]),
        )
        report["status"] = (
            "complete"
            if states <= {"accepted", "excluded"} and report["export"]["status"] == "validated"
            else "no_accepted_episodes"
            if states == {"excluded"}
            else "awaiting_approval"
            if "awaiting_approval" in states
            else "awaiting_review"
        )
        if report["status"] == "no_accepted_episodes":
            report["export"]["reason"] = "All episodes were explicitly excluded"
        if report["errors"] or report["metrics"]["failed_predictions"]:
            report["status"] = "partial_failure" if selection else "failed"
        self.persist(report)
        return report

    def prediction(self, plan, spec, identity):
        plan_path = self.owned(f"jobs/{spec.episode_id}/plan.json")
        job_plan = plan.model_copy(update={"episodes": [spec]})
        freeze(plan_path, job_plan.model_dump())
        receipt_path = self.owned(f"jobs/{spec.episode_id}/receipt.json")
        provider = self.owned(f"jobs/{spec.episode_id}/provider")
        failure = self.owned(f"jobs/{spec.episode_id}/failure.json")
        if not receipt_path.exists() and not provider.exists() and not failure.exists():
            try:
                self.runtime.predict(self.config, plan_path, self.store, provider)
            except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
                # A published provider result survives an interrupted process response.
                if not provider.exists():
                    freeze(failure, {"error": str(exc)[:1000] or type(exc).__name__})
        if provider.exists():
            manifest_path = self.owned(f"jobs/{spec.episode_id}/provider/run.json")
            manifest = json.loads(manifest_path.read_text())
            files = manifest["files"]
            if "predictions.json" not in files or len(files) > 200:
                raise ValueError("Provider did not publish a bounded prediction manifest")
            for name, expected in files.items():
                if Path(name).name != name:
                    raise ValueError("Provider artifact name must be local to its result directory")
                path = self.owned(f"jobs/{spec.episode_id}/provider/{name}")
                if file_hash(path) != expected:
                    raise ValueError("Immutable provider artifact changed")
            _, predictions = read_inputs(
                plan_path.read_bytes(), (provider / "predictions.json").read_bytes()
            )
            if (
                len(predictions.episodes) != 1
                or predictions.episodes[0].episode_id != spec.episode_id
            ):
                raise ValueError("Provider must return exactly its assigned episode")
            if predictions.producer != identity["producer"]:
                raise ValueError("Provider identity differs from the requested provider")
            row = predictions.episodes[0]
            if manifest["failed_predictions"] != int(row.error is not None):
                raise ValueError("Provider failure count differs from its prediction artifact")
            receipt = {
                "plan": job_plan.plan_sha256,
                "provider": identity,
                "training_episode_ids": predictions.training_episode_ids,
                "tuning_episode_ids": predictions.tuning_episode_ids,
                "files": {
                    **files,
                    "run.json": file_hash(manifest_path),
                },
            }
        else:
            error = json.loads(failure.read_text())["error"]
            row = PredictionEpisode(episode_id=spec.episode_id, segments=[], error=error)
            receipt = {
                "plan": job_plan.plan_sha256,
                "provider": identity,
                "failure_sha256": file_hash(failure),
            }
        freeze(receipt_path, receipt)
        return row, receipt

    def refresh(self, item, bridge):
        key = item["episode_id"]
        binding = self.store.native_binding(key)["binding"]
        task = self.sdk.tasks.get(str(binding["task_id"])).model_dump(mode="json")
        annotations = [a for a in task.get("annotations") or [] if not a.get("was_cancelled")]
        latest = self.store.latest(key)
        chosen = latest.get("native", {}).get("annotation_id")
        if chosen is None:
            if len(annotations) != 1:
                item["state"] = (
                    "awaiting_review" if not annotations else "annotation_selection_required"
                )
                return
            chosen = annotations[0]["id"]
        observation = bridge.observe(key, chosen)
        if latest.get("native", {}).get("snapshot_hash") != observation["native"]["snapshot_hash"]:
            latest = bridge.refresh(
                key,
                chosen,
                latest["revision"],
                "semantic-workflow",
                "Import saved native correction as a draft; explicit acceptance is still required",
            )
        item.update(
            revision=latest["revision"],
            annotation_id=chosen,
            state=latest["status"]
            if latest["status"] in {"accepted", "excluded"}
            else "awaiting_approval",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args()
    try:
        config = Configuration.model_validate_json(args.configuration.read_bytes())
        key = os.environ.get("LABEL_STUDIO_API_KEY")
        sdk = None
        if key:
            from label_studio_sdk import LabelStudio

            sdk = LabelStudio(base_url=config.instance, api_key=key, timeout=30)
        report = Workflow(args.workspace, config, sdk=sdk).run()
        print(json.dumps(report, indent=2))
        raise SystemExit(exit_status(report))
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
