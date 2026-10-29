"""Check one local DROID episode through Workflow with explicitly authorized fixture review.

The default provider is the pinned local Qwen model. This is an acceptance test,
not a production approval command or an annotation-accuracy benchmark.
"""

import argparse
import json
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from shutil import which

from integrations.semantic.assets import PIN
from integrations.semantic.assets import preflight as model_preflight
from robo.core import atomic_json, file_hash
from robo.droid import CAMERAS, RELEASE
from robo.native_review import NativeReviewBridge, prediction_identity
from robo.semantic_pipeline import Configuration, Workflow, exit_status
from robo.semantic_runtime import REPO, Runtime
from scripts.semantic_api_verify import (
    CORRECTED_TEXT,
    authenticated_backend,
    fetch_task,
    groups,
    saved_annotation,
)

STAGES = ("environment", "workflow", "correction", "fixture_approval", "export", "official_loader")
DEVICE_PROBE = """
import json, platform, sys, torch
requested = sys.argv[1]
available = (torch.backends.mps.is_available() if requested == 'mps' else
             torch.cuda.is_available() if requested == 'cuda' else True)
result = {'requested_device': requested, 'available': available,
          'torch': torch.__version__, 'platform': platform.platform(), 'model_inference': False}
if available:
    value = torch.ones(1, device=requested) + 1
    assert value.item() == 2 and value.device.type == requested
    result['tensor_device'] = str(value.device)
print(json.dumps(result))
"""


class Unavailable(RuntimeError):
    pass


def new_report(provider):
    return {
        "schema_version": 1,
        "status": "running",
        "provider_requested": provider,
        "stages": {name: {"status": "pending"} for name in STAGES},
        "model_execution": {"status": "pending" if provider == "qwen" else "skipped"},
        "fixture_edits": 0,
        "fixture_approvals": 0,
        "production_automatic_approvals": 0,
        "pretrained_qwen_run": False,
        "semantic_accuracy_measured": False,
        "independent_reference_labels_created": False,
        "human_efficiency_measured": False,
        "browser_interaction_tested_here": False,
        "session_cleanup": False,
        "driver_sha256": file_hash(Path(__file__)),
    }


def finish(report, path, status, stage=None, error=None):
    report["status"] = status
    if stage:
        message = (
            f"{type(error).__name__} (HTTP {error.status_code})"
            if hasattr(error, "status_code")
            else str(error)[:1000] or type(error).__name__
        )
        report["stages"][stage] = {"status": status, "reason": message}
    for entry in report["stages"].values():
        if entry["status"] == "pending":
            entry.update(status="skipped", reason="An earlier required stage did not complete")
    if report["model_execution"]["status"] == "pending":
        report["model_execution"] = {"status": "skipped", "reason": "Inference was not reached"}
    report["exit_status"] = (
        0 if status == "passed" else 3 if status in {"unavailable", "skipped"} else 2
    )
    atomic_json(path, report)
    return report


def check_environment(config, directory, runtime):
    projects = [".", "integrations/droid", "integrations/lerobot"]
    if config.provider == "qwen":
        projects.append("integrations/semantic")
    for project in projects:
        if not (REPO / project / ".venv/bin/python").is_file():
            raise Unavailable(f"Sync the locked {project} runtime first")
    for command in ("ffmpeg", "ffprobe"):
        if which(command) is None:
            raise Unavailable(f"Required local command is missing: {command}")
    if not all((config.source / name).is_file() for name in ("dataset_info.json", "features.json")):
        raise Unavailable("Supply an existing local DROID TFDS source directory")
    if config.provider == "baseline":
        return {"provider": "baseline", "model_assets_checked": False}
    missing = [name for name in PIN["files"] if not (config.model_dir / name).is_file()]
    if missing:
        raise Unavailable("Pinned Qwen assets are missing: " + ", ".join(missing))
    logs = directory / "logs/device-probe"
    runtime.call("integrations/semantic", ["-c", DEVICE_PROBE, config.device], logs)
    device = json.loads((logs / "stdout.log").read_text())
    if not device["available"]:
        raise Unavailable(f"Requested {config.device} device is unavailable; no model fallback")
    assets = model_preflight(config.model_dir)
    result = {"provider": "qwen", "device_probe": device, "assets": assets}
    atomic_json(directory / "model-preflight.json", result)
    return result


def provider_execution(workflow, first):
    predictions = json.loads((workflow.root / "predictions.json").read_text())
    if predictions["producer"] != first["provider"]["producer"]:
        raise ValueError("Requested and recorded provider differ")
    key = first["items"][0]["episode_id"]
    if workflow.config.provider == "baseline":
        return {"status": "skipped", "reason": "Explicit robot-state baseline; no model inference"}
    path = workflow.root / f"jobs/{key}/provider/run.json"
    manifest = json.loads(path.read_text())
    assets = manifest["configuration"]["assets"]
    if (
        manifest["failed_predictions"] != 0
        or assets["model_id"] != PIN["model_id"]
        or assets["revision"] != PIN["revision"]
        or assets["weight_bytes"] != PIN["files"]["model.safetensors"]["bytes"]
        or assets["files_sha256"]["model.safetensors"]
        != PIN["files"]["model.safetensors"]["sha256"]
        or assets["device"].split(":")[0] != workflow.config.device
        or not predictions["episodes"][0]["segments"]
        or any(s["provenance"]["kind"] != "model" for s in predictions["episodes"][0]["segments"])
    ):
        raise ValueError("Qwen execution evidence does not match the requested pinned model/device")
    return {
        "status": "passed",
        "assets": assets,
        "run_manifest": str(path),
        "run_manifest_sha256": file_hash(path),
        "semantic_accuracy_measured": False,
    }


def exercise(workflow, client, report_path, *, approve_fixture, report=None):
    """Use the production workflow; the only test decisions are explicit native edits/approval."""
    report = report or new_report(workflow.config.provider)
    if not approve_fixture:
        return finish(
            report,
            report_path,
            "skipped",
            "fixture_approval",
            "Pass --approve-fixture to authorize this disposable test decision",
        )
    if report["stages"]["environment"]["status"] == "pending":
        report["stages"]["environment"] = {"status": "skipped", "reason": "Existing test harness"}
    stage = "workflow"
    sdk = workflow.sdk
    try:
        first = workflow.run()
        atomic_json(report_path.parent / "workflow-first.json", first)
        report["workflow"] = {
            "root": str(workflow.root),
            "first_status": first["status"],
            "failed_predictions": first["metrics"].get("failed_predictions"),
            "errors": first["errors"],
        }
        provider_failed = first["metrics"].get("failed_predictions") or any(
            error["stage"] == "prediction" for error in first["errors"]
        )
        if provider_failed and workflow.config.provider == "qwen":
            report["model_execution"] = {
                "status": "failed",
                "reason": "Provider returned a failed episode",
                "artifacts": str(workflow.root / "jobs"),
            }
        if first["status"] != "awaiting_review" or exit_status(first) != 3:
            raise ValueError(
                f"Workflow returned {first['status']}; see workflow-first.json and jobs"
            )
        if first["metrics"]["imported_episodes"] != 1:
            raise ValueError("The acceptance driver requires exactly one episode")
        if workflow.config.provider == "qwen":
            report["model_execution"] = {
                "status": "failed",
                "reason": "Generation returned; its execution evidence has not passed validation",
            }
        report["model_execution"] = provider_execution(workflow, first)
        report["pretrained_qwen_run"] = report["model_execution"]["status"] == "passed"
        report["stages"][stage] = {"status": "passed"}
        report["evaluation_status"] = first["evaluation"]["status"]
        episode_id = first["items"][0]["episode_id"]
        prediction_sha = file_hash(workflow.root / "predictions.json")
        binding = workflow.store.native_binding(episode_id)["binding"]
        task_id, project_id = binding["task_id"], binding["project_id"]
        assert workflow.run()["status"] == "awaiting_review"
        assert len(list(sdk.tasks.list(project=project_id))) == 1
        stage = "correction"
        task = fetch_task(sdk, task_id)
        machine_before = [prediction_identity(p) for p in task["predictions"]]
        media = client.get(task["data"]["video"])
        assert media.status_code == 200 and media.headers["Content-Type"].startswith("video/")
        media.close()
        corrected = deepcopy(task["predictions"][0]["result"])
        groups(corrected)[0]["subtask_text"]["value"]["text"] = [CORRECTED_TEXT]
        annotation = sdk.annotations.create(task_id, result=corrected)
        report["fixture_edits"] = 1
        saved_annotation(
            sdk,
            task_id,
            annotation.id,
            corrected,
            machine_before,
            report_path.parent / "saved-annotation.json",
        )
        draft = workflow.run()
        assert draft["status"] == "awaiting_approval", draft["errors"]
        assert draft["items"][0]["annotation_id"] == annotation.id
        assert draft["items"][0]["revision"] == 1, "Fresh correction must create draft revision 1"
        report["stages"][stage] = {
            "status": "passed",
            "kind": "automated_fixture",
            "text": CORRECTED_TEXT,
            "annotation_id": annotation.id,
        }
        stage = "fixture_approval"
        NativeReviewBridge(sdk, workflow.store, workflow.config.instance).decide(
            episode_id,
            draft["items"][0]["revision"],
            "accept",
            "automated-workflow-test",
            "Explicit acceptance-driver fixture approval; not validated ground truth",
        )
        report["fixture_approvals"] = 1
        report["stages"][stage] = {"status": "passed", "kind": "automated_fixture"}
        stage = "export"
        completed = workflow.run()
        assert completed["status"] == "complete" and exit_status(completed) == 0, completed[
            "errors"
        ]
        assert completed["export"]["status"] == "validated", "Export validation did not complete"
        repeated = workflow.run()
        assert repeated["status"] == "complete" and repeated["export"] == completed["export"]
        assert len(list(sdk.tasks.list(project=project_id))) == 1
        assert file_hash(workflow.root / "predictions.json") == prediction_sha
        assert [
            prediction_identity(p) for p in fetch_task(sdk, task_id)["predictions"]
        ] == machine_before
        report["export"] = completed["export"]
        report["stages"][stage] = {"status": "passed"}
        stage = "official_loader"
        evidence = report_path.parent / "loader.json"
        workflow.runtime.call(
            "integrations/lerobot",
            ["-m", "integrations.lerobot.verify_pipeline", workflow.root, "--evidence", evidence],
            workflow.root / "logs/verify-training",
        )
        loaded = json.loads(evidence.read_text())
        assert (
            loaded["official_lerobot_and_dataloader"]
            and loaded["prediction_sha256"] == prediction_sha
        )
        report.update(loaded)
        report["stages"][stage] = {"status": "passed", "evidence": str(evidence)}
        report.update(
            native_task_count=1,
            native_video_response=200,
            saved_correction_refreshed_automatically=True,
            explicit_test_approval=True,
            repeated_run_keeps_export_and_predictions=True,
            automatic_approvals=0,
        )
        return finish(report, report_path, "passed")
    except Exception as exc:
        return finish(report, report_path, "failed", stage, exc)


def run(config, output, *, approve_fixture):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Choose a fresh acceptance output directory")
    output = output.resolve()
    for source in (config.source, config.model_dir):
        if source and (output.is_relative_to(source) or source.is_relative_to(output)):
            raise ValueError("Keep acceptance output separate from source data and model assets")
    output.mkdir(parents=True)
    path = output / "report.json"
    report = new_report(config.provider)
    atomic_json(output / "configuration.json", config.model_dump(mode="json"))
    if not approve_fixture:
        return finish(
            report,
            path,
            "skipped",
            "fixture_approval",
            "Explicit fixture approval was not authorized",
        )
    stage = "environment"
    try:
        runtime = Runtime()
        report["environment"] = check_environment(config, output, runtime)
        report["stages"][stage] = {"status": "passed"}
        workflow = Workflow(output / "workflow", config, runtime=runtime)
        stage = "workflow"
        with (output / "backend.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            with authenticated_backend(output, workflow.store) as (sdk, client):
                workflow.sdk = sdk
                report = exercise(workflow, client, path, approve_fixture=True, report=report)
        report["session_cleanup"] = True
        atomic_json(path, report)
        return report
    except Unavailable as exc:
        return finish(report, path, "unavailable", stage, exc)
    except Exception as exc:
        return finish(report, path, "failed", stage, exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("qwen", "baseline"), default="qwen")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--release", default=RELEASE)
    parser.add_argument("--camera", choices=CAMERAS, default=CAMERAS[0])
    parser.add_argument("--split", choices=("train", "development"), default="development")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--approve-fixture",
        action="store_true",
        help="Authorize one unvalidated test correction and approval in a fresh local backend",
    )
    args = parser.parse_args()
    try:
        config = Configuration(
            source=args.source,
            release=args.release,
            camera=args.camera,
            guidelines=REPO / "docs/semantic-guidelines.md",
            provider=args.provider,
            model_dir=args.model_dir,
            device=args.device if args.provider == "qwen" else None,
            split=args.split,
            allow_partial=args.allow_partial,
            max_episodes=1,
            instance="http://testserver",
        )
        report = run(config, args.output, approve_fixture=args.approve_fixture)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": str(args.output / "report.json"),
                    "model_execution": report["model_execution"]["status"],
                    "exit_status": report["exit_status"],
                }
            )
        )
        raise SystemExit(report["exit_status"])
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)[:1000]}))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
