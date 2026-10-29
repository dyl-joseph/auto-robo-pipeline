"""Run the local DROID workflow through real Label Studio and LeRobot, without a model download."""

import json
from copy import deepcopy
from pathlib import Path

from robo.core import atomic_json, file_hash
from robo.droid import RELEASE
from robo.native_review import NativeReviewBridge
from robo.semantic_pipeline import Configuration, Workflow, exit_status


def verify_pipeline(sdk, directory, source, client):
    guidelines = directory / "workflow-guidelines.txt"
    guidelines.write_text("Automated workflow test: baseline edits are unvalidated demo inputs.")
    config = Configuration(
        source=source,
        release=RELEASE,
        guidelines=guidelines,
        provider="baseline",
        instance="http://testserver",
        allow_partial=True,
    )
    workflow = Workflow(directory / "semantic-pipeline", config, sdk=sdk)
    first = workflow.run()
    assert first["status"] == "awaiting_review", first
    assert exit_status(first) == 3 and first["metrics"]["imported_episodes"] == 1
    assert first["metrics"]["failed_predictions"] == 0
    assert first["evaluation"]["status"] == "blocked"
    episode_id = first["items"][0]["episode_id"]
    prediction_sha = file_hash(workflow.root / "predictions.json")
    binding = workflow.store.native_binding(episode_id)["binding"]
    task_id, project_id = binding["task_id"], binding["project_id"]
    assert workflow.run()["status"] == "awaiting_review"
    assert len(list(sdk.tasks.list(project=project_id))) == 1
    task = sdk.tasks.get(str(task_id))
    media = client.get(task.data["video"])
    assert media.status_code == 200 and media["Content-Type"].startswith("video/")
    media.close()
    corrected = deepcopy(task.model_dump(mode="json")["predictions"][0]["result"])
    text = next(r for r in corrected if r["from_name"] == "subtask_text")
    text["value"]["text"] = ["Demonstration correction: move the arm toward the container."]
    annotation = sdk.annotations.create(task_id, result=corrected)
    draft = workflow.run()
    assert draft["status"] == "awaiting_approval", draft
    assert draft["items"][0]["annotation_id"] == annotation.id
    assert draft["items"][0]["revision"] == 1
    bridge = NativeReviewBridge(sdk, workflow.store, config.instance)
    bridge.decide(
        episode_id,
        1,
        "accept",
        "automated-workflow-test",
        "Explicit test decision for plumbing verification, not validated ground truth",
    )
    completed = workflow.run()
    assert completed["status"] == "complete" and exit_status(completed) == 0, completed
    assert completed["export"]["status"] == "validated"
    repeated = workflow.run()
    assert repeated["export"] == completed["export"]
    assert len(list(sdk.tasks.list(project=project_id))) == 1
    assert file_hash(workflow.root / "predictions.json") == prediction_sha
    evidence = Path("artifacts/semantic-pipeline.json").resolve()
    workflow.runtime.call(
        "integrations/lerobot",
        ["-m", "integrations.lerobot.verify_pipeline", workflow.root, "--evidence", evidence],
        workflow.root / "logs/verify-training",
    )
    result = json.loads(evidence.read_text())
    result.update(
        native_task_count=1,
        native_video_response=200,
        saved_correction_refreshed_automatically=True,
        explicit_test_approval=True,
        repeated_run_keeps_export_and_predictions=True,
        automatic_approvals=0,
        evaluation_status=completed["evaluation"]["status"],
    )
    atomic_json(evidence, result)
    print(f"Semantic workflow: one task, explicit approval, {result['rows_checked']} loaded rows")
