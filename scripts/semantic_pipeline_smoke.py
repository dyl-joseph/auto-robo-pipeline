"""Use the shared acceptance check with the baseline in the existing smoke harness."""

from pathlib import Path

from robo.core import atomic_json
from robo.droid import RELEASE
from robo.semantic_pipeline import Configuration, Workflow
from scripts.semantic_workflow_acceptance import exercise


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
    result = exercise(
        workflow, client, directory / "semantic-acceptance/report.json", approve_fixture=True
    )
    atomic_json(Path("artifacts/semantic-pipeline.json").resolve(), result)
    assert result["status"] == "passed", result
    assert result["rows_checked"] == 119 and result["evaluation_status"] == "blocked"
    print(f"Semantic workflow: one task, explicit approval, {result['rows_checked']} loaded rows")
