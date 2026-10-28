"""Publish a batch queue to Label Studio and recover imports after an interrupted response."""

import json
from pathlib import Path

from robo.core import atomic_json, digest
from robo.labelstudio import QUEUE_FIELDS, task_data_hash
from robo.native_review import NativeReviewBridge, config_hash, local_instance
from robo.pipeline import persist_report, pipeline_lock, read_report


def sync_queue(sdk, store, report_path, instance):
    instance = local_instance(instance)
    report = read_report(report_path, store)
    directory = Path(report["report_path"]).parent
    state_path = directory / "native-queue.json"
    with pipeline_lock(store):
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {"instance": instance, "projects": {}, "tasks": {}}
        )
        if state["instance"] != instance:
            raise ValueError("This batch is already linked to a different Label Studio instance")
        bridge = NativeReviewBridge(sdk, store, instance)
        for row in report["review_queue"]:
            episode_id = row["id"]
            existing = store.native_binding(episode_id)
            if existing:
                if existing["instance"] != instance:
                    raise ValueError("Episode is already bound to another Label Studio instance")
                task_id = existing["binding"]["task_id"]
                project_id = existing["binding"]["project_id"]
                # Detect deleted or inaccessible tasks before reporting a working review link.
                task = sdk.tasks.get(str(task_id))
                if (
                    task_data_hash(task.data, existing["binding"])
                    != existing["binding"]["task_data_hash"]
                ):
                    raise ValueError("Bound native task source data changed")
                if existing["binding"].get("mutable_task_fields") == QUEUE_FIELDS:
                    data = {
                        **task.data,
                        "review_priority": row["priority"],
                        "review_reason": "; ".join(row["reasons"]),
                    }
                    if data != task.data:
                        sdk.tasks.update(str(task_id), data=data)
            else:
                bundle = json.loads(Path(row["bundle"]).read_text())
                signature = config_hash(bundle["label_config"])[:16]
                title = f"Review {report['batch_id']}-{signature}"
                project_id = state["projects"].get(signature)
                if project_id is None:
                    matches = [project for project in sdk.projects.list() if project.title == title]
                    if len(matches) > 1:
                        raise ValueError("Multiple matching projects; inspect them before retrying")
                    project = (
                        matches[0]
                        if matches
                        else sdk.projects.create(title=title, label_config=bundle["label_config"])
                    )
                    if config_hash(project.label_config) != config_hash(bundle["label_config"]):
                        raise ValueError(
                            "Existing batch project has a different labeling configuration"
                        )
                    project_id = project.id
                    state["projects"][signature] = project_id
                    atomic_json(state_path, state)
                project = sdk.projects.get(project_id)
                if config_hash(project.label_config) != config_hash(bundle["label_config"]):
                    raise ValueError("Batch project configuration changed")
                storages = sdk.import_storage.local.list(project=project_id)
                storage_path = str(store.root / "review_proxy")
                if not any(
                    storage.path == storage_path and storage.use_blob_urls for storage in storages
                ):
                    sdk.import_storage.local.create(
                        project=project_id,
                        path=storage_path,
                        use_blob_urls=True,
                        title="Review proxies",
                    )
                bundle["task"]["data"].update(
                    {
                        "review_priority": row["priority"],
                        "review_reason": "; ".join(row["reasons"]),
                    }
                )
                bundle["trusted_origin"]["mutable_task_fields"] = QUEUE_FIELDS
                published = directory / "native-bundles" / f"{episode_id}.json"
                if published.exists():
                    # Recover the precise task payload sent before an interrupted response.
                    bundle = json.loads(published.read_text())
                else:
                    atomic_json(published, bundle)
                matches = [
                    task
                    for task in sdk.tasks.list(project=project_id)
                    if task.data.get("episode_id") == episode_id
                ]
                if len(matches) > 1:
                    raise ValueError(
                        "Multiple native tasks for one episode; inspect before retrying"
                    )
                if matches:
                    task = matches[0]
                    if digest(task.data) != digest(bundle["task"]["data"]):
                        raise ValueError("Existing native task differs from the prepared batch")
                    task_id = task.id
                else:
                    imported = sdk.projects.import_tasks(
                        project_id, request=[bundle["task"]], return_task_ids=True
                    )
                    if not imported.task_ids or len(imported.task_ids) != 1:
                        raise ValueError("Import response lacked one task ID; rerun to reconcile")
                    task_id = imported.task_ids[0]
                bridge.register(bundle, project_id, task_id)
            state["tasks"][episode_id] = {
                "project_id": project_id,
                "task_id": task_id,
                "url": f"{instance}/projects/{project_id}/data?task={task_id}",
            }
            atomic_json(state_path, state)
        for project_id in sorted(set(state["projects"].values())):
            views = sdk.views.list(project=project_id)
            name = "Review priority"
            if not any((view.data or {}).get("title") == name for view in views):
                hidden = [
                    "tasks:" + field
                    for field in (
                        "inner_id",
                        "completed_at",
                        "cancelled_annotations",
                        "total_predictions",
                        "annotators",
                        "annotations_results",
                        "annotations_ids",
                        "predictions_score",
                        "predictions_model_versions",
                        "predictions_results",
                        "file_upload",
                        "storage_filename",
                        "created_at",
                        "updated_at",
                        "updated_by",
                        "avg_lead_time",
                        "draft_exists",
                        "data.video",
                        "data.sensors",
                        "data.provenance",
                        "data.episode_id",
                    )
                ]
                sdk.views.create(
                    project=project_id,
                    data={
                        "title": name,
                        "ordering": ["-tasks:data.review_priority", "tasks:id"],
                        "columnsDisplayType": {"tasks:data.review_priority": "Number"},
                        "filters": {"conjunction": "and", "items": []},
                        "hiddenColumns": {"explore": hidden, "labeling": hidden},
                    },
                )
        report["native_tasks"] = state["tasks"]
        persist_report(report)
    return report
