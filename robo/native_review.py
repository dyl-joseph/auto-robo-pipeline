"""Approve saved Label Studio annotations after checking their current content."""

from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from xml.etree.ElementTree import canonicalize

from robo.analysis import latest_run
from robo.core import Edit, digest
from robo.export import export_dataset
from robo.labelstudio import annotation_snapshot, bind_task
from robo.store import Conflict, Store


def local_instance(value: str) -> str:
    url = urlsplit(value)
    if (
        url.scheme not in {"http", "https"}
        or url.hostname not in {"localhost", "127.0.0.1", "::1", "testserver"}
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in {"", "/"}
    ):
        raise ValueError("native bridge requires a loopback Label Studio origin")
    return value.rstrip("/")


def utc_timestamp(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("native annotation needs a version timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("native annotation timestamp must include a timezone")
    return parsed.astimezone(UTC).isoformat(timespec="microseconds")


def prediction_payload(prediction: dict) -> dict:
    return {key: prediction.get(key) for key in ("model_version", "result", "score")}


def prediction_identity(prediction: dict) -> dict:
    return {
        "id": prediction["id"],
        **prediction_payload(prediction),
        "created_at": utc_timestamp(prediction.get("created_at")),
        "updated_at": utc_timestamp(prediction.get("updated_at")),
    }


def config_hash(config: str) -> str:
    return digest(canonicalize(config, strip_text=True))


class NativeReviewBridge:
    """Caller supplies an authorized SDK client; no credentials are created or stored."""

    def __init__(self, sdk, store: Store, instance: str):
        self.sdk, self.store, self.instance = sdk, store, local_instance(instance)

    def _task(self, task_id: int) -> dict:
        task = self.sdk.tasks.get(str(task_id)).model_dump(mode="json")
        # SDK versions can represent predictions as full objects or IDs.
        task["predictions"] = [
            self.sdk.predictions.get(p).model_dump(mode="json") if type(p) is int else p
            for p in task.get("predictions") or []
        ]
        return task

    def register(self, bundle: dict, project_id: int, task_id: int) -> dict:
        binding = bind_task(bundle, project_id, task_id)
        task = self._task(task_id)
        project = self.sdk.projects.get(project_id)
        if config_hash(project.label_config) != config_hash(bundle["label_config"]):
            raise ValueError("native labeling configuration differs from the prepared review clock")
        annotation_snapshot(self.store, binding, {**task, "annotations": []}, bundle["proxy"])
        if binding.get("semantic_prediction"):
            from robo.semantic_review import bound_prediction, native_predictions

            _, row, _ = bound_prediction(self.store, binding)
            expected = native_predictions(row, binding["semantic_prediction"]["snapshot_id"])
            if bundle["task"]["predictions"] != expected:
                raise ValueError("semantic task differs from its uncorrected prediction snapshot")
        actual = [prediction_payload(p) for p in task["predictions"]]
        expected = [prediction_payload(p) for p in bundle["task"]["predictions"]]
        if actual != expected:
            raise ValueError("native predictions differ from the prepared machine suggestions")
        run_id = binding["suggestion_run"]
        run = latest_run(self.store, binding["episode_id"], run_id) if run_id else None
        registered = {
            "schema_version": 1,
            "instance": self.instance,
            "label_config_hash": config_hash(project.label_config),
            "binding": binding,
            "proxy": bundle["proxy"],
            "predictions": [prediction_identity(p) for p in task["predictions"]],
            "prediction_hash": digest([prediction_identity(p) for p in task["predictions"]]),
            "suggestion_artifact_hash": run["artifact_hash"] if run else None,
        }
        return self.store.register_native(binding["episode_id"], registered)

    def observe(self, episode_id: str, annotation_id: int) -> dict:
        registered = self.store.native_binding(episode_id)
        if registered is None or registered["instance"] != self.instance:
            raise ValueError("no trusted binding for this Label Studio instance")
        binding, proxy = registered["binding"], registered["proxy"]
        project = self.sdk.projects.get(binding["project_id"])
        if config_hash(project.label_config) != registered["label_config_hash"]:
            raise Conflict("native labeling configuration changed; review clock is no longer bound")
        task = self._task(binding["task_id"])
        if task.get("drafts") or task.get("draft_exists"):
            raise Conflict("submit or discard visible native autosave drafts before refresh")
        if (
            digest([prediction_identity(p) for p in task["predictions"]])
            != registered["prediction_hash"]
        ):
            raise Conflict(
                "native machine predictions changed; trusted provenance no longer matches"
            )
        run_id = binding["suggestion_run"]
        if run_id:
            run = latest_run(self.store, episode_id, run_id)
            if run["artifact_hash"] != registered["suggestion_artifact_hash"]:
                raise Conflict("pinned machine suggestion artifact changed")
        if not any(a.get("id") == annotation_id for a in task.get("annotations") or []):
            raise Conflict("selected native annotation is absent from the bound task")
        annotation = self.sdk.annotations.get(annotation_id).model_dump(mode="json")
        if annotation.get("task") != binding["task_id"]:
            raise Conflict("selected native annotation belongs to a different task")
        if annotation.get("was_cancelled") or annotation.get("last_action") == "rejected":
            raise Conflict("cancelled or rejected native annotations are not eligible for import")
        parent = annotation.get("parent_prediction")
        if parent is not None and parent not in {p["id"] for p in registered["predictions"]}:
            raise Conflict("native annotation references an unbound parent prediction")
        updated = utc_timestamp(annotation.get("updated_at"))
        raw = {key: value for key, value in annotation.items() if key != "created_ago"}
        raw["updated_at"] = updated
        raw["created_at"] = utc_timestamp(raw.get("created_at"))
        if updated < raw["created_at"]:
            raise Conflict("native annotation timestamp regressed before its creation")
        lead_time = raw.get("lead_time")
        if lead_time is not None:
            import math

            if isinstance(lead_time, bool) or not math.isfinite(lead_time) or lead_time < 0:
                raise ValueError("invalid native lead_time")
        snapshot = annotation_snapshot(self.store, binding, {**task, "annotations": [raw]}, proxy)
        selected = snapshot["annotations"][0]
        # Detect a torn multi-request read. This is still an observation, not a
        # transaction with Label Studio; edits after this check need a new refresh.
        confirmed = self._task(binding["task_id"])
        if (
            config_hash(self.sdk.projects.get(binding["project_id"]).label_config)
            != registered["label_config_hash"]
        ):
            raise Conflict("native labeling configuration changed during freshness observation")
        for field in ("id", "project", "data", "drafts", "draft_exists"):
            if confirmed.get(field) != task.get(field):
                raise Conflict("native task changed during freshness observation; retry refresh")
        if [prediction_identity(p) for p in confirmed["predictions"]] != [
            prediction_identity(p) for p in task["predictions"]
        ]:
            raise Conflict("native predictions changed during freshness observation; retry refresh")
        selected_versions = [
            a for a in confirmed.get("annotations") or [] if a.get("id") == annotation_id
        ]
        if (
            len(selected_versions) != 1
            or utc_timestamp(selected_versions[0].get("updated_at")) != updated
            or selected_versions[0].get("result") != raw["result"]
            or selected_versions[0].get("was_cancelled")
        ):
            raise Conflict("native annotation changed during freshness observation; retry refresh")
        source = {
            "binding_hash": digest(registered),
            "instance": registered["instance"],
            "source_binding": binding,
            "label_config_hash": registered["label_config_hash"],
            "machine_predictions": registered["predictions"],
            "annotation_id": annotation_id,
            "updated_at": updated,
            "native_reviewer_id": raw.get("completed_by"),
            "raw_annotation": raw,
            "prediction_hash": registered["prediction_hash"],
            "boundary_evidence": selected["boundary_evidence"],
            "boundary_policy": proxy["boundary_policy"],
            "fidelity": proxy["fidelity"],
            "reviewer_evidence": selected["reviewer_evidence"],
            "native_lead_time_s": lead_time,
            "active_time_policy": "unmeasured; native lead_time is not active correction time",
        }
        if "semantic_review" in selected:
            source["semantic_review"] = selected["semantic_review"]
        source["snapshot_hash"] = digest(source)
        source["observed_at"] = datetime.now(UTC).isoformat(timespec="microseconds")
        return {
            "native": source,
            "segments": selected["segments"],
            "outcome": selected["outcome"],
            "quality_flags": selected["quality_flags"],
            "suggestion_run": run_id if registered["predictions"] else None,
            "perception_run": None,
            "condition": "assisted" if registered["predictions"] else "manual",
        }

    def refresh(
        self,
        episode_id: str,
        annotation_id: int,
        expected_revision: int,
        actor: str,
        rationale: str,
    ) -> dict:
        observation = self.observe(episode_id, annotation_id)
        native = observation.pop("native")
        edit = Edit(
            expected_revision=expected_revision,
            actor=actor,
            rationale=rationale,
            active_s=None,
            wall_s=None,
            **observation,
        )
        return self.store.commit(episode_id, edit, native=native)

    def decide(
        self, episode_id: str, expected_revision: int, decision: str, actor: str, rationale: str
    ) -> dict:
        if decision not in {"accept", "reject"}:
            raise ValueError("choose an explicit accept or reject decision")
        revision = self.store.latest(episode_id)
        if revision["revision"] != expected_revision:
            raise Conflict("stale bridge revision; inspect the latest draft before deciding")
        if not revision.get("native"):
            raise ValueError("refresh a native annotation as a draft before deciding")
        observation = self.observe(episode_id, revision["native"]["annotation_id"])
        native = observation.pop("native")
        if decision == "accept" and native.get("semantic_review", {}).get("complete") is False:
            raise ValueError(
                "complete semantic coverage is required before acceptance; "
                "label unknown spans explicitly"
            )
        edit = Edit(
            expected_revision=expected_revision,
            actor=actor,
            rationale=rationale,
            active_s=None,
            wall_s=None,
            **observation,
        )
        return self.store.commit(
            episode_id, edit, "accept" if decision == "accept" else "exclude", native=native
        )

    def export(self, selection: list[dict], output: Path) -> dict:
        receipts = []
        selected_splits = {s["episode_id"]: s["split"] for s in selection}

        def validate(revision):
            current = self.observe(revision["episode_id"], revision["native"]["annotation_id"])
            if current["native"]["snapshot_hash"] != revision["native"]["snapshot_hash"]:
                raise Conflict(
                    "native annotation changed; refresh and explicitly approve a new draft"
                )
            semantic = current["native"].get("semantic_review")
            if semantic and semantic["split"] != selected_splits[revision["episode_id"]]:
                raise ValueError("semantic export must preserve the frozen evaluation split")
            receipts.append(
                {
                    "episode_id": revision["episode_id"],
                    "revision": revision["revision"],
                    "snapshot_hash": current["native"]["snapshot_hash"],
                    "observed_at": current["native"]["observed_at"],
                }
            )

        manifest = export_dataset(self.store, selection, output, native_validator=validate)
        return {"manifest": manifest, "native_freshness": receipts, "output": str(output)}
