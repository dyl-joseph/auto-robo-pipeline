"""Freeze blind DROID predictions, then compare them with publisher task descriptions."""

import argparse
import json
import re
import subprocess
import time
import unicodedata
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from pydantic import Field, model_validator

from integrations.droid.intake import verify_derivative
from robo.core import Contract, digest, file_hash
from robo.droid import CAMERAS, LocalOrigin, source_from_steps
from robo.ingest import import_episode
from robo.pipeline import pipeline_lock
from robo.semantic import (
    EvaluationPlan,
    Identifier,
    PredictionEpisode,
    Predictions,
    Sha256,
    Text,
    validate_episode_segments,
)
from robo.semantic_baseline import source_episode
from robo.semantic_pipeline import freeze
from robo.semantic_plan import build_plan
from robo.semantic_review import read_inputs
from robo.semantic_routing import review_reasons
from robo.semantic_runtime import REPO, Runtime
from robo.store import Store

POLICY = "hidden-v1"
GENERATION = {"task_context_policy": POLICY, "max_episodes": 1}
# This entire DROID day was used while debugging the published pilot.
DEBUGGED_SESSION = "droid-day-68c6e34981a8bb6168957e07"
LANGUAGE_FIELDS = ("language_instruction", "language_instruction_2", "language_instruction_3")
RUBRIC = {
    "id": "droid-task-consistency-v1",
    "correct": "The activity agrees with a publisher task, including its object and goal.",
    "partly_correct": "Some activity agrees, but an important object, action or goal is missing.",
    "incorrect": "The prediction contradicts the task or describes an unrelated activity.",
    "not_judgeable": "The descriptions or prediction do not support a defensible task judgment.",
    "scope": "Task-description consistency only; a requested task does not prove observed success.",
}


class Configuration(Contract):
    source: Path
    model_dir: Path
    release: Identifier
    device: Literal["cpu", "mps", "cuda"] = "mps"
    camera: Literal[CAMERAS] = CAMERAS[0]
    source_split: Identifier = "train"
    max_episodes: int = Field(default=3, ge=1, le=5, strict=True)
    allow_partial: bool = False
    generated_fixture: bool = False
    excluded_episode_ids: list[Identifier] = Field(default_factory=list)
    excluded_session_ids: list[Identifier] = Field(default_factory=list)
    exclusions_declared_by: Identifier
    exclusions_note: Text

    @model_validator(mode="after")
    def paths_and_exclusions(self):
        self.source = self.source.expanduser().resolve()
        self.model_dir = self.model_dir.expanduser().resolve()
        for values in (self.excluded_episode_ids, self.excluded_session_ids):
            if len(values) != len(set(values)):
                raise ValueError("Exclusion IDs must be unique")
        return self


class Judgment(Contract):
    episode_id: Identifier
    verdict: Literal["correct", "partly_correct", "incorrect", "not_judgeable"]
    reason: Text


class Judgments(Contract):
    manifest_sha256: Sha256
    references_sha256: Sha256
    predictions_sha256: Sha256
    rubric_id: Literal["droid-task-consistency-v1"]
    method: Literal["independent_manual"]
    reviewer: Identifier
    recorded_at: Text
    independent_of_prediction_author: Literal[True]
    entries: list[Judgment]


def descriptions(raw):
    """Retain the original strings; repeated RLDS step copies are not temporal labels."""
    result = []
    for field in LANGUAGE_FIELDS:
        values = [step.get(field, "") for step in raw["steps"]]
        if any(not isinstance(v, str) for v in values) or len(set(values)) != 1:
            raise ValueError(f"Expected an episode-constant string field: {field}")
        if values[0].strip():
            result.append({"field": field, "text": values[0]})
    if not result:
        raise ValueError("No publisher task descriptions; cannot freeze this reference cohort")
    return result


def lexical(prediction, reference):
    def tokens(text):
        return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())

    a, b = tokens(prediction), tokens(reference)
    overlap = sum((Counter(a) & Counter(b)).values())
    return {
        "nfkc_casefold_word_exact_match": a == b,
        "word_multiset_f1": 2 * overlap / (len(a) + len(b)) if a or b else 0.0,
        "word_set_jaccard": len(set(a) & set(b)) / len(set(a) | set(b)) if a or b else 0.0,
    }


class Benchmark:
    def __init__(self, root, *, runtime=None):
        self.root = Path(root).absolute()
        if any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise ValueError("Benchmark paths must not contain symlinks")
        self.root.mkdir(parents=True, exist_ok=True)
        self.runtime = runtime or Runtime()

    def owned(self, name):
        path = self.root / name
        if not path.is_relative_to(self.root) or ".." in path.parts:
            raise ValueError("Artifact must stay inside the benchmark")
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Benchmark artifact must not be a symlink")
        return path

    def read(self, name):
        return json.loads(self.owned(name).read_bytes())

    def identity(self):
        return self.runtime.provider_identity(SimpleNamespace(provider="qwen"))

    def prepare(self, config):
        for source in (config.source, config.model_dir):
            if source.is_relative_to(self.root) or self.root.is_relative_to(source):
                raise ValueError("Keep benchmark output separate from source and model assets")
        with pipeline_lock(self):
            freeze(self.owned("configuration.json"), config.model_dump(mode="json"))
            if self.owned("manifest.json").exists():
                return self.load()[0]
            arguments = [
                "-m",
                "integrations.droid.intake",
                config.source,
                self.owned("intake"),
                "--release",
                config.release,
                "--camera",
                config.camera,
                "--split",
                config.source_split,
                "--max-episodes",
                config.max_episodes,
            ]
            if config.allow_partial:
                arguments.append("--allow-partial")
            if config.generated_fixture:
                arguments.append("--generated-fixture")
            self.runtime.call("integrations/droid", arguments, self.owned("logs/intake"))
            intake = self.read("intake/intake.json")
            if intake["status"] == "failed":
                raise ValueError("Intake failed; no benchmark cohort was frozen")
            store = Store(self.owned("store"))
            episodes, references, mappings, excluded = [], [], [], []
            seen, external = set(), dict(intake["contract"]["inventory"]["metadata"])
            sessions = set(config.excluded_session_ids) | {DEBUGGED_SESSION}
            for shard, receipt in intake["shards"].items():
                external[shard] = receipt["sha256"]
                for index, record in enumerate(receipt["records"]):
                    directory = self.owned("intake/episodes/" + record["directory"])
                    conversion = json.loads((directory / "conversion.json").read_bytes())
                    origin = LocalOrigin.model_validate(conversion["origin"])
                    if (origin.shard, origin.record_index, origin.shard_sha256) != (
                        shard,
                        index,
                        receipt["sha256"],
                    ) or origin.release != config.release:
                        raise ValueError("Conversion origin differs from the intake record")
                    if (
                        origin.features_sha256 != external["features.json"]
                        or origin.dataset_info_sha256 != external["dataset_info.json"]
                        or origin.generated_fixture != config.generated_fixture
                    ):
                        raise ValueError("Conversion origin differs from the source metadata")
                    verify_derivative(directory, origin, config.camera, record["conversion_sha256"])
                    raw = json.loads((directory / "rlds-episode.json").read_bytes())
                    if (
                        len(raw["steps"]) != record["steps"]
                        or len(raw["image_rgb_sha256"]) != record["steps"]
                    ):
                        raise ValueError("Sidecar frame count differs from intake")
                    derived = source_from_steps(
                        raw["steps"], raw["episode_metadata"], config.camera, origin
                    )
                    derived["provenance"] += (
                        " Lossless numeric sidecar SHA256 "
                        + file_hash(directory / "rlds-episode.json")
                        + "."
                    )
                    if derived != json.loads((directory / "source.json").read_bytes()):
                        raise ValueError("Source mapping differs from verified RLDS sidecar")
                    # Text changes cannot turn the same recording or camera frames into a new trial.
                    keys = [
                        ("recording", digest(raw["episode_metadata"]["file_path"])),
                        ("media", file_hash(directory / "video.mp4")),
                        (
                            "rgb",
                            digest([frame[config.camera] for frame in raw["image_rgb_sha256"]]),
                        ),
                    ]
                    if any(key in seen for key in keys):
                        raise ValueError("Duplicate recording or camera frames in benchmark cohort")
                    seen.update(keys)
                    episode = import_episode(directory / "source.json", store)
                    episodes.append(episode)
                    reasons = []
                    if (
                        episode["id"] in config.excluded_episode_ids
                        or episode["source_id"] in config.excluded_episode_ids
                    ):
                        reasons.append("declared_debugged_episode")
                        sessions.add(episode["session"])
                    if episode["session"] in sessions:
                        reasons.append("debugged_session")
                    mapping = {
                        "episode_id": episode["id"],
                        "dataset_episode_id": episode["source_id"],
                        "session_id": episode["session"],
                        "camera_id": config.camera,
                        "source_sha256": episode["content_hash"],
                        "media_sha256": episode["video_sha256"],
                        "origin": origin.model_dump(),
                        "conversion_sha256": record["conversion_sha256"],
                        "directory": str(directory.relative_to(self.root)),
                        "sidecar_sha256": file_hash(directory / "rlds-episode.json"),
                        "original_recording_path_sha256": keys[0][1],
                    }
                    mappings.append(mapping)
                    references.append({**mapping, "descriptions": descriptions(raw)})
                    excluded.append(reasons)
            if not episodes:
                raise ValueError("No intake records; no benchmark cohort was frozen")
            plan = build_plan(
                store,
                episodes,
                REPO / "docs/semantic-guidelines.md",
                dataset="DROID",
                camera=config.camera,
                split="test",
            )
            for spec, reasons in zip(plan.episodes, excluded, strict=True):
                if spec.session_id in sessions:
                    reasons.append("debugged_session")
                    spec.split = "development"
            plan = EvaluationPlan.model_validate(plan.model_dump())
            freeze(self.owned("plan.json"), plan.model_dump())
            freeze(self.owned("generation.json"), GENERATION)
            freeze(
                self.owned("scoring/references.json"),
                {
                    "schema_version": 1,
                    "kind": "publisher_episode_task_descriptions",
                    "identity_status": "unverified",
                    "human_verified": False,
                    "temporal_reference": False,
                    "license": "CC-BY-4.0",
                    "attribution": "DROID, Khazatsky et al. (2024)",
                    "episodes": references,
                },
            )
            frozen = [
                "configuration.json",
                "plan.json",
                "generation.json",
                "scoring/references.json",
                "intake/intake.json",
            ]
            manifest = {
                "schema_version": 1,
                "benchmark": "droid-task-reference-v1",
                "selection": "First bounded TFDS records in declared order, before predictions",
                "task_context_policy": POLICY,
                "provider": self.identity(),
                "implementation_sha256": file_hash(Path(__file__)),
                "files": {name: file_hash(self.owned(name)) for name in frozen},
                "source_files": external,
                "mappings": mappings,
                "selected_episode_ids": [s.episode_id for s in plan.episodes if s.split == "test"],
                "excluded": [
                    {
                        "episode_id": s.episode_id,
                        "session_id": s.session_id,
                        "reasons": sorted(set(r)),
                    }
                    for s, r in zip(plan.episodes, excluded, strict=True)
                    if r
                ],
                "identity_audit": "unverified",
                "known_debugged_session": DEBUGGED_SESSION,
                "partial_source": bool(intake["contract"]["inventory"]["missing"]),
                "intake_status": intake["status"],
                "candidate_limit": config.max_episodes,
                "generated_fixture": config.generated_fixture,
            }
            freeze(self.owned("manifest.json"), manifest)
            self.load()
            return manifest

    def load(self):
        manifest = self.read("manifest.json")
        if (
            manifest["task_context_policy"] != POLICY
            or manifest["provider"] != self.identity()
            or manifest["implementation_sha256"] != file_hash(Path(__file__))
        ):
            raise ValueError(
                "Benchmark implementation or hidden-context provider changed; use a new run"
            )
        for name, sha in manifest["files"].items():
            if file_hash(self.owned(name)) != sha:
                raise ValueError(f"Frozen benchmark artifact changed: {name}")
        config = Configuration.model_validate(self.read("configuration.json"))
        for name, sha in manifest["source_files"].items():
            if (
                Path(name).name != name
                or (config.source / name).is_symlink()
                or file_hash(config.source / name) != sha
            ):
                raise ValueError("Original RLDS source changed")
        plan = EvaluationPlan.model_validate(self.read("plan.json"))
        store = Store(self.owned("store"))
        for mapping in manifest["mappings"]:
            verify_derivative(
                self.owned(mapping["directory"]),
                LocalOrigin.model_validate(mapping["origin"]),
                mapping["camera_id"],
                mapping["conversion_sha256"],
            )
        for spec in plan.episodes:
            source_episode(store, spec)
        return manifest, config, plan, store

    def predict(self):
        with pipeline_lock(self):
            manifest, config, plan, store = self.load()
            if not manifest["selected_episode_ids"]:
                raise ValueError(
                    "All inventoried episodes are excluded; supply an untouched cohort"
                )
            if self.owned("prediction-receipt.json").exists():
                prior = self.read("prediction-receipt.json")
                if prior["predictions_sha256"] != file_hash(self.owned("predictions.json")):
                    raise ValueError("Frozen predictions changed")
                for key, sha in prior["receipts"].items():
                    if file_hash(self.owned(f"jobs/{key}/receipt.json")) != sha:
                        raise ValueError("Prediction receipt changed")
            rows, receipts = [], {}
            for spec in plan.episodes:
                if spec.split != "test":
                    continue
                job = f"jobs/{spec.episode_id}"
                job_plan = EvaluationPlan.model_validate(
                    {
                        **plan.model_dump(),
                        "episodes": [
                            s.model_dump()
                            for s in plan.episodes
                            if s.split != "test" or s.episode_id == spec.episode_id
                        ],
                    }
                )
                freeze(self.owned(job + "/plan.json"), job_plan.model_dump())
                binding = {
                    "manifest_sha256": file_hash(self.owned("manifest.json")),
                    "plan_sha256": job_plan.plan_sha256,
                    "task_context_policy": POLICY,
                }
                receipt_path = self.owned(job + "/receipt.json")
                attempt = self.owned(job + "/attempt.json")
                output = self.owned(job + "/provider")
                if receipt_path.exists():
                    receipt = self.read(job + "/receipt.json")
                    if receipt["binding"] != binding:
                        raise ValueError("Prediction receipt belongs to another benchmark")
                    for name, sha in receipt["files"].items():
                        if file_hash(self.owned(job + "/provider/" + name)) != sha:
                            raise ValueError("Immutable prediction or provider artifact changed")
                    row = PredictionEpisode.model_validate(receipt["prediction"])
                else:
                    if output.exists() and not attempt.exists():
                        raise ValueError(
                            "Unbound provider output; legacy assisted artifacts cannot be reused"
                        )
                    error = None
                    elapsed = None
                    if not attempt.exists():
                        freeze(attempt, binding)
                        started = time.monotonic()
                        try:
                            self.runtime.call(
                                "integrations/semantic",
                                [
                                    "-m",
                                    "integrations.semantic.qwen",
                                    "--model-dir",
                                    config.model_dir,
                                    "--device",
                                    config.device,
                                    "--plan",
                                    self.owned(job + "/plan.json"),
                                    "--store",
                                    store.root,
                                    "--output",
                                    output,
                                    "--split",
                                    "test",
                                    "--config",
                                    self.owned("generation.json"),
                                ],
                                self.owned(job + "/logs"),
                            )
                        except (
                            ValueError,
                            OSError,
                            RuntimeError,
                            subprocess.SubprocessError,
                        ) as exc:
                            error = str(exc)[:1000] or type(exc).__name__
                        elapsed = time.monotonic() - started
                    else:
                        if self.read(job + "/attempt.json") != binding:
                            raise ValueError("Attempt binding changed")
                        error = "Interrupted attempt; retained as failure. Use a new run to retry."
                    files, execution = {}, None
                    if output.exists():
                        run = self.read(job + "/provider/run.json")
                        generation = run["configuration"]["generation"]
                        if (
                            run.get("task_context_policy") != POLICY
                            or generation.get("task_context_policy") != POLICY
                            or run.get("source_task_context_supplied") is not False
                        ):
                            raise ValueError(
                                "Provider is assisted or lacks explicit hidden-context provenance"
                            )
                        if any(generation.get(k) != v for k, v in GENERATION.items()):
                            raise ValueError("Provider generation configuration differs")
                        files = {**run["files"], "run.json": file_hash(output / "run.json")}
                        for name, sha in files.items():
                            if (
                                Path(name).name != name
                                or file_hash(self.owned(job + "/provider/" + name)) != sha
                            ):
                                raise ValueError("Provider artifact hash differs")
                        _, predictions = read_inputs(
                            self.owned(job + "/plan.json").read_bytes(),
                            self.owned(job + "/provider/predictions.json").read_bytes(),
                        )
                        if (
                            len(predictions.episodes) != 1
                            or predictions.episodes[0].episode_id != spec.episode_id
                            or predictions.producer != manifest["provider"]["producer"]
                            or predictions.config_sha256 != digest(run["configuration"])
                            or predictions.training_episode_ids
                            or predictions.tuning_episode_ids
                        ):
                            raise ValueError("Prediction identity or configuration differs")
                        row = predictions.episodes[0]
                        if run["failed_predictions"] != int(row.error is not None):
                            raise ValueError("Provider failure count differs")
                        if not row.error:
                            validate_episode_segments(
                                row.segments, spec, plan.normalized_actions, complete=True
                            )
                        execution = run["configuration"]
                        if error and not row.error:
                            row = PredictionEpisode(
                                episode_id=spec.episode_id, segments=[], error=error
                            )
                    else:
                        row = PredictionEpisode(
                            episode_id=spec.episode_id,
                            segments=[],
                            error=error or "Provider published no result",
                        )
                    receipt = {
                        "binding": binding,
                        "files": files,
                        "prediction": row.model_dump(),
                        "execution": execution,
                        "elapsed_seconds": elapsed,
                        "timing_scope": (
                            "Provider subprocess including model loading, decoding and generation; "
                            "not model-only latency"
                        ),
                        "memory": {"measured": False, "peak_metal_bytes": None},
                    }
                    freeze(receipt_path, receipt)
                rows.append(row)
                receipts[spec.episode_id] = file_hash(receipt_path)
            self.load()
            predictions = Predictions(
                plan_sha256=plan.plan_sha256,
                run_id=digest(receipts)[:24],
                producer=manifest["provider"]["producer"],
                code_revision="droid-task-reference-v1",
                config_sha256=digest(manifest),
                training_episode_ids=[],
                tuning_episode_ids=[],
                reference_access="none",
                episodes=rows,
            )
            freeze(self.owned("predictions.json"), predictions.model_dump())
            freeze(
                self.owned("prediction-receipt.json"),
                {
                    "manifest_sha256": file_hash(self.owned("manifest.json")),
                    "predictions_sha256": file_hash(self.owned("predictions.json")),
                    "receipts": receipts,
                },
            )
            return predictions

    def report(self, judgments=None):
        with pipeline_lock(self):
            manifest, _, plan, _ = self.load()
            receipt = self.read("prediction-receipt.json")
            binding = {
                "manifest_sha256": file_hash(self.owned("manifest.json")),
                "references_sha256": file_hash(self.owned("scoring/references.json")),
                "predictions_sha256": file_hash(self.owned("predictions.json")),
            }
            if any(
                receipt[key] != binding[key] for key in ("manifest_sha256", "predictions_sha256")
            ):
                raise ValueError("Frozen predictions or manifest changed")
            for key, sha in receipt["receipts"].items():
                if file_hash(self.owned(f"jobs/{key}/receipt.json")) != sha:
                    raise ValueError("Prediction receipt changed")
                for name, expected in self.read(f"jobs/{key}/receipt.json")["files"].items():
                    if file_hash(self.owned(f"jobs/{key}/provider/{name}")) != expected:
                        raise ValueError("Immutable provider output changed")
            _, predictions = read_inputs(
                self.owned("plan.json").read_bytes(), self.owned("predictions.json").read_bytes()
            )
            if [r.episode_id for r in predictions.episodes] != manifest["selected_episode_ids"]:
                raise ValueError("Predictions omit or reorder frozen cohort members")
            judged = {}
            judgment_sha256 = file_hash(Path(judgments)) if judgments else None
            if judgments:
                judgments = Judgments.model_validate_json(Path(judgments).read_bytes())
                if any(getattr(judgments, key) != value for key, value in binding.items()):
                    raise ValueError(
                        "Judgments do not match frozen manifest, references and predictions"
                    )
                for entry in judgments.entries:
                    if (
                        entry.episode_id in judged
                        or entry.episode_id not in manifest["selected_episode_ids"]
                    ):
                        raise ValueError("Duplicate or unknown judgment episode")
                    judged[entry.episode_id] = entry
            references = {
                r["episode_id"]: r for r in self.read("scoring/references.json")["episodes"]
            }
            specs = {s.episode_id: s for s in plan.episodes}
            items = []
            for prediction in predictions.episodes:
                key = prediction.episode_id
                if prediction.error and key in judged:
                    raise ValueError("Failed inference cannot receive a task-correctness judgment")
                text = " ".join(s.subtask_text for s in prediction.segments)
                items.append(
                    {
                        "episode_id": key,
                        "error": prediction.error,
                        "prediction_text": text,
                        "segments": [s.model_dump() for s in prediction.segments],
                        "publisher_descriptions": [
                            {
                                **ref,
                                "lexical_diagnostics": lexical(text, ref["text"])
                                if not prediction.error
                                else None,
                            }
                            for ref in references[key]["descriptions"]
                        ],
                        "reference_mapping": {
                            k: v for k, v in references[key].items() if k != "descriptions"
                        },
                        "review_reasons": review_reasons(plan, specs[key], prediction),
                        "judgment": judged[key].model_dump() if key in judged else None,
                        "execution": self.read(f"jobs/{key}/receipt.json"),
                    }
                )
            failed = sum(p.error is not None for p in predictions.episodes)
            report = {
                "schema_version": 1,
                **binding,
                "task_context_policy": POLICY,
                "generated_fixture": manifest["generated_fixture"],
                "intake_status": manifest["intake_status"],
                "status": "failed"
                if failed == len(items)
                else "partial_failure"
                if failed
                else "complete",
                "counts": {
                    "frozen_selected": len(items),
                    "failed_predictions": failed,
                    "successful_predictions": len(items) - failed,
                    "judged": len(judged),
                    "unjudged_successful": len(items) - failed - len(judged),
                    "excluded_before_prediction": len(manifest["excluded"]),
                    "review_routed": sum(bool(row["review_reasons"]) for row in items),
                },
                "judgment_counts": dict(Counter(j.verdict for j in judged.values())),
                "judgment_file_sha256": judgment_sha256,
                "judgment_provenance": judgments.model_dump(exclude={"entries"})
                if judgments
                else None,
                "rubric": RUBRIC,
                "items": items,
                "label_exposure": (
                    "Publisher task text withheld from permitted model inputs; original stores "
                    "and sidecars retain labels. This is not OS-level isolation."
                ),
                "boundary_metrics": {
                    "status": "unavailable",
                    "reason": (
                        "Episode instructions contain no temporal truth; use the separate temporal "
                        "evaluator with audited temporal references."
                    ),
                },
                "semantic_accuracy_measured": False,
                "human_labor_measured": False,
                "automatic_approvals": 0,
                "training_exports": 0,
                "limitations": [
                    "Lexical diagnostics are not semantic accuracy.",
                    "Tasks describe intent, not verified observed success.",
                    "Identity and exclusions are declared, not independently audited.",
                    "Pretrained model contamination is unknown.",
                    "Only the frozen bounded local cohort is measured; no dataset-wide claim.",
                    "No temporal boundary truth or policy-quality measurement.",
                ],
            }
            path = self.owned("scoring/reports/" + digest(report) + ".json")
            freeze(path, report)
            return report, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "predict", "report"))
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--judgments", type=Path)
    args = parser.parse_args()
    try:
        benchmark = Benchmark(args.workspace)
        if args.stage == "prepare":
            if args.config is None:
                parser.error("prepare requires --config")
            manifest = benchmark.prepare(
                Configuration.model_validate_json(args.config.read_bytes())
            )
            result = {
                "status": "prepared",
                "selected": len(manifest["selected_episode_ids"]),
                "excluded": len(manifest["excluded"]),
                "manifest_sha256": file_hash(benchmark.owned("manifest.json")),
            }
            code = 0 if result["selected"] else 3
        elif args.stage == "predict":
            predictions = benchmark.predict()
            failed = sum(row.error is not None for row in predictions.episodes)
            result = {"episodes": len(predictions.episodes), "failed_predictions": failed}
            code = 2 if failed else 0
        else:
            report, path = benchmark.report(args.judgments)
            result = {"report": str(path), **report["counts"]}
            code = 2 if report["counts"]["failed_predictions"] else 0
        print(json.dumps(result, indent=2))
        return code
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
