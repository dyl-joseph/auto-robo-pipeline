"""Controlled provider fixtures test orchestration, not pretrained label quality."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robo import task_benchmark as module
from robo.core import atomic_json, digest, file_hash
from robo.droid import CAMERAS, LocalOrigin, source_from_steps
from robo.semantic import EvaluationPlan, PredictionEpisode, Predictions
from robo.semantic_runtime import Runtime, StageFailure

SENTINEL = "REFERENCE_ONLY_SENTINEL_9e01"


class FixtureRuntime(Runtime):
    def __init__(self, video, *, count=1, fail=(), legacy=False, duplicate=False):
        self.video, self.count, self.fail = video, count, fail
        self.legacy, self.duplicate = legacy, duplicate
        self.shared_session = False
        self.prediction_calls = 0

    def call(self, project, arguments, logs):
        if project == "integrations/droid":
            self.intake_fixture(Path(arguments[2]), Path(arguments[3]))
            return
        args = dict(zip(arguments[2::2], arguments[3::2], strict=True))
        assert project == "integrations/semantic"
        assert "references" not in str(arguments) and SENTINEL not in str(arguments)
        assert json.loads(Path(args["--config"]).read_text()) == module.GENERATION
        assert SENTINEL not in Path(args["--plan"]).read_text()
        self.prediction_calls += 1
        plan = EvaluationPlan.model_validate_json(Path(args["--plan"]).read_bytes())
        spec = next(s for s in plan.episodes if s.split == "test")
        output = Path(args["--output"])
        configuration = {
            "generation": module.GENERATION,
            "evidence_kind": "controlled_provider_fixture",
        }
        if self.legacy:
            configuration = {"generation": {"max_episodes": 1}}
        failed = self.prediction_calls in self.fail
        row = PredictionEpisode(
            episode_id=spec.episode_id,
            error="Controlled inference failure" if failed else None,
            segments=[]
            if failed
            else [
                {
                    "episode_id": spec.episode_id,
                    "start_s": 0,
                    "end_s": spec.frame_boundaries_s[-1],
                    "source_frames": {
                        "start": 0,
                        "end_exclusive": len(spec.frame_boundaries_s) - 1,
                        "evidence": [0],
                    },
                    "subtask_text": "The robot moves an object",
                    "normalized_action": "motion",
                    "object_text": None,
                    "uncertainty": 0.8,
                    "provenance": {
                        "kind": "synthetic_fixture",
                        "producer": "fixture",
                        "artifact_sha256": digest("fixture"),
                    },
                }
            ],
        )
        predictions = Predictions(
            plan_sha256=plan.plan_sha256,
            run_id="fixture",
            producer=self.provider_identity(type("C", (), {"provider": "qwen"}))["producer"],
            code_revision="generated_fixture",
            config_sha256=digest(configuration),
            training_episode_ids=[],
            tuning_episode_ids=[],
            reference_access="none",
            episodes=[row],
        )
        atomic_json(output / "predictions.json", predictions.model_dump())
        atomic_json(
            output / "run.json",
            {
                "configuration": configuration,
                "task_context_policy": "provided-v1" if self.legacy else module.POLICY,
                "source_task_context_supplied": self.legacy,
                "files": {"predictions.json": file_hash(output / "predictions.json")},
                "failed_predictions": int(failed),
            },
        )
        if failed:
            raise StageFailure("Runtime exited 2")

    def intake_fixture(self, source, output):
        source.mkdir(exist_ok=True)
        for name in ("features.json", "dataset_info.json", "fixture.tfrecord"):
            (source / name).write_text("generated fixture bytes")
        records = []
        for index in range(self.count):
            origin = LocalOrigin(
                release="fixture",
                shard="fixture.tfrecord",
                shard_sha256=file_hash(source / "fixture.tfrecord"),
                features_sha256=file_hash(source / "features.json"),
                dataset_info_sha256=file_hash(source / "dataset_info.json"),
                record_index=index,
                generated_fixture=True,
            )
            directory = output / "episodes" / digest([origin.model_dump(), CAMERAS[0]])[:32]
            directory.mkdir(parents=True)
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(self.video),
                    "-vf",
                    f"hue=h={index * 30}",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "1",
                    str(directory / "video.mp4"),
                ],
                check=True,
            )
            rows = [
                {
                    "is_first": step == 0,
                    "is_last": step == 19,
                    "language_instruction": SENTINEL,
                    "language_instruction_2": "A second publisher wording",
                    "language_instruction_3": "Third wording",
                    **{
                        group: {"joint_position": [0.0] * 7, "gripper_position": [0.5]}
                        for group in ("observation", "action_dict")
                    },
                }
                for step in range(20)
            ]
            raw = {
                "episode_metadata": {
                    "file_path": (
                        f"/fixture/day-{0 if self.duplicate or self.shared_session else index}/"
                        f"episode-{0 if self.duplicate else index}/trajectory.h5"
                    )
                },
                "steps": rows,
                "image_rgb_sha256": [
                    {camera: digest([index, step, camera]) for camera in CAMERAS}
                    for step in range(20)
                ],
            }
            atomic_json(directory / "rlds-episode.json", raw)
            body = source_from_steps(rows, raw["episode_metadata"], CAMERAS[0], origin)
            body["provenance"] += (
                " Lossless numeric sidecar SHA256 "
                + file_hash(directory / "rlds-episode.json")
                + "."
            )
            atomic_json(directory / "source.json", body)
            atomic_json(directory / "training-schema.json", {"fixture": True})
            atomic_json(
                directory / "conversion.json",
                {
                    "origin": origin.model_dump(),
                    "selected_camera": CAMERAS[0],
                    "steps": 20,
                    "files": {p.name: file_hash(p) for p in directory.iterdir()},
                },
            )
            records.append(
                {
                    "directory": directory.name,
                    "conversion_sha256": file_hash(directory / "conversion.json"),
                    "steps": 20,
                }
            )
        atomic_json(
            output / "intake.json",
            {
                "status": "complete",
                "contract": {
                    "inventory": {
                        "metadata": {
                            name: file_hash(source / name)
                            for name in ("features.json", "dataset_info.json")
                        },
                        "missing": [],
                    }
                },
                "shards": {
                    "fixture.tfrecord": {
                        "sha256": file_hash(source / "fixture.tfrecord"),
                        "records": records,
                    }
                },
            },
        )


@pytest.fixture
def setup(fixture, tmp_path):
    _, manifest, body = fixture
    runtime = FixtureRuntime(manifest.parent / body["video"])
    config = module.Configuration(
        source=tmp_path / "rlds",
        model_dir=tmp_path / "weights",
        release="fixture",
        generated_fixture=True,
        exclusions_declared_by="fixture tester",
        exclusions_note="Generated records only",
    )
    benchmark = module.Benchmark(tmp_path / "benchmark", runtime=runtime)
    return benchmark, config, runtime


def test_cli_prepare_predict_report_and_resume(setup, monkeypatch, tmp_path, capsys):
    benchmark, config, runtime = setup
    monkeypatch.setattr(module, "Runtime", lambda: runtime)
    configuration = tmp_path / "config.json"
    atomic_json(configuration, config.model_dump(mode="json"))
    for stage in ("prepare", "predict", "report", "prepare", "predict", "report"):
        monkeypatch.setattr(
            sys, "argv", ["benchmark", stage, str(benchmark.root), "--config", str(configuration)]
        )
        assert module.main() == 0
    assert runtime.prediction_calls == 1
    report, path = benchmark.report()
    assert report["counts"]["frozen_selected"] == 1
    assert report["counts"]["judged"] == 0 and report["counts"]["unjudged_successful"] == 1
    assert len(report["items"][0]["publisher_descriptions"]) == 3
    assert not report["semantic_accuracy_measured"]
    assert report["boundary_metrics"]["status"] == "unavailable"
    assert report["training_exports"] == report["automatic_approvals"] == 0
    assert SENTINEL in path.read_text()
    # The actual CLI also reads an existing run in a fresh interpreter, without model access.
    result = subprocess.run(
        [sys.executable, "-m", "robo.task_benchmark", "report", str(benchmark.root)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("failure", ["partial", "all"])
def test_failures_remain_in_denominator_and_exit_nonzero(setup, monkeypatch, failure):
    benchmark, config, runtime = setup
    runtime.count, runtime.fail = 2, (1,) if failure == "partial" else (1, 2)
    benchmark.prepare(config)
    monkeypatch.setattr(module, "Runtime", lambda: runtime)
    for stage in ("predict", "report", "predict"):
        monkeypatch.setattr(sys, "argv", ["benchmark", stage, str(benchmark.root)])
        assert module.main() == 2
    report, _ = benchmark.report()
    assert report["counts"]["failed_predictions"] == len(runtime.fail)
    assert report["counts"]["frozen_selected"] == 2
    assert runtime.prediction_calls == 2
    assert all(
        row["publisher_descriptions"][0]["lexical_diagnostics"] is None
        for row in report["items"]
        if row["error"]
    )


@pytest.mark.parametrize(
    "artifact",
    [
        "plan.json",
        "scoring/references.json",
        "generation.json",
        "configuration.json",
        "predictions.json",
    ],
)
def test_changed_frozen_inputs_or_predictions_rejected(setup, artifact):
    benchmark, config, _ = setup
    benchmark.prepare(config)
    benchmark.predict()
    path = benchmark.owned(artifact)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        benchmark.report()
    with pytest.raises(ValueError, match="changed"):
        benchmark.predict()


def test_legacy_assisted_artifacts_cannot_be_reused(setup):
    benchmark, config, runtime = setup
    runtime.legacy = True
    benchmark.prepare(config)
    for _ in range(2):
        with pytest.raises(ValueError, match="assisted"):
            benchmark.predict()
    assert runtime.prediction_calls == 1


def test_interrupted_attempt_not_silently_retried(setup):
    benchmark, config, runtime = setup
    manifest = benchmark.prepare(config)
    spec = next(s for s in benchmark.load()[2].episodes if s.split == "test")
    atomic_json(
        benchmark.owned(f"jobs/{spec.episode_id}/attempt.json"),
        {
            "manifest_sha256": file_hash(benchmark.owned("manifest.json")),
            "plan_sha256": benchmark.load()[2].plan_sha256,
            "task_context_policy": module.POLICY,
        },
    )
    predictions = benchmark.predict()
    assert "Interrupted" in predictions.episodes[0].error
    assert runtime.prediction_calls == 0 and len(manifest["selected_episode_ids"]) == 1


def test_duplicate_recordings_rejected_before_prediction(setup):
    benchmark, config, runtime = setup
    runtime.count, runtime.duplicate = 2, True
    with pytest.raises(ValueError, match="Duplicate recording"):
        benchmark.prepare(config)
    assert runtime.prediction_calls == 0


def test_exclusions_apply_to_entire_session_and_cannot_change_after_prepare(setup):
    benchmark, config, runtime = setup
    runtime.count = 2
    config.excluded_session_ids = ["droid-day-" + digest("/fixture/day-0")[:24]]
    manifest = benchmark.prepare(config)
    assert len(manifest["excluded"]) == len(manifest["selected_episode_ids"]) == 1
    assert {s.split for s in benchmark.load()[2].episodes} == {"development", "test"}
    config.excluded_session_ids = []
    with pytest.raises(ValueError, match="changed"):
        benchmark.prepare(config)


def test_source_and_sidecar_changes_rejected(setup):
    benchmark, config, _ = setup
    manifest = benchmark.prepare(config)
    sidecar = benchmark.owned(manifest["mappings"][0]["directory"] + "/rlds-episode.json")
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    with pytest.raises(ValueError, match="derivative changed"):
        benchmark.predict()


def test_judgments_are_optional_bound_and_have_real_denominator(setup, tmp_path):
    benchmark, config, _ = setup
    benchmark.prepare(config)
    benchmark.predict()
    report, _ = benchmark.report()
    body = {
        key: report[key] for key in ("manifest_sha256", "references_sha256", "predictions_sha256")
    }
    body.update(
        rubric_id=module.RUBRIC["id"],
        method="independent_manual",
        reviewer="Fixture reviewer",
        recorded_at="2026-10-29",
        independent_of_prediction_author=True,
        entries=[
            {
                "episode_id": report["items"][0]["episode_id"],
                "verdict": "not_judgeable",
                "reason": "Generated fixture, not a real judgment",
            }
        ],
    )
    path = tmp_path / "judgments.json"
    atomic_json(path, body)
    scored, _ = benchmark.report(path)
    assert scored["counts"]["judged"] == 1 and scored["counts"]["unjudged_successful"] == 0
    assert scored["judgment_counts"] == {"not_judgeable": 1}
    body["predictions_sha256"] = digest("changed")
    atomic_json(path, body)
    with pytest.raises(ValueError, match="Judgments do not match"):
        benchmark.report(path)


def test_lexical_names_do_not_claim_semantics():
    assert module.lexical(" Pick UP  cup!", "pick up cup")["nfkc_casefold_word_exact_match"]
    assert module.lexical("cup", "cup cup")["word_multiset_f1"] == pytest.approx(2 / 3)


def test_session_exclusion_removes_all_views_before_prediction(setup):
    benchmark, config, runtime = setup
    runtime.count, runtime.shared_session = 2, True
    config.excluded_session_ids = ["droid-day-" + digest("/fixture/day-0")[:24]]
    manifest = benchmark.prepare(config)
    assert len(manifest["excluded"]) == 2 and manifest["selected_episode_ids"] == []
    assert {spec.split for spec in benchmark.load()[2].episodes} == {"development"}
    with pytest.raises(ValueError, match="All inventoried episodes are excluded"):
        benchmark.predict()
    assert runtime.prediction_calls == 0


def test_changed_original_shard_is_not_reused(setup):
    benchmark, config, _ = setup
    benchmark.prepare(config)
    (config.source / "fixture.tfrecord").write_text("different bytes")
    with pytest.raises(ValueError, match="Original RLDS source changed"):
        benchmark.predict()


def test_unbound_provider_directory_and_symlink_are_rejected(setup, tmp_path):
    benchmark, config, _ = setup
    manifest = benchmark.prepare(config)
    provider = benchmark.owned(f"jobs/{manifest['selected_episode_ids'][0]}/provider")
    provider.mkdir(parents=True)
    with pytest.raises(ValueError, match="Unbound provider"):
        benchmark.predict()
    provider.rmdir()
    provider.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        benchmark.predict()


def test_task_references_are_rejected_by_temporal_contract(setup):
    from robo.semantic import References

    benchmark, config, _ = setup
    benchmark.prepare(config)
    with pytest.raises(ValueError):
        References.model_validate(benchmark.read("scoring/references.json"))
