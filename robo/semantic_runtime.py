"""Use the existing isolated intake, prediction and training runtimes."""

import json
import os
import subprocess
from pathlib import Path

from robo.core import file_hash

REPO = Path(__file__).resolve().parents[1]


class StageFailure(RuntimeError):
    pass


class Runtime:
    def provider_identity(self, config):
        if config.provider == "baseline":
            from robo.semantic_baseline import PRODUCER

            paths = ["robo/semantic_baseline.py", "uv.lock"]
            producer = PRODUCER
        else:
            pin = json.loads((REPO / "integrations/semantic/assets.json").read_text())
            producer = pin["model_id"] + "@" + pin["revision"]
            paths = [
                "integrations/semantic/qwen.py",
                "integrations/semantic/assets.py",
                "integrations/semantic/assets.json",
                "integrations/semantic/uv.lock",
            ]
        return {"producer": producer, "files": {p: file_hash(REPO / p) for p in paths}}

    def call(self, project, arguments, logs):
        python = REPO / project / ".venv/bin/python"
        if not python.is_file():
            raise StageFailure(f"Missing runtime {python}; sync its locked environment first")
        logs.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "HF_HOME": str(REPO / ".cache/huggingface"),
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        }
        with (logs / "stdout.log").open("w") as out, (logs / "stderr.log").open("w") as err:
            result = subprocess.run(
                [str(python), *map(str, arguments)],
                cwd=REPO,
                env=env,
                stdout=out,
                stderr=err,
                timeout=900,
            )
        if result.returncode:
            raise StageFailure(f"Runtime exited {result.returncode}; see {logs / 'stderr.log'}")

    def intake(self, config, output):
        args = [
            "-m",
            "integrations.droid.intake",
            config.source,
            output,
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
            args.append("--allow-partial")
        self.call("integrations/droid", args, output.parent / "logs/intake")

    def predict(self, config, plan, store, output):
        if config.provider == "baseline":
            self.call(
                ".",
                [
                    "-m",
                    "robo.semantic_baseline",
                    plan,
                    store.root,
                    output,
                    "--split",
                    config.split,
                ],
                output.parent / "logs",
            )
        else:
            self.call(
                "integrations/semantic",
                [
                    "-m",
                    "integrations.semantic.qwen",
                    "--model-dir",
                    config.model_dir,
                    "--device",
                    config.device,
                    "--plan",
                    plan,
                    "--store",
                    store.root,
                    "--output",
                    output,
                    "--split",
                    config.split,
                ],
                output.parent / "logs",
            )

    def training(self, accepted, store, schema, split, output):
        self.call(
            "integrations/lerobot",
            [
                "-m",
                "robo.lerobot_export",
                "--accepted",
                accepted,
                "--source-root",
                store.root,
                "--schema",
                schema,
                "--split",
                split,
                "--output",
                output,
            ],
            output.parent / (output.name + "-logs"),
        )
