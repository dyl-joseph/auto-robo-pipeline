# Run the real DROID demo

This walkthrough downloads one official episode, exercises native review and LeRobot export, then tests recovery from known corruptions. It uses local files and disposable test sessions. The browser corrections and approvals are automated, so the resulting labels demonstrate the software path rather than human annotation quality.

Run commands from the repository root with Python 3.12, uv 0.12.23, FFmpeg and a Chromium or Chrome installation that supports H.264. The [development guide](development.md) covers browser installation.

The [recorded local run](evidence/final-demo.json) includes source hashes, all 119 loaded rows, queue results, benchmark counts and MLX CPU measurements.

## Prepare the episode

```bash
./scripts/setup.sh
uv run --locked python scripts/check_environment.py
uv run --locked python -m robo preflight
uv run --locked python -m scripts.fetch_droid
uv sync --locked --project integrations/droid
TF_CPP_MIN_LOG_LEVEL=2 uv run --locked --project integrations/droid \
  python -m integrations.droid.convert data/local/droid-sample data/local/droid-converted
```

Acquisition downloads 9,560,772 bytes including metadata, and reuses matching files on later runs. Conversion requires a new output directory. If `data/local/droid-converted` already contains a verified conversion, keep it and continue.

`conversion.json` records one episode, 119 steps and 119 decoded images for each of three cameras. `source.json` selects one camera and preserves an inferred 15 Hz clock. The [DROID guide](droid.md) documents the source, CC-BY-4.0 license and absolute-joint action schema.

## Run native correction and export

```bash
npm ci --cache .cache/npm
PLAYWRIGHT_BROWSERS_PATH=.cache/playwright npx playwright install ffmpeg
ROBO_NATIVE_EXPORT_FIXTURE=artifacts/demo-native \
  uv run --locked python -m scripts.labelstudio_smoke --browser \
  --source data/local/droid-converted/source.json
```

Set `CHROMIUM_PATH` to your browser executable if needed. Use a new `ROBO_NATIVE_EXPORT_FIXTURE` path on each run. The test opens the actual Label Studio application, moves a boundary from 0.6 to 0.4 seconds, submits and reloads the annotation. It approves a test revision and exports it with outcome `unknown` and quality `incomplete`.

The fixture contains the source media and accepted metadata needed by the next step. Its application database and session are removed when the test exits.

## Load the exported data through LeRobot

```bash
uv sync --locked --project integrations/lerobot
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
uv run --locked --project integrations/lerobot python -m integrations.lerobot.verify_droid \
  data/local/droid-converted artifacts/demo-native artifacts/demo-lerobot
```

The check loads all 119 rows through the official LeRobot reader, compares numeric features with decoded RLDS values, decodes every image and creates a DataLoader batch with action windows. It records zero omitted or repeated source frames. Use an unused output path for a separate run.

## Exercise the queue and recovery

```bash
uv run --locked python -m scripts.labelstudio_smoke --batch \
  --source data/local/droid-converted/source.json
uv run --locked python -m scripts.messy_droid \
  data/local/droid-converted/source.json artifacts/messy-droid
```

The queue test adds two episode variants, an exact duplicate and an invalid manifest. It verifies native priority order, recovers a task created before an interrupted response, changes an uncertainty reason and exports one explicitly approved test annotation. Screenshots appear at `artifacts/batch-queue.png` and `artifacts/batch-review.png`.

The benchmark creates 16 separate cases. Expected counts are four complete repairs, three data-review cases, seven rejected inputs and two unchanged controls. It runs the batch twice, checks source and input hashes, and compares retained frames, numeric values and alignment against the untouched source. See `artifacts/messy-droid/report.json` and the [benchmark guide](messy-droid.md).

To review your own annotations instead of running the fixture, follow the [batch workflow](batch-workflow.md). It uses your local Label Studio login and requires a separate acceptance decision.

## Check the MLX adapter

```bash
uv sync --locked --project integrations/mlx_pi
uv run --locked --project integrations/mlx_pi python -m pytest -q integrations/mlx_pi
uv run --locked --project integrations/mlx_pi python -m integrations.mlx_pi.verify
```

These checks use small random PI05 models on CPU and write `artifacts/mlx-pi05.json`. They cover numerical agreement, gradients, an optimizer step and checkpoint resume. The [MLX guide](mlx.md) explains how to run the explicit Metal check on the M5 Pro and what is still required for pretrained execution. This demo does not download the full checkpoint or accept a gated tokenizer agreement.
