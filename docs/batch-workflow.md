# Process an inbox and review what needs attention

The batch command imports every episode it can, generates cached motion suggestions and prepares a Label Studio queue. Invalid inputs get a report with the failed stage and a next action. They do not stop other episodes. Rerunning the inbox checks source integrity and reuses completed artifacts.

![Real DROID episode opened from the native review queue](evidence/droid-batch-review.png)

The screenshot shows an automated integration test, not human review. The video frame is from Khazatsky et al., [DROID (2024)](https://droid-dataset.github.io/), used under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). It is displayed in Label Studio with a derived review video and inferred 15 Hz timing; the test variant has missing numeric samples. The original dataset remains unchanged.

## Start with local episodes

Each episode needs a `source.json` or `*.source.json` manifest following the [data contract](data-contract.md). Video paths are relative to their manifests. Each numeric stream must include an explicit `offset_s` and `clock_evidence`. Intake sorts paired rows and removes exact duplicates; conflicting values and unknown clocks require corrected input. An inbox can contain several episode directories; keep the pipeline store outside it.

For the existing DROID sample, use `data/local/droid-converted` after following the [download/conversion steps](droid.md):

```bash
./scripts/setup.sh
uv run --locked python -m robo preflight
uv run --locked python -m robo --store data/local/work batch run data/local/droid-converted
uv run --locked python -m robo --store data/local/work batch status
```

The output includes a report path under `data/local/work/batches/`. `report.json` contains the full result; `summary.txt` lists failed inputs and pending reviews. `needs_input` means at least one input failed, even when other episodes are ready. Repeated records for the same immutable episode create one review request. Episodes with identical video but different metadata remain distinct and receive a duplicate warning.

## Open the queue in Label Studio

Start Label Studio in one terminal:

```bash
uv run --locked python -m robo --store data/local/work serve
```

Use your existing local login and authorized API key. The workflow creates no account or credential. With `LABEL_STUDIO_API_KEY` already set in another terminal:

```bash
uv run --locked python -m robo --store data/local/work batch sync
```

`sync` publishes tasks, configures local proxy storage and creates a **Review priority** tab. The table shows task, priority and reasons such as missing samples or weak motion evidence. Numeric sorting puts higher priorities first. Report links open this queue, and `summary.txt` gives the native task ID to select. Click that row to review video, sensors and proposed regions in the existing editor. Projects are grouped by labeling configuration, so different channel layouts or review frame rates do not share an incompatible editor.

If a request is interrupted after Label Studio creates a task, rerun `sync`. It finds and validates that task before binding it locally. Conflicting matches stop synchronization with an error. Repeated syncs reuse projects, storage connections, views and tasks. Queue priority and reason can change as reviews develop; source data, predictions and annotations retain their provenance checks.

Python callers can pass an authenticated SDK session to `robo.native_queue.sync_queue`. The disposable integration test uses this route.

## Correct, approve and export

Submit the annotation in Label Studio, then use the [native approval commands](native-approval.md) to import that specific annotation and explicitly accept or reject it. An uncertain annotation cannot be accepted until the uncertainty is resolved. Ranking never approves an annotation.

```bash
uv run --locked python -m robo --store data/local/work batch status
uv run --locked python -m robo --store data/local/work batch export artifacts/accepted --split train
```

Export rechecks the native annotation before writing. Unreviewed, rejected or changed annotations cannot enter a new accepted export. The result feeds the existing [LeRobot conversion](lerobot.md). These commands use the latest batch by default. Select an older batch with a report path on `status`/`sync`, or `--report PATH` on `export`.

For a DROID batch in `data/local/work`, convert the approved export with its generated schema:

```bash
uv sync --locked --project integrations/lerobot
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
uv run --locked --project integrations/lerobot python -m robo.lerobot_export \
  --accepted artifacts/accepted --source-root data/local/work \
  --schema data/local/droid-converted/training-schema.json \
  --split train --output artifacts/lerobot-train
```

Use your own explicit schema for other recordings. Approval does not override the converter's checks for missing state/actions or unresolved clocks.

## What the counts mean

Reports count inputs, unique prepared episodes, duplicate records, failures by stage, pending reviews and locally approved episodes. They measure workload and recovery behavior, not human time savings or label accuracy. Priorities use validation warnings, explicit uncertainty and heuristic motion evidence; they are not calibrated probabilities.

[Integration evidence](evidence/batch-workflow.json) uses real DROID values in four input records: two episode variants, one repeated record and one invalid manifest. It checks interrupted-import recovery, numeric sorting, opening tasks in the actual editor, changing priority after an uncertain annotation, and correction/approval/export. Reviewer actions are automated test fixtures. The [controlled corruption benchmark](messy-droid.md) separately measures repair and detection quality.

## Before pretrained PI05 use

Review/export works without PI weights. The [MLX adapter](mlx.md) has a separate runtime with local MPS and MLX Metal verification. Inspect supplied local assets without downloading anything:

```bash
uv run --locked python -m robo preflight \
  --weights /path/to/authorized/model.safetensors \
  --tokenizer /path/to/authorized/tokenizer \
  --metal-evidence artifacts/mlx-metal.json
```

The report distinguishes review readiness, local file checks and recorded small-model Metal evidence. It checks a saved Hugging Face tokenizer directory and does not execute the model. The separate [full synthetic inference command](mlx.md#run-full-inference-with-synthetic-observations) uses the official public SentencePiece asset and has passed on the M5 Pro. Full pretrained training and sustained performance remain unmeasured.
