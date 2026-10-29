# One-episode semantic check on a Mac

This checks execution on your Mac: local DROID intake, a label-free plan, baseline/Qwen predictions, and native correction/export. It does not measure annotation accuracy. Use a revision containing the semantic review adapter and `robo.semantic_plan`. Your existing PI05/MLX checkout and evidence need no changes; Qwen uses a separate PyTorch MPS environment.

## 1. Install the isolated runtimes

From the repository root, with Python 3.12, uv 0.12.23 and FFmpeg/FFprobe installed:

```bash
uv sync --locked --python 3.12
uv sync --locked --project integrations/droid
uv sync --locked --project integrations/semantic
mkdir -p artifacts/mac-semantic
uv run --locked --project integrations/semantic python -c \
  'import platform, torch; print(platform.platform(), torch.__version__); assert torch.backends.mps.is_available(), "MPS unavailable"' \
  > artifacts/mac-semantic/hardware.txt
```

The [recorded setup check](evidence/mac-semantic-setup.json) covers the real one-episode intake, plan, baseline and review-bundle commands on Linux. Full pretrained Qwen execution, memory use and performance on the M5 Pro remain for your hardware to verify. A failed accelerator check should be reported as a failure, not rerun as a successful CPU test.

## 2. Download and verify the exact Qwen assets

The public Apache-2.0 [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) snapshot is about 4.27 GB including its tokenizer. This command uses the repository's pinned revision and file list with the official [Hub downloader](https://huggingface.co/docs/huggingface_hub/guides/download). It downloads only model assets, anonymously; it sends no recordings.

```bash
HF_HUB_DISABLE_TELEMETRY=1 uv run --locked --project integrations/semantic python - <<'PY'
from huggingface_hub import snapshot_download
from integrations.semantic.assets import PIN
snapshot_download(
    repo_id=PIN["model_id"], revision=PIN["revision"],
    allow_patterns=list(PIN["files"]), local_dir="models/local/qwen3-vl-2b",
    token=False, max_workers=2,
)
PY

HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  uv run --locked --project integrations/semantic python -m integrations.semantic.qwen \
  --model-dir models/local/qwen3-vl-2b --preflight \
  > artifacts/mac-semantic/assets.json
```

If those exact files are already local, use their directory and skip the download. Preflight checks every required size and hash, including the weight file. It must succeed before inference. No new token or model-agreement acceptance is needed for this pinned public release.

## 3. Import one episode and create its plan

For a reproducible first check, use the existing official 9.54 MB pilot shard. Repeating acquisition reuses verified local files. No full DROID download is needed.

```bash
uv run --locked python -m scripts.fetch_droid
uv run --locked --project integrations/droid python -m integrations.droid.intake \
  data/local/droid-sample data/local/mac-semantic-inbox \
  --release droid_100/1.0.0 --max-episodes 1 --allow-partial

uv run --locked python -m robo --store data/local/mac-semantic-store import \
  data/local/mac-semantic-inbox/episodes/*/source.json \
  > artifacts/mac-semantic/import.json

uv run --locked python -m robo.semantic_plan \
  --store data/local/mac-semantic-store --dataset DROID \
  --camera exterior_image_1_left --split development \
  --guidelines docs/semantic-guidelines.md --output artifacts/mac-semantic/plan.json
```

The glob must match exactly one manifest. Use fresh inbox/store paths for this first check. The plan records the imported source hashes, frame clock and guideline hash. Its identity audit is `unverified`, and it contains no reference labels. It is a **development** episode so inspecting its predictions does not turn it into a claimed held-out test.

For your own downloaded folder, first confirm it contains `dataset_info.json`, `features.json` and TFRecord shards. Replace the intake source path and release name; omit `--allow-partial` for a complete shard set. The adapter does not handle raw HDF5 folders. Its byte limits remain active even with `--max-episodes 1`; see [local intake](droid-local-intake.md) before changing them. Your unprovided folder layout has not been tested.

## 4. Run the baseline and Qwen on the same episode

```bash
uv run --locked python -m robo.semantic_baseline \
  artifacts/mac-semantic/plan.json data/local/mac-semantic-store \
  artifacts/mac-semantic/baseline --split development

PYTORCH_ENABLE_MPS_FALLBACK=0 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  /usr/bin/time -l uv run --locked --project integrations/semantic \
  python -m integrations.semantic.qwen \
  --model-dir models/local/qwen3-vl-2b --device mps --split development \
  --plan artifacts/mac-semantic/plan.json --store data/local/mac-semantic-store \
  --output artifacts/mac-semantic/qwen \
  > artifacts/mac-semantic/qwen-result.json 2> artifacts/mac-semantic/qwen-time.log
```

Use a new Qwen output directory for every run. Keep `run.json` and the window artifacts: they record failures, sampled frames, token counts, asset hashes and actual device/dtype. `/usr/bin/time -l` records elapsed time and maximum resident set size for the command; that is not a measurement of peak Metal/unified-memory allocation. It includes startup and asset verification, so it is not model-only inference latency.

Inspect `failed_predictions` in `qwen/run.json`. A process exit alone does not establish successful labeling: malformed model output produces an explicit failed episode. Keep that failure and its raw reply. Even a valid reply is an unvalidated prediction until compared with independent references. Defaults keep this run to one episode, bounded windows and at most 16 supplied images per window.

## 5. Correct and explicitly export

Read the episode ID printed by plan creation, then prepare the Qwen result (or use `baseline/predictions.json` to inspect the baseline instead):

```bash
uv run --locked python -m robo.semantic_review \
  --store data/local/mac-semantic-store --plan artifacts/mac-semantic/plan.json \
  --predictions artifacts/mac-semantic/qwen/predictions.json \
  --episode EPISODE_ID --output artifacts/mac-semantic/review

uv run --locked python -m robo --store data/local/mac-semantic-store serve
```

Open `http://127.0.0.1:8000`, use your local Label Studio login, and import `review/tasks.json` with `review/review.xml` and the [local storage setup](labelstudio.md#open-a-review-project). With an existing authorized local SDK credential, [bind the task, refresh its saved annotation and explicitly accept it](native-approval.md#operator-workflow). Keep the original predictions unchanged. The [semantic review guide](semantic-review.md) explains per-region text and boundary editing.

Create `artifacts/mac-semantic/selection.json` with the actual episode ID and accepted bridge revision:

```json
[{"episode_id":"ACTUAL_EPISODE_ID","revision":2,"split":"development"}]
```

Revision 2 assumes a first draft followed by its first approval; use `native-review inspect` to obtain the actual value. Then export:

```bash
uv run --locked python -m robo --store data/local/mac-semantic-store native-review export \
  artifacts/mac-semantic/selection.json artifacts/mac-semantic/accepted
uv run --locked python -m robo validate-export artifacts/mac-semantic/accepted
```

Optional [LeRobot conversion](semantic-review.md#export-frame-aligned-text) uses the single `training-schema.json` under this inbox, the same source store and `--split development`. It preserves corrected text as annotation features; it does not train a policy or change PI05 conditioning.

## 6. Check the training path without a browser

Use a separate `train` plan for this integration check. The public pilot can be reused as a training fixture; that does not make it independent benchmark data. Development and test exports cannot be passed to the trainer. Download or supply the [full pretrained PI05 assets](mlx.md#run-full-inference-with-synthetic-observations) first, then install the dataset and training runtimes:

```bash
uv sync --locked --project integrations/lerobot
uv sync --locked --project integrations/mlx_pi
uv run --locked python -m robo.semantic_plan \
  --store data/local/mac-semantic-store --dataset DROID \
  --camera exterior_image_1_left --split train \
  --guidelines docs/semantic-guidelines.md --output artifacts/mac-semantic/train-plan.json

PYTORCH_ENABLE_MPS_FALLBACK=0 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  uv run --locked --project integrations/semantic python -m integrations.semantic.qwen \
  --model-dir models/local/qwen3-vl-2b --device mps --split train \
  --plan artifacts/mac-semantic/train-plan.json --store data/local/mac-semantic-store \
  --output artifacts/mac-semantic/qwen-train

uv run --locked python -m scripts.semantic_api_verify \
  --source data/local/mac-semantic-inbox/episodes/*/source.json \
  --plan artifacts/mac-semantic/train-plan.json \
  --predictions artifacts/mac-semantic/qwen-train/predictions.json \
  --output artifacts/mac-semantic/training-api-check

uv run --locked --project integrations/lerobot python -m robo.lerobot_export \
  --accepted artifacts/mac-semantic/training-api-check/accepted-refreshed \
  --source-root artifacts/mac-semantic/training-api-check/source \
  --schema data/local/mac-semantic-inbox/episodes/*/training-schema.json \
  --split train --output artifacts/mac-semantic/lerobot-train

MLX_ENABLE_TF32=0 PYTORCH_ENABLE_MPS_FALLBACK=0 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  uv run --locked --project integrations/mlx_pi python -m integrations.mlx_pi.train_export \
  --dataset artifacts/mac-semantic/lerobot-train --checkpoint artifacts/pi05-base \
  --output artifacts/mac-semantic/pi05-training --sample-index 16 \
  --camera-map observation.images.source=observation.images.base_0_rgb \
  --task-key annotation.subtask_text \
  --action-representation droid_absolute_joint_positions_gripper_closure
```

The API verifier uses the actual Label Studio backend and official SDK. Its edits and approvals are automated fixtures. It preserves the original Qwen predictions and checks save/reload, cleared objects, stale approvals and historical exports. It verifies persisted backend state without a rendered editor.

The trainer consumes the exported Qwen subtask text, real camera/state/action data and statistics fitted only to this train export. It runs the full frozen encoder on MPS, updates the pretrained action expert twice on MLX Metal, and requires exact checkpoint resume and finite postprocessed actions. Frame 16 lies inside a sufficiently long subtask in the [recorded check](evidence/mac-semantic-e2e.json). Other data or predictions may need a different explicit index. A chunk crossing an episode or subtask boundary is rejected. Keep the training `report.json`, `actions.npz` and checkpoint manifest.

## What to report

Keep the git revision, `hardware.txt`, preflight result, `qwen/run.json`, `qwen-time.log`, and whether native save/reload and approved export succeeded. If the run failed, retain the exact command and error. These are execution findings. No raw dataset, model weights or session credentials need to be shared.

Accuracy evaluation is a separate step: collect independently annotated temporal references, audit session/camera identities and near duplicates, freeze untouched test episodes and the annotation rules, then score **uncorrected** predictions with the [semantic evaluator](semantic-evaluation.md). Do not reuse this inspected development episode as untouched test evidence or treat assisted corrections as independent gold.
