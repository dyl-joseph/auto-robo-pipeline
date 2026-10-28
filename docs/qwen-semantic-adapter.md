# Local Qwen semantic adapter

The adapter uses the official [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) through [Transformers](https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl). It submits short sequences of sampled images with source-frame numbers and timestamps, then validates the returned temporal segments. It uses no hosted inference API and uploads no recordings.

The pinned release is `89644892e4d85e24eaac8bacfd4f463576704203`, published under Apache-2.0 without an access gate. Its weight file is 4,255,140,312 bytes. The [asset manifest](../integrations/semantic/assets.json) records the publisher's file sizes and hashes. Only 18,342 bytes of metadata/configuration were saved during development; no pretrained weights were downloaded.

## Setup and asset check

```bash
uv sync --locked --project integrations/semantic
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  uv run --locked --project integrations/semantic python -m integrations.semantic.qwen \
  --model-dir PATH_TO_PINNED_QWEN_DIRECTORY --preflight
```

Supply a local copy of that exact official release, including its tokenizer, processor and chat-template files. Preflight verifies every required asset before model loading. It rejects missing files, changed sizes/hashes and extra files that could override the pinned configuration. Loading uses safetensors and local files only, with remote code disabled. Missing weights cause a clear failure; there is no substitute model or generated-label fallback.

## Generate unedited predictions

Use the [frozen evaluation plan](semantic-evaluation.md) and an existing imported source store:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  uv run --locked --project integrations/semantic python -m integrations.semantic.qwen \
  --model-dir PATH_TO_PINNED_QWEN_DIRECTORY --device mps \
  --plan artifacts/semantic/plan.json --store data/local/droid-work \
  --output artifacts/qwen-predictions
```

Choose `mps`, `cuda` or `cpu` explicitly. An unavailable accelerator is an error. CPU uses float32 and accelerators use float16. Weight storage alone is about 4.26 GB; runtime memory also includes model buffers, vision tokens, attention cache and loading overhead. Full pretrained execution and memory use on the M5 Pro or RTX 3060 have not been measured here.

Defaults are one episode, eight-second windows, at most 16 sampled frames per window, 40 windows, 4,096 input tokens and 768 generated tokens per call. `--config` accepts a JSON `Config` with explicit bounds and any declared tuning episode IDs. The chosen split must fit the episode budget; the command never silently drops extra episodes. Source frames are decoded in order once per episode, downscaled to at most 320 pixels on the longest side without cropping, and passed to the official image processor. The prompt carries exact source-frame times, including the inferred-clock designation retained in the output. It does not use Qwen's default video-FPS inference.

Windows are processed independently. Window edges can therefore create extra predicted segment boundaries; there is no unvalidated merge across windows. Sparse images can miss brief actions. Those are properties to measure against independent references, not reasons to remove difficult input from the benchmark.

## Output and failure handling

The output directory contains immutable unedited predictions, raw replies, source-frame indices/times, RGB hashes for the supplied images, prompt hashes, token counts, model asset hashes and device/dtype information. Use a fresh directory for each model run. References are not an input to the prediction command, and stored review revisions are not read.

The response must be a JSON array covering each window in order. Actions must belong to the frozen taxonomy; every segment needs at least one supplied evidence frame. Unknown fields, gaps, overlaps, out-of-range boundaries and unsupported labels are rejected. A failed episode keeps its raw evidence and an explicit error but contributes no invented segments. It remains in `predictions.json`, so the evaluator counts its misses. The source snapshot is checked again before publishing a successful episode result.

Uncertainty is the model's own uncalibrated estimate. It can request review; it does not approve labels or establish accuracy. Corrections belong in a separate native review record, while this prediction artifact remains unchanged.

## What has been tested

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  uv run --locked --project integrations/semantic python -m pytest -q integrations/semantic/test_qwen.py
```

The tests run the official processor and a small randomly initialized Qwen3-VL through multiframe generation, including the visual encoder. They also cover asset checks, token/episode/window limits, frame sampling, strict reply parsing, source changes and explicit failed-episode records. Generated runner fixtures check artifact handling separately from model execution. CI runs these checks alongside the existing MLX job, using an isolated environment and no downloaded model weights.

On the real 119-frame DROID pilot, the sampler selected 16 chronological frames with the inferred clock intact. It created no semantic labels. [Recorded metadata](evidence/qwen-adapter.json) separates that sampling check from the tiny random-model execution. A full pretrained Qwen run, an independent reference set, semantic accuracy and hardware-fit measurements remain outstanding.
