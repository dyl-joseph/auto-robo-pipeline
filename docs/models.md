# Models and hardware

## Review suggestions

| Backend | Output | Verification |
| --- | --- | --- |
| OpenCV motion | Change boxes, sparse optical flow and motion/hold intervals | Generated clips, cache tests and real DROID batch intake |
| MobileNet-SSD / VOC | Object classes, boxes and raw detector scores | Pinned weights, CPU inference and browser evidence |

The motion baseline uses image differences and optical flow. Its regions are not object masks or persistent tracks. MobileNet-SSD has a narrow object vocabulary and does not establish grasps, contact or recovery. Both produce review candidates, and their scores are not calibrated probabilities.

MobileNet-SSD files come from [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD/tree/bb17b6c3eef36d80be441ae8e5339be66e8e3b7a), commit `bb17b6c3eef36d80be441ae8e5339be66e8e3b7a`. Its [MIT license](https://github.com/chuanqi305/MobileNet-SSD/blob/bb17b6c3eef36d80be441ae8e5339be66e8e3b7a/LICENSE) is downloaded and hash-checked with the weights. The source describes COCO pretraining and VOC0712 fine-tuning. That model's published results are separate from this project's tests.

Run `uv run --locked python -m scripts.fetch_model` to acquire the pinned files. Inference defaults to one frame every 0.5 seconds, up to 600 samples. It retains one decoded frame and one network instance. Increase `--sample-every-s` or lower `--max-frames` to reduce work; the report records truncation.

Cache keys include source content, model hashes, preprocessing, OpenCV version, implementation hash and sampling configuration. Human revisions keep the suggestion run they used. Changing an active run does not rewrite a prior review.

## PI05 and MLX

The [MLX adapter](mlx.md) trains the π₀.₅ action expert with a frozen official LeRobot encoder. All 14 local model tests use MPS and MLX Metal to compare loss, gradients, parameter updates and checkpoint resume. The [license notice](../integrations/mlx_pi/NOTICE.md) lists the implementation sources and distinguishes Apache code from the Gemma checkpoint terms.

The [full synthetic inference run](evidence/pi05-synthetic-metal.json) passed on the M5 Pro with 48 GB unified memory. The encoder and official reference use MPS, and the MLX action expert uses Metal. CPU model fallback is disabled. The RTX 3060 machine has 12 GB VRAM and 16 GB system RAM and remains untested. The separate OpenCV review backend uses CPU inference.

The full PI05 check uses the public tokenizer asset downloaded anonymously by official OpenPI and the complete pinned pretrained checkpoint. Generated images, state, task text and synthetic quantile statistics exercise the full input/output flow. Peak memory use, sustained training speed and policy quality remain unmeasured.
