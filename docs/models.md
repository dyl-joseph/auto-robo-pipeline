# Perception and hardware

| Backend | Inputs → evidence | Exercised here |
| --- | --- | --- |
| OpenCV motion v1 | Consecutive grayscale frames → change boxes, sparse optical-flow vectors, motion/hold intervals, raw heuristic scores | Six illustrated clips; cache and corruption tests |
| MobileNet-SSD / VOC | Sampled BGR frames → object classes, boxes and uncalibrated scores | Pinned weights, CPU inference, bounded sampling/cache tests, browser evidence |

The motion baseline is not learned. Its regions are not object masks or persistent identities. MobileNet-SSD is pretrained; it does not detect a robot, prove a grasp, or recognize recovery. The vocabulary is narrow, and illustrated scenes are out of distribution. Both outputs are candidates for inspection.

The model files come from [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD/tree/bb17b6c3eef36d80be441ae8e5339be66e8e3b7a), commit `bb17b6c3eef36d80be441ae8e5339be66e8e3b7a`. Its [MIT license](https://github.com/chuanqi305/MobileNet-SSD/blob/bb17b6c3eef36d80be441ae8e5339be66e8e3b7a/LICENSE) is downloaded and hash checked with the weights. The source README describes COCO pretraining and VOC0712 fine tuning. We do not repeat its benchmark as a project result. The exact hashes and preprocessing are in `robo/perception.py` and every cached run. Weights remain outside Git.

`python -m scripts.fetch_model` performs bounded public downloads into temporary files and verifies SHA256 before publication. It needs no credential. Inference defaults to one sample every 0.5 seconds, maximum 600 samples, one decoded frame in memory and one network instance. Increase `--sample-every-s` or lower `--max-frames` to bound work; truncation is explicit. Motion inference also decodes sequentially and retains only small features. Import currently retains per-frame PTS and stream metadata in memory; use short episodes, not continuous multi-hour recordings.

Cache keys include episode content, model hashes, preprocessing, OpenCV version, implementation hash and sampling config. Cached artifacts include their own hash. Human revisions retain the exact suggestion run ID. Changing the active run never rewrites past reviews.

## Hardware choice

Choose the **RTX 3060 workstation** as the primary future model-development target, given the user's either/or choice. Its 12 GB of dedicated VRAM suits one small detector or segmentation model at a time, while CUDA provides a direct path to the PyTorch research model ecosystem. Keep batch size one and sequential decoding because the workstation has 16 GB system RAM. This is a compatibility and memory-budget choice, not a measured speed comparison.

The shipped OpenCV wheel uses CPU inference on both machines and has no CUDA execution path. It requires neither accelerator stack; adding `--device cuda` would falsely suggest otherwise. The M5 Pro's 48 GB unified memory may accommodate larger models, but the relevant MPS operator support, model implementation and throughput require actual validation. We do not maintain a second untested accelerator backend.

A future semantic backend should use a separately locked PyTorch/CUDA environment, one model at a time, cached inference outside requests, reduced-resolution samples and a measured peak-memory gate below available VRAM. Use [PyTorch's installation selector](https://pytorch.org/get-started/locally/) for the installed NVIDIA driver and verify `torch.cuda.is_available()` before attempting CUDA. [MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html) describes the alternative Apple path. Neither physical computer was accessed or benchmarked. A VLM, segmentation model and CUDA adapter are not shipped in this draft.

NanoDet was also assessed from OpenCV Zoo. Its model asset endpoint failed with a proxy tunnel 403, including through supported execution permissions. No NanoDet weights were downloaded and no NanoDet integration is claimed.
