# Sources and licenses

The MLX action expert adapts the equations and parameter layout in:

- Physical Intelligence, [OpenPI](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479), commit `215abfb217dbac7d5f1273282331b9b1866c0479`: `src/openpi/models/pi0.py` and `gemma.py`.
- Physical Intelligence and the Hugging Face team, [LeRobot 0.6.1](https://github.com/huggingface/lerobot/tree/v0.6.1): `policies/pi05/modeling_pi05.py` and `policies/pi_gemma.py`.

These sources use Apache-2.0. The copyright notices are retained in the adapted model file, and the license is included in [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0). OpenPI's Gemma implementation also credits the Big Vision Authors, copyright 2024. The MLX port adds a frozen-prefix interface, training/checkpoint support and validation against the installed official implementation. It does not include community π₀.₅ ports.

MLX 0.32.4 is an [Apple project under MIT](https://github.com/ml-explore/mlx). LeRobot and Transformers remain installed dependencies; their source files are not vendored here.

The public [`lerobot/pi05_base` model card](https://huggingface.co/lerobot/pi05_base/tree/22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0) declares the **Gemma** license. This is distinct from the implementation's Apache license. Its underlying model is `google/paligemma-3b-pt-224`; see [Gemma terms](https://ai.google.dev/gemma/terms) and OpenPI's [LICENSE_GEMMA.txt](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/LICENSE_GEMMA.txt). No pretrained weights or tokenizer files are committed to this repository. The optional inspection command saves one small weight tensor under the caller's local artifact directory.

The full synthetic inference command also uses Google's public `paligemma_tokenizer.model`. [Official OpenPI](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/tokenizer.py) downloads this asset anonymously from `gs://big_vision/paligemma_tokenizer.model`. The command loads that local SentencePiece model and verifies its task/state token IDs against the original SentencePiece implementation. The [inference evidence](../../docs/evidence/pi05-synthetic-metal.json) records the asset hash and full checkpoint hash. The committed action arrays use generated observations and synthetic normalization statistics.
