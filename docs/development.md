# Set up development

Install Python 3.12, uv 0.12.23, FFmpeg with FFprobe, and Node.js for browser tests. From the repository root:

```bash
./scripts/setup.sh
uv run --locked python scripts/check_environment.py
uv run --locked pytest -q
npm ci --cache .cache/npm
npm run check:web
npm run format:check
```

Setup installs the locked Python dependencies and starts no service. The readiness check generates and decodes 12 H.264 frames, checks video timestamps, round-trips a Parquet table and sends an in-process API request.

## Run browser checks

Label Studio supplies the UI from its pinned application package. There is no frontend build. Browser tests use the locked Playwright package:

```bash
PLAYWRIGHT_BROWSERS_PATH=.cache/playwright npx playwright install chrome ffmpeg
CHROMIUM_PATH=/usr/bin/google-chrome uv run --locked python -m scripts.labelstudio_smoke --browser
CHROMIUM_PATH=/usr/bin/google-chrome uv run --locked python -m scripts.labelstudio_smoke --batch
```

Use the actual browser executable path on your machine. In the saved Linux environment, `/usr/bin/chromium` and system FFmpeg are available. The recording helper can use system FFmpeg at the path expected by the locked Playwright release:

```bash
mkdir -p .cache/playwright/ffmpeg-1013
ln -sf /usr/bin/ffmpeg .cache/playwright/ffmpeg-1013/ffmpeg-linux
```

Native tests create a disposable Label Studio database and authenticated session, exercise the UI and SDK, then remove the database. Failure reports omit credentials. The [integration guide](labelstudio.md) explains the pinned upstream regressions and the temporary token used by native export. `npm run test:e2e` covers the earlier editor's regression tests.

To use the application yourself, run `uv run --locked python -m robo serve` and open `http://127.0.0.1:8000`. Label Studio's private database lives under the selected store's `labelstudio/` directory.

## Use isolated data and model runtimes

The root lockfile includes Label Studio 1.23.2 and SDK 2.1.2. TensorFlow, LeRobot and MLX have separate lockfiles under `integrations/`. Their guides provide the exact commands:

- [DROID decoding](droid.md) uses TensorFlow CPU and TFDS.
- [LeRobot conversion](lerobot.md) uses the official writer and loader.
- [MLX checks](mlx.md) use the PI05 model implementation and official Linux CPU package.

Set `HF_HOME="$PWD/.cache/huggingface"` for Hugging Face consumers so their cache stays in the workspace. Disable telemetry with `HF_HUB_DISABLE_TELEMETRY=1`. Use `HF_HUB_OFFLINE=1` for verification that already has all required local assets. Offline mode cannot provision a missing tokenizer.

## Keep generated files local

Datasets belong in `data/local/`, weights in `models/local/`, and test output in `artifacts/`. Git ignores these directories and `.cache/`. Commit small provenance reports and permitted documentation images separately. Do not put secrets, databases or authenticated sessions in source control.

An environment snapshot can retain installed files, but processes and sessions must be started again. Preserve the platform's proxy and CA settings. Public download access does not grant access to gated model assets or private networks.

When changing dependencies, update the relevant `pyproject.toml` and lockfile together. Rerun setup, inspect the lock diff and run the checks for that runtime.
