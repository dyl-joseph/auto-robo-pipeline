# Development environment

## What is prepared

The current milestone is planning and environment preparation. Python 3.12 and uv 0.12.23 manage a locked CPU environment with FastAPI, Uvicorn, Pydantic, NumPy, OpenCV headless, and PyArrow. The development group contains pytest, pytest-cov, HTTPX, and Ruff.

FFmpeg/FFprobe provide media inspection and fixture generation. They are available in the cloud base image. Node 24 is available for the planned React editor, but no frontend dependencies or application server exist yet.

No GPU, dataset access, paid API, or model weights are needed for the current checks. PyTorch, detectors, segmentation frameworks, VLM providers, and DROID adapters are deferred until their runtime and licensing requirements are known. Installing every possible research framework now would add cost without validating a workflow.

## Install and verify

From `/workspace/auto-robo-pipeline` in the cloud, or from the checkout root locally:

```bash
./scripts/setup.sh
uv run --locked python scripts/check_environment.py
uv run --locked ruff check .
uv run --locked ruff format --check .
```

`setup.sh` verifies tool prerequisites and uses `uv sync --locked` with Python downloads disabled. It must not rewrite the lockfile. The project config uses the ignored `.cache/uv` directory inside the checkout so the default commands also work when the home directory is read-only. `UV_CACHE_DIR` can override this location. `.venv` is disposable and ignored.

If uv is absent, install the pinned tool in a separate tool environment, then rerun setup:

```bash
python3 -m venv /workspace/.tools/uv-0.12.23
/workspace/.tools/uv-0.12.23/bin/python -m pip install 'uv==0.12.23'
export PATH="/workspace/.tools/uv-0.12.23/bin:$PATH"
```

On a machine without FFmpeg, install it through the operating system's verified package manager. This project does not disable TLS, package signatures, or artifact hashes to resolve installation failures.

The readiness script verifies:

- Python version and the locked dependency imports.
- H.264 fixture generation, FFprobe metadata, timestamp count, and OpenCV frame decoding.
- A PyArrow Parquet write/read round-trip.
- JSON serialization and an HTTP request through a temporary FastAPI test app.

All generated fixtures live in a temporary directory and are removed after the check. The script does not download robotics data or launch a persistent server. These are dependency checks, not results from a robotics application.

Pytest is installed for implementation work. There is no application test suite yet. Add behavior tests in M1, then run `uv run --locked pytest`; a zero-test run does not establish readiness.

## Filesystem and runtime lifecycle

- Use the existing checkout. Each cloud task is already isolated; do not create a Git worktree unless the user requests one.
- Keep local datasets in `data/local/`, model weights in `models/local/`, and outputs in `artifacts/`. These paths are ignored by Git. Store licenses and reproducible public manifests separately from large media.
- Keep credentials in environment settings. Never put secret values in repository files, shell tracing, logs, or saved startup instructions.
- Installed files can survive an environment snapshot. Processes, sockets, and authentication sessions must be reestablished when required.
- No long-running process needs to start for this milestone. When the API/editor exist, update saved startup instructions with their real commands and functional readiness checks.

## Cloud network settings

The user requested all public internet access. The configuration draft uses `allowed_domains: ["*"]`. Draft saving does not change this running machine's policy or publish a snapshot. Review and save the configuration in environment settings, then publish the environment to activate the prepared configuration. Verify the runtime policy after activation before claiming unrestricted access is live.

Public HTTP/HTTPS access uses the platform proxy. This does not configure private-network/VPN access, arbitrary TCP routes, public inbound hosting, or credentials for gated datasets. Preserve the inherited proxy and CA trust.

If the sandbox prevents a connection to the configured proxy, use the host's supported execution approval path. A local socket denial does not by itself establish a missing GitHub credential. Native Git access is supplied by the platform; do not request a personal token solely because a token variable is absent.

## Dependency changes

Keep `pyproject.toml`, `.python-version`, and `uv.lock` consistent. Run `uv lock` deliberately when changing dependencies, inspect the diff, and rerun setup and readiness checks. Optional model and web dependencies should be pinned with the milestone that introduces their consumer.
