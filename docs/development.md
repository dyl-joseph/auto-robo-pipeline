# Development environment

## Application workflow (current)

Use the README commands for the implemented application. `python -m robo demo` prepares the local synthetic store and `python -m robo serve` binds to 127.0.0.1:8000. No process is started by setup. The material below records the original environment milestone.

Native Label Studio owns the current browser UI, served from its pinned application package. There is no project frontend build. `npm ci`, `npm run check:web` and `npm run format:check` validate browser verification and the temporary legacy editor. `pytest` covers the Python bridge; the real native integration runs with `python -m scripts.labelstudio_smoke --browser`.

In this saved Linux environment, system Chromium and FFmpeg can be reused for browser video without another download:

```bash
mkdir -p .cache/playwright/ffmpeg-1013
ln -sf /usr/bin/ffmpeg .cache/playwright/ffmpeg-1013/ffmpeg-linux
npm run test:e2e
```

This path is tied to the locked Playwright version. On another system use its documented browser install command in README. `CHROMIUM_PATH` overrides executable discovery. Browser tests create a disposable store, record their actions under `artifacts/browser-tests`, and save synthetic screenshots under `docs/evidence`. No real data or credentials are involved.


## Locked dependencies and verification

Python 3.12 and uv 0.12.23 manage core ingest/export, Label Studio 1.23.2 and SDK 2.1.2. `./scripts/setup.sh` installs the default dev and labelstudio dependency groups. The lock includes Label Studio's NumPy/PyArrow constraints. LeRobot remains a separate integration gate.

`uv run --locked python scripts/check_environment.py` checks H.264 generation/decode, FFprobe timestamp metadata, Parquet and an API request. `uv run --locked pytest -q` runs behavior tests. Native browser verification uses a temporary migrated Label Studio database and real session permissions, then deletes it. See [integration details](labelstudio.md) for the transient native-export fixture token and supported version-specific feature flag.

Optional pinned perception weights use `python -m scripts.fetch_model` and stay ignored. No model benchmark or human study follows from these software checks.

## Filesystem and runtime lifecycle

- Use the existing checkout. Each cloud task is already isolated; do not create a Git worktree unless the user requests one.
- Keep local datasets in `data/local/`, model weights in `models/local/`, and outputs in `artifacts/`. These paths are ignored by Git. Store licenses and reproducible public manifests separately from large media.
- Keep credentials in environment settings. Never put secret values in repository files, shell tracing, logs, or saved startup instructions.
- Installed files can survive an environment snapshot. Processes, sockets, and authentication sessions must be reestablished when required.
- Setup starts no long-running process. Run `uv run --locked python -m robo serve` explicitly for the native GUI. Its private application database lives beneath the selected local store.

## Cloud network settings

The user requested all public internet access. The configuration draft uses `allowed_domains: ["*"]`. Draft saving does not change this running machine's policy or publish a snapshot. Review and save the configuration in environment settings, then publish the environment to activate the prepared configuration. Verify the runtime policy after activation before claiming unrestricted access is live.

Public HTTP/HTTPS access uses the platform proxy. This does not configure private-network/VPN access, arbitrary TCP routes, public inbound hosting, or credentials for gated datasets. Preserve the inherited proxy and CA trust.

If the sandbox prevents a connection to the configured proxy, use the host's supported execution approval path. A local socket denial does not by itself establish a missing GitHub credential. Native Git access is supplied by the platform; do not request a personal token solely because a token variable is absent.

## Dependency changes

Keep `pyproject.toml`, `.python-version`, and `uv.lock` consistent. Run `uv lock` deliberately when changing dependencies, inspect the diff, and rerun setup and readiness checks. Optional model and web dependencies should be pinned with the milestone that introduces their consumer.
