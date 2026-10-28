# Decision and verification record

Recorded October 27, 2026, America/Chicago. This record covers the planning and environment milestone, not an implemented annotation system.

## Decisions

| Observation | Decision | Evidence and remaining uncertainty |
| --- | --- | --- |
| The remote repository initially advertised no refs or files | Initialize `main` with a roadmap and development tooling | Read-only native Git access succeeded; the requested branch was absent before this change |
| The source notes combine a broad pipeline, portfolio motivation, and hypothetical metrics | Center the project on human review efficiency with a quality constraint | Scope and milestones are in `plan.md`; no robotics performance has been measured |
| The pilot is small and has no existing services | Prefer a local Python application with offline model runs, SQLite revisions, and a later React editor | Two design candidates compared sequentially in `architecture.md`; scale limits remain unmeasured |
| Python 3.12, uv, FFmpeg, and Node are available; no GPU runtime is required for M0 | Install a locked CPU environment; defer model runtimes and dataset acquisition | Dependency readiness passed; actual DROID access, licensing, and model behavior are unverified |
| The home directory is read-only | Configure uv to use the ignored project cache | Initial default-cache command failed; final documented commands pass without home-directory writes |
| A synchronous FastAPI fixture hung in the execution sandbox | Verify it through supported execution permissions and use an async fixture for the final in-process readiness check | The synchronous check passed with approved execution; the final async check passes in the normal sandbox; live servers may need execution approval |
| The user requested all public internet access | Save `allowed_domains: ["*"]` in the environment draft | Draft save confirmed; it does not activate policy or publish the environment |

## Checks completed

- `./scripts/setup.sh` installed 28 locked packages into `.venv`. A repeat run reused the environment successfully. The resolver's 30-package graph also includes the virtual project and platform-specific dependencies.
- `uv run --locked python scripts/check_environment.py` passed H.264 generation, FFprobe timestamps, OpenCV decoding of all 12 expected frames, a Parquet round-trip, and a FastAPI JSON request. The final script passes in the normal sandbox.
- `uv run --locked ruff check .` and `uv run --locked ruff format --check .` passed after correction of one long output line.
- `bash -n scripts/setup.sh` passed. Repository-local Markdown links were checked before commit.
- Workspace discovery found one project checkout. `/workspace/.git` is an empty platform sentinel and `.cache/uv/sdists-v9/.git` belongs to the dependency cache, not another project. No submodules were present.

No application tests, robot datasets, model inference, review UI, training exports, or human annotation experiments were run. Pytest is installed, but there is no application test suite yet. No persistent application service is required or running for this milestone.

## Reusable configuration and next work

The environment configuration stores setup and startup checks. Review and save the draft in environment settings, then publish the environment. Snapshot publication and restoration in a new task are separate operations and have not been validated by this record.

The next coding task is M1's local episode importer and timestamp validation. Follow `plan.md` and add behavior tests with the implementation. The proposed `episodes` commands are not available yet.
