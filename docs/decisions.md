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


## Implementation record (October 28, 2026 UTC)

- Read the complete project brief as supplied verbatim by the coordinating parent. The executor's Library materialization capability was absent; the parent withdrew the consumer-local transfer requirement. This executor did not materialize the original Library file.
- Chose one local application and SQLite optimistic revisions. Native JavaScript replaces the proposed React/TypeScript client; this reduces setup/build requirements at the cost of static frontend types. Real desktop/mobile browser tests cover the interactions.
- Kept manual acceptance separate from machine suggestions and edits. Save/restore create drafts, stale revisions return 409, and export freezes review eligibility in a database transaction.
- Added sequential motion/flow analysis and pinned CPU MobileNet-SSD evidence. Model source commit, file hashes, MIT terms, preprocessing, sampling caps and implementation hashes are recorded. No semantic action or contact model is claimed.
- Assessed NanoDet and DROID acquisition. Their asset/storage requests returned runtime proxy tunnel 403, with no downloaded bytes. The NanoDet request was also tried through supported execution permissions and still failed at the proxy. No approval-review rejection occurred. A distinct public MobileNet-SSD repository was reachable and its files were hash verified. DROID remains unacquired.
- Chose the RTX 3060 workstation as the future CUDA extension target based on ecosystem compatibility and bounded models, while shipping exercised CPU inference. No accelerator backend or hardware benchmark is claimed.
- Added behavior tests, actual browser correction/conflict/export recordings, and pull-request CI. Exact remote CI status belongs to the final PR handoff, not this static record.

- The user requested a more minimal editor inspired by the current [Physical Intelligence website](https://www.pi.website/). Reviewed its public page through the permitted web reader; direct executor HTTP was blocked by the proxy. Chose white surfaces, restrained serif headings, hairline separators and collapsed evidence notes, preserving independent project identity. No PI logo or affiliation is used.
- The user reiterated real-DROID testing. A supported-permission request for the official `droid_100` listing still returned proxy tunnel 403, with no bytes. This remains an acceptance blocker; the required active-host policy change is documented in the dataset card.

- Parent review found request ordering, save/navigation, passive-playback timing, pinned evidence and malformed-export boundary issues. Added explicit load tokens and edit generations, locked save transitions, visible-playback timing, revision-bound machine origins, strict export request models, and delayed-response/clock/provenance regressions. CI playback failure also led to preserving media/playhead on save and selecting Chrome for H.264 checks.
