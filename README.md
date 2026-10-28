# Robot episode curation

A local robotics data pipeline built around **Label Studio**, with project code focused on source alignment, provenance, assisted-review integration and evaluation.

![Native Label Studio review — synthetic fixture](docs/evidence/labelstudio-native.png)

The real Label Studio application and official SDK import video/sensor tasks and predictions, save corrections, and connect them to explicit approval and immutable accepted exports. New approvals/exports check the selected native revision and machine provenance. The earlier bespoke editor is a development prototype, available only through `--legacy-editor`; native-bound episodes cannot bypass the approval bridge through it.

## Run the native GUI

Requires Python 3.12, uv 0.12.23, FFmpeg/FFprobe. The lockfile includes Label Studio 1.23.2 and SDK 2.1.2.

```bash
./scripts/setup.sh
uv run --locked python -m robo demo
# Choose an episode ID printed above:
uv run --locked python -m robo prepare-labelstudio EPISODE_ID
uv run --locked python -m robo serve
# Open http://127.0.0.1:8000
```

`serve` starts native Label Studio on loopback using its normal local authentication. The installer does not provision an account or persistent API credential. Use the application's local setup/login workflow.

For this first slice, create a Label Studio project, paste `artifacts/labelstudio-bundle/review.xml` into its labeling configuration, and import `tasks.json`. Add Local Files source storage at the `local_storage_path` printed by the preparation command, with **Treat every object as a source file** enabled. Do not sync the proxy directory as new tasks; the supplied tasks already bind video and sensor files. [Integration guide](docs/labelstudio.md) explains SDK automation, frame precision and verification; [design notes](docs/design.md) document the PI-inspired styling and retained Label Studio branding.

The demonstration contains six generated illustrated episodes, eight seconds each. Its state/action streams are generated fixtures. A separate [real DROID pilot](docs/droid.md) downloads one pinned 9.54 MB official shard and exercises native correction/export plus official LeRobot loading on its 119-step episode. These automated integration checks do not establish human annotation savings or reference-label quality.

After native Submit/Update, use the [approval workflow](docs/native-approval.md) to bind the task, refresh the chosen annotation into a draft, inspect it and explicitly accept or reject that revision. This thin CLI/SDK gate keeps dataset eligibility separate from saving an annotation. It uses existing authorized access and creates no credentials. The guide includes copyable commands and the precise freshness limitations.

## Ownership

For several episodes, the [batch workflow](docs/batch-workflow.md) runs import and suggestions together, reports failed inputs and creates a native Label Studio queue. Reruns recover interrupted work; explicit approval still controls export.

```mermaid
flowchart LR
    A[Video + explicit source clock basis] --> B[Robotics alignment / provenance bridge]
    B --> C[CFR display proxy + explicit original-PTS map]
    C --> D[Native Label Studio video / timeline / sensor GUI]
    E[Cached prediction evidence] --> D
    D --> F[Official SDK / native JSON export]
    F --> G[Authenticated refresh + immutable draft]
    G --> H[Explicit approval + fresh native check / split gates]
    H --> I[Official LeRobot v3 writer / loader + provenance]
```

| Boundary | Current state |
| --- | --- |
| Source ingest | Immutable video snapshots, source PTS or explicitly inferred nominal clocks, optional streams, declared units/offsets and missing-data masks |
| Mature review UI | Native Video, TimelineLabels and synchronized TimeSeries; project theme uses supported Style/View configuration |
| SDK bridge | Actual project/task/prediction import, correction, native JSON export and trusted task/source/proxy validation exercised |
| Timing/approval | Explicit accept/reject with revision freshness, pinned native provenance and checked new exports. Native `lead_time` is retained; active human time stays null and is excluded from efficiency estimates |
| Curation/evaluation | Resumable batch intake, per-input failure reports, native priority queue and explicit review/export; workload counts remain separate from human-efficiency measurements |
| Training format | Official LeRobot 0.6.1 create/finalize/load and DataLoader roundtrip on synthetic and real DROID native exports; explicit schema, causal alignment and hashes |
| Real data | One official DROID episode downloaded and decoded: 119 steps, three cameras checked, one selected view reviewed/exported; inferred 15 Hz timing retained |
| MLX / PI family | [π₀.₅ action expert](docs/mlx.md) with a frozen official encoder; CPU loss/gradient/update/resume checks pass on small models. Full pretrained and Apple runs remain open |

The constant-rate review proxy is deliberately separate from raw observations. Its manifest maps every display frame to the source frame, integer PTS and source time. Sensor plots use a zero-anchored display grid with nearest samples and blank gaps, while original streams remain intact. Temporal exports include source frames bracketing each boundary and identify inferred clocks. The default proxy retains all source frames up to 60 fps; an explicit coarser rate reports omitted frames. The grid is not a lossless representation of arbitrary source timestamps. [Data contract](docs/data-contract.md) · [Dataset card](docs/dataset-card.md).

The [controlled DROID benchmark](docs/messy-droid.md) tests row ordering, duplicates, clock offsets, missing samples and damaged media against an untouched reference. Its evidence separates complete repairs from review requests and rejected inputs.

## Training dataset

The [LeRobot guide](docs/lerobot.md) provides the isolated locked runtime, explicit feature schema and conversion command. An accepted version becomes a local v3 dataset with original PTS/sample provenance and one selected split. CI loads the native review fixture through the official loader and includes a bounded [DROID gate](docs/droid.md). The real-data test compares every loaded state/action vector with decoded RLDS values. The [MLX adapter](docs/mlx.md) adds PI05 action-expert training and checkpoint resume, tested against small official models. Full pretrained training is still pending.

## Verify

```bash
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
npm ci --cache .cache/npm
npm run check:web
npm run format:check
# Uses a disposable database/session; no persistent API credential:
uv run --locked python -m scripts.labelstudio_smoke --browser
```

The native smoke test uses installed system Chromium by default. Set `CHROMIUM_PATH` for another installation; CI uses Chrome for H.264. It disables telemetry and blocks non-local browser requests. Legacy browser regression tests remain under `npm run test:e2e`; see [development notes](docs/development.md) for the locked Playwright recording prerequisite.

[Verification and limits](docs/results.md) · [Original study protocol](docs/evaluation.md) · [Model provenance](docs/models.md). No human time reduction, real-data quality improvement, policy performance or Apple training result is claimed. This project is independent of Physical Intelligence.
