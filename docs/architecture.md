# Architecture

The current product uses self-hosted Label Studio 1.23.2 with official SDK 2.1.2. Project code handles robotics source clocks, immutable provenance, assistance, eligibility and evaluation. LeRobot is the selected next training-facing integration. This supersedes the original custom-editor design.

```mermaid
flowchart LR
  A[Source video + robot streams] --> B[Validated ingest / measured PTS]
  B --> C[Immutable source store]
  C --> D[CFR review proxy + explicit PTS map]
  C --> E[Cached machine evidence]
  D --> F[Native Label Studio]
  E --> F
  F --> G[SDK JSON export]
  G --> H[Trusted draft snapshot]
  H -. pending approval bridge .-> I[Accepted revision / split gates]
  I -. next adapter .-> J[LeRobotDataset]
```

| Owner | Contract |
| --- | --- |
| `robo/ingest.py` | Validate manifests, snapshot source bytes, preserve measured integer PTS/time base and original optional streams |
| `robo/analysis.py`, `robo/perception.py` | Cache bounded machine runs with exact config/model/source hashes; never mutate human decisions |
| `robo/labelstudio.py` | Create a separate CFR display proxy, map native frame spans to source intervals, publish through SDK and verify exported task/source/proxy bindings |
| Native Label Studio | Authentication, video/timeline editing, synchronized sensor plots, annotation persistence and native export |
| `robo/store.py`, `robo/export.py` | Existing immutable revision/acceptance, stale-write and grouped split gates; native promotion is still pending |
| `robo/evaluation.py` | Temporal matching, error accounting and optional real reviewer measurements; fixture outputs are labeled synthetic |

The source clock is authoritative. Review proxies hold the last measured frame and record each display frame's original source index, integer PTS and episode time. No FPS-derived clock silently replaces source evidence. Display sensor CSV uses declared offsets and tolerance-bound nearest samples with blank gaps; original actions/state remain intact.

Native completion produces a draft snapshot. Label Studio's `lead_time` is raw metadata, not measured active correction time. The next bridge must explicitly reconcile updates, invalidate prior acceptance, preserve suggestion origins and reject stale approvals. Upstream annotation completion alone cannot establish these guarantees.

Use local files and project-scoped proxy storage on loopback for this bounded pilot. Avoid a second UI/database such as FiftyOne until a distinct curation requirement justifies it. The deprecated standalone Label Studio frontend is not embedded. Styling uses supported Style/View configuration; the upstream application shell remains maintained by Label Studio.

The old FastAPI/editor path survives temporarily behind `--legacy-editor` for regression evidence while approval/timing behavior migrates. It is not a second primary product. Remove it when the native path owns its useful behaviors.

See [native integration](labelstudio.md), [data contract](data-contract.md) and [verified outcomes and remaining gates](results.md).
