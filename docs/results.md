# Verification evidence and research limits

This draft ships a working local pipeline and a reproducible evaluation harness. It does not contain a human study or a real robot quality benchmark.

| Evidence | Observed outcome |
| --- | --- |
| Python behavior tests | 72 passed, including twelve native proxy cases, twenty-five native approval cases and optional pretrained model smoke with downloaded pinned weights |
| Native Label Studio 1.23.2 + SDK 2.1.2 | Actual project/storage/task/prediction import, annotation correction and native JSON export passed in a disposable authenticated application |
| Native browser review | H.264 playback, two synchronized sensor plots, sensor-to-video seeking, temporal end-frame edit, submit, persisted API reread and reload passed without script errors |
| Native task loading regression | Controlled detail-before-list responses reproduce the pinned cold-link bug; native table entry recovers prediction review, correction, submit, reload and SDK export under 6× browser CPU throttling |
| Native sync regression | Official time-formatted CSV, numeric JSON and resized themed view seek correctly; two pinned upstream Collapse initialization failures reproduced as controls |
| Native approval lifecycle | Official SDK refresh → explicit acceptance → idempotent export → native edit blocks a new export → historical artifact validates → refreshed draft rejected; outcome, quality, notes and null active time preserved |
| Native temporal export | Source frames 5/6 deliberately removed; native frames 1–6 export to generated review interval [0, 0.5), with end-boundary measured-source bracket [0.333333…, 0.583333…) and width 0.25 s |
| Legacy desktop Chromium | Import fixture → playback/seek → candidates → correction → save → stale-tab conflict → accept → download export → edit/reopen passed |
| Legacy browser race/provenance/timing regressions | Delayed out-of-order loads, edits during load, repeated saves, navigation during save, restored origins and >30s playback timing passed |
| Legacy mobile Chromium, 390×844 | Manual condition, segment creation, responsive layout and no horizontal overflow passed |
| Dependency readiness | 12 H.264 frames decoded, Parquet round-trip, FastAPI request passed |
| Code checks | Ruff lint/format, browser JavaScript syntax, Prettier check |
| Export consumer | JSONL/Parquet equality, file/semantic/version hashes, latest acceptance and split leakage checks tested |
| Learned perception | MobileNet-SSD executed on synthetic illustrations; bounded sampling, cache and provenance exercised |
| DROID | No acquisition; public storage discovery blocked by proxy tunnel 403 |
| Human time savings, agreement and error recall | Unmeasured; automation timings and fixture metrics do not establish these |
| RTX 3060 / M5 Pro performance | Unmeasured; neither machine accessed |

The browser evidence uses automated actions and synthetic media. The notes identify automation. The current primary application is [native Label Studio](evidence/labelstudio-native.png), with a [recorded temporal correction](evidence/labelstudio-walkthrough.webm). [Desktop screenshot](evidence/labelstudio-native-desktop.png), [machine-readable review evidence](evidence/native-review-verification.json) [sync regression observations](evidence/native-sync-contract.json) and [native approval evidence](evidence/native-approval-verification.json) accompany the current native path. Earlier custom-editor screenshots and recording are historical prototype evidence. The native approval bridge now checks immutable selected snapshots at refresh, approval and new-export boundaries. Active human timing remains unmeasured; prototype browser save-lock/timer tests do not establish equivalent native-editor behavior.

## Reproduce the evaluation machinery

```bash
uv run --locked python -m robo evaluate examples/synthetic-evaluation.json artifacts/evaluation.json
uv run --locked python -m robo validate-export PATH_TO_EXPORTED_VERSION
```

[Recorded fixture evaluator output](evidence/synthetic-evaluation.json) and [pretrained smoke provenance](evidence/perception-smoke.json) are provided as reproducible machinery checks. The fixed-budget fixture deliberately has missed errors: prioritized recall is 0.5 while seeded random recall is 1.0. Those are specified synthetic inputs, not a claim that either strategy performs that way on robot data.

The small evaluation example intentionally includes a wrong class and boundary offsets. It exercises matching and missed-error accounting with declared synthetic inputs. It is not a model benchmark. The report leaves human efficiency and agreement null when no real reviewer evidence is supplied.

To analyze real reviewer artifacts, extend the protocol's episode records with `condition` (`manual` or `assisted`), `group` (session), `reviewer`, `accepted`, and `review_history` containing every event's `active_s` and `wall_s`. Use independently adjudicated `reference` labels. Optional `reviewer_a` and `reviewer_b` label lists measure pre-adjudication agreement. The harness sums correction and acceptance events, reports group counts, bootstraps seconds/accepted-segment by session when at least two groups exist, and emits descriptive time reduction only when both conditions are present. A counterbalanced design, setup costs and reviewer assignments remain the experimenter's responsibility; the code cannot manufacture independence.

For fixed-budget error selection, provide `error_pool` records with `id`, `priority`, positive `review_s`, adjudicated boolean `erroneous`, and `budget_s`. The harness compares priority order with a seeded random order and counts all missed errors. A fully labeled pool is required; selected-item labels alone cannot establish recall. This implementation does not extrapolate from a weighted audit sample.

Temporal matches maximize cardinality and then total IoU within each episode and class. Reports include micro/macro and per-class F1 at 0.3, 0.5 and 0.7, plus per-episode matched boundary median/p90 and explicit false-positive/negative counts. There is no automatic quality noninferiority decision or population claim. Follow the [frozen study design](evaluation.md) before collecting and interpreting human results.

## Remaining acceptance gates

The original M1–M4 prototype and M5 software harness run. The ecosystem migration now verifies native review mechanics, trusted native snapshots, explicit approval and fresh accepted exports. Measured active human time, queue integration and actual LeRobot consumption remain open. M5's research gate also remains open: real public episodes, independent reference reviewers, counterbalanced timing, confidence intervals for quality preservation, and evidence that prioritization finds real errors. Learned action recognition, object masks/identities, near-duplicate embeddings, multi-camera adapters and downstream policy training are not implemented.
