# Verification results

The tests cover source import, review, approval, training-format conversion and controlled data recovery. The DROID evidence uses one real episode. Browser actions and approvals are automated test fixtures; no human study or robot policy evaluation was performed.

| Check | Result and evidence |
| --- | --- |
| Core Python behavior | 117 tests passed, covering import, clocks, provenance, review, exports, recovery and batch retries |
| Native review | Actual Label Studio video, sensor seeking, correction, submit, API reread and reload passed; [browser evidence](evidence/native-review-verification.json) |
| Native approval | Refresh, explicit acceptance, repeat export, stale-export rejection and historical validation passed; [approval evidence](evidence/native-approval-verification.json) |
| Native loading and plot regressions | Pinned cold-link and collapsed-plot defects reproduced; supported table navigation and visible plots passed; [sync evidence](evidence/native-sync-contract.json) |
| Batch workflow | Four inputs produced two unique tasks and one input failure. Interrupted import, numeric queue order, changed uncertainty and accepted export passed; [recorded run](evidence/batch-workflow.json) |
| Official LeRobot | Five isolated integration tests passed, including held-out split isolation, decoded images, action windows, hashes and failed-decode cleanup |
| Synthetic native export | 96 rows loaded through LeRobot, with zero source frames omitted and two repeated across a deliberate gap; [recorded run](evidence/lerobot-roundtrip.json) |
| Real DROID export | All 119 loaded state/action vectors matched decoded RLDS values; every selected-camera image checked; zero omitted or repeated source frames; [recorded run](evidence/droid-roundtrip.json) |
| Controlled corruptions | Two seeds each classified 16 cases correctly: four complete repairs, three data-review cases, seven rejections and two clean controls; [benchmark](messy-droid.md) |
| MLX PI05 | 14 small-model tests passed on CPU. Loss error was `2.38e-7`, maximum gradient error `2.55e-7`, and resumed parameters matched exactly; [recorded run](evidence/mlx-pi05.json) |
| Code and browser checks | Ruff, formatting and JavaScript syntax passed; seven browser tests cover desktop/mobile review and response diagnostics across navigation |

The [real DROID screenshot](evidence/droid-batch-review.png) shows a missing-sample variant in native review, with an automated-test label and source attribution. The [synthetic walkthrough](evidence/labelstudio-walkthrough.webm) records a temporal correction. Older custom-editor images document the earlier prototype.

The [demo guide](demo.md) reproduces acquisition, native correction and export, LeRobot loading, batch recovery and the benchmark. Each integration uses its locked environment.

## Scope of the measurements

The corruption benchmark uses seeded perturbations of one organized episode. Its clean reference and answer key stay outside the recovery process. Zero silent corruption in this suite does not establish recovery quality on arbitrary recordings. Incomplete and rejected cases remain visible in the report and are not counted as repaired.

The MLX result uses random small-model weights and the official encoder implementation. Full pretrained training has not run. Official tokenizer requests returned HTTP 401, so pretrained preprocessing needs authorized local assets. Neither the M5 Pro nor the RTX 3060 machine was available; memory use and accelerator performance remain unmeasured.

Human annotation time, agreement, semantic-label accuracy and policy quality remain unmeasured. Native `lead_time` is retained as raw metadata, while active human time is null. The pipeline does not turn automated test durations into efficiency estimates.

## Evaluate supplied review artifacts

```bash
uv run --locked python -m robo evaluate examples/synthetic-evaluation.json artifacts/evaluation.json
uv run --locked python -m robo validate-export PATH_TO_EXPORTED_VERSION
```

The [fixture evaluator output](evidence/synthetic-evaluation.json) exercises matching and missed-error accounting. Its specified pool gives prioritized recall 0.5 and seeded random recall 1.0. Those numbers describe the fixture, not real review performance.

For a human study, supply `condition`, session `group`, `reviewer`, `accepted` and the complete `review_history` with `active_s` and `wall_s` for each episode. Labels in `reference` must come from independent adjudication. Optional `reviewer_a` and `reviewer_b` lists measure agreement before adjudication. The evaluator sums correction and acceptance time, reports group counts and bootstraps seconds per accepted segment by session when at least two groups exist. Counterbalancing and independent assignments remain part of the study design.

For a fixed review budget, supply `error_pool` records with `id`, `priority`, positive `review_s`, adjudicated `erroneous`, and `budget_s`. Priority and seeded random orders are compared against the complete labeled pool, including missed errors. Selected-item labels alone cannot establish recall; the current evaluator does not extrapolate from a weighted audit sample.

Temporal matching maximizes the number of matches, then total IoU, within each episode and class. Reports include micro, macro and per-class F1 at IoU 0.3, 0.5 and 0.7, matched boundary errors, false positives and false negatives. Follow the [study protocol](evaluation.md) before interpreting these measurements as a quality or efficiency result.
