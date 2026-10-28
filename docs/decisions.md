# Design decisions

This record describes the choices behind the current implementation. The [results page](results.md) contains the corresponding checks.

## Use Label Studio for annotation

The first prototype included a custom editor. Label Studio replaced it as the primary UI because it already provides video timelines, synchronized plots, annotation history and authentication. Project code now handles robot clocks, provenance and export eligibility. Styling uses supported configuration rather than an application fork.

Two defects in pinned Label Studio 1.23.2 affect the supported workflow. Open tasks from the Data Manager table because cold direct links can lose predictions. Keep sensor plots visible because collapsed plots can retain a zero canvas width and fail to seek. Browser regressions reproduce both defects and verify the working paths.

## Keep source data and review media separate

Label Studio expects a constant-rate video. The bridge creates a display proxy and records its frame mapping instead of rewriting the source recording. Training uses original streams and causal alignment. Missing samples remain missing, and a derived video with an unresolved clock origin cannot pass training conversion.

## Require a separate approval

Saving a correction and approving a dataset have different consequences. The bridge records an immutable draft of the selected native annotation, then requires an accept or reject decision. Approval and new exports recheck the native snapshot. Historical exports keep their original hashes when later edits occur.

## Reuse the training ecosystem

LeRobot owns dataset creation and loading. A separate lockfile avoids conflicts with Label Studio's dependencies. The PI05 adapter reuses official preprocessing and a frozen PyTorch encoder; only the action expert runs and trains in MLX. This avoids maintaining a second tokenizer or vision encoder.

## Test recovery without supplying the answer

The corruption benchmark keeps the clean DROID episode and corruption key with the evaluator. Recovery receives opaque input directories and declared metadata. It may sort rows or remove exact duplicates, but it cannot recover missing values or choose between conflicting observations. The report counts rejected inputs separately from complete repairs.

## Keep the experiment local

Source data, weights, databases and sessions live in ignored directories. The pinned DROID sample is 9.54 MB, which is enough to test the complete path without downloading the full dataset. A single attributed screenshot documents the native review UI. Human-efficiency and full-policy claims require separate measurements.
