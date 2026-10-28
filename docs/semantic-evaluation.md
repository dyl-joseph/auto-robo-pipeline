# Semantic segments and independent evaluation

The semantic contract describes text-labeled temporal segments and compares unedited predictions with separately supplied references. It is the input/output boundary for the planned robot-state baseline and small video-language adapter. Those predictors are not implemented by this change. No independently verified semantic reference set has been loaded, so the current tests establish evaluator behavior, not labeling accuracy.

## Three separate inputs

`robo.semantic` defines validated Python models; each also reads and writes JSON through Pydantic. Unknown fields and non-finite numbers are rejected.

| Input | Required information |
| --- | --- |
| `EvaluationPlan` | Frozen taxonomy and guideline hash; episode, dataset, release, original episode, session and camera identities; source/media hashes; source frame boundaries and clock type; split assignments and identity audit |
| `Predictions` | Plan hash, producer, run, code revision, configuration hash, declared training/tuning episodes and one result for every evaluated episode; results must be uncorrected and declare no reference access |
| `References` | Same plan hash and episode population; complete reference segments; annotation source, file hash, license, annotators, verification status, method and whether annotators saw the predictions |

The plan contains no labels. Keep reference files out of the prediction process. Freeze the plan, taxonomy, boundary rules and review threshold before the held-out run. The evaluator checks their hashes and declarations; it cannot prove that a person kept a file private or that a pretrained model never encountered related data.

`episode_spec(...)` maps an existing imported episode into the plan using its content/media hashes, session and integer video PTS. The caller supplies the dataset's original episode identity and camera name so multiple views can be linked correctly. Inferred DROID timing stays inferred. Include all training, development and test episodes in the catalog: shared original episodes, sessions, exact source/media hashes or declared near-duplicate groups cannot cross splits. Repackaging the same episode under a new release does not make it independent. Near-duplicate groups require a separate audit; hash checks alone cannot find them.

## Segment fields

Every `SemanticSegment` carries:

- `episode_id`, `start_s`, `end_s` and `source_frames`. Frame ranges are half-open: `start` is included and `end_exclusive` is not. `evidence` lists the source frames actually used inside that segment.
- `subtask_text`, `normalized_action` and nullable `object_text`. The normalized action is a semantic label such as `grasp`, not a robot command vector. Free text is retained alongside the fixed taxonomy.
- `uncertainty` in `[0, 1]`, required for predictions and nullable for references. Higher values request more review; the contract does not claim these scores are calibrated.
- `provenance`: the producer, method kind and source artifact SHA256. Reference segment hashes must agree with the reference audit.

Times must match the plan's source-frame boundaries within one microsecond. Segments must be ordered, non-overlapping and inside the episode. References must cover every frame, using `unknown` or `not_observable` explicitly where needed. Predictions may leave gaps, but the evaluator reports those gaps and still counts unmatched reference segments as misses.

The existing Label Studio and LeRobot workflow continues to use its current annotation schema. This contract does not silently replace those annotations or add semantic fields to an accepted export. The later model/review slice will connect these fields to that workflow while preserving explicit approval.

## Run the comparison

```bash
uv run --locked python -m robo.semantic_evaluation \
  artifacts/semantic/plan.json artifacts/semantic/predictions.json \
  artifacts/semantic/references.json artifacts/semantic/report.json
```

The default split is `test`; `--split development` can evaluate development episodes that were not used to train or tune this run. Supply only the chosen split in the prediction and reference files, while retaining the complete split catalog in the plan. A failed prediction needs an explicit episode record with an error and no segments. Omitting that episode is an error. Repeating an identical evaluation is safe; a different result requires a new output path. Inputs are local JSON files, capped at 64 MiB each. The command never loads pickle or downloads a dataset.

The evaluator reuses the existing one-to-one temporal matcher. At IoU thresholds 0.3, 0.5 and 0.7 it reports precision, recall, F1 and per-action counts in two forms: normalized action alone, and normalized action plus exact object text. Text comparison ignores case and repeated whitespace. It does not treat paraphrases or object synonyms as equivalent. Exact subtask-text agreement is a separate diagnostic over matched segments, not a measure of free-form semantic correctness. Boundary median/p90 errors also cover matched segments only; false positives and misses remain in the counts.

All raw segments are scored, including uncertain ones. Failed episodes contribute misses. Review counts come from the frozen uncertainty threshold, unknown labels, unlabeled frames and failures. They do not filter the benchmark or imply automatic approval, calibrated confidence or human time savings. Corrected annotations need a separate future evaluation; the primary command accepts only uncorrected predictions.

## Reference acceptance

A real reference evaluation requires a declared human-verified identity audit and independently human-verified episode annotations. Blind manual annotations and audited publisher annotations are supported; references exposed to the evaluated predictions or derived through model-assisted correction are rejected. Source hashes must differ from the prediction artifacts. These checks validate the supplied audit record, not the truth of its claims.

DROID task instructions are episode-level context, not temporal ground truth. The publisher's [RoboInter demo documentation](https://github.com/InternRobotics/RoboInter/tree/main/RoboInterData-Demo) is a candidate source to audit separately. No RoboInter data has been downloaded or accepted as a reference, and no adapter assumes an identity match from a filename alone. License/access checks, episode/session/camera matching and human-verification status must be resolved first.

Generated fixtures declare `synthetic_fixture` throughout, and their reports retain that designation. Run `uv run --locked pytest -q tests/test_semantic.py` to check matching counts, failure accounting, frame alignment, split leakage, independent-reference gates and the actual JSON CLI. The tests deliberately include wrong objects, shifted boundaries and a failed episode; none of their labels are presented as human ground truth.
