# Robot-state baseline

The baseline turns joint and gripper changes into coarse text-labeled segments. It provides a simple comparison for the planned video-language model. It uses observed state only: no images, task text, command vectors or reference annotations. Object identity and contact remain unknown.

```bash
uv run --locked python -m robo.semantic_baseline \
  artifacts/semantic/plan.json data/local/droid-work artifacts/state-predictions
```

The [evaluation plan](semantic-evaluation.md) must describe episodes already imported into the source store. Its taxonomy needs `unknown`, `not_observable`, `motion`, `hold`, `grasp` and `release`. The command checks each source snapshot, media hash and frame clock against that plan. Use a fresh output path for a new run; repeating the same run checks and reuses the existing output.

## How it labels state

The default streams are `state_joint_position` (seven radians) and `state_gripper_position` (one normalized closure value). They match the [DROID adapter](droid.md). Each video frame uses the latest state sample within the stream's declared tolerance, through the same causal alignment function used by LeRobot conversion. Missing or stale samples fail that episode and remain visible in `predictions.json`; values are not filled in.

Rates use changes between distinct observed samples divided by the elapsed source time. A video frame that reuses the same state sample is `unknown`, since reuse does not establish stillness. The first frame is also `unknown` because it has no earlier sample for a rate estimate.

Gripper closing/opening creates `grasp`/`release` candidates. The text explicitly says contact is unverified. Otherwise, joint movement creates `motion`, and small changes create `hold`. These are coarse candidates, not detected object interactions. Each segment retains its source-frame range, all contributing frames and the hash of the state evidence artifact. DROID's inferred timing stays inferred.

Defaults are hand-set: joint speed 0.12 rad/s, gripper change 0.2 per second, and uncertainty 0.85. Unknown segments use uncertainty 1. These values have not been validated against independent annotations. The score is a review heuristic, not a calibrated probability. All candidates require review under the default evaluation threshold of 0.6.

An optional `--config config.json` can change the stream names, thresholds and uncertainty. Declare any episodes used to choose those settings in `tuning_episode_ids`. They must appear outside the evaluated split in the frozen plan. The runner copies that declaration into its prediction artifact; it cannot independently verify how someone selected a threshold.

## Compare and review

The output contains `predictions.json`, per-episode state evidence and `run.json` with file hashes, configuration and failure counts. The prediction process has no reference-file argument and does not read stored review revisions. Its output can be scored with the [semantic evaluator](semantic-evaluation.md#run-the-comparison) once independent references exist. Failed episodes stay in the prediction population and count as misses.

This command creates prediction artifacts. It does not write native annotations or accept them into an export. The semantic review integration is a separate step; the existing native approval workflow remains unchanged.

## Checks

```bash
uv run --locked pytest -q tests/test_semantic_baseline.py
```

Generated state tests cover motion, stillness, gripper changes, missing samples, wrong units/shapes, held-sample uncertainty, untouched review labels and repeat runs. On the existing real DROID pilot, the CLI produced nine candidate segments covering all 119 frames with no episode failure: four `motion`, two `grasp`, one `release`, one `hold` and one `unknown`. These counts show that the state path runs; they do not establish that the labels are correct. [Recorded metadata](evidence/semantic-state-baseline.json) preserves that distinction. Source footage/data: Khazatsky et al. (2024), [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
