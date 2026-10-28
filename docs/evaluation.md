# Evaluation protocol

Status: proposed human-study protocol. The matching, timing, agreement and fixed-budget harness is implemented in `robo/evaluation.py` and exercised with explicit fixtures. No human study or real robot benchmark has been run. See [results](results.md) for the protocol input fields and observed software checks. Freeze numeric thresholds, split manifests, and analysis rules before a held-out run.

## Hypothesis and decision rule

Assisted annotation should reduce active human time per accepted segment without materially degrading temporal label quality. Review prioritization should find more genuine errors at the same human time budget than random sampling.

Proposed engineering targets, subject to a pilot before freezing the protocol:

- At least 30% lower median human time per accepted segment than manual annotation.
- No more than 2 percentage points lower segment F1 at temporal IoU 0.5 than manual annotation against an adjudicated reference.
- Higher error recall than random selection at the same human review time budget, with precision reported alongside it.

These are goals, not predicted or measured improvements. If the data cannot support a quality-preservation conclusion, report the experiment as inconclusive. Do not hide an accuracy loss behind aggregate time savings.

## Dataset and reference labels

1. Establish a 10–20 episode pilot to refine the taxonomy and collect timing variance. Pilot episodes cannot enter the held-out test set.
2. Aim for 100–200 short episodes if acquisition and reviewer capacity allow. Record the actual episode count, duration, task distribution, and missing modalities.
3. Group near-duplicates, shared sessions, and related trajectories before splitting. Split by group, not frames. Record deterministic train/development/test manifests. Keep test media and labels out of threshold/model selection.
4. Make a reference subset independently annotated by two reviewers, blinded to machine suggestions. Write observable label definitions and boundary rules. Adjudicate disagreements and retain the pre-adjudication versions.
5. Determine the reference-set size from pilot variability and available independent groups. Report uncertainty; a small convenience sample is a pilot, not population-level evidence.

Reference guidelines must define what counts as reach, grasp, release, reposition, task-specific manipulation, recovery, unknown evidence, and task completion. A camera view alone may not prove physical contact. Use available evidence and record ambiguity.

## Experimental conditions

| Condition | Labels initially shown | Queue policy | Purpose |
| --- | --- | --- | --- |
| Manual | None | Fixed randomized order | Human time and quality baseline |
| Assisted | Cached machine suggestions | Same ordering policy as manual | Effect of draft annotations |
| Assisted + prioritized | Same cached suggestions | Frozen transparent ranking | Added effect of review selection |
| Random review budget baseline | Same cached suggestions | Seeded random sample | Comparison at fixed human time budget |

Use the same editor and taxonomy across conditions. Counterbalance condition and episode allocation across reviewers. Avoid having a reviewer label the same clip in multiple conditions; memory would bias timing. Analyze episode and reviewer clustering. With only one reviewer, describe the study as a preliminary usability experiment and do not claim inter-annotator evidence.

A prioritization ablation must hold suggestions constant. Compare explicit signals such as low detector score, track discontinuity, inconsistent event transitions, and alignment warnings. Report model-only and corrected labels separately so human effort is visible.

## Timing and metrics

Start active timing when the reviewer opens an item. Include playback, seeking, boundary editing, rejection, and acceptance. Record pauses and idle time using a rule frozen before analysis. Report active time and wall time separately, including setup/training time as a separate cost. Do not omit the expensive clips or include abandoned reviews as successful annotations.

| Metric | Definition or reporting rule |
| --- | --- |
| Human efficiency | Active seconds per accepted segment and per reviewed video minute; include correction/rejection time |
| Time reduction | `1 - assisted_time / manual_time` on comparable, prespecified aggregates; report sample sizes and uncertainty |
| Throughput | Accepted segments per human hour; report annotation quality beside it |
| Temporal label quality | Per-class and macro segment F1 at temporal IoU thresholds 0.3, 0.5, 0.7 |
| Segment matching | One-to-one matches within an episode and class, maximizing temporal IoU above the threshold; unmatched predictions/reference items are false positives/negatives |
| Boundary accuracy | Median and 90th percentile absolute start/end error in seconds on matched segments; unmatched rates reported separately |
| Reviewer agreement | Per-class segment agreement before adjudication plus boundary deviations; state matching rules |
| Error selection | Precision and recall of adjudicated erroneous items at fixed review time; quantify missed errors in unreviewed items |
| Calibration | Reliability bins or calibration error on development labels if a probability is claimed; raw model scores remain scores |
| Dataset coverage | Task, object, outcome, recovery, session, and quality-flag distribution before/after selection |
| System cost | Decode/inference latency, CPU/GPU type, memory, storage, API expense if used, and failed-run rate |

For recall estimation, audit a seeded random sample of the unreviewed pool as well as the prioritized items. Apply known sampling weights when estimating pool-level totals. Evaluating only selected items cannot measure missed-error recall. Treat unresolved reference items explicitly and report exclusions.

Bootstrap uncertainty at the episode/session-group level, respecting reviewer assignment. Report paired estimates only for a genuinely paired design. State the number of independent groups; correlated frames do not increase the independent sample count. A quality noninferiority claim needs its confidence interval to satisfy the predeclared margin, not just a point estimate.

## Export acceptance

An export passes only when all selected sources resolve, hashes match, intervals are valid, annotation revisions are eligible, label/schema versions agree, and split groups do not leak. The manifest records code/config versions, dataset/model provenance, inclusion/exclusion reasons, and artifact hashes. Rebuilding from the manifest must reproduce semantic contents and canonical content hashes; volatile timestamps are kept outside that comparison.

A downstream policy-learning ablation is optional after the workflow works. It requires a pinned training recipe, a fixed compute budget, multiple seeds, and held-out robot/task evaluation. Do not claim improved robot performance based on annotation metrics alone.

## Release evidence

Publish the frozen protocol, acquisition and split manifests to the extent licensing permits, anonymized review timings, matching/metric code, dataset card, representative corrected failures, and a table of observed results. Separate passed targets, failed targets, unavailable measurements, and limitations.

The source notes link [Physical Intelligence research](https://www.pi.website/blog/pi07). Its current contents and detailed claims have not been verified during this setup. Verify any research statement against the primary source before citing it in a project report. The experiment does not depend on that claim being true.
