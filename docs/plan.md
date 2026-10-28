# Build roadmap

## Objective

Build a credible proof of work for visual data and annotation automation in robotics. Demonstrate engineering judgment through reliable multimodal alignment, useful draft annotations, efficient human review, explicit quality criteria, and reproducible dataset exports.

The primary outcome is **accepted, correct temporal annotations per human review hour**. Throughput gains count only if label accuracy remains within a predeclared quality margin. The attached concept's illustrative time savings are not evidence and will not be used as results.

## Product journey

1. Import a small set of robot episodes with videos, timestamps, task descriptions, and any available robot state/action streams.
2. Inspect missing frames, timestamp gaps, coverage, and source metadata before inference.
3. Generate coarse action segments, object interaction candidates, and quality warnings. Keep machine suggestions separate from human decisions.
4. Show a prioritized review queue with the reason each item was selected.
5. Let a reviewer play synchronized video, adjust segment boundaries and labels, accept or reject suggestions, and flag uncertain evidence.
6. Export only eligible annotations with their provenance, review history, split membership, and quality metadata.
7. Compare annotation time and held-out quality with a manual workflow using the same editor.

## Scope and defaults

| Area | First complete demonstration | Later work |
| --- | --- | --- |
| Data | A licensed, explicitly selected subset of public robot episodes | New collection protocols and live robot capture |
| Scale | 10–20 episodes to prove the workflow; 100–200 if acquisition and review effort permit | Distributed processing of entire corpora |
| Perception | One replaceable baseline with cached outputs | Multiple detectors, trackers, segmentation models, and VLM comparisons |
| Labels | Task-specific temporal subtasks, outcome, quality flags, and explicit unknowns | Dense masks and exhaustive object tracking |
| Review | Single reviewer workflow with revision history; two reviewers for the reference subset | Multi-tenant collaboration and identity management |
| Curation | Transparent checks, uncertainty sampling, duplicate groups, coverage report | Learned acquisition policies |
| Training integration | Validated JSONL/Parquet exports with alignment and provenance | Policy training, downstream ablations, additional dataset adapters |

Do not train a foundation model, promise autonomous annotation, or build distributed infrastructure to prove this first experiment. Recovery and lower-quality behavior remain useful data; quality flags must not silently discard them.

## Dataset acquisition

Start by checking whether a small DROID subset can be downloaded and used under its current access and licensing terms. Use the [DROID project](https://droid-dataset.github.io/) and [official repository](https://github.com/droid-dataset/droid) as entry points. These links are acquisition leads, not confirmation that this environment has a licensed copy or a working downloader.

Before downloading:

- Record dataset release, license/terms URL, permitted redistribution, source URLs, hashes, and storage estimate in a local acquisition manifest.
- Confirm video codecs, camera timestamps, robot stream units, and frame-to-state alignment semantics from the release documentation.
- Acquire one episode first, measure disk and processing cost, then cap the pilot by bytes and duration. Keep at least 5 GB of workspace free.
- Select task families with observable action boundaries. Prefer a bounded tabletop manipulation task if shirt folding is unavailable. The shirt example in the source notes is illustrative.
- If access is gated, finish the import and review workflow with deterministic synthetic fixtures, then select a legally accessible robot dataset. Synthetic fixtures validate mechanics, not annotation research claims.

No raw recordings, proprietary data, or model weights belong in this repository. Do not redistribute a dataset unless its terms allow it.

## Milestones and gates

Each milestone is an independently reviewable change. The estimates describe work order, not a promised calendar. Keep a tagged or committed usable state at each gate.

| Milestone | Deliverable and owner boundary | Acceptance gate | Recovery boundary |
| --- | --- | --- | --- |
| M0: plan and environment | This repository, locked CPU dependencies, setup and readiness checks | Environment checks and lint pass; roadmap available on `main` | Revert documentation/tooling commit; no dataset state affected |
| M1: import one episode | Python core owns `Episode`, source manifests, timestamp validation, and local artifact layout | Valid clip imports; variable timestamps preserved; corrupt media, duplicate IDs, and nonmonotonic streams fail explicitly; missing modality is represented | Delete only the failed run's temporary output; source remains immutable |
| M2: manual review slice | API owns revisions; web UI owns playback and editing; export consumes only reviewed revisions | Import → play → label → save → reopen → export works; stale edit rejected; export cannot include unreviewed suggestions | Retain immutable revisions; disable new UI/API surface without altering raw data |
| M3: assisted annotation | Offline runner owns baseline predictions and quality signals; human editor remains authoritative | One baseline produces cached suggestions; user can correct or reject them; model version/config recorded; rerun reuses valid cache | Select manual workflow; never overwrite prior human labels |
| M4: prioritized curation | Pure ranking logic owns review selection and coverage reports | Fixed budget comparison against random review; report selection reasons, missed errors, duplicate groups, and recovery coverage | Use deterministic random queue with recorded seed |
| M5: evaluation and release | Evaluation code owns split manifests and metrics; report cites artifact hashes | Reference subset adjudicated; timings include correction; confidence intervals and limitations reported; clean export reproduced | Publish partial results honestly; keep failed targets visible |

The dependency order is `M0 → M1 → M2 → M3 → M4 → M5`. Acquisition and reference guidelines can progress alongside M1/M2, but test-set tuning cannot.

## First implementation task

Implement M1 only after reviewing the [proposed contracts](architecture.md). Create a minimal Python package and an `episodes import` command that accepts a local video plus explicit metadata. The command is proposed and does not exist yet.

Required behavior:

- Assign a stable episode ID from the source identity and content, not the absolute local path.
- Inspect video duration and presentation timestamps with FFprobe. Preserve original time bases.
- Validate every modality independently and record offsets, tolerances, and unavailable streams.
- Persist an immutable source manifest and a validation report using an atomic write.
- Repeating the same import returns the same identity and does not duplicate artifacts. Conflicting content under the same declared source identity is rejected.
- Return actionable errors for corrupt video, missing timestamps, unsupported units, and invalid alignment. Do not invent robot states from video-only data.
- Test a short generated video, a video-only episode, irregular timestamps, timestamp regression, missing media, and retry after a partial write.

Then implement M2 with manual labels before adding automated suggestions. This gives the evaluation a usable manual baseline and avoids optimizing models before there is an end-to-end workflow.

## Portfolio release

The release should include a short walkthrough of a real episode with uncertain/recovery behavior, reproducible scripts, a dataset card, exact source/model versions, and a results table. Show an actual corrected error and its audit trail. Publish only media permitted by the source license.

The report should answer:

- How much human time did assistance save, at what label quality?
- Which errors did prioritization find, and which did it miss?
- Did curation preserve underrepresented failure and recovery behavior?
- Can another engineer reproduce the export and the reported metrics?

Use prior tracking, segmentation, annotation, and product experience as motivation. Evidence from this project must stand on its own. Do not describe a prototype as Physical Intelligence's production pipeline or repeat unverified research/job claims from the source notes.
