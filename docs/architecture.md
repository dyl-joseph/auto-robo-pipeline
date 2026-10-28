# Proposed architecture

Status: original design retained below for context. The implementation now lives in `robo/`, `web/` and `tests/`. The shipped CLI uses `python -m robo`; see the README for exact commands. The native JavaScript editor replaces the proposed React client to avoid a second application build for the first single-page workflow. SQLite owns review transactions, immutable local files own source/analysis artifacts, and FastAPI serves loopback media with byte ranges. M5 human research gates remain open; see [results](results.md).

## Caller experience

The intended command sequence is:

```text
episodes import --manifest data/local/episode-source.json
episodes suggest --episode <episode-id> --config configs/baseline.toml
episodes serve --host 127.0.0.1 --port 8000
episodes export --selection <selection-id> --output artifacts/<version>
episodes evaluate --protocol configs/evaluation.toml
```

Import errors must explain which media, timestamp, or unit is invalid. Suggestion failures leave the manual workflow usable. Export rejects unresolved annotations and prints a validation report. Commands return nonzero status on failure.

## Design alternatives

Two alternatives were compared sequentially in this planning pass. They are design candidates, not measured prototypes or independent reviews.

| Criterion | A: local application with offline processing | B: distributed event pipeline |
| --- | --- | --- |
| Shape | Python core, offline commands, FastAPI, React editor, SQLite, local artifacts | Queue, worker services, object storage, database, separate API and editor |
| Ownership | One process owns review writes; offline runners own immutable run directories | Services own separate stores and exchange versioned events |
| Failure handling | Atomic file publication and optimistic review revision checks | Retry queues, delivery deduplication, distributed consistency |
| Evaluation | Same cached predictions and local manifests across conditions | Reproducibility also depends on service/event snapshots |
| Fit for 100–200 episodes | Small operational footprint; video/model latency remains an empirical question | More deployment and state-management work before the core experiment |
| Growth cost | Storage and job interfaces can be replaced if measured bottlenecks require it | Higher initial cost; useful if concurrent workloads outgrow one machine |

Choose A. Retain B's explicit job IDs, immutable artifacts, and idempotent processing. Defer queues and remote object stores until measured capacity or concurrency requires them. This trades horizontal scale for a workflow that can be reproduced on one machine.

## Data and state ownership

| Record | Proposed fields | Invariants and writer |
| --- | --- | --- |
| `Episode` | Schema version, source ID/release, content hashes, task, camera streams, optional robot streams, source license reference | Importer writes once; media references identify immutable source objects |
| `Stream` | Modality, URI, units, original timestamps/time base, reference clock offset, alignment policy | Importer rejects nonmonotonic timestamps and undeclared units; absent streams are explicit |
| `SuggestionRun` | Episode hash, model/weights hash, code version, config hash, seed, output hashes, status | Offline runner owns one immutable run directory; failed output is never published as complete |
| `SegmentSuggestion` | Run ID, label candidate, start/end in episode time, raw score, evidence references | Machine output only; raw scores are not called calibrated probabilities |
| `AnnotationRevision` | Episode ID, base revision, segments, outcome, quality flags, reviewer ID, timestamp, rationale | Review service validates and atomically writes the next revision; human acceptance never mutates a suggestion |
| `ReviewEvent` | Event ID, revision, actor, action, previous/new values or revision references, suggestion origin, active duration | Review service appends; prior decisions remain auditable |
| `DatasetVersion` | Selected episode/revision IDs, source and artifact hashes, split groups, export schema version, code/config versions, validation report | Exporter publishes an immutable manifest only after validation succeeds |

Keep task outcome, recording quality, annotation status, and execution quality separate. A successful episode can contain a recovery. A low-quality recording may be unsuitable for a particular export without implying bad robot behavior.

Use half-open temporal intervals `[start, end)` on the reference episode clock. Require `0 <= start < end <= duration`. Preserve presentation timestamps for variable-frame-rate video; do not substitute `frame_number / nominal_fps`. An index is a playback convenience, not the time authority.

Map source clocks into episode time with recorded offsets and any documented drift correction. Store original values. For nearest-neighbor alignment, record maximum allowed error and unmatched samples. Never silently interpolate categorical events or robot actions. Missing alignment evidence must be visible to reviewers and export consumers.

A task taxonomy declares whether labels are mutually exclusive, hierarchical, or allowed to overlap. Use `unknown` and `not_observable` instead of forced guesses. Keep recovery and outcome vocabularies separate from movement primitives.

The review state transition is `unreviewed → in_review → accepted | needs_adjudication | excluded`. An edit to an accepted revision creates a new unreviewed revision. Exclusion requires a reason. Export eligibility is an explicit policy over the latest selected revision, not a side effect of a model score.

## Module boundaries and proposed signatures

The following pseudocode describes contracts, not shipped APIs:

```python
import_episode(source: SourceManifest, store: ArtifactStore) -> ImportResult
validate_alignment(episode: Episode, policy: AlignmentPolicy) -> ValidationReport
generate_suggestions(episode: Episode, config: ModelConfig) -> SuggestionRun
rank_review_items(items: list[ReviewItem], budget: ReviewBudget) -> ReviewSelection
commit_review(episode_id: EpisodeId, expected_revision: RevisionId,
              edit: ReviewEdit, actor: ReviewerId) -> AnnotationRevision
export_dataset(selection: ExportSelection, policy: ExportPolicy) -> DatasetVersion
evaluate_run(protocol: EvaluationProtocol, artifacts: EvaluationArtifacts) -> MetricsReport
```

- `core/` owns domain types, temporal rules, ranking, and export policy. It does not import FastAPI or a model framework.
- `ingest/` owns media probing, external dataset parsing, and boundary validation.
- `models/` owns replaceable inference adapters and cache keys. It does not decide whether human labels are accepted.
- `review/` owns SQLite transactions, revision checks, and audit events.
- `api/` adapts HTTP requests to core/review operations, validates input, and serves permitted local media.
- `web/` will contain a React/TypeScript editor with keyboard controls and synchronized timeline playback. Add its lockfile when M2 starts.
- `evaluation/` owns split creation, matching rules, timing analysis, and reports. Evaluation manifests are not training inputs.

Start with a few cohesive modules inside one Python package rather than separate deployable packages. Add a model adapter only when a concrete backend needs it.

## Review and serving behavior

The editor has video, temporal tracks, an editable label form, a review queue, and an audit panel. A queue item explains why it was selected. Keyboard editing, visible save state, undo via revisions, and an explicit uncertain state are part of the first usable editor.

The API is local-only by default. Serving a media path must resolve it beneath the configured dataset root; reject arbitrary filesystem access and unsupported remote URLs. Do not expose a public service until authentication and authorization are implemented. Read-only video access must support seeking.

Two browser tabs may edit the same episode. `expected_revision` prevents one tab from overwriting the other. A stale save returns a conflict that the editor can resolve. SQLite is appropriate for one writer and a small team pilot; do not create shared JSON-file writes and then add a global lock.

## Processing and provenance

- Cache inference by media hash, preprocessing version, model weights/config, and code version. Changing any of these invalidates the cache entry.
- Write results into a run-specific temporary directory, validate them, then publish atomically. Retries reuse complete artifacts and isolate incomplete outputs.
- Record sampled-frame choices, model license, hardware, timing, seed, and nondeterministic operations. Reproducible provenance does not imply bitwise deterministic GPU inference.
- Keep model inference outside HTTP requests. The review UI loads cached suggestions and remains usable if inference fails.
- Preserve timestamp ranges and validity masks in training-facing exports. Do not claim support for a training framework until its consumer validates an exported sample.

## Verification boundaries

M1 tests external input validation and repeated imports. M2 tests save/reopen/export behavior and stale edits through public API/UI surfaces. M3 uses a small fixed inference fixture and verifies suggestion provenance and human override. M4 checks a held-out, fixed-budget review simulation. M5 reproduces an export and metrics from their manifests.

Revisit this architecture if measured video/model workloads exceed the local resource budget, multiple review writers cause sustained contention, or a training consumer requires a different alignment contract. Separate immutable processing outputs before introducing shared distributed state.
