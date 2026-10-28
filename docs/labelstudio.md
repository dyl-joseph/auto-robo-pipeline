# Label Studio integration

Label Studio 1.23.2 and its official Python SDK 2.1.2 are locked in the default `labelstudio` dependency group. Native Video, TimelineLabels and TimeSeries own playback, temporal editing and sensor visualization. This replaces the bespoke editor as the primary application path.

The labeling XML is [review.xml](../integrations/labelstudio/review.xml). Its cream background, compact monospace typography, serif title and square black media border follow the [reference-derived design](design.md), applied through the supported Style/View tags. Label Studio branding remains visible. The application chrome and authentication remain Label Studio's own. There is no frontend fork or deprecated standalone frontend dependency.

## Local use

`python -m robo serve` runs native Label Studio on 127.0.0.1:8000. Its data lives under the configured robotics store's `labelstudio/` directory. It does not automatically create users or API credentials. `prepare-labelstudio EPISODE_ID` creates a native task import JSON, labeling XML, explicit frame mapping and provenance sidecar.

The first-slice manual path uses the normal Label Studio project setup/import UI and a project-scoped Local Files storage rooted at `STORE/review_proxy`. Open imported tasks by clicking their row in the native Data Manager table. The application document root is the selected store, while the project storage authorizes only the review proxy subdirectory. Do not authorize the full filesystem or expose the development server publicly.

For automation, `publish_bundle()` accepts an already authorized official SDK client. It creates a project, authorizes its proxy storage and imports the task/predictions, returning a trusted source/task/proxy binding. The bridge does not create an API credential for that client. `annotation_snapshot()` validates native SDK JSON exports against that binding and emits draft snapshots. Promotion of these snapshots to the existing accepted-export workflow remains a subsequent integration gate.

## Timing and frame semantics

Label Studio recommends constant-frame-rate video. The bridge produces a separate display proxy by holding the last measured source frame until its next original PTS. Its default rate is the ceiling of the reciprocal of the shortest measured frame interval, up to 60 fps. It checks that every source frame appears at least once. If that cannot be achieved within the pilot limit, preparation fails rather than silently dropping observations. An explicit `--fps` opts into a different grid; the manifest and task banner report omitted source frames. This retains frame coverage, not lossless video pixels or arbitrary timestamp precision: the proxy is H.264 re-encoded.

Every display frame records its original source index, integer PTS and episode time. Raw video and robot streams remain unchanged. Native timeline spans are one-based and inclusive: `[a,b]` becomes `[(a-1)/fps, min(b/fps, episode_duration))` on the **generated review clock**. Adjacent ranges share the same boundary even when they divide held copies of one source frame. We do not expand both ranges to that entire source frame's interval, which would create overlaps.

Machine boundaries snap to the nearest shared grid edge, with numerical displacement at most half a grid period. Candidates that collapse to zero width are omitted and counted in trusted provenance. Human edit resolution is one grid period. Neither bound is a claim about true event accuracy: missing source observations may create much wider uncertainty. Exported `boundary_evidence` records the preceding/following measured PTS, whether the boundary exactly coincides with a measured frame, and the full source-time bracket. A recording-end boundary has no following frame.

For example, the browser fixture removes source frames at 5/12 and 6/12 seconds. A native edit to frames 1–6 exports `[0, 0.5)` on the 12 fps review grid. Its end lies between measured observations at 4/12 and 7/12 seconds, an explicit 0.25-second bracket. It does **not** assert that a measured image or real action boundary exists at 0.5 seconds. Unit tests also cover adjacent native ranges within a held frame, 30/60 fps coverage, 30000/1001 fps input, coarse-rate loss, empty predictions and collapsed candidates.

The display sensor CSV starts at relative zero, preventing the native TimeSeries first-row offset rule from shifting delayed streams. It uses declared clock offsets and tolerance-bound nearest samples, with blanks for missing values. The task embeds those display values as numeric JSON with null gaps and numeric seconds formatting; this avoids a calendar axis. Original state/actions remain the training-facing source.

Label Studio's native `lead_time` is retained as raw metadata and is not advertised as active correction time. Browser test durations are automated software checks, not human-study results. Imported completion does not automatically approve a training annotation.

## Reproducible real-application check

```bash
uv run --locked python -m scripts.labelstudio_smoke --sync-contract
uv run --locked python -m scripts.labelstudio_smoke --browser
```

Install the locked browser/recording dependency with `npx playwright install chrome ffmpeg`, or use the system Chromium/FFmpeg setup in [development notes](development.md).

This starts the actual Django application with a disposable SQLite database, an unusable-password fixture user and a short-lived authenticated browser session. SDK calls use that session through its supported custom HTTP client. SDK 2.x requires a constructor key string; the test's non-secret sentinel is removed by HTTPX authentication before requests, and the server enforces its actual session/CSRF permissions.

Label Studio auto-creates a user token, which the test immediately removes. Its native JSON export implementation dereferences `user.auth_token` even without resource downloads, so the fixture creates a temporary token for that one export call and revokes it in `finally`. No persistent API credential is retained, no token is printed, and the entire database is removed at exit. Normal application authentication is not disabled.

Telemetry, Sentry and automatic update checks are disabled. The browser blocks all non-local requests. The supported feature-flag setting `fflag_fix_front_dev_3391_interactive_view_all=false` disables Compare All: in this pinned version, that mode leaves a per-annotation synchronization manager uninitialized during mounting. This is a bounded runtime workaround, not a patch to upstream code. Single-task annotation and sensor synchronization remain enabled.

The browser opens the native task table, clicks the imported task, selects the prediction and uses native **Create an annotation**. Prediction results explicitly carry `interactive_mode=false`, which the pinned frontend requires when copying imported results. It then edits, submits, checks persisted API data, reloads and independently exports via the SDK. Native exports represent predictions as IDs; the harness retrieves those IDs through the SDK and checks that the original machine results and run identity are unchanged.

All browser child processes have a 90-second ceiling, with shorter per-action timeouts. Failure captures include a screenshot, accessible UI snapshot and request status trace without session credentials. These fixtures exercise actual application routes and browser actions, never internal-state setters.

## Native task loading limitation

Cold navigation directly to `data?task=ID` is unreliable in pinned Label Studio 1.23.2. CI and a local run with 6× browser CPU throttling received the full prediction in the task response but rendered zero predictions and one empty human annotation. The project model version matched the imported prediction and `show_collab_predictions` was true; changing those settings or remote flags was not the remedy.

The controlled regression delivers both real task-detail responses before the real task-list response. List rows omit predictions, and the native Data Manager replaces the detailed task entry with that metadata-only row. Read-only observations confirm the task store loses its prediction; if replacement occurs before editor initialization finishes, the editor also has zero predictions. The bundled upstream `DataStore.setList` replacement and `LSFWrapper.setLSFTask` paint wait explain this ordering. No upstream code or setters are patched. The supported workflow for this pilot is to open the Data Manager table first and click a task row. The complete prediction → correction → submit → reload → SDK export check passes through that route under the same local slowdown. Ordinary page reload after saving remains part of the test. Cold direct links remain an upstream limitation, not a fixed capability of this bridge.

To exercise slower browser initialization without altering application state:

```bash
LS_CPU_RATE=6 uv run --locked python -m scripts.labelstudio_smoke --browser
# Reproduce the pinned cold-open bug, then recover through native navigation:
LS_TASK_LOADING_CONTRACT=1 LS_CPU_RATE=6 uv run --locked python -m scripts.labelstudio_smoke --browser
```

CI runs the second command. It asserts the pinned upstream failure as a negative control, clicks the native project breadcrumb and task row, and then requires every review/persistence assertion to pass. Revisit the negative control when upgrading Label Studio. Request delays reorder unmodified server responses; they do not manufacture labels. This uses Chromium's CPU throttling test API, not a performance benchmark. Read-only diagnostics record the delivered prediction IDs/counts, project settings and native store counts. The workflow still requires actual native controls and persisted backend results; it does not add predictions through internal state or skip missing suggestions.

## Native synchronization regression

`--sync-contract` reduces the official synchronized-video example to one local video and one sensor channel. It exercises a time-formatted CSV, our numeric JSON, a closed Collapse, an initially-open Collapse resized while closed, and the actual themed configuration. Browser actions perform the seeks. Read-only native state observations explain failures; the test never invokes native setters or injects seek events.

In pinned Label Studio 1.23.2, both visible minimal configurations seek correctly. The closed cases retain `canvasWidth=0` while the expanded DOM has positive width; `isNotReady` then suppresses seeking. The bundled TimeSeries implementation measures width on mount/window resize, but Collapse visibility changes do not reliably refresh that stored width. This is an upstream component interaction triggered by our layout. Changing numeric values or plot selectors did not fix it. The integration avoids placing TimeSeries inside Collapse, and verifies seeking after a real viewport resize. No CSS visibility workaround or frontend fork is used.

The test asserts the two known upstream failures as controls for this pinned release and verifies the supported configuration succeeds. Revisit those controls when upgrading Label Studio; a fixed upstream version can restore optional collapsing. Its JSON and screenshots are CI artifacts.

## Open gates

Real DROID acquisition/validation, native annotation approval/timing orchestration, actual LeRobot creation and loader round-trip, and PI-family MLX parity/training are not complete. The training target is now the user's M5 Pro 48 GB machine; prior CUDA planning is historical. Existing community inference ports have not been installed or reused without resolved permissions. MLX's documented Linux CPU support may enable genuine small-shape gradient/parity/checkpoint tests before separate Apple Metal verification.

## Primary references

[Native TimelineLabels](https://labelstud.io/tags/timelinelabels), [synchronized video and time series](https://labelstud.io/templates/timeseries_audio_video), [supported Style tag](https://labelstud.io/tags/style) and [CFR Video requirements](https://labelstud.io/tags/video) informed this adapter. The standalone frontend is [deprecated upstream](https://labelstud.io/guide/frontend); this project uses the maintained application.
