# Label Studio integration

Label Studio 1.23.2 and Python SDK 2.1.2 are locked in the default environment. Native Video, TimelineLabels and TimeSeries provide playback, temporal editing and sensor plots. The [review configuration](../integrations/labelstudio/review.xml) applies the project's [theme](design.md) through supported Style and View tags. Authentication, navigation and annotation history remain part of the upstream application.

## Open a review project

Use the [batch workflow](batch-workflow.md) to prepare and publish a queue. `python -m robo serve` starts Label Studio at `127.0.0.1:8000`, with its database under the selected store's `labelstudio/` directory. The command creates no account or API credential.

For a single episode, `prepare-labelstudio EPISODE_ID` writes a task JSON, labeling XML, frame map and original bundle. Import the JSON through the normal project UI and add Local Files storage rooted at `STORE/review_proxy`. Open tasks from the Data Manager table. The project needs access only to its proxy directory.

Python callers can pass an authenticated official SDK client to `publish_bundle()`. The function creates the project, storage and task, then returns a binding that records the source, task and proxy. `NativeReviewBridge` uses that binding to read a saved annotation and check it before approval or export. See [native approval](native-approval.md).

## Review clocks and source clocks

Label Studio recommends constant-frame-rate video. The bridge creates a display proxy by holding each source frame until the next original timestamp. The default rate is the ceiling of the reciprocal of the shortest frame interval, up to 60 fps. Preparation fails if that rate cannot include every source frame. An explicit `--fps` can use a coarser grid, with omissions reported in the manifest and task banner.

The proxy is an H.264 re-encode. Every display frame records its original source index, integer PTS and episode time. Raw video and robot streams stay unchanged.

Native timeline spans are one-based and inclusive. A span `[a,b]` becomes `[(a-1)/fps, min(b/fps, episode_duration))` on the generated review clock. Adjacent ranges share a boundary even when they divide copies of the same held source frame. Expanding both ranges to that frame's full duration would create an overlap.

Machine boundaries snap to the nearest grid edge, at most half a grid period away. Candidates that collapse to zero width are omitted and counted. Human editing has one-grid-period resolution. Neither bound measures the accuracy of a real event boundary. Exported `boundary_evidence` records the surrounding source timestamps and marks whether timing was measured or inferred.

For example, the browser fixture removes observations at 5/12 and 6/12 seconds. Editing native frames 1 through 6 produces `[0, 0.5)` on the review clock. The end lies between source observations at 4/12 and 7/12 seconds, leaving a 0.25-second bracket. DROID uses an inferred step clock, so its corresponding evidence does not claim measured capture times.

Sensor displays start at zero and apply each stream's declared offset. Nearest samples must satisfy the declared tolerance; gaps remain null. Numeric seconds formatting avoids a calendar axis. Training reads the original streams, not the display samples.

## Verify the application

Install the browser dependencies described in [development setup](development.md), then run:

```bash
uv run --locked python -m scripts.labelstudio_smoke --sync-contract
uv run --locked python -m scripts.labelstudio_smoke --browser
```

The test starts the actual Django application with a disposable database and authenticated session. SDK requests use that session through a custom HTTP client. SDK 2.x requires a constructor key string, but HTTPX removes the test sentinel before sending requests. Native JSON export also dereferences a user token, so the test creates one for that call and revokes it in `finally`. The database is removed at exit.

The browser blocks non-local requests. Telemetry and update checks are disabled. The pinned release's Compare All mode leaves a synchronization manager uninitialized, so the supported feature flag `fflag_fix_front_dev_3391_interactive_view_all=false` disables that mode for the test. Single-task annotation remains enabled.

The test clicks a task, copies its prediction through **Create an annotation**, edits a boundary and submits. It checks stored API data, reloads the saved annotation and exports through the SDK. Imported predictions carry `interactive_mode=false`, as required by this frontend version. The SDK retrieves prediction IDs from the native export and verifies their original contents.

After a reload, the test selects the saved annotation rather than relying on the default tab. It waits for the video to attach before polling media readiness. Submitted TextArea notes appear as saved results, while the input becomes blank. Failure evidence includes a screenshot, accessible UI snapshot and request statuses without session credentials.

Response diagnostics record the API path, method and status synchronously. Chromium can discard a response body during navigation, so reading JSON inside an async response listener can reject after a successful save. The checks for predictions and saved corrections use awaited API requests. `npm run test:e2e -- tests/browser/native-trace.spec.js` reproduces the discarded-body error after navigation, checks that diagnostics survive it, and verifies that application errors and malformed JSON remain visible to assertions.

## Known task-loading defect

Cold navigation to `data?task=ID` can lose predictions in Label Studio 1.23.2. The task-detail response contains predictions, but a later task-list response can replace it with a row that omits them. If this happens before the editor mounts, the editor shows an empty annotation.

Open the Data Manager first and click a task row. This route passes the complete correction, submit, reload and export test, including at 6× browser CPU throttling. Ordinary reload after saving remains covered. The bridge does not patch upstream stores or inject predictions.

```bash
LS_CPU_RATE=6 uv run --locked python -m scripts.labelstudio_smoke --browser
LS_TASK_LOADING_CONTRACT=1 LS_CPU_RATE=6 uv run --locked python -m scripts.labelstudio_smoke --browser
```

CI uses the second command. It reorders unmodified server responses to reproduce the defect, then recovers through the project table and requires the full review sequence to pass. Revisit the negative control when upgrading Label Studio. CPU throttling tests loading order; it is not a hardware performance measurement.

## Known collapsed-plot defect

`--sync-contract` tests visible CSV and numeric JSON plots, two Collapse configurations and the themed editor. Visible plots seek correctly. Collapsed plots can retain `canvasWidth=0` after opening, which causes `isNotReady` to suppress seeking. The component measures width on mount and window resize but does not reliably refresh it when Collapse visibility changes.

The review configuration keeps sensor plots visible and tests seeking after a real viewport resize. Both failing Collapse cases remain negative controls for the pinned version. No native setter or injected seek event replaces the browser action.

## References

The adapter follows the official [TimelineLabels](https://labelstud.io/tags/timelinelabels), [synchronized video and time series](https://labelstud.io/templates/timeseries_audio_video), [Style](https://labelstud.io/tags/style) and [Video](https://labelstud.io/tags/video) contracts. The standalone frontend is [deprecated](https://labelstud.io/guide/frontend), so this project uses the maintained application. Current verification results are in [results](results.md).
