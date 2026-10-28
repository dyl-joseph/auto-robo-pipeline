# Label Studio integration

Label Studio 1.23.2 and its official Python SDK 2.1.2 are locked in the default `labelstudio` dependency group. Native Video, TimelineLabels and TimeSeries own playback, temporal editing and sensor visualization. This replaces the bespoke editor as the primary application path.

The labeling XML is [review.xml](../integrations/labelstudio/review.xml). Its white background, restrained title typography and spacing are applied through the supported Style/View tags. The application chrome and authentication remain Label Studio's own. There is no frontend fork or deprecated standalone frontend dependency.

## Local use

`python -m robo serve` runs native Label Studio on 127.0.0.1:8000. Its data lives under the configured robotics store's `labelstudio/` directory. It does not automatically create users or API credentials. `prepare-labelstudio EPISODE_ID` creates a native task import JSON, labeling XML, explicit frame mapping and provenance sidecar.

The first-slice manual path uses the normal Label Studio project setup/import UI and a project-scoped Local Files storage rooted at `STORE/review_proxy`. The application document root is the selected store, while the project storage authorizes only the review proxy subdirectory. Do not authorize the full filesystem or expose the development server publicly.

For automation, `publish_bundle()` accepts an already authorized official SDK client. It creates a project, authorizes its proxy storage and imports the task/predictions, returning a trusted source/task/proxy binding. The bridge does not create an API credential for that client. `annotation_snapshot()` validates native SDK JSON exports against that binding and emits draft snapshots. Promotion of these snapshots to the existing accepted-export workflow remains a subsequent integration gate.

## Timing and frame semantics

Label Studio recommends constant-frame-rate video. The bridge produces a deterministic display proxy by holding the last measured source frame until its next original PTS. The proxy records every display index, source index, original integer PTS and original episode time. Raw video and robot streams remain unchanged.

Native timeline frame spans are one-based and inclusive. Decoding uses the explicit source map and the next measured source-frame timestamp for the half-open interval end. Repeated proxy frames can snap to the same source frame; a source timestamp gap is not interpolated evidence. Boundary precision remains limited by measured source frames. Unit fixtures cover irregular PTS and duplicated display frames. The real browser test edits the first native range to frames 1–6, submits, reloads and independently exports through the SDK. Its source deliberately omits frames 5/6. The result is [0, 0.583333) in measured source time, not [0, 0.5) inferred from proxy FPS. Keep this check in release verification as editor versions change.

The display sensor CSV starts at relative zero, preventing the native TimeSeries first-row offset rule from shifting delayed streams. It uses declared clock offsets and tolerance-bound nearest samples, with blanks for missing values. This display CSV is not a replacement for original training-facing state/actions.

Label Studio's native `lead_time` is retained as raw metadata and is not advertised as active correction time. Browser test durations are automated software checks, not human-study results. Imported completion does not automatically approve a training annotation.

## Reproducible real-application check

```bash
uv run --locked python -m scripts.labelstudio_smoke --browser
```

Install the locked browser/recording dependency with `npx playwright install chrome ffmpeg`, or use the system Chromium/FFmpeg setup in [development notes](development.md).

This starts the actual Django application with a disposable SQLite database, an unusable-password fixture user and a short-lived authenticated browser session. SDK calls use that session through its supported custom HTTP client. SDK 2.x requires a constructor key string; the test's non-secret sentinel is removed by HTTPX authentication before requests, and the server enforces its actual session/CSRF permissions.

Label Studio auto-creates a user token, which the test immediately removes. Its native JSON export implementation dereferences `user.auth_token` even without resource downloads, so the fixture creates a temporary token for that one export call and revokes it in `finally`. No persistent API credential is retained, no token is printed, and the entire database is removed at exit. Normal application authentication is not disabled.

Telemetry, Sentry and automatic update checks are disabled. The browser blocks all non-local requests. The supported feature-flag setting `fflag_fix_front_dev_3391_interactive_view_all=false` disables Compare All: in this pinned version, that mode leaves a per-annotation synchronization manager uninitialized during mounting. This is a bounded runtime workaround, not a patch to upstream code. Single-task annotation and sensor synchronization remain enabled.

## Open gates

Real DROID acquisition/validation, native annotation approval/timing orchestration, actual LeRobot creation and loader round-trip, and PI-family MLX parity/training are not complete. The training target is now the user's M5 Pro 48 GB machine; prior CUDA planning is historical. Existing community inference ports have not been installed or reused without resolved permissions. MLX's documented Linux CPU support may enable genuine small-shape gradient/parity/checkpoint tests before separate Apple Metal verification.

## Primary references

[Native TimelineLabels](https://labelstud.io/tags/timelinelabels), [synchronized video and time series](https://labelstud.io/templates/timeseries_audio_video), [supported Style tag](https://labelstud.io/tags/style) and [CFR Video requirements](https://labelstud.io/tags/video) informed this adapter. The standalone frontend is [deprecated upstream](https://labelstud.io/guide/frontend); this project uses the maintained application.
