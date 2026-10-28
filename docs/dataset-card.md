# Data and acquisition record

## Shipped demonstration

Six generated 2D illustrated tabletop episodes, 8 seconds each, 12 fps, 640×360, H.264. Five include generated gripper and command streams on the same fixture clock; one is deliberately video only. One clip contains a blurred interval. The generator and original PTS are reproducible. These are software fixtures without physical dynamics, real sensor measurements, or human reference labels. Generated media is offered under CC0-1.0; external models retain their own terms.

No raw dataset, model weight, credential or third-party robot video is committed. Committed screenshots and walkthrough recordings show only our synthetic collection and automation-generated decisions. They are evidence of application behavior, not human annotation results. Real-data browser evidence remains in ignored local artifacts.

## DROID pilot: acquired in the published environment

The prior executor's proxy tunnel 403 was resolved by the user's newly published environment. The official GCS listing and generation-pinned download now succeed. The [DROID guide](droid.md) records the exact source URI, bytes, checksums, CC BY 4.0 attribution, actual decoded fields and reproduction commands.

One complete 9,541,347-byte shard contains one 119-step real episode. All three camera streams were decoded; one selected view was converted for native review. All non-image RLDS values and per-camera RGB hashes remain in a local sidecar, and the original shard remains local. The source manifest and accepted export retain license, attribution and transformation provenance. No dataset bytes or real-video screenshots are committed.

The RLDS schema has no capture timestamp channel. Nominal 15 Hz timing is explicitly inferred from step indices, not presented as measured synchronization. The generic action description disagrees with the sampled values, so the adapter uses explicitly named absolute joint-position commands plus normalized gripper closure. Review and LeRobot preserve that mapping and its clock qualification.

The automated native browser correction, test approval and actual LeRobot loading passed for all 119 frames, with no omitted/repeated source frames. Outcome remains unknown; quality is incomplete. The approval is integration-test evidence, not a human-reviewed training release. [Recorded result](evidence/droid-roundtrip.json).

Start a quality study only after selecting independent sessions and collecting adjudicated reference labels. One small episode establishes interoperability, not dataset coverage, annotation efficiency or policy performance. Multi-camera training and raw HDF5/MP4 clock validation remain separate work.
