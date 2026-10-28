# Dataset card

## Real DROID sample

The pilot uses one official 9,541,347-byte shard containing a 119-step episode of putting a cable into a storage box. All three camera streams were decoded. The converter selects one 180×320 view for review and preserves every non-image RLDS field in a local JSON sidecar.

The [DROID guide](droid.md) records the source URI, pinned generation, checksums, decoded fields and reproduction commands. DROID is by Khazatsky et al. (2024), released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). The source manifest carries that attribution and the conversion history.

The RLDS release has no capture-timestamp field. The adapter uses step index divided by the nominal 15 Hz rate and labels the resulting clock as inferred. Its action schema selects seven absolute joint positions in radians and normalized gripper closure. The generic `action` field is excluded because its description disagrees with the values in this episode.

The automated native review test corrected a boundary, approved a test revision and loaded all 119 rows through LeRobot. Outcome remains unknown and quality is incomplete. These are integration-test annotations, not independently reviewed reference labels. [Recorded result](evidence/droid-roundtrip.json).

## Controlled variants

The [corruption benchmark](messy-droid.md) creates 16 variants from the same immutable episode. The variants retain real pixel and numeric data but use synthetic perturbations: shuffled rows, duplicates, declared offsets, missing observations, metadata errors and damaged media. They test repair and detection behavior on known cases. They do not represent the frequency or variety of naturally messy recordings.

## Generated demonstration

The synthetic demo contains six illustrated tabletop episodes, each eight seconds long at 12 fps and 640×360. Five have generated gripper and command streams; one is video only. One clip includes blur. The generator supplies no physical dynamics or real sensor measurements. Generated media is offered under CC0-1.0.

## Storage and permitted use

Raw data, derived videos, weights and review sessions stay in ignored local directories. The repository includes metadata reports, synthetic UI recordings and one attributed [DROID review screenshot](evidence/droid-batch-review.png). The screenshot shows an automated test with missing numeric samples.

One episode demonstrates that the formats and tools work together. It cannot establish dataset coverage, annotation quality, human time savings or policy performance. A quality study needs independent sessions and adjudicated reference labels; the [study protocol](evaluation.md) describes those requirements.
