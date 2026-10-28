# Import a local DROID RLDS directory

Use this command when you already have a TFDS-generated DROID directory containing `dataset_info.json`, `features.json` and TFRecord shards. It reads local files through the official TFDS and TensorFlow APIs. It does not download data or handle raw HDF5 recordings. The user's own downloaded folder has not been supplied or tested.

```bash
uv sync --locked --project integrations/droid
uv run --locked --project integrations/droid python -m integrations.droid.intake \
  PATH_TO_TFDS_DIRECTORY data/local/droid-inbox \
  --release droid_100/1.0.0 --max-episodes 10

uv run --locked python -m robo --store data/local/droid-work \
  batch run data/local/droid-inbox/episodes
```

The release name is required provenance supplied by the caller. A local directory is not authenticated as an official download. For the hash-pinned pilot, use the [acquisition command](droid.md#acquire-and-decode) first. That directory contains only one of the 31 declared shards, so its intake needs an explicit partial-source option:

```bash
uv run --locked --project integrations/droid python -m integrations.droid.intake \
  data/local/droid-sample data/local/droid-pilot-inbox \
  --release droid_100/1.0.0 --max-episodes 1 --allow-partial
```

Continue with the [batch review and approval workflow](batch-workflow.md). Intake creates source manifests, videos, numeric sidecars and training schemas. It does not create semantic reference labels or approve annotations.

## Limits and outputs

The default limits are 10 episodes, 4,500 steps per episode, 128 MiB per shard and 512 MiB across selected shards. An episode limit can be raised to 200, matching the batch inbox limit. Other options are `--max-steps`, `--max-shard-bytes`, `--max-selected-bytes`, `--split` and `--camera`. Byte limits count the sizes of selected source shards, including cached shards checked during resume. Hash verification can read those bytes more than once.

Records are read in the split's shard order without shuffling or caching. Each serialized RLDS episode is materialized before its steps are decoded. Step decoding uses one worker with automatic prefetch disabled; only the numeric fields and camera hashes accumulate across the episode. Memory therefore depends on the largest encoded episode and its numeric sidecar. This is bounded intake, not a measured peak-memory guarantee.

The supported feature mapping remains DROID's three 180×320 RGB cameras, named seven-joint absolute positions and gripper closure. One chosen camera becomes the review video. Every camera's decoded frames are hashed; all non-image fields are retained. Time is inferred from step index at nominal 15 Hz. The [pilot guide](droid.md#fields-observed-in-the-downloaded-episode) explains the units, generic-action discrepancy and CC-BY-4.0 attribution.

`intake.json` records the metadata hashes, shard inventory, processed record ordinals, derivative hashes and counts. Its status distinguishes a completed split, `partial_source`, `episode_limit` and `failed`. A bad record stops the run with its shard, record index and error. Earlier published episodes remain available; missing or rejected input is never counted as converted.

## Resume

Run the same command again with the same output directory. Completed episodes are hash-checked and reused. Increase `--max-episodes` to continue further. An interruption after episode publication but before the checkpoint is recoverable without rewriting that episode. A lock prevents two writers from sharing an output.

Inputs and conversion settings must remain unchanged. Adding shards to a partial directory, changing metadata, or changing the selected camera requires a new output directory. Moving an otherwise identical source directory is supported. A changed derivative fails verification instead of being silently rebuilt. Only complete episode folders appear under `episodes/`; temporary work stays outside the batch inbox.

## Verification

```bash
uv run --locked --project integrations/droid python -m pytest -q integrations/droid/test_intake.py
uv run --locked --project integrations/droid python -m integrations.droid.verify_intake \
  data/local/droid-sample artifacts/droid-intake-check
```

Choose a fresh verification directory. Generated fixtures test multiple records and shards, resume, interruption, altered files, partial sources, limits and failures. They contain generated pixels and values, and are marked synthetic. The real-data regression compares the new intake with the pinned converter: all 119 steps, numeric fields, three-camera RGB hashes, selected-camera video bytes and training schema must match. Only local identity and provenance change. CI runs both checks; [recorded pilot evidence](evidence/droid-local-intake.json) contains metadata, not dataset bytes.

The runtime pins protobuf below version 7 because TFDS 4.9.9 still uses the removed `FieldDescriptor.label` API. See the upstream [protobuf migration guide](https://protobuf.dev/support/migration/) and [TFDS local-directory API](https://www.tensorflow.org/datasets/api_docs/python/tfds/builder_from_directory).
