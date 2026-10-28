# Real DROID pilot

The published environment can download the official DROID debug subset. The pilot pins **one complete shard containing one 119-step episode**, plus its original feature/schema metadata. Dataset bytes and real-video screenshots stay in ignored local directories.

## Acquire and decode

From the repository root, after `./scripts/setup.sh`:

```bash
uv run --locked python -m scripts.fetch_droid
uv sync --locked --project integrations/droid
TF_CPP_MIN_LOG_LEVEL=2 uv run --locked --project integrations/droid \
  python -m integrations.droid.convert data/local/droid-sample data/local/droid-converted

uv run --locked python -m robo import data/local/droid-converted/source.json
# Use the printed episode ID with the normal suggest / prepare-labelstudio commands.
```

Use an unused conversion output directory. Acquisition reuses only files matching the pinned hashes; a conflicting file is an error. The download checks a hard byte budget, SHA256 and MD5 before publishing each file. Conversion verifies its local inputs before/after decoding. It neither invokes TFDS download preparation nor discovers a remote dataset builder.

| Official object under `gs://gresearch/robotics/droid_100/1.0.0/` | Bytes | SHA256 |
| --- | ---: | --- |
| `r2d2_faceblur-train.tfrecord-00009-of-00031` | 9,541,347 | `645d500e23e7359c21121b12fe1a1395157ba421b1b326eee7df829eb608a03d` |
| `features.json` | 18,665 | `4c556e2e38e179e41089e07cf54273b1540d67d3c78a420898370329813c9a25` |
| `dataset_info.json` | 760 | `58646aa75fd054a0d8670edd676620267adb32550dc228545d70a3418abb1834` |

The listing contained 31 shards totaling 2,192,595,669 bytes; shard 9 was the smallest. Its immutable GCS generation is `1710528094394407`. Total pinned payload is 9,560,772 bytes. This is a bounded connectivity/interoperability pilot, not a representative quality sample.

## License and attribution

The [official documentation](https://droid-dataset.github.io/droid/the-droid-dataset.html) identifies the bucket. The [authors' paper](https://arxiv.org/html/2403.12945v2), sections I and III, releases the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution: **Khazatsky, Pertsch, Nair et al., DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset (2024)**, [DROID project](https://droid-dataset.github.io/). The source manifest retains this attribution, the license, source URI, generation, hash and transformations. The repository's software dependencies have separate licenses.

## Fields observed in the downloaded episode

The original builder name is `r2d2_faceblur`, even though its bucket directory is `droid_100`. Official TensorFlow Datasets 4.9.9 and TensorFlow CPU 2.20.0 decode the original `features.json` directly.

| Field | Actual type / per-step shape |
| --- | --- |
| `episode_metadata.file_path`, `recording_folderpath` | Strings; opaque provenance paths, never opened |
| `steps.observation.{exterior_image_1_left,exterior_image_2_left,wrist_image_left}` | uint8 RGB, 180×320×3; 119 images per camera |
| `steps.observation.joint_position` | float64, 7 |
| `steps.observation.cartesian_position` | float64, 6 |
| `steps.observation.gripper_position` | float64, 1 |
| `steps.action_dict.{joint_position,joint_velocity}` | float64, 7 each |
| `steps.action_dict.{cartesian_position,cartesian_velocity}` | float64, 6 each |
| `steps.action_dict.{gripper_position,gripper_velocity}` | float64, 1 each |
| `steps.action` | float64, 7; ambiguous description, excluded from the training mapping |
| `steps.language_instruction{,_2,_3}` | Three strings, constant across this episode |
| `steps.{is_first,is_last,is_terminal}` | bool scalars; first at 0, last/terminal at 118 |
| `steps.{reward,discount}` | float32 scalars; final reward 1, other rewards 0; discount 1 |

The task is putting a cable in a storage box. The unmodified shard retains all imagery. The converter decodes and hashes all three camera streams, writes one explicitly selected camera as H.264 CRF18/yuv420p, and preserves every non-image field in `rlds-episode.json`. That numeric sidecar's hash is included in the source provenance. There is no crop or resize, but H.264 encoding is lossy. This pipeline currently reviews/trains from one camera per converted episode.

**Action discrepancy:** both the official web schema and downloaded feature metadata describe generic `action` as six joint velocities plus gripper position. Across all 119 downloaded steps, its first six values instead exactly equal `action_dict.cartesian_position`, and its last value equals `action_dict.gripper_position`. The maximum difference from named Cartesian velocity is approximately 3.6143. This observation applies to this shard only. The adapter explicitly selects seven commanded absolute joint positions plus normalized gripper closure, and never treats generic `action` as velocity.

Position fields use radians and normalized gripper closure (0 open, 1 closed), following the [publisher's robot implementation](https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/franka/robot.py). The schema lists every component, units and meaning. It does not assert that this action representation is appropriate for an arbitrary policy checkpoint.

## Timing and acceptance

The RLDS schema contains **no measured capture timestamps**. The paper states a nominal recording/control rate of 15 Hz. The adapter assigns step `i` the inferred time `i / 15` and aligns fields by their shared RLDS step. Generated MP4 PTS are exact for that nominal grid; they are not evidence of physical camera/robot synchronization. The source manifest, review banner, native boundary evidence and LeRobot alignment provenance retain this distinction. Exact coincidence with a derived frame is recorded separately from a measured source boundary.

An integer-PTS duration fix prevents rounded FFprobe decimal fields from inventing an extra tail frame for this 119-frame clip. LeRobot Parquet retains `source.timestamp` as float64; its current torch loader casts the value to float32. Integer frame indices and the alignment JSON retain the original mapping. Verification checks the stored doubles and the loader's actual float32 behavior separately.

To exercise the native application and official loader locally:

```bash
npm ci --cache .cache/npm
PLAYWRIGHT_BROWSERS_PATH=.cache/playwright npx playwright install ffmpeg
ROBO_NATIVE_EXPORT_FIXTURE=artifacts/droid-native-fixture \
  uv run --locked python -m scripts.labelstudio_smoke --browser \
  --source data/local/droid-converted/source.json

uv sync --locked --project integrations/lerobot
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
uv run --locked --project integrations/lerobot python -m integrations.lerobot.verify_droid \
  data/local/droid-converted artifacts/droid-native-fixture artifacts/droid-lerobot
```

The browser harness uses a disposable local database/session. It edits the first temporal boundary from 0.6 to 0.4 seconds, submits, reloads, and checks the persisted correction. A named test actor approves the immutable export. Outcome stays `unknown`, quality is `incomplete`, and the native note identifies automated integration testing. These are **automated test decisions, not human-approved reference labels**. Normal uncertainty/freshness gates remain enforced.

The verification compares all 119 loaded state/action vectors to the original RLDS numeric sidecar, decodes every selected-camera image, checks frame/time/label correspondence, and forms a DataLoader batch and action window. It also checks idempotence and episode-end action padding. Evidence includes zero omitted or repeated source frames. Screenshots and recordings at `artifacts/droid-native*.png` and `artifacts/droid-walkthrough.webm` show the actual application and stay untracked.

No human annotation efficiency, real-label quality improvement, optimizer step, policy evaluation, PI checkpoint compatibility or Apple/MLX result is established by this gate.
