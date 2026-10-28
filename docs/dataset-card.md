# Data and acquisition record

## Shipped demonstration

Six generated 2D illustrated tabletop episodes, 8 seconds each, 12 fps, 640×360, H.264. Five include generated gripper and command streams on the same fixture clock; one is deliberately video only. One clip contains a blurred interval. The generator and original PTS are reproducible. These are software fixtures without physical dynamics, real sensor measurements, or human reference labels. Generated media is offered under CC0-1.0; external models retain their own terms.

No raw dataset, model weight, credential or third-party robot video is committed. Screenshots and walkthrough recordings show only our synthetic collection and automation-generated decisions. They are evidence of application behavior, not human annotation results.

## DROID pilot: pending acquisition

The [official data documentation](https://droid-dataset.github.io/droid/the-droid-dataset.html) provides the 100-episode debugging subset at `gs://gresearch/robotics/droid_100`, approximately 2 GB. The [authors' paper](https://arxiv.org/html/2403.12945v2), introduction, explicitly releases the dataset under CC-BY 4.0. Attribute **Khazatsky, Pertsch, Nair et al., DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset (2024)** and retain source/release metadata.

Attempted read-only acquisition discovery through `storage.googleapis.com` returned `Tunnel connection failed: 403 Forbidden` from this runtime's network proxy. No objects or episodes were downloaded. The official `droid_100` listing was also attempted through supported execution permissions and returned the same proxy tunnel 403. This was a network response, not an automatic approval-review rejection. Raw-source video and HDF5 synchronization semantics were not verified with actual bytes.

On a machine with authorized access and gsutil installed, first inspect the documented subset:

```bash
gsutil ls -l gs://gresearch/robotics/droid_100/**
# Review size; keep at least 5 GB free. Then acquire only the small subset:
gsutil -m cp -r gs://gresearch/robotics/droid_100 data/local/droid-download
```

These commands are acquisition guidance, not a claim that the RLDS adapter exists. The subset needs an RLDS reader in a separate optional environment before conversion to the project's source schema. Prefer one raw episode with `trajectory.h5`, its camera MP4, and verified source clocks for testing true multimodal alignment. Record the selected object URLs, generations/releases, byte sizes, SHA256, license URL, transformations and camera/trajectory timing evidence in its source manifest.

The displayed RLDS schema does not provide an explicit timestamp channel. Do not relabel a nominal step frequency as measured synchronization. If only extracted frames or a re-encoded clip are available, mark that time basis and leave robot streams absent until alignment evidence is available. Multi-camera ingest and the RLDS/HDF5 adapter remain future work.

Start with one episode, then 10–20 episodes with observable boundaries and independent sessions. Expand to 100 only after storage, inference and review costs are measured. Do not use test-set labels to choose thresholds. Recovery and failed behavior must remain visible in coverage reports.


## Required unblock for real-DROID acceptance

Enable authorized HTTPS access to `storage.googleapis.com` in the **active executor's network policy**, then rerun the official subset listing. A saved configuration draft alone does not activate access. This executor has no tool to change that policy. The smallest supported handoff is an environment with that host allowed, or a locally supplied DROID camera MP4 plus its source identity/license metadata. Do not change proxies, use an alternate host or executor to route around this restriction. Real-DROID verification remains an explicit open acceptance gate.
