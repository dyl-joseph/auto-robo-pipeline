# Project scope and status

The goal is to turn robot recordings into structured datasets with automated preparation and a small, explicit review queue. The current implementation handles local video and declared robot streams. It repairs order and exact duplicates, detects several kinds of missing or ambiguous data, proposes motion segments and exports approved annotations through LeRobot.

The implemented slices and their evidence are listed below. The next approved milestone is DROID → automatic text-labeled temporal segments → independent ground-truth comparison → review/export. Work starts with bounded local RLDS intake and semantic reference/evaluation contracts, followed by a robot-state baseline and a small video-language adapter. Broader messy-data recovery comes after that comparison.

| Work | Evidence |
| --- | --- |
| Import and source integrity | Immutable snapshots, hashes, timestamp validation and retry tests |
| Native review and approval | Label Studio correction, persisted reload, explicit approval and fresh export checks |
| Real DROID import | One pinned shard, 119 decoded steps, three cameras inspected and one selected for review |
| Local RLDS intake | Generated multi-episode/shard tests, hash-checked resume, partial-source reports and the unchanged real pilot |
| Semantic evaluation contract | Source-frame segments, separate predictions/references, split and independence checks, and generated evaluator tests; no real accuracy result yet |
| Robot-state baseline | Coarse text-labeled candidates, ten generated tests and execution on all 119 real pilot frames; semantic accuracy remains unmeasured |
| Local Qwen adapter | Official tiny-model multiframe execution, pinned-asset checks and strict output contracts; full pretrained inference and hardware fit remain untested |
| LeRobot conversion | Every loaded DROID state/action vector compared with the source; official loader and DataLoader tested |
| MLX compatibility | PI05 action-expert loss, gradients, updates and resume compared with small official models on MPS and MLX Metal |
| Full PI05 inference | Pinned pretrained checkpoint, generated raw observations, official preprocessing and postprocessing, and complete action chunks verified on the M5 Pro |
| Batch workflow | Interrupted-import recovery, native priority queue, isolated failures and accepted export |
| Controlled corruption benchmark | Two seeds, 16 cases each, four complete repairs, three data-review cases and seven rejections |

The [demo](demo.md) reproduces the workflow. The [results](results.md) link the recorded evidence.

Full pretrained synthetic inference now passes on the M5 Pro. The [recorded run](evidence/pi05-synthetic-metal.json) uses the public official tokenizer and full checkpoint. Pretrained training, peak memory use, sustained speed and policy quality remain unmeasured.

The original proposal also asked whether assistance saves human time while preserving label quality. No human study was conducted. The [study protocol](evaluation.md) remains available for that separate experiment, including independent reference labels and matched manual/assisted conditions. Automated fixture approvals and queue counts cannot substitute for those measurements.
