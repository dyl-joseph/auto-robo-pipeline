# Prototype semantic annotation rules

These rules define the initial coarse annotation contract. They are a starting point for an independently audited reference set, not evidence that the labels or model are accurate. Freeze their file hash in the evaluation plan before comparing predictions.

Describe observed activity with short subtask text and a half-open source-frame interval `[start, end)`. A boundary marks a visible change in activity. Record evidence frames inside each segment. Cover the episode without gaps or overlapping regions, including periods that cannot be interpreted. Retain whether the source clock is measured or inferred.

| Action | Meaning |
| --- | --- |
| `reach` | The robot approaches an apparent target without established transport. |
| `grasp` | An apparent attempt to acquire an object. State-only gripper closure is a candidate; say when contact is unverified. |
| `transport` | The robot visibly moves an object it appears to hold. |
| `release` | An apparent attempt to let go of an object. Say when release/contact is unverified. |
| `recover` | A visible corrective attempt after an apparent mistake; describe the evidence. |
| `motion` | Motion is observable, but a more specific action cannot be established. |
| `hold` | The available observations show little change; a reused stale sample alone is insufficient. |
| `unknown` | Evidence is insufficient to decide what the robot is doing. |
| `not_observable` | The relevant activity is hidden or outside the available view. |

Use an object description only when supported by the observations. Otherwise leave it absent. Do not infer successful completion from the episode instruction. Explain ambiguous contact, occlusion or sparse sampling in the subtask text; uncertainty values are not calibrated probabilities.

Independent reference annotators must not see the tested model's predictions or assisted corrections. Record the actual annotator, session/camera identity checks and adjudication evidence. Keep related sessions and near-duplicate recordings in the same split. Review corrections made after seeing predictions remain model-assisted output, even when a person approves them.
