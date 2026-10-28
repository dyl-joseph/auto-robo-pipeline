# Native review design

The visual reference is the [Physical Intelligence homepage](https://www.pi.website/), inspected on 2026-10-28. The review uses its cream paper color, compact monospace typography and black rules while retaining Label Studio's name, wordmark and application hierarchy. It does not imply affiliation.

| Reference token | Review configuration |
| --- | --- |
| Paper | `#f5f4ef` |
| Body/navigation | System monospace, 14px / 20px, weight 400 |
| Heading | 30px / 36px serif, weight 400 |
| Primary text | `#000` |
| Secondary text | `#686868`, 12px / 16px |
| Dividers | `#d4d3cb`; stronger muted border `#c0bdad` |
| Featured panel | White, square corners, 1px black border and 3px black offset shadow |

PI uses Signifier for its brand heading. This project uses the system's Georgia/Times serif fallback. No proprietary font files or PI logos are copied. Annotation needs a wider working area than PI's roughly 576px homepage column, so video and timeline use the available native editor width.

The supported [Style tag](https://labelstud.io/tags/style) applies CSS and native design variables when the project review is mounted. View wrappers supply the media border and compact rows. Native inline Choices replace the long vertical form; robot signals use two compact, visible plots. A native Collapse initialization defect in the pinned release prevents reliable seeking after opening a hidden chart, so those plots stay visible; [the minimal reproduction](labelstudio.md#native-synchronization-regression) records the reason. Required outcomes and quality fields remain outside that panel. The upstream shell is retained, including navigation, annotation history and persistent submit/update controls. Login and unrelated project settings retain upstream styling.

[Desktop capture](evidence/labelstudio-native-desktop.png) shows the actual application with synthetic fixture media. Styling is not evidence that the remaining synchronization, approval or real-data gates have passed. See [verification](results.md) for their status.
