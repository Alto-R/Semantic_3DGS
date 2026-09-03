# Legacy DINOv2 multiview fusion

This document describes the earlier DINOv2 ViT-L/14 ADE20K base. Its scheduler
and modules remain available for reproducing older results, but the maintained
production route now uses DINOv3 and dino.txt. Historical DINOv2 prototypes are
under `archive/`.

## Data flow

```text
real camera RGB
     |
     v
DINOv2 ADE20K logits
     |
     +--> per-pixel class confidence
     |
     v
FlashSplat visibility/lift
     |
     v
per-Gaussian, per-view semantic evidence
     |
     v
exact cross-view aggregation
     |
     v
agreement + evidence + support gates
     |
     v
class label or abstention
```

## Per-view prediction

`scripts/task1/dinov2/dinov2_segment_views.py` runs the ADE20K head and saves a class
prediction and confidence for each pixel. Pixels below
`MIN_PIXEL_CONFIDENCE=0.50` do not contribute a semantic vote.

`scripts/task1/dinov2/lift_dinov2_view_votes.py` uses FlashSplat camera geometry to map
valid image evidence to Gaussians. The pipeline preserves visibility separately
from semantic confidence so weak visible pixels are not misinterpreted as unseen
geometry.

## Exact fusion

`scripts/task1/dinov2/fuse_dinov2_multiview_votes.py` aggregates the saved per-view
evidence without approximating the vote history in memory. For each Gaussian it
tracks:

- the number of contributing semantic views;
- the winning ADE20K class;
- agreement with the winning class;
- accumulated semantic evidence;
- visibility without sufficient semantic confidence.

The documented DINOv2 mode is `separate_abstain`. A label is assigned only when
all relevant gates pass:

```text
MIN_VIEWS=2
MIN_AGREEMENT=0.50
MIN_SEMANTIC_EVIDENCE=0.50
MIN_ASSIGNED_THING_GAUSSIANS=5000
MIN_ASSIGNED_STUFF_GAUSSIANS=10000
adaptive class thresholds enabled
```

Otherwise the Gaussian retains an abstained or unseen state. The distinction is
important during QA and later refinement.

## Ontology

The DINOv2 base ontology is ADE20K.
`scripts/task1/dinov2/dinov2_ontology.py` provides normalized names and
thing/stuff metadata used by fusion and config validation.

GroundingDINO is disabled for this base configuration. The older workflow used
adaptive instance-guard v5 for ADE20K completion and a separate reviewed merge
for identities absent from ADE20K.

## Outputs

The base job writes intermediate artifacts under:

```text
stages/01_real_camera_views/
stages/02_flashsplat_votes/
stages/03_exact_fusion/
```

Published material is grouped under `deliverables/`, `visualizations/`,
`validation/`, and `logs/`. The primary semantic contract is
`gaussian_labels.npy` plus `label_map.json`. A semantic PLY, when intentionally
published, must contain one integer label per source Gaussian.

## Why all real cameras

Use `VIEW_COUNT=0` to select every camera in `cameras.json`. This maximizes
available evidence without inventing poses and reduces the chance that a class
decision depends on one selected viewpoint. Multiple views do not guarantee
completeness. An occluded object or consistently weak prediction can still
abstain.

## QA interpretation

Use original/overlay contact sheets to verify spatial meaning. Fused class counts
help identify collapse or disappearance, but do not establish correctness on
their own. Overlay coverage is likewise a visibility proxy, not an accuracy
metric.
