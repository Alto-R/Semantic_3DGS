# PKU 3DGS VR Semantic Annotation

This repository tracks the EyeNavGS semantic annotation and gaze-target labeling work.
The immediate priority is Task 1: add object-level semantic labels to 3D Gaussian
Splatting scenes from the public EyeNavGS dataset.

## Project Goal

Convert EyeNavGS from raw VR navigation and gaze traces into a supervised
gaze-target dataset:

- D1: semantic 3DGS scenes, with one integer `label` property per Gaussian and a
  `label_map.json` per scene.
- D2: per-frame gaze-target annotations derived from ray intersection against
  labeled scenes.
- D3: quality report with hit rate, manual spot checks, geometric error analysis,
  and semantic snapping ablation.

## Current Status

- Repository paths are project-relative; large data, model, dependency, and
  output roots remain outside Git. Git history is the source of truth for code
  and documentation.
- The pipeline reuses the `gaussian_grouping_true` Conda environment and the
  downloaded GraphDeco models. Models are available for 8/12 EyeNavGS scenes;
  `nyc`, `london`, `berlin`, and `alameda` still need separate 3DGS sources.
- The four-scene Task 1 pilot minimum is complete. Accepted outputs are exposed
  under `outputs/eyenavgs_task1/accepted/`.
- The scene-configurable entry point is
  `scripts/slurm/slurm_task1_semantic_scene.sbatch`; the bicycle-specific script
  remains a compatibility wrapper.

| Scene | Accepted target and result | Visible-overlay proxy |
| --- | --- | --- |
| Bicycle | `bicycle -> ../bicycle_semantic_targeted_v1`; 6,131,954 Gaussians across 70 views, 599 masks, and 583 lifted proposals. It retains one 119,570-Gaussian bicycle, one 139,727-Gaussian bench, and eight tree IDs totaling 1,096,092 Gaussians; 58.40% remain unlabeled. | 89.57% pooled; 92.42% on the original 50 views and 82.45% on the 20 difficult additions. The original-view minimum improved from 70.32% to 72.82%. |
| Train | `train -> ../train_semantic_targeted_v3`; 1,026,508 Gaussians across 70 views. It retains two train instances and a strongly supported sky group while pruning the shipping-container `building` false positive. | 99.03% pooled, 89.62% minimum; 99.56% on the original 50 views and 97.70% on the 20 additions. |
| Room | `room -> ../room_semantic_targeted_v2`; 1,593,376 Gaussians across 70 views. It retains a clean 7,779-Gaussian television and a separate 9,054-Gaussian table, removing the earlier table-as-television error. | 81.38% pooled; 84.22% on the original 50 views (62.24% minimum) and 74.28% on the additions (35.03% minimum). The lowest view is dominated by unlabeled wall and ceiling. |
| Truck | `truck -> ../truck_semantic_targeted_v1`; 2,541,226 Gaussians across 70 views. Four wheel IDs contain 8,887, 13,343, 9,253, and 4,389 Gaussians with support from 19, 22, 24, and 24 views; a weak 95-Gaussian fragment is pruned and 71.57% remain unlabeled. | 94.85% pooled; 95.88% on the original 50 views and 92.27% on the 20 additions. Acceptance is based on stronger multiview support and clean visual QA, not proxy improvement. |

The accepted pipeline incorporates the following scene-neutral corrections:

- Signed positive/negative multiview evidence suppresses leakage that survived
  spatial pruning alone.
- Merge-only consolidation may join connected same-class IDs but never split an
  accepted instance; this preserves valid disconnected parts and tree coverage.
- Support-adaptive pruning replaces per-scene thresholds. Strongly supported
  stuff groups use 75% of the normal cutoff, while thing cutoffs scale by
  `max(0.5, 1 - source_view_ratio)` without crossing the global minimum.
- Reuse mode inherits the exact camera indices and count from the Grounded-SAM
  manifest, so all targeted views are validated.
- Phrase resolution rejects unrelated multi-class matches, connected fragments
  consolidate before size pruning, and deterministic CIELAB-separated colors
  improve inspection without changing semantic ownership.

Overlay-difference coverage is a visible-tint proxy, not semantic ground truth
or gaze-hit accuracy. Acceptance also requires multiview support and focused
visual QA. Detailed methods and run evidence remain in
`docs/TASK1_SEMANTIC_ANNOTATION.md`; the source task documents are
`TASK_BRIEF_EyeNavGS_Semantic_Annotation.md` and `INTERNSHIP_SCHEDULE.md`.

## Superseded Task 1 Run Record

These outputs may be deleted from cluster storage. "Superseded" does not mean
failed: several were valid baselines replaced by more complete runs. Where the
project record lacks a specific diagnosis, none is inferred.

- Bicycle outputs: `bicycle`, `bicycle_auto`,
  `archive/bicycle_semantic_10views`,
  `archive/bicycle_semantic_50views_hard_priority`,
  `archive/bicycle_semantic_50views_multiview_q003`,
  `archive/bicycle_semantic_50views_pre_spatial_pruning`,
  `archive/bicycle_semantic_50views_spatial_only`, `bicycle_semantic`,
  `bicycle_semantic_identity_test`, and `bicycle_semantic_identity_test_v2`.
  The legacy/manual and class-agnostic paths were replaced by named semantic
  fusion. The first three archived experiments have no retained diagnosis.
  Later runs progressed from fragmented identities, through spatial-only
  leakage, to signed evidence and consolidation. The first identity test split
  13 tree groups into 26 components and discarded 95,584 tree Gaussians; the
  merge-only correction preserved them. The valid 50-view baseline was finally
  superseded by the accepted 70-view targeted run.
- Train outputs: `train_semantic_baseline_v1`, `train_semantic_baseline_v2`,
  `train_semantic_targeted_v1`, and `train_semantic_targeted_v2`. The initial
  fixed 10,000-Gaussian cutoff removed an 8,462-Gaussian sky group supported by
  43/50 views; an interim 8,000 cutoff was replaced by generic adaptive pruning.
  The first targeted run also retained a shipping container as `building`, and
  the second validated only 50 of 70 views. Consistent stuff handling and exact
  manifest-camera reuse produced the accepted run.
- Room outputs: `room_semantic_baseline_v1`, `room_semantic_baseline_v2`, and
  `room_semantic_targeted_v1`. The initial vocabulary omitted piano,
  television, speakers, media console, and curtains. The corrected 50-view
  baseline still lacked difficult views and had minor television spill and a
  curtain/chair display-color collision. Targeted expansion exposed an
  ambiguous-phrase error that labeled table geometry as television; stricter
  phrase resolution and pre-pruning fragment consolidation corrected it.
- Truck outputs: `truck_semantic_baseline_v1` and
  `truck_semantic_baseline_v2`. The fixed 5,000-Gaussian thing cutoff removed a
  real 3,738-Gaussian wheel supported by 15 views; generic support-adaptive
  pruning retained it while rejecting weak fragments. The valid corrected
  50-view baseline was superseded by the accepted 70-view targeted run.

Keep the final accepted targets and compact reports:

- `bicycle_semantic_targeted_v1`
- `train_semantic_targeted_v3`
- `room_semantic_targeted_v2`
- `truck_semantic_targeted_v1`
- `accepted/`, whose scene links point to those results
- `logs/`, `manifests/`, and small validation/report JSON files needed for the
  final report

Generated `semantic_point_cloud_rgb_debug.ply` files and their inspection JSON
are also safe to delete from every old and accepted run. They are derived,
redundant visualization artifacts rather than required deliverables. The
pipeline no longer creates them. For accepted scenes, keep
`deliverables/semantic_point_cloud.ply`, `label_map.json`, run summaries,
validation reports, and `semantic_point_cloud_supersplat_debug.ply`.

## Workflow Summary

Windows is the control workspace for code, docs, configs, and small samples.
The Linux GPU cluster is the execution workspace for dataset download, 3DGS
rendering, FlashSplat/SAGA processing, and generated outputs.

Large files are intentionally excluded from Git. Keep EyeNavGS data, 3DGS model
folders, rendered frames, masks, checkpoints, and third-party repo clones outside
tracked files.

Future semantic runs no longer generate the redundant
`semantic_point_cloud_rgb_debug.ply` export or its JSON inspection. SuperSplat
debug PLYs remain the supported colorized 3D visualization, while
`deliverables/semantic_point_cloud.ply` remains the required labeled output.

See:

- `scripts/README.md` for the maintained script layout.
- `docs/TASK1_SEMANTIC_ANNOTATION.md` for the semantic annotation pipeline.
- `configs/paths.example.yaml` for path conventions.

## First Milestone

The four-scene Task 1 pilot is complete for `bicycle`, `train`, `room`, and
`truck`. Each accepted result uses 50 base views plus 20 automatically selected
difficult views, passes structural validation, and includes semantic PLY,
label-map, overlay, and visual-QA artifacts under
`outputs/eyenavgs_task1/accepted/`.

Next: extend the validated pipeline to the remaining scenes and define a
downstream gaze-hit acceptance criterion.
