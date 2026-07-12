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

- Local project folder: `C:\Users\dhana\PROJECTS\PKU 3DGS VR`
- GitHub remote: `https://github.com/DhanaKresnawijaya237/pku-3dgs-vr`
- Cluster account: `ssh -p 10022 cse12312032@172.18.34.25`
- Cluster root: `/lab/haoq_lab/cse12312032`
- The old cluster bare Git remote is deprecated; use GitHub as the source of
  truth.
- Reused semantic Conda env: `gaussian_grouping_true`
- GraphDeco official pretrained models downloaded and extracted on the cluster.
- Current Task 1 model coverage: 8/12 EyeNavGS scenes matched; `nyc`,
  `london`, `berlin`, and `alameda` still need separate 3DGS model sources.
- Bicycle identity baseline job `92369` established the first accepted result
  at commit `95ef1d6`: 50 views, 411 GroundingDINO + SAM masks, 399 FlashSplat
  proposals, and 6,131,954 validated Gaussians.
- Job `92344` removed the bicycle-colored road streaks and bench-colored
  vegetation patches visible in the prior spatial-only run while preserving the
  bicycle and bench across all 50 validation views.
- Automatic targeted-camera job `92426` is the current accepted bicycle result:
  it screened 144 safe unused cameras, evaluated a 100-camera candidate pool,
  automatically added 20 low-coverage/pose-diverse views, and validated a
  70-view run with 599 GroundingDINO + SAM masks and 583 FlashSplat proposals.
- Job `92426` retains one 119,570-Gaussian bicycle, one 139,727-Gaussian bench,
  and eight tree IDs totaling 1,096,092 Gaussians. The apparent fence label in
  job `92369` was reassigned primarily to the bench, matching the clean focused
  overlays and the detector's earlier `bench fence` ambiguity.
- The current result remains conservative in 3D: 58.40% of raw Gaussians use
  label `0`, down from 60.25%. On the same original 50 cameras, visible coverage
  is effectively unchanged at 92.42% while the worst frame improves from
  70.32% to 72.82%. The 20 difficult additions measure 82.45% pooled coverage.
- Identity-test job `92353` merged the bicycle and bench correctly, but its
  first consolidation rule also split 13 accepted tree groups into 26
  components and ultimately discarded 95,584 tree Gaussians. That result is
  retained as a diagnostic, not accepted as the new baseline.
- Consolidation treats accepted instances as atomic: connected geometry may
  merge same-class IDs but may not split them. Job `92426` passed this check and
  is exposed at `outputs/eyenavgs_task1/accepted/bicycle` on the cluster.
- Train reuse job `92469` is the accepted 50-view train baseline. Lowering the
  stuff-class minimum from 10,000 to 8,000 retained the strongly supported sky
  group (43 source views, 8,462 final Gaussians) without changing either train
  instance. Structural validation passed for all 1,026,508 Gaussians, and
  `outputs/eyenavgs_task1/accepted/train` points to
  `train_semantic_baseline_v2`.
- The train overlay-difference proxy measures 99.32% pooled visible coverage
  across 50 views, with a 94.74% minimum. This measures visible semantic tint,
  not semantic ground truth or gaze-hit accuracy; the train and train-versus-
  track contact sheets provide the accompanying visual QA.
- The accepted workflow is now scene-configurable through
  `scripts/slurm_task1_semantic_scene.sbatch`; the historical bicycle script is
  a compatibility wrapper. Targeted train job `92470` added 20 difficult views
  but is diagnostic only: it pruned sky at the default stuff cutoff and labeled
  a shipping container as `building`. Scene-configured per-class thresholds now
  retain sky at 8,000 Gaussians and require 8,000 for building; a reuse run of
  the 70-view masks/proposals is the next scheduler checkpoint.
- Task docs imported:
  - `TASK_BRIEF_EyeNavGS_Semantic_Annotation.md`
  - `INTERNSHIP_SCHEDULE.md`

## Workflow Summary

Windows is the control workspace for code, docs, configs, and small samples.
The Linux GPU cluster is the execution workspace for dataset download, 3DGS
rendering, FlashSplat/SAGA processing, and generated outputs.

Large files are intentionally excluded from Git. Keep EyeNavGS data, 3DGS model
folders, rendered frames, masks, checkpoints, and third-party repo clones outside
tracked files.

See:

- `docs/WORKFLOW.md` for local/cluster setup and command conventions.
- `docs/TASK1_SEMANTIC_ANNOTATION.md` for the semantic annotation pipeline.
- `configs/paths.example.yaml` for path conventions.

## First Milestone

The `bicycle` pilot has completed end to end:

1. Original `point_cloud.ply` inspected.
2. Fifty representative views rendered at 960-pixel width.
3. GroundingDINO + SAM masks generated and exported as 8-bit binary PNGs.
4. FlashSplat proposals fused using signed confidence-weighted multi-view
   evidence.
5. Semantic PLY and `label_map.json` exported in the D1 format.
6. Full and bicycle-versus-bench overlays visually checked.

Task 1 is not complete: the hard minimum remains four fully labeled and
validated scenes. The next work is a downstream gaze-hit acceptance criterion,
plus user-controlled train targeted-view expansion and additional scenes.
