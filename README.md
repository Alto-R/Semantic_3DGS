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
- Bicycle signed-evidence baseline completed on the cluster as Slurm job
  `92344` from commit `385f8ba`: 50 views, 411 GroundingDINO + SAM masks, 399
  FlashSplat proposals, 6,131,954 Gaussians validated, and 25 nonzero labels.
- Job `92344` removed the bicycle-colored road streaks and bench-colored
  vegetation patches visible in the prior spatial-only run while preserving the
  bicycle and bench across all 50 validation views.
- The accepted high-confidence result remains conservative: 60.25% of
  Gaussians use label `0` (`unlabeled`), and small same-class fragments still
  split the foreground bicycle and bench across multiple instance IDs.
- Automatic connected-component instance consolidation is implemented but must
  pass the cluster identity-test run before replacing the job `92344` baseline.
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
validated scenes. The current bicycle result still needs validated instance
consolidation and an explicit decision about acceptable unlabeled coverage.
