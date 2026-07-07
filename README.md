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
- Cluster account: `ssh -p 10022 cse12312032@172.18.34.25`
- Cluster root: `/lab/haoq_lab/cse12312032`
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

Run one pilot scene end to end, preferably `bicycle` if the required 3DGS model
and camera/render metadata are available:

1. Inspect the original `point_cloud.ply`.
2. Render representative views.
3. Generate masks.
4. Run FlashSplat baseline.
5. Convert labels into project format.
6. Validate with semantic overlay renders.

