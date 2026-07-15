# Script Layout

Run project commands from the repository root so the relative paths used by
the Slurm entry points resolve consistently.

- `task1/`: maintained Task 1 Python pipeline, validation, visualization, and
  inspection tools. These files stay together because several import shared
  local modules.
- `slurm/`: maintained Slurm entry points. Use
  `slurm/slurm_task1_semantic_scene.sbatch` for scene-configurable runs; the
  bicycle entry point is a compatibility wrapper.
- `cluster/`: cluster bootstrap, health-check, dependency-clone, and dependency
  inventory helpers.
- `setup/`: renderer and semantic-extension installation helpers.

The alternative prompt-free DINOv2 route is implemented by:

- `task1/render_task1_views.py`: render all real `cameras.json` views (or an
  explicit smoke-test subset).
- `task1/dinov2_segment_views.py`: official ViT-L/14 ADE20K linear-head
  inference with raw class/confidence maps and QA overlays.
- `task1/lift_dinov2_view_votes.py`: one compact indexed FlashSplat lift per
  view, including low-confidence abstention.
- `task1/fuse_dinov2_multiview_votes.py`: exact disk-backed vote fusion,
  thresholding, thing-instance components, pruning, and D1 export.
- `slurm/slurm_task1_dinov2_scene.sbatch`: separate DINOv2 and FlashSplat Conda
  environments plus the maintained validation/visualization stages.
- `setup/install_dinov2_segmentation.sh`: install the separate segmentation
  environment and download the official ViT-L/14 ADE20K linear artifacts.

The v1 defaults preserve all 150 ADE20K classes and use only real cameras. See
`docs/DINOV2_MULTIVIEW_VOTING.md` for the vote definition and fixed scope.

Superseded experiments and utilities are retained under
`archive/task1_legacy/scripts/` and `archive/task1_prompt_pilot/scripts/` for
provenance. They are not active pipeline entry points.

Scene vocabularies live in `configs/task1_semantic_classes.<scene>.json`. If a
scene-specific file is absent, the generic scheduler falls back to
`configs/task1_semantic_classes.example.json`.
