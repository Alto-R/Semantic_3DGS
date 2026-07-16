# Script Layout

Run project commands from the repository root so the relative paths used by
the Slurm entry points resolve consistently.

- `task1/`: maintained Task 1 Python pipeline, validation, visualization, and
  inspection tools. These files stay together because several import shared
  local modules.
- `slurm/`: maintained Slurm entry points. Use
  `slurm/slurm_task1_dinov2_scene.sbatch` for the production DINOv2 route and
  its optional continuous GroundingDINO extension. Use
  `slurm/slurm_task1_semantic_scene.sbatch` for standalone GroundingDINO
  debugging; the bicycle entry point is a compatibility wrapper.
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
- `task1/build_grounding_guard_config.py`: builds the automatic GroundingDINO
  vocabulary from DINO classes present in the fused scene plus the selected
  custom extension classes. DINO-derived entries are competition-only guards.
- `task1/run_grounding_semantic_branch.py`: shared GroundingDINO/SAM,
  FlashSplat, and semantic-fusion branch used by both scheduler paths.
- `task1/merge_semantic_extensions.py`: deterministic Gaussian-label merge for
  reviewed classes outside the ADE20K ontology.
- `slurm/slurm_task1_dinov2_scene.sbatch`: production entry point. Set
  `ENABLE_GROUNDINGDINO=1` to run the extension branch continuously after the
  DINOv2 base; the default `0` publishes DINOv2 alone.
- `slurm/slurm_task1_recolor_output.sbatch`: visualization-only stable-color
  regeneration for an existing output; it does not rerun either semantic model.
- `setup/install_dinov2_segmentation.sh`: install the separate segmentation
  environment and download the official ViT-L/14 ADE20K linear artifacts.
- `setup/install_semantic_extensions.sh`: rebuild the CUDA extensions for
  native RTX 8000 (`sm_75`) execution and L40-compatible `sm_86` PTX by
  default; override with `TORCH_CUDA_ARCH_LIST` only when required.

The v1 defaults preserve all 150 ADE20K classes and use only real cameras. See
`docs/DINOV2_MULTIVIEW_VOTING.md` for the vote definition and fixed scope.

Superseded experiments and utilities are retained under
`archive/task1_legacy/scripts/` and `archive/task1_prompt_pilot/scripts/` for
provenance. They are not active pipeline entry points.

Scene vocabularies live in `configs/task1_semantic_classes.<scene>.json`. If a
scene-specific file is absent, the generic scheduler falls back to
`configs/task1_semantic_classes.example.json`.

Semantic colors are keyed by normalized class name, not label ID or run-local
label ordering. `COLOR_MODE=class` is the comparison default; `instance` is an
optional debug view. Every run writes a JSON and PNG color legend.
