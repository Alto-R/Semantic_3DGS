# DINOv3 ADE20K Pipeline

Status: maintained. The DINOv3 abstention-recovery materialization is the
preferred review artifact, and the general end-to-end pipeline is the
maintained route for scenes with new DINOv3 evidence. Results and the
acceptance rule are in `docs/TASK1_DINOV3_RECOVERY_REPORT.md`; the full
research history, including rejected experiments, is archived under
`archive/dinov3_recovery_experiments/`.

## Scope

This route replaces only the 2D ADE20K segmenter of the maintained semantic
pipeline. It reuses the existing real-camera rendering, identity-preserving
ADE20K ontology, FlashSplat vote lifting, and immutable-output policy.
DINOv3 is a closed-set ADE20K segmenter in this route; it does not add
arbitrary custom identities.

## Required upstream files

Use the official DINOv3 PyTorch-Hub checkpoint format. The adapter requires
these exact files:

```text
data/models/dinov3/
├── dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
└── dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth
```

- Backbone: `dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth`
- ADE20K Mask2Former head:
  `dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth`

The pinned DINOv3 repository commit is
`6876159a11b4df116f30f667f8c9888617df0751`; the segmenter verifies the
repository commit, clean state, and checkpoint SHA-256 digests before
running. Install the environment with `scripts/setup/install_dinov3_semantic.sh`.

## Maintained stages

### 1. Per-scene DINOv3 evidence

`scripts/slurm/slurm_task1_dinov3_scene.a100.sbatch` renders real camera
views, runs DINOv3 Mask2Former, lifts dense per-view FlashSplat votes, and
optionally fuses. `REPORT_ONLY=1` is the safe default.

### 2. Audited hard-vote consensus

The selected-view cache, visibility audit, camera ranking, and round-trip
fidelity audit produce the immutable hard-vote consensus:

- `slurm_task1_dinov3_all_camera_visibility_scene.sbatch`
- `slurm_task1_dinov3_visibility_camera_selection_scene.sbatch`
- `slurm_task1_dinov3_selected_view_cache_scene.sbatch`
- `slurm_task1_dinov3_round_trip_fidelity_audit_scene.sbatch`

The hard vote is one equal identity vote per camera per Gaussian with a
strict majority (>= 2 cameras, unique winner, > 50%); everything else
abstains. Materialization is gated:
`slurm_task1_dinov3_hard_vote_materialize_scene.sbatch`.

### 3. Abstention recovery

Additional cameras are selected automatically by abstention visibility and
pose diversity, DINOv3 evidence is lifted for them, and recovery candidates
are produced with reliability calibration:

- `slurm_task1_dinov3_detected_abstention_recovery_scene.sbatch`
- `slurm_task1_dinov3_abstention_round_trip_validation_scene.sbatch`
- `slurm_task1_dinov3_abstention_recovery_materialize_scene.sbatch`

Recovery labels are materialized only after the paired held-out gates pass.

### 4. General end-to-end pipeline

For scenes without existing DINOv3 evidence, one invocation runs the full
chain: render -> segment -> lift -> hard consensus -> reliability
calibration -> weighted abstention recovery -> materialize labels/label
map/semantic PLY/SuperSplat PLY/summary:

```bash
SCENE=<scene> \
OUTPUT_NAME=<scene>_dinov3_abstention_recovery_review_v1 \
bash scripts/slurm/slurm_task1_dinov3_end_to_end_recovery_scene.sbatch
```

The script works with or without Slurm. It accepts `CAMERA_INDICES` and
`VIEW_COUNT` (default: all real cameras) and exposes the same DINOv3
environment variables as the staged schedulers. `CONFIG_ONLY=1` resolves
paths and parameters without running anything.

Key environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `SCENE` | required | scene name used to derive the model directory |
| `MODEL_DIR` | `<workspace>/data/3dgs_models/graphdeco/<SCENE>` | scene model root |
| `OUTPUT_NAME` | `<SCENE>_dinov3_abstention_recovery_review_v1` | output root name |
| `CAMERA_INDICES` | empty | explicit comma-separated camera indices |
| `VIEW_COUNT` | 0 | camera count; 0 = every real camera |
| `RENDER_MAX_WIDTH` | 960 | render width for views and vote lifting |
| `ITERATION` | 30000 | Gaussian model iteration |
| `GAUSSIAN_ENV` | `semantic_3dgs_renderer` | conda env for render/lift/materialize |
| `DINOV3_ENV` | `dinov3_semantic` | conda env for DINOv3 inference |
| `DINOV3_ROOT` | `<workspace>/external/dinov3` | pinned DINOv3 repository |
| `DINOV3_BACKBONE` | `<workspace>/data/models/dinov3/...pretrain...pth` | ViT-7B backbone checkpoint |
| `DINOV3_SEGMENTOR` | `<workspace>/data/models/dinov3/...m2f_head...pth` | ADE20K Mask2Former head |
| `DINOV3_PRECISION` | `bfloat16` | inference precision (`bfloat16` or `float32`) |
| `DINOV3_CROP_SIZE` | 896 | sliding-window crop (multiple of 32) |
| `DINOV3_STRIDE` | 596 | sliding-window stride |
| `DINOV3_CHECKPOINT_LOAD_MODE` | `local_mmap` | `standard` or `local_mmap` |
| `DINOV3_MAX_CUDA_MEMORY_GIB` | 42 | CUDA allocator memory cap |
| `ONTOLOGY` | `configs/ade20k_to_project.json` | project ontology |
| `CONFIG_ONLY` | 0 | resolve paths/parameters and exit without running |
| `RESET_OUTPUT` | 0 | allow replacing an existing output root |

## Output policy

- Accepted hard anchors are immutable; zero-camera Gaussians are never
  filled; no scene-specific or class-specific rules are applied.
- Recovery outputs are review materializations only.
- Models, datasets, and generated outputs stay on the cluster; only code,
  tests, and reports are committed.
