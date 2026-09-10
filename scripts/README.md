# Script inventory

Operational scripts are grouped by role. Retired utilities, audit schedulers,
and rejected fusion methods are preserved under `archive/`.

## Maintained entry points

| Entry point | Role |
|---|---|
| `slurm/slurm_task1_dinov3_end_to_end_recovery_scene.sbatch` | One-invocation end-to-end DINOv3 abstention-recovery pipeline for any scene |
| `task1/dinov3/run_oov_multiclass_scene.sh` | Fuse dino.txt/SAM masks against a DINOv3 base and publish semantic and SuperSplat PLYs |

The complete Graphdeco-to-final-PNG command sequence is in the
[quickstart](../docs/DINOV3_DINOTXT_SAM_QUICKSTART.md).

## Experimental SAM3 instance route

| Entry point | Role |
|---|---|
| `slurm/slurm_task1_sam3_instance_scene.sbatch` | SAM3 concept segmentation, per-concept FlashSplat lift, cross-view association, per-instance consensus, scene graph, and QA overlays |

The route consumes an existing render stage and is under pilot evaluation;
see [the design](../docs/plans/2026-09-10-sam3-instance-layer-design.md).
Cluster verification of the SAM3 checkpoint is pending.

## Legacy alternative entry points

| Scheduler | Role |
|---|---|
| `slurm/slurm_task1_dinov2_scene.sbatch` | Render real cameras, run DINOv2 ADE20K, lift votes, and build an immutable multiview base |
| `slurm/slurm_task1_ade_refinement_scene.sbatch` | Run adaptive instance-guard v5 from an immutable DINOv2 base |
| `slurm/slurm_task1_ade_refinement_replay_scene.sbatch` | Reapply the v5 merge to cached refinement evidence |
| `slurm/slurm_task1_semantic_scene.sbatch` | Generate standalone GroundingDINO+SAM evidence for classes absent from ADE20K |
| `slurm/slurm_task1_reviewed_extensions_scene.sbatch` | Merge an explicit allowlist of reviewed custom classes into a new output |
| `slurm/slurm_task1_recolor_output.sbatch` | Recolor an existing accepted semantic output |
| `slurm/slurm_gpu_check.sbatch` | Verify the cluster GPU environment |

The DINOv2, DINOv3, ADE v5, custom-source, reviewed-extension, and recolor
schedulers support configuration inspection. Use `CONFIG_ONLY=1` with `bash`
to resolve paths, inputs, and parameters without submitting a job. The cached
ADE replay scheduler does not expose this mode.

## Task 1 package

The repository code is split by responsibility:

```text
task1/
  common/      shared FlashSplat camera, PLY, and palette utilities
  dinov2/      rendering, ADE20K inference, vote lifting, and exact fusion
  dinov3/      DINOv3 segmentation, vote lifting, recovery, dino.txt/SAM classification, and OOV fusion
  grounding/   camera selection, GroundingDINO+SAM, and 3D proposal fusion
  merge/       legacy guarded ADE v5 and reviewed custom extensions
  qa/          PLY publishing, overlays, contact sheets, summaries, and validation
  sam3/        experimental SAM3 instance route: segmentation, mask lifting, association, membership, hierarchy, overlays
```

The retired DINOv3 3D-first and hybrid-refinement modules are preserved under
`archive/dinov3_3d_first_experiments/`.

Schedulers invoke these as Python modules from the repository root, for example:

```bash
python -m scripts.task1.dinov2.fuse_dinov2_multiview_votes --help
python -m scripts.task1.dinov3.dinotxt_sam_mask_pilot --help
python -m scripts.task1.dinov3.compose_oov_multiclass_votes --help
python -m scripts.task1.qa.validate_task1_outputs --help
```

Package imports are absolute (`scripts.task1...`), so tests and child processes
use the same import graph as cluster jobs. The guarded ADE and reviewed-extension
merges belong to the legacy DINOv2 workflow.

## Setup and cluster helpers

- `setup/` installs or verifies project dependencies.
- `cluster/` contains repository and workspace helpers.

These are support scripts, not semantic algorithms. They must derive runtime
paths from the repository/workspace and must not embed personal hosts or account
names.

## Configuration

The maintained dino.txt/SAM target configurations use:

```text
configs/task1_dinotxt_sam.<scene>_<target>.json
```

Legacy GroundingDINO scene vocabularies use:

```text
configs/task1_semantic_classes.<scene>[.<version>].json
```

Legacy reviewed-extension policies use:

```text
configs/task1_hybrid_extensions.<scene>[.<version>].json
```

Versioned source configs may exist before their classes are accepted. An empty
hybrid default means the evidence is still pending review, not that a merge
should infer all available source classes.
