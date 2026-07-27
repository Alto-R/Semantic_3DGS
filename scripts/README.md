# Script Inventory

Operational and research scripts are grouped by role. Retired utilities and
superseded schedulers are preserved under `archive/`. DINOv3 audit schedulers
remain experimental unless their documentation explicitly records an accepted
materialization checkpoint; the multiview spatial-core route is permanently
report-only.

## Slurm entry points

| Scheduler | Role |
|---|---|
| `slurm/slurm_task1_dinov2_scene.sbatch` | Render real cameras, run DINOv2 ADE20K, lift votes, and build an immutable multiview base |
| `slurm/slurm_task1_dinov3_scene.a100.sbatch` | Audit DINOv3 ViT-7B ADE20K on A100; report-only by default, with optional maintained lift/fusion |
| `slurm/slurm_task1_dinov3_boundary_identity_audit_scene.sbatch` | Report-only DINOv3 query-boundary, DINOv2 identity, immutable-v5, and adaptive 3D audit |
| `slurm/slurm_task1_dinov3_3d_first_scene.sbatch` | Independent report-only DINOv3 full-query-evidence, class-agnostic 3D association, and component identity audit; reads no prior semantic output |
| `slurm/slurm_task1_dinov3_targeted_50plus20_scene.sbatch` | Select 50 even plus 20 low-coverage, pose-diverse cameras from a reviewed DINOv3 label source, then run a fresh independent 3D-first report on those 70 cameras |
| `slurm/slurm_task1_dinov3_anchored_association_audit_scene.sbatch` | Reuse cached 24- and larger-view DINOv3 proposal supports to audit immutable-anchor association, one-view-outlier consensus, and multiview extensions without labels or PLY |
| `slurm/slurm_task1_dinov3_multiview_spatial_core_audit_scene.sbatch` | Reuse a cached larger-view DINOv3 report, exclude one stable dissenting camera, and audit multiview-supported adaptive spatial cores without labels or PLY |
| `slurm/slurm_task1_dinov3_incremental_spatial_core_fill_audit_scene.sbatch` | Subtract immutable preferred labels from resolved spatial cores, re-split unlabeled residuals, and render exact matched-view fill masks without labels or PLY |
| `slurm/slurm_task1_dinov3_automatic_anchor_guard_audit_scene.sbatch` | Apply global class-consistent nearest-anchor profiles to cached incremental residuals and render exact report-only diagnostics without scene/component exceptions |
| `slurm/slurm_task1_dinov3_3d_first_materialize_scene.sbatch` | Materialize reviewed DINOv3 3D components with cross-class conflict abstention |
| `slurm/slurm_task1_dinov3_region_vote_scene.sbatch` | Fuse cached DINOv3 query regions directly with strict multiview majority |
| `slurm/slurm_task1_dinov3_dense_vote_scene.sbatch` | Lift every cached dense DINOv3 pixel class and fuse complete per-camera class mass, with end-to-end visual QA |
| `slurm/slurm_task1_dinov3_seeded_dense_scene.sbatch` | Preserve direct-region DINOv3 seeds and fill only matching, conflict-free dense 3D components, with provenance and end-to-end visual QA |
| `slurm/slurm_task1_dinov3_plurality_dense_scene.sbatch` | Give each camera one equal component-identity vote, optionally resolve cross-class overlap by a unique maximum supporting-camera count, combine with direct-region seeds, and run guarded dense propagation plus visual QA |
| `slurm/slurm_task1_ade_refinement_scene.sbatch` | Run adaptive instance-guard v5 from an immutable DINOv2 base |
| `slurm/slurm_task1_ade_refinement_replay_scene.sbatch` | Reapply the v5 merge to cached refinement evidence |
| `slurm/slurm_task1_semantic_scene.sbatch` | Generate standalone GroundingDINO+SAM evidence for classes absent from ADE20K |
| `slurm/slurm_task1_sam_mask2former_hybrid_scene.sbatch` | Run SAM-seeded boundary refinement, one controlled 3D propagation round, and 3D audit in one job |
| `slurm/slurm_task1_reviewed_extensions_scene.sbatch` | Merge an explicit allowlist of reviewed custom classes into a new output |
| `slurm/slurm_task1_recolor_output.sbatch` | Recolor an existing accepted semantic output |
| `slurm/slurm_gpu_check.sbatch` | Verify the cluster GPU environment |

The DINOv2, DINOv3, ADE v5, custom-source, reviewed-extension, and recolor
schedulers support configuration inspection. Use `CONFIG_ONLY=1` with `bash`
to resolve paths, inputs, and parameters without submitting a job. The cached
ADE replay scheduler does not currently expose this mode.

## Task 1 package

The semantic implementation is split by responsibility:

```text
task1/
  common/      shared FlashSplat camera, PLY, and palette utilities
  dinov2/      rendering, ADE20K inference, vote lifting, and exact fusion
  dinov3/      contract-compatible DINOv3 ADE20K segmentation
  grounding/   camera selection, GroundingDINO+SAM, and 3D proposal fusion
  hybrid_refinement/  SAM/Mask2Former matching, guarded 3D propagation, and base QA
  merge/       guarded ADE v5 and reviewed custom-extension merges
  qa/          PLY publishing, overlays, contact sheets, summaries, and validation
```

Schedulers invoke these as Python modules from the repository root, for example:

```bash
python -m scripts.task1.dinov2.fuse_dinov2_multiview_votes --help
python -m scripts.task1.merge.merge_semantic_extensions --help
python -m scripts.task1.qa.validate_task1_outputs --help
```

Package imports are absolute (`scripts.task1...`), so tests and child processes
use the same import graph as cluster jobs. The guarded ADE merge implements the
active class-neutral v5 policy; the reviewed-extension merge handles only
explicitly accepted identities that are absent from the base ontology.

## Setup and cluster helpers

- `setup/` installs or verifies project dependencies.
- `cluster/` contains repository and workspace helpers.

These are support scripts, not semantic algorithms. They must derive runtime
paths from the repository/workspace and must not embed personal hosts or account
names.

## Configuration

Scene semantic vocabularies live in:

```text
configs/task1_semantic_classes.<scene>[.<version>].json
```

Reviewed hybrid policies live in:

```text
configs/task1_hybrid_extensions.<scene>[.<version>].json
```

Versioned source configs may exist before their classes are accepted. An empty
hybrid default means the evidence is still pending review, not that a merge
should infer all available source classes.
