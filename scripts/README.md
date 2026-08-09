# Script Inventory

Operational and research scripts are grouped by role. Retired utilities and
superseded schedulers are preserved under `archive/`. DINOv3 audit schedulers
that are not part of the maintained abstention-recovery route are preserved
under `archive/dinov3_recovery_experiments/scripts/slurm/`.

## Slurm entry points

| Scheduler | Role |
|---|---|
| `slurm/slurm_task1_dinov2_scene.sbatch` | Render real cameras, run DINOv2 ADE20K, lift votes, and build an immutable multiview base |
| `slurm/slurm_task1_dinov3_scene.a100.sbatch` | Audit DINOv3 ViT-7B ADE20K on A100; report-only by default, with optional maintained lift/fusion |
| `slurm/slurm_task1_dinov3_all_camera_visibility_scene.sbatch` | Report-only Gaussian visibility audit over every real reconstruction camera |
| `slurm/slurm_task1_dinov3_visibility_camera_selection_scene.sbatch` | Deterministic greedy camera ranking from the all-camera visibility audit |
| `slurm/slurm_task1_dinov3_selected_view_cache_scene.sbatch` | Build the automatic 99% two-view coverage DINOv3 cache from the camera ranking |
| `slurm/slurm_task1_dinov3_round_trip_fidelity_audit_scene.sbatch` | Held-out leave-one-camera-out DINOv3 fidelity audit of the hard-vote consensus |
| `slurm/slurm_task1_dinov3_detected_abstention_recovery_scene.sbatch` | Automatic additional-camera DINOv3 evidence and report-only abstention recovery |
| `slurm/slurm_task1_dinov3_abstention_round_trip_validation_scene.sbatch` | Held-out validation of the abstention-recovery candidate |
| `slurm/slurm_task1_dinov3_hard_vote_materialize_scene.sbatch` | Materialize the audited hard-vote consensus and inspection PLYs |
| `slurm/slurm_task1_dinov3_abstention_recovery_materialize_scene.sbatch` | Materialize recovery labels only after the paired held-out gate passes |
| `slurm/slurm_task1_dinov3_end_to_end_recovery_scene.sbatch` | One-invocation end-to-end DINOv3 abstention-recovery pipeline for any scene |
| `slurm/slurm_task1_ade_refinement_scene.sbatch` | Run adaptive instance-guard v5 from an immutable DINOv2 base |
| `slurm/slurm_task1_ade_refinement_replay_scene.sbatch` | Reapply the v5 merge to cached refinement evidence |
| `slurm/slurm_task1_semantic_scene.sbatch` | Generate standalone GroundingDINO+SAM evidence for classes absent from ADE20K |
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
  dinov3/      DINOv3 ADE20K segmentation, vote lifting, hard-vote audit, and abstention recovery
  grounding/   camera selection, GroundingDINO+SAM, and 3D proposal fusion
  merge/       guarded ADE v5 and reviewed custom-extension merges
  qa/          PLY publishing, overlays, contact sheets, summaries, and validation
```

The rejected DINOv3 experiment modules and the SAM/Mask2Former hybrid modules
are preserved under `archive/dinov3_recovery_experiments/scripts/task1/`.

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
