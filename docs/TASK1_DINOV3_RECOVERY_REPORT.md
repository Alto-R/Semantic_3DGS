# Task 1 DINOv3 Black-Spot Recovery Report

Status: reviewed. The hard-vote materialization is the accepted base. The
DINOv3 abstention-recovery materialization is the preferred review artifact
and the general end-to-end pipeline is the maintained route for scenes with
new DINOv3 evidence. Recovery outputs are review materializations, not
auto-accepted semantic labels.

## Goal

Fill camera-observed black (unlabeled) Gaussians correctly, as best as
possible, using per-camera DINOv3 ADE20K evidence lifted to the 3D Gaussian
model with FlashSplat.

## Method

The maintained pipeline runs every stage in one invocation per scene:

1. render every real camera view (or an explicit subset);
2. DINOv3 ViT-7B + ADE20K Mask2Former segmentation of each view;
3. dense FlashSplat vote lift (one normalized distribution per
   camera/Gaussian);
4. equal-camera strict-majority consensus -> hard labels + status
   (>= 2 cameras, unique winner, > 50%);
5. leave-one-camera-out reliability calibration (95% Wilson lower bound);
6. calibrated weighted strict-majority recovery for abstentions
   (single camera, exact tie, no strict majority), requiring >= 2 cameras;
7. materialize labels, label map, semantic PLY, SuperSplat debug PLY,
   and a summary.

No manual camera or Gaussian selection is used; no scene-specific or
class-specific rules are applied; zero-camera Gaussians are never filled;
accepted hard anchors are immutable.

Entry points:

- `scripts/slurm/slurm_task1_dinov3_end_to_end_recovery_scene.sbatch`
  (stages 1-7; works with or without Slurm);
- `scripts/task1/dinov3/recover_dinov3_abstentions_pipeline.py`
  (stages 4-7 from an existing DINOv3 vote manifest).

The audited hard-vote and abstention-recovery stages used for the reviewed
Playroom/Dr. Johnson artifacts remain under
`scripts/slurm/slurm_task1_dinov3_{hard_vote_materialize,
abstention_recovery_materialize,...}_scene.sbatch`.

## Acceptance rule

Fill precision is measured by rendering newly labeled black Gaussians into
each held-out baseline camera and comparing with that camera's cached
DINOv3 map (consistency check, not human ground truth). The retained
acceptance bar is held-out fill precision >= 0.50 plus the four
non-regression gates (projected coverage, overall agreement, interior
agreement, boundary agreement).

## Results

### Reviewed Playroom / Dr. Johnson (DINOv3)

| Scene | Hard-vote accepted | Abstention-recovery accepted | Recovery added | Strict candidate precision |
|---|---:|---:|---:|---:|
| Playroom | 1,911,015 | 2,128,752 | 217,737 | 41.5% |
| Dr. Johnson | 2,740,365 | 2,893,994 | 153,629 | 55.2% |

The strict DINOv2-gated candidate never passed the acceptance bar; the
abstention-recovery materialization is the best reviewed result.

### Counter / Kitchen (general end-to-end pipeline)

New DINOv3 evidence generated end to end on 240 / 279 real cameras:

| Scene | Gaussians | Cameras | Hard accepted | Recovered | Accepted total | Unlabeled |
|---|---:|---:|---:|---:|---:|---:|
| Counter | 1,222,956 | 240 | 783,948 | 33,426 | 817,374 | 405,582 |
| Kitchen | 1,852,335 | 279 | 1,642,598 | 13,248 | 1,655,846 | 196,489 |

The DINOv2 abstention-recovery reference for the same scenes: Counter
751,098 accepted / 471,858 unlabeled; Kitchen 1,397,595 accepted / 454,740
unlabeled. The DINOv3 runs label 66,276 more Counter Gaussians and 258,251
more Kitchen Gaussians; the model difference, not the recovery rule, drives
the coverage gain.

Outputs are kept on the cluster under
`outputs/eyenavgs_task1/{scene}_dinov3_abstention_recovery_review_v1/`.
Review copies of the SuperSplat PLYs are retained locally under
`scratch/dinov3_abstention_recovery_supersplat/`.

## Rejected alternatives

All alternative recovery rules and independent mask sources were measured
with the same held-out protocol and rejected; the full audit trail is in
`archive/dinov3_recovery_experiments/README.md` and the archived progress
log.

## Constraints

- Recovery outputs are review materializations only.
- The acceptance measurement uses held-out DINOv3 maps, not human labels.
- Models and generated outputs stay on the cluster; the repository commits
  code, tests, and reports only.
