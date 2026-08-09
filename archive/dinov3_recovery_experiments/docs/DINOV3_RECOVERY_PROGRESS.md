# DINOv3 Black-Spot Recovery — Results and Progress

Last updated: 2026-08-10

## Goal and acceptance rule

Fill the camera-observed black (unlabeled) Gaussians **correctly**, as best
as possible.  The retained acceptance rule is deliberately strict:

- both scenes pass the four held-out non-regression gates (projected
  coverage, overall agreement, interior agreement, boundary agreement);
- both scenes' directly measured fill precision is at least 0.50;
- class-agnostic, automatic, and report-only until accepted;
- no materialization without a separate reviewed materializer.

Fill precision is measured by rendering the newly labeled black Gaussians
into each held-out baseline camera and comparing with that camera's cached
DINOv3 map (the same model family, so the reference is a consistency check,
not human ground truth).

## Accepted base

The original hard-vote materialization remains the accepted result: one
equal identity vote per camera per Gaussian, strict-majority consensus
(>= 2 cameras, unique winner, > 50%), everything else black.  No recovery
candidate has passed the acceptance rule, so nothing new has been
materialized.

## DINOv2 abstention recovery on other scenes (Counter/Kitchen, 2026-08-10)

The abstention-recovery method that produced the better-looking
`drjohnson_dinov3_abstention_recovery_materialized_v1` was adapted to the
DINOv2 per-view vote streams for scenes without DINOv3 evidence.  A single
end-to-end invocation now runs every stage (no incremental hand-holding):

1. load DINOv2 FlashSplat per-view votes (weights normalized per
   camera-Gaussian totals);
2. equal-camera strict-majority consensus -> hard labels + status;
3. leave-one-camera-out reliability calibration (95% Wilson lower bound);
4. calibrated weighted strict-majority recovery for abstentions
   (single camera, exact tie, no strict majority), requiring >= 2 cameras;
5. materialize labels, label map, semantic PLY, and SuperSplat debug PLY;
6. write a summary report.

Pipeline: `scripts/task1/dinov2/recover_dinov2_abstentions_pipeline.py`
(commit `1eb509d`, branch `feature/dinov3-recovery-roundtrip`).  It uses
the existing DINOv2 all-view vote streams from the old cluster; no new
inference is run.  These are review materializations, not auto-accepted.

Results on the target cluster
(`outputs/eyenavgs_task1/{scene}_dinov2_abstention_recovery_review_v1/`):

| Scene | Gaussians | Cameras | Hard accepted | Recovered | Accepted total | Unlabeled |
|---|---:|---:|---:|---:|---:|---:|
| Counter | 1,222,956 | 240 | 702,693 | 48,405 | 751,098 | 471,858 |
| Kitchen | 1,852,335 | 279 | 1,361,624 | 35,971 | 1,397,595 | 454,740 |

Both SuperSplat debug PLYs were verified to match `gaussian_labels.npy`
exactly (label count, no non-finite values).  Local copies for visual
review:
`scratch/dinov2_abstention_recovery_supersplat/{counter,kitchen}/semantic_point_cloud_supersplat_debug.ply`.

Status: pending review in SuperSplat; no promotion decision made yet.
Garden and Bonsai can be run the same way if these look right.

## General DINOv3 end-to-end pipeline (2026-08-10)

The DINOv3 route (the one that produced the better
`drjohnson_dinov3_abstention_recovery_materialized_v1`) was run start to end
on the other scenes instead of the DINOv2 variant.  A general per-scene
pipeline now runs the full chain in one invocation for any scene:

1. render every real camera view (or an explicit `CAMERA_INDICES` subset);
2. DINOv3 ViT-7B + ADE20K Mask2Former segmentation of each view;
3. dense FlashSplat vote lift (one normalized distribution per
   camera/Gaussian);
4. equal-camera strict-majority consensus -> hard labels + status;
5. leave-one-camera-out reliability calibration (95% Wilson lower bound);
6. calibrated weighted strict-majority recovery for abstentions
   (single camera, exact tie, no strict majority), requiring >= 2 cameras;
7. materialize labels, label map, semantic PLY, SuperSplat debug PLY,
   and summary.

Files (commit `177610b`, branch `feature/dinov3-recovery-roundtrip`):

- `scripts/task1/dinov3/recover_dinov3_abstentions_pipeline.py` (stages 4-7);
- `scripts/slurm/slurm_task1_dinov3_end_to_end_recovery_scene.sbatch`
  (stages 1-7, works with or without Slurm);
- `tests/test_dinov3_recovery_pipeline.py`.

The shared consensus code now counts per-class camera evidence in `uint16`,
so scenes with more than 255 cameras (Kitchen has 279) no longer wrap.

No new DINOv3 evidence existed for Counter/Kitchen, so these runs generate
it: all-camera rendering + segmentation + lifting on the target cluster
(`{scene}_dinov3_abstention_recovery_review_v1`).  Bonsai/Garden/Flowers
models are not on the target cluster and require a transfer from the
original cluster before the same pipeline can run on them.

### Counter/Kitchen DINOv3 results (2026-08-10)

Both scenes completed end to end in one invocation each (240 and 279 real
cameras, 960 px, DINOv3 ViT-7B bf16, crop 896 / stride 596).  Validation
passed for both (`status: ok`).  Outputs:
`outputs/eyenavgs_task1/{scene}_dinov3_abstention_recovery_review_v1/`
(labels, label map, semantic PLY, SuperSplat PLY, summary).

| Scene | Gaussians | Cameras | Hard accepted | Recovered | Accepted total | Unlabeled |
|---|---:|---:|---:|---:|---:|---:|
| Counter | 1,222,956 | 240 | 783,948 | 33,426 | 817,374 | 405,582 |
| Kitchen | 1,852,335 | 279 | 1,642,598 | 13,248 | 1,655,846 | 196,489 |

Same scenes under the DINOv2 abstention recovery for reference: Counter
751,098 accepted / 471,858 unlabeled; Kitchen 1,397,595 accepted / 454,740
unlabeled.  The DINOv3 runs label 66,276 more Counter Gaussians and 258,251
more Kitchen Gaussians; the model difference, not the recovery rule, drives
the coverage gain.  These are review materializations, not auto-accepted.

Local SuperSplat copies:
`scratch/dinov3_abstention_recovery_supersplat/{counter,kitchen}/`.

The uint16 consensus fix was also verified by rerunning the Counter/Kitchen
DINOv2 recovery (identical numbers to the original runs, so the old kitchen
figures were not actually corrupted).

## Experiment summary

| Experiment | Playroom fill precision | Dr. Johnson fill precision | Verdict |
|---|---:|---:|---|
| Strict DINOv2 agreement-gate candidate | 41.5% | 55.2% | best tested; paired gate still fails |
| Plurality runner-up cap 0.20 | 28.5% | 31.6% | worse than strict |
| Plurality runner-up cap 0.30 | 27.3% | 39.4% | worse than strict |
| Plurality runner-up cap 0.40 | 28.8% | 39.6% | worse than strict |
| Class-aware grouping + cap 0.30 | 29.1% | 40.9% | slightly better than KNN cap, still worse than strict |
| Independent masks as labels | 20.8% | 27.9% | worse than strict |
| SAM masks as components + strict vote | 21.1% | 46.1% | worse than strict |
| SAM masks as components + DINOv2-gated vote | 26.9% | 35.9% | worse than strict |
| SAM masks as components + plurality cap 0.30 | 27.5% | 36.1% | worse than strict |

All measurements used the same leave-one-camera-out protocol and the same
expanded evidence (baseline + additional cameras).  All experiments were
report-only; no labels or PLYs were written.

## Strict candidate details

| Metric | Playroom | Dr. Johnson |
|---|---:|---:|
| Eligible report-only Gaussians | 3,310 (289 components) | 3,238 (249 components) |
| Recovered (per-fold sum) | 5,431 | 4,267 |
| Fill pixels | 470,867 | 703,376 |
| Fill precision | 41.5% | 55.2% |
| Four non-regression gates | pass | pass |

The DINOv2 agreement gate cut the ungated expanded candidates from 6,506 to
3,310 (Playroom) and 8,797 to 3,238 (Dr. Johnson).

## What the diagnosis found

- The remaining error is concentrated in the same ADE20K "thing" classes:
  door, windowpane, rug, cabinet, chair, plaything, box, shelf, ceiling.
- Stuff vs thing precision is bimodal: Playroom stuff 65.0% / thing 14.6%;
  Dr. Johnson stuff 78.8% / thing 11.4%.
- Restricting to classes with measured precision >= 0.50 would retain 44.8%
  of Playroom fill pixels at 79.7% precision and 58.8% of Dr. Johnson pixels
  at 86.6% (class-conditioned acceptance, not adopted).
- Wrong components often have high winner margins and 4-11 cameras; agreement
  strength does not separate right from wrong.
- Base DINOv3, added DINOv3, and DINOv2 all split or disagree on the same
  objects (e.g., the Playroom wardrobe: DINOv2 42.5% wardrobe vs 30.5%
  cabinet), so no vote or grouping rule built from these sources resolves
  them.
- The old GroundingDINO+SAM masks were generated on only 50/70 cameras
  (Playroom/Dr. Johnson) while the black-spot pipeline uses 159/213 cameras;
  masks covered only 55-59% of black Gaussians.

## Current work: independent masks (Plan B)

The remaining untested variant is new GroundingDINO+SAM inference on the
same camera set as the DINOv3 pipeline with focused prompts (wardrobe
separated from cabinet; door/window/rug/chair/etc. as separate classes).

Progress:

- Grounding stack migrated to the target cluster
  (`external/grounding/`): source 220 MB,
  GroundingDINO weights 694 MB, SAM ViT-H weights 2.56 GB.
- Conda env `gaussian_grouping_true` was recreated on the target cluster
  (Python 3.9 +
  torch 2.0.0+cu117, numpy 1.24.4, opencv 4.8, transformers 4.35, editable
  FlashSplat rasterizer hooks copied from the renderer env, BERT tokenizer
  cache transferred from the old cluster). The 14 GB environment was not
  transferred.
- Focused class configs drafted (scratch, not yet committed):
  `scratch/b_experiment/task1_semantic_classes.{playroom,drjohnson}.json`.
- Inference launched (tmux `grounding_pr` / `grounding_dj`): GroundingDINO+SAM
  on the full same-camera set (159 Playroom / 213 Dr. Johnson cameras) at
  512 px with the focused configs; outputs
  `{scene}_dinov3_black_independent_masks_samecameras_v1`.
- Completed: 1,603 / 2,617 proposals and fused labels for both scenes.

### Same-camera independent-mask results (2026-08-09)

Held-out fill precision of the new same-camera masks:

| Usage | Playroom precision | Dr. Johnson precision |
|---|---:|---:|
| Masks as labels | 13.0% | 24.6% |
| Masks as components, DINOv2-gated majority | 26.2% | 35.9% |
| Masks as components, plurality cap 0.30 | 26.1% | 36.0% |
| Masks as components (merge 0.50), DINOv2-gated | 27.7% | **49.3%** |
| Masks as components (merge 0.50), plurality cap 0.30 | 27.5% | 34.9% |
| Masks as components (merge 0.70), DINOv2-gated | 32.0% | 44.0% |
| Masks as components (merge 0.70), plurality cap 0.30 | 26.4% | 36.0% |
| Masks as components (unmerged), DINOv2-gated | 32.5% | 44.9% |

Strict DINOv2-gated candidate reference: 41.5% / 55.2%. Same-camera coverage
improved (Playroom 55.5% -> 60.5% of black covered; Dr. Johnson ~59%) but
precision did not improve with coarse merge (0.20). Finer components
(0.50 overlap, 37-39 components) improve Dr. Johnson substantially
(36% -> 49.3% DINOv2-gated) while Playroom stays ~28%. A merge-0.70 sweep
continues the Playroom improvement (32.0%) while Dr. Johnson backs off
(44.0%); the unmerged control plateaus (Playroom 32.5%, Dr. Johnson 44.9%).
The curve is non-monotonic and scene-dependent.

### Plan B conclusion

Same-camera masks with focused prompts were fully tested both as labels and
as instance components across a merge granularity sweep. Coverage improved
but no configuration reached the strict candidate's fill precision
(41.5% / 55.2%) or the 0.50 acceptance bar; the best relaxed results were
Playroom 32.5% (unmerged, DINOv2-gated) and Dr. Johnson 49.3% (merge 0.50,
DINOv2-gated). The label evidence inside the objects remains the bottleneck;
independent masks change coverage and component structure but not the
fundamental correctness of the DINOv3/DINOv2 votes on the failing classes.

## Constraints

- All experiments are report-only; the strict candidate remains the best
  tested result but is not accepted for materialization.
- Fill precision is measured against held-out DINOv3 maps, not human labels.
- Cluster work stays under the shared project root; no Slurm on the target
  cluster.
