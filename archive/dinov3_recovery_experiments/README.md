# Archived DINOv3 Recovery Experiments

Status: rejected research experiments; not part of the maintained DINOv3
semantic pipeline.

The maintained route is the DINOv3 abstention-recovery materialization and
its general end-to-end pipeline (see
`docs/TASK1_DINOV3_RECOVERY_REPORT.md`). This archive preserves the later
DINOv3 recovery experiments for provenance and future research without
exposing their flags, imports, stages, or tests through the maintained
scripts.

## Why These Experiments Are Archived

- **Strict DINOv2 agreement-gate candidate**: the best report-only fill
  candidate on Playroom/Dr. Johnson (41.5% / 55.2% held-out precision) but
  still below the 0.50 acceptance bar and never materialized.
- **Plurality runner-up cap (0.20/0.30/0.40)**: fills 8-43x more Gaussians
  but precision drops to ~27-29% (Playroom) and ~31-40% (Dr. Johnson);
  never reaches the bar.
- **Class-aware grouping + cap 0.30**: 29.1% / 40.9%, still worse than the
  strict candidate.
- **Independent GroundingDINO+SAM masks as labels**: 20.8% / 27.9%
  precision; the source is not a reliable label producer.
- **SAM masks as instance components** (unmerged and merged at 0.20/0.50/
  0.70): best 32.5% / 49.3%; no configuration reached the strict candidate
  or the 0.50 bar. The label evidence inside the objects, not the grouping,
  was the bottleneck.
- **Black-evidence camera expansion and component-graph audits**: diagnosed
  the failure modes but every global identity/component rule still accepted
  obvious cross-class lookalikes. Report-only, no labels or PLY.
- **Core-first, spatial-core, incremental fill, anchored association,
  region/query, calibrated/soft probability, 3D-first, plurality, and
  boundary-identity routes**: superseded by the maintained abstention
  recovery or rejected after held-out audits; all report-only.
- **SAM-Mask2Former hybrid refinement**: validated and rejected as a
  semantic-label source because large coherent false components survived
  its global gates.

## Layout

- `scripts/task1/dinov3/`: the rejected DINOv3 experiment modules.
- `scripts/task1/hybrid_refinement/`: the rejected SAM/Mask2Former hybrid
  modules.
- `scripts/slurm/`: the experiment schedulers.
- `tests/`: focused tests that accompanied the archived modules. They are
  outside the default `tests/` discovery root on purpose.
- `docs/`: the running recovery progress log (with cluster-specific paths
  removed) and the retired hybrid/prototype method docs.

Archived modules may import maintained modules that remain at their active
paths; archived-to-archived imports are not rewired. Reproducing an archived
experiment requires explicitly composing the archived modules with the
ordinary Task 1 dependencies, exactly as recorded in the original commits.

## Retained Evidence Names

Key cluster outputs referenced by these experiments (retained separately,
not in this repository):

- `{scene}_dinov3_observed_black_fill_precision_audit_plurality_cap{20,30,40}_v1`
- `{scene}_dinov3_observed_black_component_graph_audit_v{2,3}`
- `{scene}_dinov3_observed_black_component_graph_round_trip_validation_v7`
- `{scene}_dinov3_black_independent_masks_samecameras_v1`
- `{scene}_dinov3_mask_components_samecameras_merge{50,70,100}_eval_v1`
- `{scene}_dinov3_black_evidence_recovery_v1`
