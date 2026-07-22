# Archived Task 1 Singleton Experiments (v6-v10)

Status: rejected research experiments; not part of the active Task 1 pipeline.

Adaptive instance-guard v5 remains the active automatic ADE refinement. This
archive preserves the later singleton-recovery experiments for provenance and
future research without exposing their flags, imports, stages, or tests through
the maintained scripts.

## Why These Experiments Are Archived

- v6 appended synthetic cross-view SAM masks and reran global proposal
  clustering. It changed already-correct group topology and produced visible
  leakage.
- v7-v8 intersected singleton seed support with alternate views. The approach
  protected points outside the seed but could repeat the same projection error
  in adjacent cameras and still failed to recover the intended objects safely.
- v9 was report-only and used independent GroundingDINO boxes in separated
  views. It exposed same-class physical-instance confusion.
- v10 added a balanced two-view identity gate and adaptive 3D seed-touching
  components. The identity gate rejected the known Playroom different-door
  match, but the component rule still accepted obvious cross-class lookalikes.

No v9 or v10 semantic labels or PLY were written. The final v10 decision was to
keep v5 and stop before label generation.

## Layout

- `scripts/task1/`: the standalone v6-v10 recovery and audit modules.
- `scripts/slurm/`: the report-only v9/v10 scheduler. The `.a100_run` copy
  preserves the cluster-side partition override used for the completed audit;
  the unsuffixed copy preserves the repository-side RTX 8000 configuration.
- `tests/`: focused tests that accompanied the archived modules.
- `snapshots/shared_pipeline/`: exact end-state snapshots of maintained files
  that contained the optional v6-v8 hooks before those hooks were removed from
  the active tree.

The snapshots are historical references, not alternate maintained entry
points. They intentionally keep version suffixes so they cannot shadow active
modules. Reproduction requires explicitly composing the archived modules with
their recorded shared snapshots and the ordinary Task 1 dependencies.

## Retained Evidence Names

- `playroom_independent_multiview_evidence_v9`
- `drjohnson_independent_multiview_evidence_v9`
- `playroom_identity_component_audit_v10`
- `drjohnson_identity_component_audit_v10`

Cluster outputs are retained independently and were not moved or deleted by
this repository cleanup. Detailed measurements remain in
`docs/TASK1_SEMANTIC_IMPLEMENTATION_HISTORY.md`.
