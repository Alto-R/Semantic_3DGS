# SAM-Mask2Former Hybrid Refinement

Status: report-only cluster validation completed; rejected for semantic
materialization and retained for research reproduction.

This experimental route combines prompted semantic identity with
prompt-independent boundaries:

- GroundingDINO supplies the requested class and candidate box.
- SAM supplies the class-associated object mask.
- Mask2Former supplies class-agnostic query regions.
- FlashSplat first lifts accepted SAM-seeded masks into 3D, then lifts the
  final masks after one guarded reprojection round.

Mask2Former's ADE20K class prediction is diagnostic only. The hybrid matcher
never uses it to assign identity. A Mask2Former-only region cannot create a
semantic mask.

## Why this route exists

The direct Mask2Former audit produced sharp boundaries but confused shuttered
windows with doors. The existing GroundingDINO+SAM extension path has the
opposite limitation: SAM can outline a prompted object accurately, but it
cannot recover a view where GroundingDINO supplied no useful box.

The hybrid route separates identity from boundary evidence. It keeps SAM's
prompt identity and uses a Mask2Former region only when their geometry agrees.
Conflicting identities, unsafe expansion, and weak overlap abstain.

## One-pass pipeline

`scripts/slurm/slurm_task1_sam_mask2former_hybrid_scene.sbatch` runs every
computational stage in one job:

```text
render + Mask2Former semantic/region inference
    -> GroundingDINO + SAM on the same RGB views
    -> guarded SAM-to-region matching
    -> initial FlashSplat proposal lifting
    -> one multiview 3D-seed reprojection into missed views
    -> final FlashSplat proposal lifting
    -> immutable-base overlap report
    -> multiview grouping + adaptive 3D component audit
    -> contact sheets and manifests
```

The individual Python modules remain callable for debugging:

```text
scripts/task1/dense_seg/segment_views_semantic.py
scripts/task1/hybrid_refinement/refine_sam_with_mask2former_regions.py
scripts/task1/grounding/run_flashsplat_mask_proposals.py
scripts/task1/hybrid_refinement/propagate_regions_from_3d_seeds.py
scripts/task1/hybrid_refinement/audit_proposals_against_base.py
scripts/task1/grounding/cluster_semantic_flashsplat_proposals.py
```

## Safety modes

`REPORT_ONLY=1` is the default. It runs through 3D grouping and adaptive
component pruning but writes only reports, proposal supports, overlays, and
contact sheets. It writes no Gaussian semantic labels, label map, or semantic
PLY and never modifies the base.

`REPORT_ONLY=0` additionally writes isolated candidate labels and a candidate
label map. It still does not merge into the base or write a semantic PLY.
It fails closed before candidate fusion when cross-class 3D proposal
containment exceeds the global identity-conflict threshold. Final adoption
remains a separate reviewed operation.

Fresh output names are required unless `RESET_OUTPUT=1` is explicitly set.
The scheduler never modifies the selected base output.

## Global matching policy

For each SAM mask and overlapping Mask2Former region, the matcher measures:

- intersection-over-union;
- containment and SAM coverage;
- class-agnostic region confidence;
- region-to-SAM area ratio;
- competing semantic identities;
- overlap with SAM masks of other classes.

A region boundary replaces a SAM mask only when all global gates pass.
Otherwise the original SAM mask is retained with an abstention reason in the
manifest. The thresholds are shared by every scene and class; scene files
contain only declarative vocabularies and prompts.

The controlled propagation stage builds each class seed only from Gaussians
supported by at least two distinct source views. It renders that seed into the
same audited cameras and may append a class-agnostic region when projection
coverage, containment, confidence, identity margin, and cross-class overlap
gates all pass. It never uses a Mask2Former class prediction, never creates a
mask without an independently confirmed 3D seed, and never feeds propagated
masks back into the seed builder. The pipeline therefore performs exactly one
bounded propagation round rather than an iterative feedback loop.

The output keeps the standard Grounded-SAM mask-stack/manifest contract, so
the maintained FlashSplat proposal and 3D grouping implementations are reused.
The one-pass audit enables adaptive connected-component pruning for every
selected target type, including ontology entries represented as stuff; this
avoids treating windows and doors as exempt from spatial consistency checks.
It also reports pairwise cross-class 3D containment rather than resolving a
window/door conflict through a hard-coded class priority.

## Main scheduler variables

```text
SCENE, OUTPUT_NAME
INCLUDE_CLASSES                 required comma-separated target vocabulary
CLASS_CONFIG                    defaults to the scene semantic config
BASE_OUTPUT_NAME                defaults to the scene's immutable v5 output
VIEW_COUNT                      default 24 evenly spaced views
CAMERA_INDICES                  optional exact camera list
MASK2FORMER_MODEL               local path or cached model id
FINAL_MAX_MASKS_PER_VIEW        default 0; retain all final masks
REPORT_ONLY                     default 1
RESET_OUTPUT                    default 0
```

Mask2Former region-export and hybrid-matching thresholds are exposed as
`REGION_*` and `REFINE_*` variables. Controlled reprojection thresholds are
exposed as `PROPAGATION_*`. They must remain global during the audit; do not
tune them per scene or per class.

## Review outputs

```text
stages/01_mask2former_regions/        semantic and class-agnostic region maps
stages/02_grounded_sam/               original prompted masks
stages/03_hybrid_masks/               refined masks and abstention evidence
stages/04_seed_flashsplat_proposals/  initial per-view 3D supports
stages/05_propagated_hybrid_masks/    one-round masks and propagation report
stages/06_final_flashsplat_proposals/ final per-view 3D supports
stages/08_3d_audit/                   multiview/component summary
validation/proposal_base_overlap.json immutable-base transition evidence
visualizations/contact_sheets/        region, SAM, refined, and propagated QA
experiment_mode.txt                   write/no-write contract
```

No output should be treated as accepted merely because the job succeeds.
Review the per-mask statuses, base-overlap transitions, 3D groups, and contact
sheets before authorizing candidate labels or a later merge.

## Validation outcome

The four-view runtime canary and both 24-view scene audits completed without
writing semantic labels, a label map, or a semantic PLY. The route was
rejected for materialization:

- class-wide seeds mixed physical instances before reprojection;
- identity-conflict and no-region-match masks could still contribute to a
  seed;
- propagated regions absorbed large wall, floor, cabinet, rug, and furniture
  surfaces;
- adaptive connected components removed small islands but could not reject
  large connected false surfaces;
- door/window proposal unions overlapped unrelated immutable-base classes by
  roughly 79-89% in the reviewed scenes.

Do not run `REPORT_ONLY=0` or merge these candidates. The implementation is
kept to reproduce the negative result and its safety contracts.
