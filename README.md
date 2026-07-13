# PKU 3DGS VR Semantic Annotation

This repository tracks the EyeNavGS semantic annotation and gaze-target labeling work.
The immediate priority is Task 1: add object-level semantic labels to 3D Gaussian
Splatting scenes from the public EyeNavGS dataset.

## Project Goal

Convert EyeNavGS from raw VR navigation and gaze traces into a supervised
gaze-target dataset:

- D1: semantic 3DGS scenes, with one integer `label` property per Gaussian and a
  `label_map.json` per scene.
- D2: per-frame gaze-target annotations derived from ray intersection against
  labeled scenes.
- D3: quality report with hit rate, manual spot checks, geometric error analysis,
  and semantic snapping ablation.

## Current Status

- Local project folder: `C:\Users\dhana\PROJECTS\PKU 3DGS VR`
- GitHub remote: `https://github.com/DhanaKresnawijaya237/pku-3dgs-vr`
- Cluster account: `ssh -p 10022 cse12312032@172.18.34.25`
- Cluster root: `/lab/haoq_lab/cse12312032`
- The old cluster bare Git remote is deprecated; use GitHub as the source of
  truth.
- Reused semantic Conda env: `gaussian_grouping_true`
- GraphDeco official pretrained models downloaded and extracted on the cluster.
- Current Task 1 model coverage: 8/12 EyeNavGS scenes matched; `nyc`,
  `london`, `berlin`, and `alameda` still need separate 3DGS model sources.
- Bicycle identity baseline job `92369` established the first accepted result
  at commit `95ef1d6`: 50 views, 411 GroundingDINO + SAM masks, 399 FlashSplat
  proposals, and 6,131,954 validated Gaussians.
- Job `92344` removed the bicycle-colored road streaks and bench-colored
  vegetation patches visible in the prior spatial-only run while preserving the
  bicycle and bench across all 50 validation views.
- Automatic targeted-camera job `92426` is the current accepted bicycle result:
  it screened 144 safe unused cameras, evaluated a 100-camera candidate pool,
  automatically added 20 low-coverage/pose-diverse views, and validated a
  70-view run with 599 GroundingDINO + SAM masks and 583 FlashSplat proposals.
- Job `92426` retains one 119,570-Gaussian bicycle, one 139,727-Gaussian bench,
  and eight tree IDs totaling 1,096,092 Gaussians. The apparent fence label in
  job `92369` was reassigned primarily to the bench, matching the clean focused
  overlays and the detector's earlier `bench fence` ambiguity.
- The current result remains conservative in 3D: 58.40% of raw Gaussians use
  label `0`, down from 60.25%. On the same original 50 cameras, visible coverage
  is effectively unchanged at 92.42% while the worst frame improves from
  70.32% to 72.82%. The 20 difficult additions measure 82.45% pooled coverage.
- Identity-test job `92353` merged the bicycle and bench correctly, but its
  first consolidation rule also split 13 accepted tree groups into 26
  components and ultimately discarded 95,584 tree Gaussians. That result is
  retained as a diagnostic, not accepted as the new baseline.
- Consolidation treats accepted instances as atomic: connected geometry may
  merge same-class IDs but may not split them. Job `92426` passed this check and
  is exposed at `outputs/eyenavgs_task1/accepted/bicycle` on the cluster.
- Train reuse job `92469` established the accepted 50-view train baseline. Lowering the
  stuff-class minimum from 10,000 to 8,000 retained the strongly supported sky
  group (43 source views, 8,462 final Gaussians) without changing either train
  instance. Structural validation passed for all 1,026,508 Gaussians, and
  it prepared the scene-local targeted expansion.
- Final train job `92483` is now accepted at
  `outputs/eyenavgs_task1/accepted/train -> ../train_semantic_targeted_v3`.
  It validates 1,026,508 Gaussians across 70 views, retains two train instances
  and an automatically supported sky group, and prunes the shipping-container
  `building` false positive.
- The accepted train overlay-difference proxy is 99.03% pooled across all 70
  views, with an 89.62% minimum. The original 50 views measure 99.56%; the 20
  difficult additions measure 97.70%. These are visible-tint measurements, not
  semantic ground truth or gaze-hit accuracy.
- The accepted workflow is now scene-configurable through
  `scripts/slurm_task1_semantic_scene.sbatch`; the historical bicycle script is
  a compatibility wrapper. Targeted train job `92470` added 20 difficult views
  but is diagnostic only: it pruned sky at the default stuff cutoff and labeled
  a shipping container as `building`. Fusion now adapts the stuff cutoff from
  multi-view support for every scene: a stuff group seen in at least half the
  source views uses 75% of the normal cutoff. `building` is treated consistently
  as stuff rather than an object instance. Corrected reuse job `92482` validated
  that fusion result, but its overlays exposed a reuse-mode bug: validation
  defaulted to 50 evenly spaced cameras instead of inheriting all 70 manifest
  cameras. Reuse mode now infers both camera indices and count from the reused
  Grounded-SAM manifest. Job `92483` passed that complete validation.
- Room baseline job `92490` is structurally valid but diagnostic only. Its
  sofa, chair, table, rug, plant, door, and window labels are visually coherent,
  but the initial vocabulary omitted the prominent piano, television, speakers,
  media console, and curtains. Pooled visible-tint coverage is 67.01%, with the
  lowest electronics-dominated view at 7.73%.
- Corrected room job `92492` reran detection with the expanded vocabulary and
  is accepted at
  `outputs/eyenavgs_task1/accepted/room -> ../room_semantic_baseline_v2`.
  It validates all 1,593,376 Gaussians with 18 nonzero labels. The unlabeled
  ratio fell from 72.56% to 68.49%, while the 50-view visible-tint proxy rose
  from 67.01% to 81.14% and its minimum rose from 7.73% to 56.81%.
- Visual QA shows consistent piano and television localization and plausible
  room furniture/surface labels. The hardest close views have minor television
  spill onto adjacent cabinet/wall geometry, so this remains an accepted
  baseline rather than a semantic-ground-truth claim. Automatic targeted-view
  expansion is the next room checkpoint.
- SuperSplat inspection exposed a display-palette collision: curtain and chair
  were separate labels but used similar purple shades. Debug PLYs and overlays
  now allocate deterministic per-label colors with a CIELAB separation target,
  including distinct colors for multiple instances of one class. This changes
  only visualization colors, not semantic IDs or Gaussian ownership.
- Targeted room job `92498` validated all 70 views and reduced the unlabeled
  ratio to 65.30%, but the separated colors exposed a real cross-class error:
  ambiguous phrases such as `television stand table desk` were assigned to the
  first matching class and colored coffee/side-table geometry as television.
  The run is diagnostic and `accepted/room` remains on the 50-view baseline.
- Phrase resolution now rejects unrelated multi-class matches while preserving
  specific compounds such as `floor speaker`. Connected same-class fragments
  are also consolidated before the global object-size cutoff so valid parts of
  one object are evaluated together without adding class-specific thresholds.
- Corrected targeted room job `92513` is accepted at
  `outputs/eyenavgs_task1/accepted/room -> ../room_semantic_targeted_v2`.
  It validates 1,593,376 Gaussians across 70 views, retains a clean
  7,779-Gaussian television and a separate 9,054-Gaussian table, and removes the
  table-as-television failure from job `92498`.
- Accepted room visible-tint coverage is 81.38% pooled. The original 50 views
  measure 84.22% with a 62.24% minimum; the 20 automatically selected difficult
  views measure 74.28% with a 35.03% minimum. The lowest view is dominated by
  unlabeled wall/ceiling rather than visible class leakage.
- `truck` is the fourth-scene pilot: 2,541,226 Gaussians and 251 cameras. Its
  scene configuration prioritizes the truck and wheels while retaining common
  outdoor context. The first checkpoint is an evenly spaced 50-view baseline.
- Task docs imported:
  - `TASK_BRIEF_EyeNavGS_Semantic_Annotation.md`
  - `INTERNSHIP_SCHEDULE.md`

## Superseded Task 1 Run Record

This table preserves the reason each non-final run may be deleted from cluster
storage. A row marked "superseded" is not necessarily a failed run: some were
valid intermediate baselines that were replaced by a more complete accepted
result. The three early archive experiments without a retained diagnosis are
identified explicitly rather than assigned a speculative failure.

| Deletable output | What was wrong or incomplete | How the following run addressed it |
| --- | --- | --- |
| `bicycle` | Legacy review/finalization-path output; the retained project record does not identify a specific visual defect. It is not the accepted automatic semantic result. | The GroundingDINO + SAM + FlashSplat semantic pipeline replaced this fallback path and produced auditable class proposals and validation artifacts. |
| `bicycle_auto` | Class-agnostic SAM grouping produced `object_candidate` groups rather than final semantic class names, so it still required manual naming/review. | The semantic pipeline used scene class prompts and automatic cross-view fusion to produce named labels. |
| `archive/bicycle_semantic_10views_job92182` | This was an early 10-view experiment. The tracked notes do not preserve a more specific failure diagnosis, and it was never accepted. | Later bicycle runs used 50 source views, followed by automatic difficult-view expansion to 70 views. |
| `archive/bicycle_semantic_50views_job92250_hard_priority` | Archived hard-priority experiment; the tracked notes do not preserve its exact rejection diagnosis. | It was superseded by the subsequent multiview and spatial-pruning experiments, then by signed multiview evidence. |
| `archive/bicycle_semantic_50views_job92251_multiview_q003` | Archived multiview quality-threshold experiment; the tracked notes do not preserve its exact rejection diagnosis. | Job `92261` became the first documented useful 50-view baseline, after which failures were diagnosed explicitly. |
| `archive/bicycle_semantic_50views_job92261_pre_spatial_pruning` | First useful automatic baseline, but foreground object identities were fragmented. | Job `92331` added spatial connected-component pruning to remove small disconnected label islands. |
| `archive/bicycle_semantic_50views_job92331_spatial_only` | Spatial pruning removed small islands, but larger bicycle-colored road streaks and bench-colored vegetation patches survived because fusion accumulated positive evidence only. | Job `92344` added signed positive/negative multiview evidence and removed the obvious leakage. |
| `bicycle_semantic` (job `92344`) | Clean class localization, but the same physical bicycle remained split into 2 IDs and the bench into 4 IDs. | Job `92353` introduced connected identity consolidation; job `92369` corrected that rule so accepted instances could merge but not split. |
| `bicycle_semantic_identity_test` (job `92353`) | The first consolidation rule merged bicycle and bench correctly but split 13 accepted tree groups into 26 components; later size pruning discarded 95,584 tree Gaussians. | Job `92369` treated accepted instances as atomic and used merge-only consolidation, preserving all accepted tree coverage. |
| `bicycle_semantic_identity_test_v2` (job `92369`) | Valid accepted 50-view baseline; no semantic failure was recorded. It became superseded because it did not test automatically selected difficult views. | Job `92426` added 20 automatically selected low-coverage/pose-diverse cameras and became the accepted 70-view bicycle result. |
| `train_semantic_baseline_v1` (job `92467`) | The fixed 10,000-Gaussian stuff cutoff discarded a sky group supported by 43 of 50 views. | Job `92469` lowered the train experiment's stuff cutoff to 8,000 and retained the 8,462-Gaussian sky group without changing the train instances. This scene-specific experiment was later replaced by a generic adaptive rule. |
| `train_semantic_baseline_v2` (job `92469`) | Valid accepted 50-view baseline; superseded because it did not include difficult-view expansion and its 8,000 cutoff was an interim scene-specific setting. | Targeted runs added 20 cameras, and the generic fusion rule now lowers the cutoff automatically for strongly supported stuff classes in every scene. |
| `train_semantic_targeted_v1` (job `92470`) | The default cutoff again pruned sky, while a long shipping container was falsely retained as `building`. | Job `92482` added generic support-adaptive stuff pruning and treated `building` consistently as stuff; it kept sky and pruned the container. |
| `train_semantic_targeted_v2` (job `92482`) | Fusion was correct, but reuse-mode overlay validation silently fell back to 50 evenly spaced cameras instead of validating the full 70-camera manifest. | Reuse mode was changed to inherit the exact camera indices and count; job `92483` validated all 70 views and became accepted. |
| `room_semantic_baseline_v1` (job `92490`) | The initial vocabulary omitted piano, television, speakers, media console, and curtains; electronics-dominated views therefore had very low visible-tint coverage. | Job `92492` expanded the vocabulary and phrase aliases and reran detection/fusion. |
| `room_semantic_baseline_v2` (job `92492`) | Valid accepted 50-view baseline; minor television spill remained, difficult views had not been added, and similar curtain/chair debug colors obscured visual inspection even though their label IDs were separate. | Targeted expansion added 20 cameras, and a deterministic perceptually separated palette made different labels visibly distinct. |
| `room_semantic_targeted_v1` (job `92498`) | The clearer palette exposed a real semantic error: ambiguous phrases such as `television stand table desk` assigned table geometry to television. True television fragments were then individually lost under the object-size cutoff. | The phrase resolver now rejects unrelated multi-class phrases, and connected same-class fragments consolidate before pruning. Job `92513` retained a clean television separately from the table and became accepted. |

Keep the final accepted targets and their compact reports:

- `bicycle_semantic_targeted_v1` (job `92426`)
- `train_semantic_targeted_v3` (job `92483`)
- `room_semantic_targeted_v2` (job `92513`)
- `accepted/`, whose scene links point to those results
- `logs/`, `manifests/`, and small validation/report JSON files needed for the
  final report

Generated `semantic_point_cloud_rgb_debug.ply` files and their inspection JSON
are also safe to delete from every old and accepted run. They are derived,
redundant visualization artifacts rather than required deliverables, and commit
`244787f` stopped future runs from creating them. Keep
`deliverables/semantic_point_cloud.ply`, `label_map.json`, run summaries,
validation reports, and `semantic_point_cloud_supersplat_debug.ply` for accepted
scenes.

## Workflow Summary

Windows is the control workspace for code, docs, configs, and small samples.
The Linux GPU cluster is the execution workspace for dataset download, 3DGS
rendering, FlashSplat/SAGA processing, and generated outputs.

Large files are intentionally excluded from Git. Keep EyeNavGS data, 3DGS model
folders, rendered frames, masks, checkpoints, and third-party repo clones outside
tracked files.

Future semantic runs no longer generate the redundant
`semantic_point_cloud_rgb_debug.ply` export or its JSON inspection. SuperSplat
debug PLYs remain the supported colorized 3D visualization, while
`deliverables/semantic_point_cloud.ply` remains the required labeled output.

See:

- `docs/WORKFLOW.md` for local/cluster setup and command conventions.
- `docs/TASK1_SEMANTIC_ANNOTATION.md` for the semantic annotation pipeline.
- `TASK2_HANDOFF.md` for the isolated Task 2 gaze-target implementation
  contract and new-thread starting prompt.
- `configs/paths.example.yaml` for path conventions.

## First Milestone

The `bicycle` pilot has completed end to end:

1. Original `point_cloud.ply` inspected.
2. Fifty representative views rendered at 960-pixel width.
3. GroundingDINO + SAM masks generated and exported as 8-bit binary PNGs.
4. FlashSplat proposals fused using signed confidence-weighted multi-view
   evidence.
5. Semantic PLY and `label_map.json` exported in the D1 format.
6. Full and bicycle-versus-bench overlays visually checked.

Task 1 is not complete: the hard minimum remains four fully labeled and
validated scenes. The next work is room targeted-view validation, a downstream
gaze-hit acceptance criterion, and at least one additional scene.
