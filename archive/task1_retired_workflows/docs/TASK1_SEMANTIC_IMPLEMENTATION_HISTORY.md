# Task 1 Semantic Implementation and Run History

This document is a short historical record of the semantic-annotation methods
and evaluation runs. It is not the operational guide; the current procedure is
documented in `docs/TASK1_SEMANTIC_ANNOTATION.md` once a method is finalized.

## 1. Minimal-view GroundingDINO baseline

- Started with 50 evenly spaced reconstruction views.
- GroundingDINO supplied open-vocabulary object detections, SAM refined their
  masks, and the masks were lifted and fused into the Gaussian scene.
- This established the first usable semantic point clouds, but small or rarely
  visible objects could be missed and some masks leaked across object
  boundaries.

## 2. Targeted GroundingDINO views

- Added 20 automatically selected views to the 50-view baseline, prioritizing
  low-coverage regions and useful camera poses.
- This improved several hard objects, but the Playroom all-camera experiment
  showed that simply adding every camera could also increase noisy coverage.
- The reviewed GroundingDINO comparison runs were:

| Scene | Retained run | Views | Unlabeled fraction | Labeled-vote proxy |
|---|---|---:|---:|---:|
| Dr. Johnson | `drjohnson_semantic_targeted_v1` | 70 | 0.7293 | 0.8160 |
| Playroom | `playroom_semantic_baseline_v3` | 50 | 0.6854 | 0.8516 |
| Stump | `stump_semantic_targeted_v1` | 70 | 0.7664 | 0.7673 |
| Treehill | `treehill_semantic_targeted_v1` | 70 | 0.8074 | 0.8192 |

## 3. DINOv2 all-camera base

- Added an automatic dense base using the official DINOv2 ViT-L/14 ADE20K
  linear head on every real reconstruction camera.
- Used exact multiview vote accumulation and a minimum Gaussian-group size of
  500 to keep the output practical.
- This gave broader automatic scene coverage, but the closed ADE20K vocabulary
  could not name scene-specific objects and the original fusion produced many
  black, abstained regions.
- The first four-scene DINOv2 runs were:

| Scene | Run | Unlabeled fraction | Labeled-vote proxy |
|---|---|---:|---:|
| Dr. Johnson | `drjohnson_dinov2_auto_allviews_v1` | 0.3459 | 0.9628 |
| Playroom | `playroom_dinov2_auto_allviews_v1` | 0.5950 | 0.8054 |
| Stump | `stump_dinov2_auto_allviews_v1` | 0.4298 | 0.8735 |
| Treehill | `treehill_dinov2_auto_allviews_v1` | 0.3742 | 0.7966 |

## 4. DINOv2 + GroundingDINO hybrid

- Kept DINOv2 as the dense automatic base and used GroundingDINO only for
  selected vocabulary extensions such as piano and speaker.
- The first unguarded Room hybrid mislabeled the television as piano.
- Added DINO-derived semantic guards before accepting extension masks. The
  cached guarded Room run removed the television false positive while keeping
  the intended piano and speakers, and was visually accepted.
- Important Room comparisons are `room_dinov2_control_allviews_v2`,
  `room_hybrid_piano_speaker_allviews_v1`, and
  `room_hybrid_guarded_cached_branch_v1`.

## 5. Abstain-confidence correction

- Found that rejected DINOv2 pixels were all contributing abstain confidence
  `1.0`, while semantic votes retained their real confidence. This gave
  abstention an unintended advantage during multiview fusion.
- Changed abstain votes to use the actual mean confidence of rejected pixels.
- Reran Playroom and Treehill with corrected weights. The correction reduced
  the bias but did not fully eliminate the black-region problem under joint
  fusion.

## 6. Separate-abstain fusion

- Separated two decisions that joint fusion had mixed together: which semantic
  class wins, and whether enough semantic evidence exists to label the group.
- Semantic class agreement is now measured only over semantic votes. Abstain
  evidence is checked separately using the semantic-evidence ratio.
- On Playroom this filled 222,686 previously black points, removed only 41
  labeled points, and made no class changes where both outputs were labeled.
- The aggressive `semantic_only` diagnostic labeled too freely and was
  rejected. The useful comparison trio is:

| Fusion behavior | Run | Final unlabeled fraction | Labeled-vote proxy | Result |
|---|---|---:|---:|---|
| Corrected joint | `playroom_dinov2_abstain_confidence_v1` | 0.5008 | 0.8891 | Improved, still too black |
| Separate abstain | `playroom_dinov2_separate_abstain_fusion_v1` | 0.4134 | 0.9633 | Visually solved the black-region problem |
| Semantic only | `playroom_dinov2_semantic_only_fusion_v1` | 0.3652 | 0.9834 | Diagnostic only; too aggressive |

## 7. Global agreement-0.45 ablation

- Replayed the saved separate-abstain vote arrays with one global semantic
  agreement threshold lowered from 0.50 to 0.45.
- All eight scenes gained labeled coverage and no Gaussian switched between
  two nonzero semantic classes relative to 0.50.
- The ablation did not solve the actual semantic confusions: Bicycle retained
  bicycle-on-bench leakage, Dr. Johnson gained both correct windowpane and
  wrong wall labels, and Treehill path remained earth.
- The outputs are retained as
  `<scene>_dinov2_separate_abstain_agreement045_fusion_v1` for comparison, but
  0.50 remains the base for later refinement experiments.

## 8. Automatic same-ontology ADE refinement

- Added a cached-DINO experimental route that derives its Grounded-SAM
  vocabulary automatically from ADE classes already supported by the DINO
  scene. Missing-vocabulary labels remain disabled.
- The first merge rejected ambiguous multi-class claims and required each
  Grounding group to overlap a same-class DINO anchor. It corrected the main
  Bicycle and Dr. Johnson failures, but allowed a small valid anchor to
  authorize a disconnected wrong region on Train and a connected broad
  ceiling mask to overwrite Playroom objects.
- Added two global guards with no scene or class exception list:
  - keep only scale-derived 3D components with at least 10% same-instance DINO
    anchor density;
  - allow Grounding `stuff` to fill abstentions or refine other `stuff`, but
    never overwrite an existing DINO `thing` instance.
- The final experimental comparison outputs are named
  `<scene>_dinov2_ade_refinement_auto_spatial_thing_guard_v3`.

Key retained-region measurements for spatial thing-guard v3 are:

| Scene / region | Separate-abstain base | Automatic ADE v3 | Result |
|---|---:|---:|---|
| Bicycle bench | 62.12% bench / 27.98% bicycle | 76.05% bench / 18.84% bicycle | Leakage reduced; true bicycle remains 98.97% bicycle |
| Dr. Johnson window | 17.03% windowpane | 93.22% windowpane | Main miss corrected |
| Room television | 48.08% television | 94.06% television | Main miss corrected |
| Room window | 80.44% windowpane | 91.06% windowpane | Boundary improved |
| Truck body | 67.74% truck | 79.21% truck | Car/building leakage reduced |
| Train body | 77.47% truck / 1.69% building | 80.72% truck / 1.49% building | Wrong building refinement blocked; `train` still unavailable |
| Treehill fence | 44.22% railing | 63.52% railing | Fence split improved |
| Treehill path | 0.44% path | 0.44% path | Not solved; remains mostly earth |
| Playroom chair | 34.18% chair | 49.26% chair | Improved without thing-instance erasure |

This remains an ablation rather than the production hybrid default. In
particular, missing-vocabulary objects are intentionally unresolved, Treehill
path still lacks safe evidence, and some Playroom abstained/wall support is
still accepted as ceiling. `docs/TASK1_SEMANTIC_ANNOTATION.md` is therefore not
changed yet.

## 9. Instance-geometry guard and reviewed extensions

The v3 visual review exposed two different failure modes: broad Grounded-SAM
regions still crossed object boundaries, while missing ADE vocabulary made it
impossible for same-ontology refinement to produce `speaker` or `train`.
The cached v4 replay adds global, class-neutral guards:

- a robust oriented envelope around a DINO thing anchor limits changes outside
  the instance geometry;
- strong multiview groups may recover a repeated, unanchored thing only when a
  compatible anchored prototype exists and no competing DINO thing dominates;
- a large accepted parent thing may absorb a smaller nested competing thing
  only with at least 80% geometric containment, a 2x source-size ratio, and
  at least 10% direct parent overlap coverage;
- an ambiguous group may correct one systematically wrong existing thing only
  when its full 3D geometry matches an anchored same-class prototype, at least
  90% of the candidate is the same competing thing, and it has strong
  multiview support;
- reviewed missing-vocabulary groups are composed afterward from saved
  Grounded-SAM fusion labels rather than being forced into an ADE class.

Measured final replay effects were:

| Scene / group | Measured v4 result |
|---|---|
| Bicycle `bench_01` | 32,461 points changed to bench, including 26,585 formerly bicycle; 1,316 out-of-envelope changes rejected |
| Room speakers | 14,443 points merged across three groups; 2,772 formerly television points changed to speaker |
| Train | 449,733 points merged across two train groups; 359,302 formerly truck and 6,697 formerly building points changed to train |
| Truck `car_01` | 87,162-point car anchor changed to truck after 1.0 containment and 0.2696 parent-overlap coverage |
| Dr. Johnson `fireplace_01` | wall-to-fireplace changes reduced from 16,406 in v3 to 7,841; 8,653 out-of-envelope changes rejected |
| Dr. Johnson windows | the first two groups changed 2,959 and 10,270 points; the third recovered 4,602 points, including 4,593 cabinet-to-windowpane corrections |
| Playroom `door_02` | 7,927 points recovered using the strong-view repeated-instance fallback |
| Playroom `wardrobe_01` | 4,196 points changed; 1,136 out-of-envelope changes rejected |
| Playroom `curtain_02` | rejected because an existing competing thing accounted for 0.6823 of the candidate region |
| Playroom ambiguity guard | rejected a stroller-like chair candidate (0.0103 systematic competing-thing fraction) and a chair-like window candidate (0.6905), below the 0.90 correction gate |

The contact sheets visually indicate coherent Bicycle/bench, Train, Room
speaker, and Truck regions. This is visual QA, not semantic ground truth; the
final SuperSplat inspection remains the acceptance step. Treehill and Stump
were intentionally left unchanged. All six regenerated deliverables have
structural validation status `ok`, and the cluster repository suite passed 92
tests.

## 10. Adaptive v5 and report-only singleton evidence audits

The residual Playroom and Dr. Johnson review motivated a sequence of global
singleton experiments. Adaptive v5 remains the immutable merge baseline: it
rejects the false Playroom wardrobe/door recovery and permits one conservative
connected-component completion for the strongly anchored Dr. Johnson window.
The later v6 mask-augmentation and v8 point-intersection outputs were visually
rejected because adjacent or repeated views could confirm the same lifting
error, and because same-class detections could switch to a different physical
instance.

The v9 audit therefore stopped writing labels and required each alternate box
to come from a fresh GroundingDINO detection in a geometrically separated
camera. Positive-versus-negative FlashSplat contribution exposed the exact
identity failure: Playroom proposal `310` intersected mainly an already-correct
different door, while proposal `100` and Dr. Johnson proposals `8`, `341`, and
`376` had balanced source-seed contribution across two independent views.

The report-only v10 audit added the global identity gate and adaptive 3D
components. The second view must support at least 20% of the source seed and at
least half as much as the best view; components use twice the measured local
4-neighbor Gaussian spacing and are retained only when they contain a source-
seed Gaussian. Near-threshold quality failures are lifted for diagnosis but
remain excluded from selection.

Both v10 runs preserve all no-write invariants. Playroom reports 21 component-
supported, 1 component-abstained, 7 identity-abstained, and 19 semantic-
abstained scheduled candidates; Dr. Johnson reports 22 component-supported, 5
identity-abstained, and 21 semantic-abstained candidates. The targeted findings
are:

- Playroom `310` is correctly identity-abstained (best/second seed fractions
  0.739/0.109). Proposal `100` passes (0.437/0.399), but its retained 1,483
  Gaussians still mix wall, unlabeled, and two existing door instances.
- Dr. Johnson `8`, `341`, and `376` pass the identity gate. Their component-
  retained counts are 6,162, 7,022, and 12,191; `341` and `376` still expand to
  1.73x and 3.42x their source seeds.
- Proposal `879` remains semantic-abstained. Its near-threshold frame `00211`
  contributes 6,600 source-seed Gaussians (0.251), while its only quality-
  passing view contributes 9. This diagnoses a useful shutter mask but not a
  second valid independent confirmation.
- Seed-touching connectivity is not a sufficient global semantic guard. The
  same audit marks obvious lookalikes as component-supported, including window
  candidates whose retained regions are 99.8-99.9% an existing door, 98.6% a
  painting, or 97.5-98.2% a fireplace. These current-label histograms are proxy
  evidence rather than ground truth, but the known repeated-door/window failure
  and overlays make global promotion unsafe.

No v10 semantic labels or PLY were written. Adaptive v5 remains unchanged, and
the singleton route stops at report-only evidence pending a stronger global
physical-instance/semantic-conflict guard.

## Current status

- The scheduler supports DINOv2-only runs and DINOv2 plus guarded
  GroundingDINO extensions, with a stable class palette and separate final and
  debug point-cloud outputs.
- `separate_abstain` is the current DINOv2 fusion candidate based on the
  Playroom result.
- The eight separate-abstain scene jobs completed successfully:

| Scene | Run | Unlabeled fraction | Labeled-vote proxy |
|---|---|---:|---:|
| Bicycle | `bicycle_dinov2_separate_abstain_allviews_v1` | 0.4022 | 0.8990 |
| Train | `train_dinov2_separate_abstain_allviews_v1` | 0.4438 | 0.9032 |
| Room | `room_dinov2_separate_abstain_allviews_v1` | 0.4239 | 0.9642 |
| Truck | `truck_dinov2_separate_abstain_allviews_v1` | 0.5029 | 0.9612 |
| Dr. Johnson | `drjohnson_dinov2_separate_abstain_allviews_v1` | 0.2790 | 0.9982 |
| Playroom | `playroom_dinov2_separate_abstain_allviews_v1` | 0.4133 | 0.9635 |
| Stump | `stump_dinov2_separate_abstain_allviews_v1` | 0.3495 | 0.9971 |
| Treehill | `treehill_dinov2_separate_abstain_allviews_v1` | 0.2925 | 0.9375 |

- Separate-abstain is the accepted DINO base for the current experiments. The
  automatic same-ontology refinement has passed structural validation and
  cross-scene QA as an ablation, but its remaining limitations prevent it from
  being presented as the final global implementation in
  `docs/TASK1_SEMANTIC_ANNOTATION.md`.

## Retained comparison outputs

The output cleanup removed superseded baselines, rejected all-camera
GroundingDINO runs, smoke tests, duplicate `current` reruns, and cached
single-class Room QA branches. The retained set is intentionally conservative:

- Accepted pilot GroundingDINO outputs for Bicycle, Room, Train, and Truck.
- Reviewed GroundingDINO outputs for Dr. Johnson, Playroom, Stump, and
  Treehill.
- Original DINOv2 and current separate-abstain outputs for all four scenes.
- Corrected-joint DINOv2 outputs for Playroom and Treehill.
- The corrected-joint, separate-abstain, and semantic-only Playroom fusion
  comparisons, including the full current Playroom rerun.
- The Room DINO control, rejected unguarded hybrid, and accepted guarded hybrid
  comparison trio.
- One stable-color Room GroundingDINO visualization reference.

All retained production/comparison runs still have validation status `ok` and
their final and SuperSplat debug point clouds. The cleanup reclaimed about
55.3 GB while preserving the accepted pointers and their targets.
