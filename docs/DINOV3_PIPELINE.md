# DINOv3 ADE20K Pipeline

Status: experimental research branch. The reviewed 24-view v2 output is the
preferred DINOv3 checkpoint. Larger-view association and spatial-core methods
remain report-only; the production semantic decision remains DINOv2 ADE v5.

## Scope

This route replaces only the maintained pipeline's 2D DINOv2 ADE20K
segmenter. It reuses the existing real-camera rendering, identity-preserving
ADE20K ontology, exact FlashSplat vote lifting, connected-component fusion,
and immutable-v5 output policy.

DINOv3 remains a closed-set ADE20K segmenter in this route. It can improve
features or boundaries for ADE20K classes, but it does not add arbitrary
custom identities.

## Required upstream files

Use the official DINOv3 PyTorch-Hub checkpoint format. The adapter requires
these exact files:

```text
data/models/dinov3/
├── dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
└── dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth
```

- Backbone: `dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth`
- ADE20K Mask2Former head:
  `dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth`

Request both official `.pth` downloads from the
[Meta DINOv3 download page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).
The official
[Hugging Face ViT-7B repository](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m)
contains a Transformers-format sharded backbone, not the PyTorch-Hub `.pth`
backbone, and does not contain the ADE20K Mask2Former head. Those
`safetensors` files are not inputs to this route.

If Meta's download page is unavailable in the user's region, first ask a
supervisor or lab member with authorized access to copy the two original
files into the shared cluster model directory. A third-party
[Hugging Face mirror](https://huggingface.co/jaychempan/dinov3/tree/main)
also contains files with the exact official names. The mirror is not
maintained by Meta, so accept the original DINOv3 license and verify both
files against these full hashes before use:

```text
a955f4ea3bec4fcd666bf363630da4386383069b482c8a927e17a3e1154965b7  dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
bf307cb1c2fd95046feb1bf9a8a13dae60a746bddd8f5297134da95525dbcb42  dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth
```

The segmenter and scheduler enforce these full hashes by default and record
the file sizes, hashes, and paths in `dinov3_manifest.json`.

## Pinned code and environment

The official repository is pinned to:

```text
https://github.com/facebookresearch/dinov3.git
6876159a11b4df116f30f667f8c9888617df0751
```

It must be a clean checkout under `external/dinov3`. The scheduler and
segmenter both fail closed if the commit differs or tracked files are
modified.

Use a separate `dinov3_semantic` environment based on the official DINOv3
runtime, with Python 3.11, PyTorch 2.7.1, and torchvision 0.22.1. Do not
upgrade the `gaussian_grouping_true` environment: rendering and vote lifting
continue to run there.

The code-only setup script clones and pins the official repository, creates
the separate environment, validates imports, and prints the missing
checkpoint paths. It never downloads model weights:

```bash
bash scripts/setup/install_dinov3_semantic.sh
```

It uses the official PyTorch 2.7.1 CUDA 12.6 wheel index by default. Override
`PYTORCH_INDEX_URL` only when the cluster's NVIDIA driver requires another
official PyTorch wheel variant.

The default inference settings are:

- official `dinov3_vit7b16_ms` ViT-7B plus ADE20K Mask2Former head;
- native CUDA bfloat16;
- 896-pixel sliding crops;
- 596-pixel stride, matching the official ADE20K evaluation configuration;
- the maintained zero-based ADE20K-to-one-based project ontology;
- raw `uint8` ADE20K IDs;
- `float16` relative top-1/top-2 margin as the reusable confidence field;
- diagnostic `float16` maximum-softmax probability, absolute top-1/top-2
  margin, and one-minus-normalized-entropy maps.

Mask2Former's official semantic score path is not calibrated like the DINOv2
linear head. Its maximum 150-class softmax probability can remain far below
`0.5` even when the argmax segmentation is spatially coherent. The adapter
therefore defines confidence globally as:

```text
(top1_probability - top2_probability) / max(top1_probability, epsilon)
```

This bounded relative separation is class-agnostic and independent of the
small absolute output scale. The manifest records its distribution and the
three diagnostic distributions for every view. Report-only calibration runs
default to `MIN_PIXEL_CONFIDENCE=0.0`, so they never hide predictions before a
global policy has been reviewed. The completed ten-view Playroom/Dr. Johnson
audit rejected every positive global hard threshold: low-margin predictions
include valid entire classes, not only errors. Keep the hard threshold at zero
and use relative margin only as continuous evidence during multiview audits.

The adapter rejects bfloat16 when the selected GPU does not report native
support. The full ViT-7B route has also been validated on a 48 GB RTX 8000
using float32, 512-pixel crops, and a 384-pixel stride. Those reduced settings
are evaluation settings, not a change to the model.

## Execution modes

`scripts/slurm/slurm_task1_dinov3_scene.a100.sbatch` runs the stages in one
Slurm allocation while switching Conda environments per stage.

The safe defaults are:

```text
VIEW_COUNT=1
REPORT_ONLY=1
WRITE_SEMANTIC_PLY=0
RESET_OUTPUT=0
MIN_PIXEL_CONFIDENCE=0.0
```

Report-only mode produces:

```text
stages/01_real_camera_views/
├── rgb_renders/
├── dinov3_segments/
├── dinov3_overlays/
├── view_manifest.json
└── dinov3_manifest.json

visualizations/contact_sheets/
├── rgb_views.png
└── dinov3_ade20k.png
```

It exits before vote lifting and writes no Gaussian semantic labels, label
map, or semantic PLY.

`REPORT_ONLY=0` enables the maintained exact FlashSplat lift and fusion stages
in the same job. Semantic PLY generation remains independently disabled
unless `WRITE_SEMANTIC_PLY=1`. This mode must not be used until the 2D
contact sheets have been accepted explicitly.

Every output uses a new `OUTPUT_NAME`. Existing outputs are never reused
unless `RESET_OUTPUT=1` is deliberately supplied, and immutable v5 inputs are
never modified in place.

## Boundary/identity audit

Direct DINOv3 labels are not promoted. The matched-view audit found much
cleaner boundaries than DINOv2, but the same Dr. Johnson shuttered opening was
classified as `windowpane` in one camera and predominantly `door` in another.

The separate scheduler
`scripts/slurm/slurm_task1_dinov3_boundary_identity_audit_scene.sbatch`
tests the useful part without trusting the failed identity:

```text
render selected cameras
  -> DINOv3 sliding semantic diagnostics
  -> DINOv3 whole-image Mask2Former query boundaries
  -> independent DINOv2 weighted identity votes
  -> FlashSplat proposal supports
  -> immutable-v5 overlap report
  -> adaptive multiview 3D component audit
```

The DINOv3 query's ADE20K class is diagnostic only. A region receives a target
identity only when maintained DINOv2 evidence:

- covers enough of the region above the existing DINOv2 confidence floor;
- has sufficient weighted class share;
- has sufficient margin over the strongest competing ADE20K class; and
- selects a class listed declaratively in `INCLUDE_CLASSES`.

The scheduler is permanently report-only. It refuses `REPORT_ONLY=0`, writes
no Gaussian labels or label map, and never writes a semantic PLY. It reuses
the existing proposal lifting, immutable-base audit, and adaptive 3D
component code instead of adding parallel implementations.

Review these outputs before considering any later candidate mode:

```text
stages/01_dinov3_query_regions/dinov3_region_overlays/
stages/02_dinov2_identity_masks/boundary_identity_audit.json
stages/02_dinov2_identity_masks/identity_overlays/
stages/04_multiview_3d_audit/semantic_group_summary.json
validation/proposal_base_overlap.json
visualizations/contact_sheets/
```

## Acceptance sequence

1. Run `CONFIG_ONLY=1` after the repository, environment, and two checkpoints
   are present.
2. Run one report-only view to validate model loading and GPU memory.
3. Run identical report-only Playroom and Dr. Johnson views.
4. Compare DINOv3 and DINOv2 predictions for door, windowpane, screen door,
   blind, wall, and cabinet confusion.
5. Reject DINOv3 if it merely changes the wrong class or increases broad
   architectural leakage.
6. If direct DINOv3 identity fails but its query boundaries improve, run the
   boundary/identity audit with DINOv2 identity and immutable-v5 protection.
7. Only after explicit component-level acceptance, design a separate no-PLY
   candidate merge. Do not modify v5 during any audit.

## Independent 3D-first query pipeline

`scripts/slurm/slurm_task1_dinov3_3d_first_scene.sbatch` is a separate,
permanently report-only route that has no prior-semantic input. It starts from
the original Gaussian model and cameras and does not load a base label array,
base label map, semantic PLY, or prior semantic output.

The ordering is deliberately different from the boundary/identity audit:

```text
render selected cameras
  -> DINOv3 query masks, full ADE20K probabilities, and query features
  -> prepare every query region without assigning a class
  -> class-agnostic FlashSplat support lift
  -> class-agnostic physical 3D association
  -> component-level robust DINOv3 semantic fusion
  -> report and visualization only
```

The segmenter captures the input to Mask2Former's final class head and stores
the selected, L2-normalized 2,048-dimensional query features. Each compact
region also stores its complete 150-class probability vector conditional on
being an object and its separate no-object probability. These arrays are
small per query and avoid storing dense 150-channel per-pixel logits.

Association uses sparse Gaussian support overlap, query-feature similarity,
mutual-best matches between view pairs, and a cannot-link constraint for
regions that are distinct in the same view. ADE20K class probabilities are
not read while building components. This prevents two co-visible physical
objects from merging merely because a 2D model assigned the same class.

After association, every view contributes one normalized class distribution
to its component. Fusion uses the median of per-view log probabilities. A
component identity is accepted only when every contributing view has the same
winning class and that winner survives every leave-one-view-out fusion.
There is no per-view pixel-share or fixed class-margin gate. Unstable
components remain explicit report-only ambiguities.

The component stage computes candidate Gaussian assignment counts in memory
to expose component conflicts, but does not serialize an assignment array.
The only persistent products are sparse region supports, JSON reports, and
PNG overlays/contact sheets. Accepted regions use the semantic palette;
ambiguous or unlifted regions are gray.

Fresh DINOv3 inference is required once for this route because earlier caches
contain only the winning diagnostic query class, not full query probabilities
or query features. After the new evidence cache exists, association and
fusion changes can be evaluated without rerunning the ViT-7B model.

After explicit review of a completed 3D-first report, the separate
`scripts/slurm/slurm_task1_dinov3_3d_first_materialize_scene.sbatch` scheduler
can materialize a fresh deliverable without rerunning rendering, DINOv3, or
FlashSplat. It never reads v5 or DINOv2 outputs. The final assignment policy is
global and deterministic:

- accepted components are exactly those already recorded by the reviewed
  report-only association stage;
- overlapping components with the same project class keep the component with
  the strongest accumulated FlashSplat support at each Gaussian;
- every Gaussian supported by accepted components from different project
  classes remains unlabeled;
- no scene-specific rule, class-specific threshold, or offline threshold
  sweep is used.

The materializer writes a new output root containing instance labels, direct
project-class IDs, the cross-class conflict mask, a label map, a summary, and
an optional semantic PLY. It never modifies the reviewed report output or the
source Gaussian model.

## Direct multiview region voting

When semantic class coverage is the goal, physical region-to-region
association is not required. The cached scheduler
`scripts/slurm/slurm_task1_dinov3_region_vote_scene.sbatch` directly fuses
every lifted DINOv3 query region at each Gaussian:

```text
all lifted 2D query regions
  -> strongest region support once per camera and Gaussian
  -> 150 ADE20K probabilities plus explicit no-object probability
  -> strict camera majority
  -> agreement with the summed soft-probability winner
  -> semantic class label or abstain
```

Each camera can vote at most once for a Gaussian, even when Gaussian splatting
causes supports from disjoint 2D regions to overlap. This prevents large or
fragmented masks from receiving multiple votes in one view. A semantic label
is accepted when at least two cameras support the Gaussian, one semantic class
has a strict majority of the per-camera winners, and the summed soft evidence
has the same winner. Ties, a no-object majority, and hard/soft disagreement
remain unlabeled.

This route does not use region association, v5, DINOv2, scene-specific rules,
class-specific thresholds, or an offline threshold sweep. It consumes the
existing DINOv3 query-evidence and FlashSplat caches and writes a fresh class
label array, diagnostic vote arrays, label map, summary, and optional semantic
PLY.

The completed Dr. Johnson and Playroom checks showed that this direct-region
route preserves many strong 2D boundaries but leaves roughly 70% of Gaussians
unlabeled because most Gaussians receive zero or one lifted query-region view.
The SuperSplat black regions are therefore real unlabeled output, not a color
export bug.

## Complete dense DINOv3 vote fusion

When broad class coverage is the priority, use
`scripts/slurm/slurm_task1_dinov3_dense_vote_scene.sbatch` instead of the
direct query-region materializer. It reuses the existing 3D-first DINOv3 cache
and does not rerun ViT-7B inference:

```text
cached DINOv3 dense ADE20K argmax maps
  -> FlashSplat fractional class mass for every rendered pixel
  -> one normalized class distribution per camera and Gaussian
  -> unique soft class-mass winner
  -> semantic labels, semantic PLY, overlays, and contact sheet
```

This route deliberately ignores the compact query-region filters during
materialization. It does not use v5, DINOv2, query-region association,
scene-specific rules, class-specific thresholds, or threshold sweeps. The
default `MIN_VIEWS=1` is intentional: single-view Gaussians should receive the
best available DINOv3 class instead of becoming black holes in SuperSplat. The
output still records `supporting_views`, `winner_share`, `winner_margin`, and
abstain reason arrays so later QA can separate one-view guesses from stronger
multi-view labels.

The completed Dr. Johnson and Playroom runs raised assigned-Gaussian coverage
to 67.49% and 73.32%, respectively. That result confirmed that the cached dense
maps contain much more liftable evidence, but it also exposed systematic dense
identity errors: Dr. Johnson shutters were promoted as doors and Playroom
clutter fragmented across several ADE20K classes. Dense fusion is therefore a
coverage diagnostic and propagation candidate, not the preferred final output
by itself.

## Region-seeded dense 3D propagation

Use `scripts/slurm/slurm_task1_dinov3_seeded_dense_scene.sbatch` to combine the
cleaner direct-region result with the broader dense result without rerunning
rendering, DINOv3, or FlashSplat:

```text
direct region-vote project classes = immutable semantic seeds
dense project classes = candidate fill
  -> split every dense class into adaptive 3D voxel components
  -> accept a component only when it contains a matching region seed
  -> reject the component when it contains any conflicting nonzero seed
  -> preserve every original region seed
  -> fill only seed-unlabeled Gaussians in accepted components
```

Connectivity uses the existing 26-neighbor `voxel_components` implementation.
The voxel size is derived independently for each dense class from the median
Gaussian scale, using the same global `4.0` scale multiplier and `0.01` to
`0.20` bounds used by maintained semantic geometry stages. There is no minimum
component size, retained-share gate, scene-specific rule, class-specific
threshold, or offline threshold sweep.

The component guard intentionally treats a component containing both matching
and conflicting seeds as conflicting. Components without any seed remain
unlabeled. This means the route can expand a small, correct seed through a
coherent dense surface while preventing unsupported dense-only identities from
becoming final labels. A missing class with no region seed, such as the current
Dr. Johnson `windowpane`, remains unlabeled rather than being relabeled from a
different dense class.

The finalizer writes the class labels and semantic PLY plus explicit provenance:
the immutable seed mask, propagated-fill mask, label-source codes, globally
unique dense-component IDs, component status codes, per-class component
statistics, and an end-to-end overlay/contact-sheet review set. The region-vote
and dense-vote source outputs remain unchanged.

The completed Dr. Johnson and Playroom runs showed that this seed guard is
structurally reliable but too conservative as a final method. It raised
assigned-Gaussian coverage only from 29.51% to 31.19% for Dr. Johnson and from
27.70% to 30.60% for Playroom. It cannot create a semantic identity that is
missing from the direct-region seeds, so it recovered no Dr. Johnson
`windowpane` labels and barely changed the important Playroom objects.

## Equal-camera plurality component seeds

Use `scripts/slurm/slurm_task1_dinov3_plurality_dense_scene.sbatch` to recover
semantic identities that are present across the cached query views but were
lost by pooled soft-probability fusion:

```text
class-agnostic physical components
  -> one equal ADE argmax vote per source camera
  -> accept at least two cameras and a unique plurality; ties abstain
  -> component identity overrides direct-region identity on that support
  -> combined labels become immutable seeds
  -> guarded dense 3D propagation and end-to-end visual QA
```

The vote count, not logit magnitude, determines component identity. This fixes
the diagnosed Dr. Johnson shutter component where five camera winners were
`windowpane, door, wall, windowpane, signboard`: the equal-camera plurality is
`windowpane`, while pooled soft probabilities had selected `door`. A
Playroom wardrobe component with camera winners
`cabinet, wardrobe, wall, wardrobe, cabinet` remains unlabeled because its
plurality is tied. Single-camera components also remain unlabeled.

The route reuses the class-agnostic 3D component report, direct-region labels,
and dense labels. It does not rerun rendering, DINOv3, or FlashSplat, and does
not use v5, DINOv2, scene-specific rules, class-specific thresholds, soft
confidence gates, or an offline threshold sweep. It writes component-selection
records, component additions and overrides, seed provenance, final semantic
labels and PLY, a SuperSplat debug PLY, and 24-view overlays/contact sheets.

The completed `cross_class_abstain` runs raised assigned-Gaussian coverage to
46.69% for Dr. Johnson and 47.12% for Playroom without the dense-vote
fragmentation. They also corrected the Dr. Johnson shutter component from
`door` to `windowpane`. However, cross-class support overlap caused 247,351
Dr. Johnson and 203,536 Playroom Gaussians to abstain. Only 4,013 of the
57,697-Gaussian corrected Dr. Johnson component survived as `windowpane`.

### Unique supporting-camera ownership

Set `COMPONENT_OWNERSHIP=unique_camera_support` to resolve that overlap using
only independent geometric observations:

```text
accepted semantic plurality components overlap at one Gaussian
  -> count source-camera regions supporting that Gaussian per component
  -> uniquely largest camera count owns the Gaussian
  -> equal cross-class camera counts abstain
```

This ownership rule is separate from semantic identity. It never compares
semantic probability magnitudes or FlashSplat support strength across classes.
FlashSplat strength may break an instance tie only after both components
already have the same project class. The result is order-independent at the
project-class level.

The policy has no minimum camera-support threshold, margin, class-specific
rule, scene-specific rule, or offline threshold sweep. It continues to require
at least two cameras and a unique equal-camera plurality for component
identity. It writes the full cross-class-overlap mask, camera-count tie mask,
maximum component camera count, per-component camera-support histograms, and
resolved-overlap statistics. The default `cross_class_abstain` mode remains
available for exact reproduction of the completed v1 outputs.

## Targeted 50+20 camera experiment

Use `scripts/slurm/slurm_task1_dinov3_targeted_50plus20_scene.sbatch` to test
whether camera placement, rather than camera count alone, reduces the black
regions in the preferred 24-view result. The experiment reuses the historical
targeted-camera policy:

```text
all real reconstruction cameras
  -> 50 deterministic evenly spaced baseline cameras
  -> projection-safe screening of unused cameras
  -> 100 pose-diverse candidates
  -> render the reviewed 24-view DINOv3 labels into candidate cameras
  -> choose 20 cameras with low visible label coverage and pose novelty
  -> run a fresh independent DINOv3 3D-first report on the exact 70 cameras
```

The reviewed labels are a camera-selection diagnostic only. They identify
views that expose geometry left unlabeled by the current result. They are not
passed to DINOv3 inference, FlashSplat proposal lifting, physical component
association, or semantic identity fusion. The output therefore records both
`camera_selection_used_prior_labels=1` and
`semantic_inference_used_prior_labels=0`; it must not be described as fully
prior-independent camera selection.

The defaults reproduce the earlier selection policy: 50 baseline views, 20
additions, 100 screened candidates, 50,000 geometry samples, projection margin
1.5, low-coverage quantile 0.35, coverage weight 0.8, and pose-novelty weight
0.2. Both the selection root and the DINOv3 report root must be fresh. The
accepted 24-view v2 labels and PLY remain unchanged.

## Immutable-anchor association audit

Use `scripts/slurm/slurm_task1_dinov3_anchored_association_audit_scene.sbatch`
to diagnose the graph-percolation and exact-unanimity failures of larger-view
3D-first reports without rerunning DINOv3 or FlashSplat:

```text
strict accepted components from the 24-view report
  -> immutable aggregate Gaussian-support anchors
cached proposals from a larger-view report
  -> discard cameras already present in the anchor cache
  -> direct candidate-to-original-anchor overlap only
  -> unique mutual-best anchor per proposal and one proposal per anchor/camera
  -> no transitive envelope growth and no anchor-to-anchor merge
  -> strict, exactly-one-outlier, or mixed identity audit
  -> multiview outside-anchor support proposal and review overlays
```

The anchor envelope never changes while matching candidates. A proposal that
only overlaps an earlier extension therefore cannot attach, and a proposal
with two near-equal anchor matches remains ambiguous. Candidate support must
have at least 0.50 weighted containment in the original anchor and a best to
second-best anchor score ratio of at least 1.10. Proposed outside-anchor
support requires matching identity from at least two new cameras and at least
500 Gaussians. These are global, class-neutral report gates.

Identity reporting preserves exact unanimity as a separate tier. It also
records a conservative diagnostic tier when exactly one view disagrees, the
fused winner remains the immutable anchor identity, and every leave-one-view-
out fusion retains that winner. Mixed components continue to abstain. The
audit writes JSON, boolean/owner-count proposal masks, compressed sparse
supports, and PNG review sheets. It writes no semantic label array, label map,
or PLY, and it never changes the accepted 24-view outputs.

The completed Dr. Johnson and Playroom audit showed that immutable anchors are
too restrictive for broad recovery. It proposed only 9,943 and 8,544 new
Gaussians, respectively: 0.55% and 0.65% of the geometry that remained black
in the preferred outputs. The principal loss occurred before the extension
size gate because only a small fraction of novel proposals directly attached
to an old anchor. Disconnected surfaces and genuinely new support cannot pass
an overlap requirement against the original anchor. These fills are therefore
diagnostic only and must not be materialized.

## Multiview spatial-core audit

Use
`scripts/slurm/slurm_task1_dinov3_multiview_spatial_core_audit_scene.sbatch`
to test the cached targeted 70-view report without requiring an old anchor and
without rerunning DINOv3 or FlashSplat:

```text
cached class-agnostic 70-view components
  -> retain strict identity or exactly one dissenting view with stable
     leave-one-out winners
  -> remove the one dissenting proposal before geometry is counted
  -> one presence vote per agreeing camera and Gaussian
  -> keep support present in at least two agreeing cameras
  -> split support into adaptive 26-neighbor voxel components
  -> require at least 500 Gaussians and two independent cameras
  -> resolve cross-class overlap by a unique maximum camera count
  -> equal cross-class counts abstain
```

The voxel size uses the same global scale-adaptive rule as the seeded dense
pipeline: median maximum Gaussian scale times `4.0`, bounded to `0.01` through
`0.20`. The identity, support, size, camera, and geometry rules are identical
for every class and both scenes. The method is designed to retain the reliable
spatial cores of cases such as a 26/27-view door or 24/25-view floor while
discarding the dissenting camera and single-camera transitive tails.

This audit is permanently report-only. It writes a JSON report, boolean and
owner-count masks, cross-class overlap/tie masks, compressed sparse core
supports, matched proposal overlays, and a contact sheet. It writes no
semantic label array, project-class array, label map, or PLY. Its output must
be reviewed before any separate materialization method is considered.

The PNG overlays are proposal-level QA: they show the full cached 2D source
regions whose components produced retained cores. They are not an exact
projection of the filtered 3D core footprint. Exact retained support is stored
in the boolean masks and compressed sparse support file.

## Incremental spatial-core fill audit

Use
`scripts/slurm/slurm_task1_dinov3_incremental_spatial_core_fill_audit_scene.sbatch`
after the multiview spatial-core audit to test only additions that cannot
change the accepted 24-view v2 result:

```text
resolved post-ownership spatial-core supports
  -> subtract every Gaussian with a nonzero preferred-v2 label
  -> re-split each unlabeled residual with adaptive 26-neighbor voxels
  -> require at least 500 Gaussians and two independent source cameras
  -> retain sparse fill supports for report-only review
  -> render the exact accepted 3D supports into the matched cached RGB views
```

The preferred label array is opened read-only, hashed before and after the
audit, and never copied into an output label array. The spatial-core overlap
decision is not rerun: this stage consumes only the already-exclusive sparse
supports from the reviewed spatial-core report. After subtracting preferred
labels, only the global residual-size and independent-camera gates remain.
The scale-adaptive voxel rule keeps the same `4.0` multiplier and `0.01` to
`0.20` bounds used by the source audit.

This audit is also permanently report-only. It writes an exact incremental
fill mask, an unlabeled residual-candidate mask, an excluded-preferred-label
mask, compressed sparse residual supports, a JSON report, exact matched-view
mask renders, semantic-palette overlays, and contact sheets. Class colors are
constructed in memory directly from the sparse supports. No semantic label
array, project-class array, label map, or PLY is written.

The exact mask renders, unlike the source spatial-core proposal overlays, show
only accepted residual Gaussians after immutable-label subtraction and spatial
re-splitting. A separate fresh v3 fill-only materializer may be considered
only after both scene audits pass visual review. Such a materializer must
preserve every nonzero v2 label and fill only v2-unlabeled Gaussians.
