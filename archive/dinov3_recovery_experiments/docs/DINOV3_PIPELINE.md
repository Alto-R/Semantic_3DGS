# DINOv3 ADE20K Pipeline

Status: archived research history. This is the original full research
pipeline document, including rejected routes. The maintained DINOv3 route is
documented in `docs/DINOV3_PIPELINE.md`; results and the acceptance rule are
in `docs/TASK1_DINOV3_RECOVERY_REPORT.md`.

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
- official aspect-preserving resize of the input's shorter side to the crop
  size before sliding inference;
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
support. Crop sizes must be divisible by 32 so every crop is compatible with
the convolutional and deformable-attention spatial layouts. Short-side resize
guarantees that sliding inference never sends a smaller, partially sized crop
to the adapter. The manifest records original and resized dimensions for every
view.

The full ViT-7B route has also been validated on a 48 GB RTX 8000
using float32, 512-pixel crops, and a 384-pixel stride. Those reduced settings
are evaluation settings, not a change to the model.

Shared-GPU diagnostics may set `DINOV3_MAX_CUDA_MEMORY_GIB` in any scheduler
that directly runs the DINOv3 adapter. The scheduler forwards the value as
`--max-cuda-memory-gib`; before model construction, the adapter converts the
requested GiB value to a fraction of the GPU's total visible memory and calls
PyTorch's per-process CUDA caching-allocator limit. The configured ceiling,
device capacity, allocator fraction, free memory before model construction,
and observed allocated/reserved peaks are recorded in `dinov3_manifest.json`.
The option is unset by default and never raises its own limit after an OOM.
It is a process-level allocator guard, not a hard hardware partition, so it
does not guarantee capacity for another process whose future allocations are
unknown.

The maintained scene scheduler can also wait for shared host and GPU resources
without holding a PyTorch CUDA context. Set
`DINOV3_MIN_HOST_AVAILABLE_GIB` and/or `DINOV3_MIN_GPU_FREE_GIB` to enable the
gate. It checks once before creating the output tree and again immediately
before loading DINOv3. `DINOV3_RESOURCE_WAIT_INTERVAL_SECONDS` controls the
poll interval; `DINOV3_RESOURCE_WAIT_TIMEOUT_SECONDS=0` waits indefinitely.
The GPU defaults to the first entry in `CUDA_VISIBLE_DEVICES`, or may be
selected explicitly with `DINOV3_RESOURCE_GPU_ID`. Every poll is logged as a
JSON resource snapshot. Passing a threshold is only an availability gate, not
a memory reservation: unrelated processes may allocate more memory after a
check passes.

For the shared 48 GB RTX 6000 one-view diagnostic, the approved settings are
an 80 GiB host-available threshold, a 45 GiB GPU-free threshold, a 60-second
poll interval, and the separate 35 GiB PyTorch allocator ceiling. The host
threshold protects the transient CPU checkpoint/model overlap; crop size does
not reduce that model-loading requirement.

`DINOV3_CHECKPOINT_LOAD_MODE=local_mmap` is an opt-in, low-host-memory
alternative for every maintained scheduler that directly invokes the DINOv3
adapter, including the scene, 3D-first, and boundary/identity routes. The
adapter first completes
its existing full SHA-256 verification, then temporarily intercepts only the
two verified local `file://` loads made by the pinned DINOv3 Hub entry and
uses `torch.load(..., map_location="cpu", weights_only=True, mmap=True)`.
DINOv3's existing `load_state_dict` calls, strict backbone key check, and
Mask2Former missing/unexpected-key assertions remain unchanged. The original
PyTorch Hub loader is restored even when model construction raises, the
external DINOv3 worktree is not modified, and nonlocal or unrelated checkpoint
loads are delegated to the standard loader. The manifest records the mode and
the exact intercepted paths.

The mmap mode removes the eager in-memory checkpoint-storage copy, but it does
not reduce the model's parameter count or GPU requirement. It is disabled by
default. Before inference, validate it on the target PyTorch build with a real
checkpoint mmap probe and a model-load-only canary; derive a lower host-RAM
wait threshold from the observed peak rather than assuming a fixed reduction.

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

## All-camera Gaussian visibility audit

Before selecting more semantic cameras or changing fusion, use
`scripts/slurm/slurm_task1_dinov3_all_camera_visibility_scene.sbatch` to
measure whether every Gaussian can receive direct pixel evidence from the
real reconstruction cameras:

```text
every camera in cameras.json
  -> render one all-zero FlashSplat mask
  -> sum used_count over the mask rows
  -> mark every Gaussian with support above the global threshold
  -> retain the complete camera-by-Gaussian visibility matrix
  -> report zero-view, one-view, and multiview coverage
```

The default threshold is exactly `0.0`, so any positive contribution counts.
This definition deliberately matches the lifting mechanism used by dense
DINOv3 fusion: a Gaussian marked visible in a camera can receive dense pixel
evidence from that camera. The audit does not equate projection into the
camera frustum with evidence when the Gaussian makes no rasterized
contribution.

The output includes per-Gaussian view counts, total and maximum support, first
and strongest camera indices, explicit zero- and one-view index arrays, and a
packed Boolean camera-by-Gaussian visibility matrix. The packed matrix is kept
so a later camera set-cover stage can choose cameras from measured Gaussian
coverage without rerendering or consulting semantic labels.

This stage runs no DINO model, reads no prior semantic labels, and writes no
semantic labels, label map, or PLY. It uses every real reconstruction camera
in `cameras.json` order; a small evenly spaced subset is not an equivalent
audit. Gaussians with zero support across all real cameras must not later be
described as directly DINO-observed. They require a separately reported
artifact-removal or spatial-propagation policy.

Example:

```bash
SCENE=playroom \
OUTPUT_NAME=playroom_dinov3_all_camera_visibility_v1 \
GAUSSIAN_ENV=semantic_3dgs_renderer \
RENDER_MAX_WIDTH=960 \
VISIBILITY_SUPPORT_THRESHOLD=0.0 \
bash scripts/slurm/slurm_task1_dinov3_all_camera_visibility_scene.sbatch
```

## Automatic visibility camera selection

Use
`scripts/slurm/slurm_task1_dinov3_visibility_camera_selection_scene.sbatch`
after a completed all-camera visibility audit. This CPU-only report consumes
the packed camera-by-Gaussian matrix without rerendering any view:

```text
globally observable Gaussians with no selected camera
  -> greedily select the camera with the largest new single-view coverage
  -> break equal gains by new second-view coverage
  -> break any remaining tie by cameras.json row order
after complete single-view coverage
  -> greedily maximize second-view coverage of multiview-capable Gaussians
  -> retain every remaining camera in source order as a zero-gain suffix
```

The output ranks all source cameras and records the exact coverage curve after
every prefix. It reports the first prefix reaching 90%, 95%, 99%, and 100%
single-view coverage of the globally observable set; 90%, 95%, and 99%
two-view coverage are reported both against all observable Gaussians and
against only Gaussians that are globally capable of two-view coverage. The
two denominators remain separate because a Gaussian visible in only one of
all reconstruction cameras cannot ever receive a second independent view.

The selector validates the packed padding and reconstructs the exact saved
per-Gaussian view counts before ranking. It hashes every input before and
after selection, uses no semantic labels or scene/class rules, and does not
choose a final DINOv3 camera budget. The report contains the complete ordered
camera list so a later reviewed run can use any measured prefix without
manual camera picking.

This stage runs no renderer or DINO model and writes no semantic label array,
label map, or PLY. Example:

```bash
SCENE=playroom \
VISIBILITY_OUTPUT_NAME=playroom_dinov3_all_camera_visibility_v1 \
OUTPUT_NAME=playroom_dinov3_visibility_camera_selection_v1 \
GAUSSIAN_ENV=semantic_3dgs_renderer \
bash scripts/slurm/slurm_task1_dinov3_visibility_camera_selection_scene.sbatch
```

## Threshold-selected DINOv3 view cache

Use
`scripts/slurm/slurm_task1_dinov3_selected_view_cache_scene.sbatch` after the
automatic visibility selector has completed. This stage turns a measured
coverage target into a reusable 2D DINOv3 cache without introducing a manual
camera list:

```text
saved deterministic full-camera ranking and coverage curve
  -> validate every ranking array and selection-report step
  -> read the first prefix reaching 99% two-view coverage
     of globally multiview-capable Gaussians
  -> validate that the prefix came from the 960-width visibility audit
  -> render those exact cameras in ranked order at width 960
  -> run DINOv3 ADE20K Mask2Former at 512/384, bfloat16, local mmap
  -> validate the exact rendered and segmented camera order
  -> stop before FlashSplat vote lifting, fusion, labels, or PLY output
```

The scheduler rejects `CAMERA_INDICES` and `VIEW_COUNT`; both are derived from
the saved threshold record. It also checks that the threshold prefix matches
the complete ranking, that the saved coverage curve reaches the reported
count for the first time at that prefix, and that `cameras.json` still has the
same camera count. The output includes the exact selected indices and a
provenance report under `selection/`.

For the completed Playroom 960-width visibility selection, this global rule
resolves to 57 cameras. The number 57 is an observed result, not a hardcoded
budget: a different valid scene or coverage curve may resolve to a different
prefix. This cache still does not claim direct evidence for globally
single-view or zero-view Gaussians.

Example configuration check followed by execution:

```bash
SCENE=playroom \
CAMERA_SELECTION_OUTPUT_NAME=playroom_dinov3_visibility_camera_selection_960_v1 \
OUTPUT_NAME=playroom_dinov3_selected_99pct_two_view_cache_512_v1 \
GAUSSIAN_ENV=semantic_3dgs_renderer \
DINOV3_ENV=dinov3_semantic \
CONFIG_ONLY=1 \
bash scripts/slurm/slurm_task1_dinov3_selected_view_cache_scene.sbatch

SCENE=playroom \
CAMERA_SELECTION_OUTPUT_NAME=playroom_dinov3_visibility_camera_selection_960_v1 \
OUTPUT_NAME=playroom_dinov3_selected_99pct_two_view_cache_512_v1 \
GAUSSIAN_ENV=semantic_3dgs_renderer \
DINOV3_ENV=dinov3_semantic \
CONFIG_ONLY=0 \
bash scripts/slurm/slurm_task1_dinov3_selected_view_cache_scene.sbatch
```

The fixed 40 GiB logical allocator ceiling is above the measured 512-pixel
one-view peak on the rental 48 GB RTX 4090 and below the device capacity. This
is a PyTorch per-process allocator limit, not a hardware reservation. Use a
fresh output name for every run; this stage never resets an existing output.

## Cached 2D-to-3D round-trip fidelity audit

Use
`scripts/slurm/slurm_task1_dinov3_round_trip_fidelity_audit_scene.sbatch`
after the threshold-selected DINOv3 cache has passed its final validation.
This stage measures whether the cached 2D identities survive an independent
3D lift; it does not try to create a new accepted semantic model:

```text
validated automatic selected-view DINOv3 cache
  -> rerun FlashSplat lifting only; do not rerun DINOv3
  -> normalize every camera/Gaussian distribution to one unique class winner
     (exact per-camera ties abstain)
  -> give each camera exactly one identity vote per Gaussian
  -> exclude one camera completely
  -> require at least two remaining cameras, one unique class, and a strict
     camera majority; all other Gaussians abstain
  -> render that temporary 3D consensus into the held-out camera
  -> compare with the held-out DINOv3 map
  -> repeat for every cached camera
```

The camera exclusion is genuine leave-one-camera-out evaluation: the target
camera's lifted identity is removed before the 3D consensus is computed. A
large weight in one camera can select that camera's local identity, but it
cannot outweigh other cameras because the cross-camera vote is always one.
Zero-camera, single-camera, tied, and non-majority Gaussian populations are
reported separately.

Pixel projection uses eight binary per-Gaussian project-ID feature renders.
The output reports projection coverage, agreement on projected pixels, the
full class-confusion matrix, per-class agreement, and separate four-connected
DINOv3 boundary and interior agreement. It also writes per-camera masks,
overlays, red/green disagreement views, contact sheets, and a Gaussian
agreement diagnostic containing only camera counts, winner count/share, and
an abstention status. The temporary leave-one-out class arrays stay in memory.

The scheduler rejects manual camera lists and tunable consensus thresholds.
It writes no accepted Gaussian labels, Gaussian project-class array, label
map, or PLY. Example:

```bash
SCENE=playroom \
SOURCE_CACHE_OUTPUT_NAME=playroom_dinov3_selected_99pct_two_view_cache_512_v1 \
OUTPUT_NAME=playroom_dinov3_round_trip_fidelity_audit_v1 \
GAUSSIAN_ENV=semantic_3dgs_renderer \
CONFIG_ONLY=1 \
bash scripts/slurm/slurm_task1_dinov3_round_trip_fidelity_audit_scene.sbatch

SCENE=playroom \
SOURCE_CACHE_OUTPUT_NAME=playroom_dinov3_selected_99pct_two_view_cache_512_v1 \
OUTPUT_NAME=playroom_dinov3_round_trip_fidelity_audit_v1 \
GAUSSIAN_ENV=semantic_3dgs_renderer \
CONFIG_ONLY=0 \
bash scripts/slurm/slurm_task1_dinov3_round_trip_fidelity_audit_scene.sbatch
```

## Cached same-camera FlashSplat self-round-trip audit

Use
`scripts/slurm/slurm_task1_dinov3_self_round_trip_fidelity_audit_scene.sbatch`
after the cached leave-one-camera-out audit has produced its complete dense
FlashSplat vote manifest. This diagnostic isolates the per-camera lift and
reprojection path from cross-camera semantic fusion:

```text
one cached DINOv3 hard class map
  -> reuse that camera's existing normalized FlashSplat Gaussian class mass
  -> select one unique class winner per Gaussian; exact ties abstain
  -> render those identities immediately back into the same camera
  -> compare with that same camera's cached DINOv3 map
  -> repeat independently for every automatically selected camera
```

The stage reports same-camera projection coverage and agreement, separate
boundary and interior agreement, the per-camera Gaussian winning-mass
distribution, binary project-ID decoding margins, full class confusion, and
per-camera masks and contact sheets. It performs no cross-camera consensus.
These measurements distinguish three cases:

- low Gaussian winning mass indicates mixed semantic ownership during dense
  lifting or per-Gaussian hardening;
- high winning mass but low same-camera agreement indicates a lift/reprojection
  attribution problem;
- high same-camera agreement but low leave-one-camera-out agreement indicates
  that cross-view DINOv3 identity inconsistency and fusion are dominant.

This audit measures the existing hard-argmax path; it does not measure a future
full-probability lift. It reruns neither DINOv3 nor FlashSplat lifting and
requires both source output names so it cannot silently consume unrelated
evidence. Manual camera lists and diagnostic thresholds are rejected. It
writes no accepted Gaussian labels, Gaussian project-class array, label map,
or PLY. Example:

```bash
SCENE=playroom \
SOURCE_CACHE_OUTPUT_NAME=playroom_dinov3_selected_99pct_two_view_cache_512_v1 \
SOURCE_VOTE_OUTPUT_NAME=playroom_dinov3_round_trip_fidelity_audit_v1 \
OUTPUT_NAME=playroom_dinov3_self_round_trip_fidelity_audit_v1 \
GAUSSIAN_ENV=semantic_3dgs_renderer \
CONFIG_ONLY=1 \
bash scripts/slurm/slurm_task1_dinov3_self_round_trip_fidelity_audit_scene.sbatch

SCENE=playroom \
SOURCE_CACHE_OUTPUT_NAME=playroom_dinov3_selected_99pct_two_view_cache_512_v1 \
SOURCE_VOTE_OUTPUT_NAME=playroom_dinov3_round_trip_fidelity_audit_v1 \
OUTPUT_NAME=playroom_dinov3_self_round_trip_fidelity_audit_v1 \
GAUSSIAN_ENV=semantic_3dgs_renderer \
CONFIG_ONLY=0 \
bash scripts/slurm/slurm_task1_dinov3_self_round_trip_fidelity_audit_scene.sbatch
```

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

## Core-first semantic-identity audit

Use
`scripts/slurm/slurm_task1_dinov3_core_first_semantic_identity_audit_scene.sbatch`
to test the opposite order on every cached 70-view proposal graph, including
graphs whose whole-component identity was mixed or unstable:

```text
cached class-agnostic proposal graph
  -> one presence vote per camera and Gaussian
  -> retain support present in at least two cameras
  -> split into adaptive 26-neighbor voxel cores
  -> find the independent camera proposals intersecting each core
  -> fuse DINOv3 identity independently inside that core
  -> if exactly one stable outlier exists, remove it
  -> recompute multiview support and spatially re-split
  -> require at least 500 Gaussians and two independent cameras
  -> resolve cross-class overlap by a unique maximum camera count
  -> equal cross-class counts abstain
```

This differs from the earlier multiview spatial-core audit at the decisive
step: no source graph is rejected for mixed identity before its first spatial
split. The report separately counts how many mixed source graphs produce
accepted per-core identities and how many Gaussians they recover.

All thresholds are global and class-neutral. The stage reuses cached DINOv3
query probabilities and FlashSplat proposal supports, performs no inference or
lifting rerun, and makes no scene, class, or component decisions. It writes an
audit report, boolean and ownership masks, compressed sparse accepted supports,
and exact matched-view masks and overlays. It writes no semantic labels,
project-class arrays, label map, or PLY and is not a materialization path.

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

The completed audits retained 234,908 Dr. Johnson and 160,674 Playroom
Gaussians after residual re-splitting. Their exact projections showed that a
blanket fill was still unsafe: Dr. Johnson painting components included broad
surrounding-wall halos, while Playroom had a broad door halo and one hanging
stair item classified as lamp. The report-only audit therefore remains
unmaterialized as a whole.

## Automatic class-consistent anchor guard

Do not convert exact visual-QA findings into scene names, component IDs, class
exceptions, or manual fill decisions. The next cached-evidence stage is a
report-only automatic anchor-guard audit:

```text
accepted incremental residual supports
  + immutable preferred-v2 labels and 3D coordinates
  -> query nearby preferred nonzero anchors for every proposed fill Gaussian
  -> require enough same-class anchors
  -> require a global same-class-neighbor fraction
  -> require the nearest same-class anchor to beat the nearest competing class
  -> re-split retained support spatially
  -> reapply global size and independent-camera gates
  -> render exact matched-view masks and overlays for global radius profiles
```

The profiles vary only the radius as a global multiple of the source
component's adaptive voxel size. Neighbor count, same-class fraction, distance
competition, retained-component size, and independent-camera gates are shared
across scenes and classes. The audit writes no semantic labels, label map, or
PLY. Its purpose is to determine whether local immutable-label evidence can
automatically remove wall halos and abstain on unsupported objects while
retaining useful fills.

No candidate v3 may be materialized until one global profile passes both
scenes. Per-scene profile selection is also prohibited. Completely unanchored
objects are intentionally abstained by this incremental route; recovering
them must be evaluated through the separate automatic core-first
semantic-identity audit, not a manual exception.

## Dense per-Gaussian core cross-validation

Exact projections from the core-first and automatic-anchor audits showed that
spatially coherent support can still carry systematic 2D semantic leakage
across painting/wall, wall/ceiling, door/wall, railing, floor, and stair
boundaries. The next gate therefore checks the proposed core identity at every
currently unlabeled Gaussian using independent dense camera evidence:

```text
core-first sparse support
  + immutable preferred-v2 labels
  -> remove every Gaussian already labeled by preferred v2
  -> lift cached dense DINOv3 class evidence with explicit abstention
  -> choose at most one reliable semantic winner per camera and Gaussian
  -> require the pooled dense winner to match the core proposal
  -> require a strict majority of reliable camera winners
  -> require global winner, runner-up, and boundary-class margins
  -> spatially re-split only after semantic validation
  -> require global component-size and independent-camera support
  -> render exact masks and overlays
```

The original
`scripts/slurm/slurm_task1_dinov3_core_first_dense_cross_validation_audit_scene.sbatch`
used three global profiles with fixed absolute thresholds for relative margin,
maximum softmax probability, and normalized entropy confidence. Those
thresholds were not calibrated to the released DINOv3 Mask2Former head:
Playroom retained no pixels and Dr. Johnson retained only negligible evidence,
so every profile produced zero validated Gaussians. That scheduler remains
available only to reproduce the rejected diagnostic and is not the current
route.

## Joint empirical dense-confidence calibration

Use
`scripts/slurm/slurm_task1_dinov3_core_first_calibrated_dense_cross_validation_audit_scene.sbatch`
for the automatic successor. Before lifting either target, it streams all
cached pixels from both complete 70-view scene manifests and builds one joint
empirical CDF for each stored metric:

- relative top-1/top-2 margin;
- maximum softmax probability;
- normalized entropy confidence.

Each pixel receives the minimum of its three empirical percentile ranks. This
weakest-metric score prevents one unusually favorable metric from hiding weak
support in another metric. Four class-neutral nested profiles are then derived
from the joint score distribution:

| Profile | Target joint pixel retention |
|---|---:|
| `baseline` | 100% |
| `permissive` | 50% |
| `balanced` | 25% |
| `strict` | 10% |

These are global retained quantiles, not manually chosen probability cutoffs,
scene-specific tuning, or class-specific thresholds. Calibration reads the
existing complete confidence maps and does not rerun DINOv3 inference.

For each camera, the lift encodes the confidence level together with the
project class. Because FlashSplat accumulation is linear, all four nested
profiles are reconstructed from one render per camera. Pixels below a
profile's level remain explicit abstention mass rather than disappearing
through semantic renormalization.

Camera validation separates two questions that the fixed-threshold audit had
combined:

1. accepted semantic coverage must exceed an automatic floor equal to half the
   profile's actual joint retained ratio;
2. winner share and winner margin are measured within the accepted semantic
   mass.

Pooled cross-camera winner share remains normalized by pooled semantic mass.
The proposed class must also pass strict camera majority, pooled runner-up
margin, and wall/ceiling/floor competitor-margin gates. Only then is support
spatially re-split and subjected to the shared size and camera-count rules.

The calibrated scheduler writes calibration histograms, per-profile sparse
votes, reason masks, reports, exact masks and overlays, and a comparison
summary. It explicitly disables automatic profile selection. Preferred-v2
labels are read-only, and the stage writes no semantic label array,
project-class array, label map, or PLY.

This calibrated route is prepared for report-only evaluation. It is not an
accepted materialization path. No profile may be selected or converted into a
new label output until the same global behavior has passed exact visual review
across both calibration scenes.

## Probability-preserving round-trip audit

The hard-map same-camera audit reconstructs cached DINOv3 interiors at
98.8720%, while the hard leave-one-camera-out audit reaches only 73.8705%.
This isolates the dominant loss to camera-dependent hard identities and their
subsequent cross-camera fusion.  Use
`scripts/slurm/slurm_task1_dinov3_soft_probability_round_trip_scene.sbatch`
to test the next automatic alternative:

```text
automatic 99% two-view visibility prefix
  -> rerun the identical 512/384 DINOv3 inference
  -> retain every 150-class pixel probability as float16
  -> deterministically estimate probability-weighted FlashSplat support
  -> normalize one full class distribution per camera and Gaussian
  -> exclude each target camera in turn
  -> sum the remaining equal-camera distributions
  -> argmax only after fusion; exact ties and fewer than two cameras abstain
  -> render into the held-out camera and compare with its DINOv3 hard map
```

FlashSplat's public mask API accepts one integer class per pixel; it does not
accept a weighted class vector.  The lift therefore uses a fixed eight-sample
circular stratified categorical estimator.  Its expectation equals the full
pixel distribution.  A fixed integer hash supplies a reproducible per-pixel
phase, and every camera report includes first-four-versus-all-eight winner
agreement as a convergence diagnostic.  The external FlashSplat checkout is
not modified.

The new probability run is also compared pixel-for-pixel with the previously
completed automatic hard cache.  The comparison records every hard-class
difference and hashes both manifests; it never changes either cache or applies
a correction.

If execution stops after that comparison but before the soft FlashSplat lift
creates its output directory, set `RESUME_FROM_SOFT_LIFT=1` and reuse the same
`OUTPUT_NAME`.  Resume mode does not rerun rendering or DINOv3.  Before stage
7 it revalidates the automatic camera prefix, both manifests, every segment
map, every complete probability tensor, the saved camera-index array, and the
hard-cache comparison without rewriting them.  It refuses to continue if a
partial or completed soft-vote or round-trip directory already exists; use a
new output name instead of deleting or mixing partial evidence.

```bash
SCENE=playroom \
CAMERA_SELECTION_OUTPUT_NAME=playroom_dinov3_visibility_camera_selection_960_v1 \
SOURCE_CACHE_OUTPUT_NAME=playroom_dinov3_selected_99pct_two_view_cache_512_v1 \
OUTPUT_NAME=playroom_dinov3_soft_probability_round_trip_v1 \
GAUSSIAN_ENV=semantic_3dgs_renderer \
DINOV3_ENV=dinov3_semantic \
RESUME_FROM_SOFT_LIFT=1 \
CONFIG_ONLY=0 \
bash scripts/slurm/slurm_task1_dinov3_soft_probability_round_trip_scene.sbatch
```

If stage 7 completes but stage 8 fails before its final report, use
`RESUME_FROM_SOFT_AUDIT=1` with the same output name.  This mode does not
rerun DINOv3 or the eight-sample FlashSplat probability lift.  Stage 8
revalidates the soft-vote manifest and every per-camera Gaussian probability
distribution while rebuilding the equal-camera evidence.  The failed primary
stage-8 directory and log remain untouched.  Recovery writes to the fixed
`stages/03_soft_probability_round_trip_retry_v1` directory and a separate
retry log, then creates the normal soft contact sheets from that validated
retry.  The two resume flags are mutually exclusive.

```bash
SCENE=playroom \
CAMERA_SELECTION_OUTPUT_NAME=playroom_dinov3_visibility_camera_selection_960_v1 \
SOURCE_CACHE_OUTPUT_NAME=playroom_dinov3_selected_99pct_two_view_cache_512_v1 \
OUTPUT_NAME=playroom_dinov3_soft_probability_round_trip_v1 \
GAUSSIAN_ENV=semantic_3dgs_renderer \
DINOV3_ENV=dinov3_semantic \
RESUME_FROM_SOFT_AUDIT=1 \
CONFIG_ONLY=0 \
bash scripts/slurm/slurm_task1_dinov3_soft_probability_round_trip_scene.sbatch
```

The scheduler fixes the successful 48 GB rental settings: 960-pixel renders,
512 crop, 384 stride, bfloat16, local-mmap checkpoint loading, and a 40 GiB
PyTorch allocator ceiling.  It defaults to the rental renderer environment
`semantic_3dgs_renderer`.  Camera selection, sample count, confidence gates,
majority thresholds, and class-specific rules cannot be overridden.  The
complete probability cache is large (approximately 10.4 GB for the completed
57-view Playroom run), so it is stored only when explicitly requested by this
audit.

This remains a diagnostic, not a label-production path.  It writes the
probability cache, per-camera soft Gaussian distributions, held-out masks,
confusion matrix, overlays, contact sheets, and provenance reports.  It writes
no accepted Gaussian label array, project-class array, label map, or PLY.

## Automatic calibrated probability fusion

The completed Playroom plain-soft audit reached 68.3404% held-out agreement,
69.1292% interior agreement, and 34.0550% boundary agreement.  All three are
worse than the corresponding hard-vote results (73.8705%, 74.5391%, and
44.9340%).  Equal-camera summation of all 150 lifted probabilities is therefore
rejected: weak probability tails accumulate across cameras and can overwhelm
stronger local identities.

Use
`scripts/slurm/slurm_task1_dinov3_calibrated_probability_scene.sbatch` for the
next report-only experiment.  It consumes the existing soft Gaussian vote
cache and evaluates a fixed, scene-neutral family of 18 policies containing:

- the rejected full-soft policy and a top-1 anchor;
- global temperatures 1.0, 0.75, and 0.5;
- top-k retention for k=1, 2, 3, and 5, retaining all exact kth-place ties;
- uniform, top-1/top-2-margin, and normalized-entropy camera/Gaussian weights.

Candidate values cannot be supplied through the scheduler.  Each candidate is
evaluated by leaving one camera out, predicting that camera's unique local
Gaussian winner from the remaining cameras, and counting every eligible
observation in the denominator.  Abstention therefore lowers the selection
score instead of artificially improving agreement.  The winner maximizes
correct predictions over all policy-independent eligible observations, then
prediction coverage, then the documented fixed candidate order.

Playroom is the development scene (`POLICY_MODE=select`).  The scheduler saves
the complete candidate table and automatically selected policy, then performs
one full pixel-space round-trip audit for that winner.  DrJohnson is the locked
external validation scene (`POLICY_MODE=validate`): it copies the Playroom
policy report into its own output and applies that exact policy without
retuning.  This is a global train/validation split, not per-scene policy
selection.

If a scene does not yet have a soft Gaussian vote cache, run the existing soft
probability scheduler with `STOP_AFTER_SOFT_LIFT=1`.  It completes and validates
the probability inference and eight-stratum FlashSplat lift, then stops before
the already-rejected plain-soft pixel audit.  It writes no semantic labels or
PLY.

After both calibrated audits and both matching hard-vote audits exist, run
`scripts.task1.dinov3.compare_calibrated_probability_scenes`.  Its fixed gate
accepts the calibrated policy only when the same policy is used on both scenes,
overall/interior/boundary agreement does not regress on either scene, and at
least one scene has a strict overall improvement.  Otherwise its decision is
`retain_hard_vote_baseline`.

The calibrated route remains report-only.  It writes the policy sweep, final
held-out report, confusion matrix, overlays, disagreement images, contact
sheets, and provenance.  It never writes an accepted Gaussian label array,
project-class array, label map, or semantic PLY.  Real Playroom and DrJohnson
execution is required before the policy can be accepted.

### Audited hard-vote inspection PLYs

The report-only hard-vote audit can be materialized for visual inspection with
`scripts/slurm/slurm_task1_dinov3_hard_vote_materialize_scene.sbatch`. The
materializer reuses the audit's exact camera-collapse and global consensus
functions, verifies the recorded vote-manifest and ontology hashes, and
requires its consensus status counts to reproduce the audit exactly.

The output contains two PLY files:

- `deliverables/semantic_point_cloud.ply` preserves the trained Gaussian data
  and adds an integer `label` vertex property.
- `visualizations/semantic_point_cloud_colored.ply` bakes the project-class
  palette into the spherical-harmonic DC color fields for direct viewing in
  SuperSplat and similar viewers.

This is an inspection materialization, not an automatic completion stage.
Only Gaussians with at least two camera identities and a unique strict majority
receive a semantic label. Unobserved, single-camera, tied, and non-majority
Gaussians remain label `0`; no fill, propagation, scene-specific rule, or
manual camera/class/Gaussian selection is used.

## Automatic detected-abstention recovery

Use
`scripts/slurm/slurm_task1_dinov3_detected_abstention_recovery_scene.sbatch`
to test additional evidence for the detected label-zero groups without using
v5, DINOv2, or a manual camera/class/Gaussian selection. The source hard-vote
audit remains immutable: every original accepted strict-majority identity is
locked and cannot be changed by additional cameras.

The selector reads the saved all-camera FlashSplat visibility matrix. For each
single-camera, exact-tie, or no-strict-majority Gaussian, it computes the
larger of the two-camera deficit and the exact strict-majority deficit. It then
selects unused cameras by newly fulfillable evidence units. Global camera-pose
novelty and source camera order are deterministic tie-breaks. Selection stops
at the fixed 99% coverage of all requirements that unused cameras can
geometrically fulfill. There is no external `CAMERA_INDICES` or `VIEW_COUNT`
input.

Only the selected additional cameras run DINOv3. Their dense hard maps are
lifted with the existing complete FlashSplat vote path. Recovery is evaluated
in two ordered tiers:

1. Apply the original equal-camera, unique strict-majority rule to the new
   combined evidence, but only for originally rejected detected Gaussians.
2. For remaining detected abstentions, estimate each baseline camera from its
   leave-one-camera-out agreement and each additional camera from agreement
   with the immutable anchors. The conservative 95% Wilson lower bound is the
   camera weight. A secondary diagnostic candidate is allowed only with at
   least two camera winners and a unique weighted strict majority after
   multiplying camera reliability by within-camera winning mass.

The result remains report-only. It writes a clearly named diagnostic candidate
array and recovery-source codes so the next held-out audit can be performed,
but it writes no accepted label array, label map, or PLY. The selector and
resolver contain no scene-specific or class-specific rules.

Run the same scheduler independently for Playroom and DrJohnson. After both
reports exist, `scripts.task1.dinov3.compare_detected_abstention_scenes`
verifies the paired report contracts and summarizes recovery. It can only mark
the pair ready for the next held-out round-trip validation; it never accepts
materialization. Promotion remains blocked until candidate projections do not
regress overall, interior, or boundary agreement in either scene.

## Calibrated recovery audit for camera-observed black Gaussians

Use
`scripts/slurm/slurm_task1_dinov3_observed_black_calibrated_spatial_audit_scene.sbatch`
after the hard-vote audit and additional-view recovery already exist. This is
a cache-only diagnostic: it reruns neither DINOv3 nor FlashSplat lifting. It
examines every Gaussian that remains label zero in the recovery candidate,
including Gaussians that were originally unobserved but gained evidence from
the additional cameras.

The audit deliberately keeps zero-camera and one-camera Gaussians black. A
multicamera candidate must have a unique reliability-weighted strict majority,
and its raw camera winner must agree with the weighted identity. Reliability
is then checked at three independent levels:

- class-wide leave-one-camera-out recovery of immutable hard anchors;
- class plus semantic-camera-count calibration;
- winner-share, runner-up-margin, and 3D boundary/interior calibration.

Every calibration gate uses a 95% Wilson lower bound. The original hard-audit
held-out confusion matrix adds recall, precision, and incoming-confusion gates
for every class. A class that is systematically confused with another class
therefore cannot be hidden by strong aggregate accuracy. In particular, a
catastrophically unreliable source class blocks target classes that absorb a
substantial fraction of its held-out pixels.

Spatial evidence is corroborative only. The semantic cameras select the class
before any neighbor query. Nearby immutable anchors may support that same
class, and same-class semantic candidates are split into automatic voxel
components. Proximity never invents or changes a class. Ontology `stuff`
classes such as wall, floor, and ceiling receive stronger share, margin,
entropy, calibration, and same-class-interior requirements; spatial support
cannot relax those semantic thresholds.

Outputs are one JSON report and one compressed diagnostic archive containing
only the currently black Gaussian indices, evidence statistics, calibrated
lower bounds, neighbor/component measurements, and rejection or eligibility
reason codes. No accepted Gaussian label array, full project-class array,
label map, or PLY is written.

Run the fixed audit independently for Playroom and DrJohnson, then use
`scripts.task1.dinov3.compare_observed_black_calibrated_spatial_scenes`. The
comparator verifies that both scenes used the identical common policy and can
only mark the pair ready for a new held-out round-trip validation. It never
accepts materialization directly.

## Class-agnostic component-graph audit for camera-observed black Gaussians

Use
`scripts/slurm/slurm_task1_dinov3_observed_black_component_graph_audit_scene.sbatch`
to audit the same cache inputs with object-level grouping. Like the calibrated
spatial audit, it is cache-only and report-only: it reruns neither DINOv3 nor
FlashSplat lifting and writes no accepted labels, full class array, label map,
or PLY.

The audit first forms conservative components from a sparse mutual-kNN graph.
Its common edge score combines 3D distance, spherical-harmonic DC appearance,
Gaussian scale and orientation when available, camera co-visibility, and the
similarity of each Gaussian's complete camera-evidence distribution. Class
names do not define graph edges or thresholds. Immutable accepted anchors
calibrate one shared edge threshold by measuring same-identity versus
cross-identity anchor pairs; the threshold is never adjusted per class or
scene.

Only after grouping does the audit aggregate normalized evidence, with at most
one semantic vote per camera per component. Candidate labels come exclusively
from those camera votes. The component must have a unique weighted strict
majority that agrees with its raw camera majority. Nearby immutable anchors
can corroborate the camera-supported class through support normalized by
component size, but proximity cannot invent or change a class. Component
scores combine semantic consensus, node-level fit, internal graph affinity,
normalized anchor support, and a soft held-out class-reliability weight;
boundary pressure subtracts from the score. No global class veto is applied.

Existing accepted hard-vote Gaussians calibrate a common component score after
one deterministic observing baseline camera is removed from each sampled
component. This permits a one-camera Gaussian to gain evidence only when it
belongs to a cohesive, independently multicamera component; zero-camera
Gaussians always remain black. Broad surfaces gain no population advantage
because anchor support is averaged within each component.

The output is one JSON report and one compressed diagnostic archive containing
only current-black indices, graph/component diagnostics, candidate classes,
and report-only decision codes. Run the same policy for both Playroom and
DrJohnson, then use
`scripts.task1.dinov3.compare_observed_black_component_graph_scenes`. Its
positive result only authorizes a new paired held-out round-trip validation;
it never authorizes materialization.

## Leakage-controlled component-graph round-trip validation

Use
`scripts/slurm/slurm_task1_dinov3_observed_black_component_graph_round_trip_validation_scene.sbatch`
only after the component-graph audit report and diagnostic archive exist for
the scene. The validator is cache-only and report-only: it reruns neither
DINOv3 inference nor FlashSplat lifting and writes no accepted labels, full
class array, label map, or PLY.

For each original baseline camera, the validation fold performs all of the
following before comparing with that camera's cached DINOv3 map:

1. Remove the held-out camera from the original hard-vote baseline.
2. Remove it from the combined baseline-plus-additional camera evidence.
3. Recompute every remaining camera's reliability without the held-out
   camera, using only anchors that remain accepted without that camera.
4. Rebuild the complete component-graph candidate from the remaining
   evidence: the semantic cache, the mutual-kNN graph edges, component
   votes, component scores, and report-only decisions.
5. Render the leave-one-out hard baseline and the rebuilt component
   candidate into the same held-out camera and measure projected coverage,
   overall, interior, and boundary agreement on the same pixel partition.

The audited graph edge threshold, component score threshold, distance scale,
feature scales, and the hard-audit held-out class-reliability matrix are
fixed common policy calibrated from immutable anchors; they are reused
unchanged in every fold. Zero-camera black Gaussians remain black, and the
fold candidate may only label current-black Gaussians that become eligible
from remaining-camera component evidence.

Before any fold runs, the validator reproduces the saved full-evidence
component-graph candidate exactly from the vote caches and verifies the
saved diagnostic archive: black indices, combined camera counts, observed
mask, component ids, candidate project ids, component camera counts,
component scores, and decision codes. This prevents the held-out report from
silently evaluating a different component-graph implementation. Outputs are
limited to the validation report, per-camera overlays/disagreement images,
and contact sheets.

The report adds per-class pixel recovery (source, baseline-agreed,
candidate-agreed, and newly recovered pixels per class) and per-class newly
resolved Gaussian counts aggregated over folds. Run the validator
independently for Playroom and DrJohnson, then run
`scripts.task1.dinov3.compare_observed_black_component_graph_round_trip_scenes`
on the two reports. The paired gate accepts the component candidates for
possible later materialization only when both scenes recover at least one
held-out black Gaussian and report per-class recovery, and neither scene
regresses in:

- projected coverage;
- overall agreement of projected pixels;
- interior agreement of projected pixels;
- boundary agreement of projected pixels.

Passing this gate does not itself write or approve a replacement PLY. A
separate reviewed materializer would still be required. If either scene fails
any condition, the component-graph candidates stay diagnostic only.

## Held-out fill-precision audit for black-spot fills

Use
`scripts/slurm/slurm_task1_dinov3_observed_black_fill_precision_audit_scene.sbatch`
after the component-graph round-trip validation reports exist. The whole-scene
round-trip agreement is dominated by the immutable anchors, so a few thousand
newly labeled black Gaussians barely move the pixel metric. This audit isolates
the fills: it reuses the exact leave-one-camera-out component-graph candidate
construction and renders ONLY the newly resolved black Gaussians into each
held-out camera.

Per camera it reports how many fill pixels project, how many of those pixels
have a held-out source class, how many agree with the fill class, how many are
wrong, and how many fall on unverifiable source-zero pixels. Per-class rows
report source pixels, agreed and wrong fill pixels, precision of source, and
newly resolved Gaussian counts. The report is measurement-only: it accepts no
labels, label map, or PLY, and the paired summary
`scripts.task1.dinov3.compare_observed_black_fill_precision_scenes` never
authorizes materialization by itself.

## Raw-vs-weighted winner disagreement diagnostic

Use
`scripts/slurm/slurm_task1_dinov3_observed_black_winner_disagreement_audit_scene.sbatch`
after the component-graph audit exists. The largest blocked population is
components whose raw camera winner and reliability-weighted winner disagree
(169,370 camera-observed black Gaussians in Playroom and 92,766 in
Dr. Johnson at the audited cache state). This cache-only, report-only
diagnostic rebuilds the same leave-one-camera-out component-graph candidate
and, for every conflict component, compares both winners with the held-out
camera's own normalized component vote.

Outputs include per-subtype counters (raw tied, raw-vs-weighted winner
disagreement, weighted not accepted), per-camera agreement counts for raw and
weighted winners, Gaussian-counted confusion matrices of raw and weighted
winners versus the held-out winner, a fill predicted-vs-source confusion
matrix, and fill precision totals. It reruns neither DINOv3 nor FlashSplat
lifting, writes no labels, label map, or PLY, and its paired summary
`scripts.task1.dinov3.compare_observed_black_winner_disagreement_scenes` is
measurement-only. A positive result would inform a common calibrated
disagreement-resolution rule, which would then need its own held-out
validation before any materialization.

## Automatic black-evidence camera expansion

The disagreement and fill-precision diagnostics showed that most remaining
camera-observed black Gaussians are ambiguous under the current view set, not
resolvable by a winner-rule change.  The evidence-backed direction is more
independent cameras.  Use
`scripts/slurm/slurm_task1_dinov3_black_evidence_expansion_scene.sbatch` after
the hard audit, all-camera visibility audit, and abstention recovery exist.

The expansion runs six automatic stages:

1. Select unused reconstruction cameras that add the most fulfillable
   evidence units to every Gaussian still black in the recovery candidate.
   Requirements are `max(two_camera_deficit, strict_majority_deficit)`,
   including zero-combined-camera black Gaussians, with global pose novelty
   as the deterministic tie-break and a fixed 99% coverage target.
2. Run DINOv3 on the selected cameras and cache their segment maps.
3. Lift the new dense hard votes with FlashSplat.
4. Merge the existing and new additional vote manifests into one cache with
   a merged automatic-selection report that satisfies the recovery contract.
5. Re-run the abstention recovery with the merged evidence, producing a new
   recovery candidate.
6. Rebuild the component-graph audit from the merged evidence and new
   recovery candidate.

Every stage is automatic and report-only: no manual camera, Gaussian, or
class selection, no labels, label map, or PLY.  After the expansion completes,
re-run the round-trip validation, fill-precision audit, and winner-disagreement
diagnostic against the new recovery and component-audit outputs; materialization
remains blocked until those held-out gates pass again.

## DINOv2 agreement-gate pipeline

The agreement gate adds an independent second evidence stream from the
completed DINOv2 separate-abstain all-views vote caches.  A camera's component
vote counts only when its DINOv3 component winner and its DINOv2 component
winner are the same nonzero class; either source abstaining, an exact
DINOv2 tie, or a disagreement abstains that camera's contribution.  DINOv2
never changes reliability calibration, anchors, or the component graph; only
which camera votes enter the component aggregation changes.  DINOv2 vote
weights are not normalized per Gaussian, so the second-source adapter computes
its own per-Gaussian winner and mass and never reuses the DINOv3 collapse
contract.

Use
`scripts/slurm/slurm_task1_dinov3_agreement_gate_pipeline_scene.sbatch` after
the hard audit, selected cache, and recovery outputs exist and after the
DINOv2 second-source votes have been transferred to
`outputs/eyenavgs_task1/<scene>_dinov2_separate_abstain_allviews_v1/stages/02_flashsplat_votes/vote_manifest.json`.
For one scene the pipeline runs, in order:

1. The component-graph audit with the agreement gate
   (`--dinov2-vote-manifest`), producing a fresh gated audit output.
2. The leave-one-camera-out round-trip validation against that gated audit;
   each fold removes both the held-out DINOv3 row and its DINOv2 row.
3. The fill-precision audit against the gated audit.
4. The winner-disagreement diagnostic against the gated audit.

Every stage is cache-only and report-only: no DINOv3/DINOv2 inference, no
FlashSplat lifting, no labels, label map, or PLY.  When both scenes have run,
set `PLAYROOM_ROUND_TRIP_OUTPUT_NAME` and `DRJOHNSON_ROUND_TRIP_OUTPUT_NAME`
on either scene invocation and the pipeline additionally writes the paired
round-trip, fill-precision, winner-disagreement, and agreement-gate
summaries.  The paired decision
`scripts.task1.dinov3.compare_observed_black_agreement_gate_scenes` reports
`accepted_for_materialization=true` only when both scenes used the gate, both
scenes pass the four held-out non-regression metrics with per-class recovery,
and both scenes' directly measured fill precision is at least 0.50.  Passing
the gate does not write or approve a replacement PLY; materialization still
requires a separate reviewed materializer.

## Leakage-controlled abstention round-trip validation

Use
`scripts/slurm/slurm_task1_dinov3_abstention_round_trip_validation_scene.sbatch`
only after the automatic recovery report, diagnostic candidate, and recovery
source codes exist. The validator reuses the cached baseline and additional
FlashSplat votes plus the cached DINOv3 maps; it reruns neither DINOv3 inference
nor vote lifting.

For each original baseline camera, the validation fold performs all of the
following before comparing with that camera's cached DINOv3 map:

1. Remove the held-out camera from the original hard-vote baseline.
2. Remove it from the baseline-plus-additional strict-majority evidence.
3. Recompute every remaining camera's reliability without the held-out camera.
   Baseline-camera calibration therefore uses a consensus that excludes both
   the target camera and the camera being calibrated. Additional-camera
   calibration uses only strict-majority anchors that remain after the target
   camera is removed.
4. Rebuild the two-tier recovery candidate from the remaining evidence, using
   strict majority first and the calibrated weighted rule only for unresolved
   detected abstentions.
5. Render the baseline and candidate into the same held-out camera and measure
   projected coverage plus overall, interior, and boundary agreement on the
   same pixel partition.

Before any fold runs, the validator also reproduces the saved full-evidence
diagnostic candidate and recovery-source arrays exactly from their vote caches.
This prevents the held-out report from silently evaluating a different
recovery implementation. Outputs are limited to the validation report,
per-camera overlays/disagreement images, and contact sheets. No accepted label
array, label map, or PLY is written.

Run the validator independently for Playroom and DrJohnson with their own
automatically selected caches, hard-vote audits, and recovery outputs. Then run
`scripts.task1.dinov3.compare_abstention_round_trip_scenes` on the two reports.
The paired gate accepts recovery for possible later materialization only when
both scenes recover at least one candidate and neither scene regresses in:

- projected coverage;
- overall agreement of projected pixels;
- interior agreement of projected pixels;
- boundary agreement of projected pixels.

Passing this gate does not itself write or approve a replacement PLY. A
separate reviewed materializer would still be required. If either scene fails
any condition, the original hard-vote materialization remains the accepted
inspection result and recovery stays diagnostic only.

## Gate-locked abstention recovery materialization

After the paired Playroom/DrJohnson gate reports
`accepted_for_materialization=true`, use
`scripts/slurm/slurm_task1_dinov3_abstention_recovery_materialize_scene.sbatch`
to export one scene at a time. The materializer refuses an unaccepted gate or
a scene that did not independently pass. It also verifies that the selected
scene's held-out report reproduced the recovery candidate and that the
candidate and recovery-source arrays still match their validated SHA-256
hashes.

Before writing labels, the materializer recomputes the original baseline
hard-camera consensus from the cached FlashSplat votes. Its consensus status
must reproduce both the original diagnostic status array and the hard-audit
status counts. Original strict-majority anchors must match the recovery
candidate exactly. Recovery source codes may add labels only to the original
single-camera, exact-tie, or no-strict-majority groups. Unobserved Gaussians
and all unresolved detected abstentions remain label `0`; there is no
zero-camera fill, spatial propagation, scene-specific rule, class-specific
rule, or manual camera/Gaussian selection.

The output contains the validated label arrays, recovery-source and original
consensus-status arrays, a provenance-rich summary and label map, the semantic
PLY, a class-colored SuperSplat PLY, and a color legend. Materialization does
not rerun DINOv3 or FlashSplat lifting.
