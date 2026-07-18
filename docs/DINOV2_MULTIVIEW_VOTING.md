# Task 1 Alternative Front-End: DINOv2 Multi-View Voting

This document specifies the DINOv2 semantic-labeling front-end for the EyeNavGS
3DGS scenes. Its default mode replaces the hand-written-vocabulary detector
with **prompt-free, dense DINOv2 semantic segmentation** and assigns one label
per Gaussian by **multi-view voting**. An optional continuous GroundingDINO/SAM
branch can enrich reviewed classes missing from ADE20K.

The output contract is unchanged: each Gaussian still receives one integer
`label`, and each scene still ships a `label_map.json`. This route only changes
how the 2D semantic evidence is produced and how it is fused into 3D. It is a
drop-in alternative to the `01_grounded_sam` and (optionally) `02_flashsplat`
stages, not a rewrite of the whole pipeline.

The implemented v1 scope is deliberately fixed: the official DINOv2 ViT-L/14
ADE20K **linear** head, all 150 ADE20K identities preserved, and only real
training cameras from `cameras.json`. Orbit views and gaze-hit coverage remain
deferred until their geometry can be validated.

## 1. Motivation

The current front-end detects objects only from a **hand-written class
vocabulary** (`configs/task1_semantic_classes.<scene>.json`). In large scenes
this causes a high unlabeled ratio (58%-71% in the accepted bicycle/room/truck
runs) for two structural reasons:

1. **Closed input vocabulary.** Anything not named in the config is never
   detected. Large scenes have a long tail of objects that are easy to forget.
2. **Detector recall and background stuff.** GroundingDINO returns only
   box-level "thing" detections; broad background surfaces (wall, floor, ground,
   sky, grass, road) are weakly covered and often left unlabeled.

DINOv2 with its released semantic-segmentation head addresses both:

- **No prompt / no manual vocabulary.** The model segments the whole image and
  emits per-pixel class labels directly.
- **Dense stuff coverage.** The head is trained on ADE20K (150 classes) which
  contains many background/stuff categories, so large surfaces get labeled
  instead of being dropped.

## 2. Core Idea

> Use Facebook's DINOv2 (no text prompt required). It segments the image and
> emits a label per pixel from its built-in class set. Render the scene from a
> large number of viewpoints, covering the scene from every direction. For each
> viewpoint run DINOv2, then have every view vote on each Gaussian. If most
> views agree a Gaussian is (for example) a bicycle, assign it the `bicycle`
> label.

Formally, for every Gaussian `g` we accumulate a class-vote vector across all
rendered views, then assign `label(g) = argmax(votes[g])` subject to minimum
support thresholds; otherwise `g` stays `unlabeled` (id `0`).

## 3. What to Reuse vs. Replace

Reuse the existing repository infrastructure to keep the output identical and
the change small:

| Component | Action | Existing code |
| --- | --- | --- |
| Camera loading / selection | Reuse | `scripts/task1/flashsplat_cameras.py` |
| 3DGS rendering from `cameras.json` | Reuse | `flashsplat_cameras.py::render_flashsplat` |
| 2D class map -> Gaussian vote lifting | Adapt | `scripts/task1/lift_dinov2_view_votes.py` |
| Pruning / connected-component / instance IDs | Reuse | `scripts/task1/cluster_semantic_flashsplat_proposals.py` |
| `label_map.json` build + PLY writing | Reuse | `cluster_semantic_flashsplat_proposals.py`, `add_labels_from_npy.py` |
| Structural validation | Reuse | `scripts/task1/validate_task1_outputs.py` |
| **2D semantic evidence** | **Replace** | new: `dinov2_segment_views.py` |
| **3D fusion / voting** | **Replace** | `scripts/task1/fuse_dinov2_multiview_votes.py` |

The `01_grounded_sam` stage is replaced by DINOv2 dense segmentation. The
FlashSplat renderer is reused in a dedicated lifting stage that feeds one
compact integer class map per view. The final fusion stage is replaced by exact
per-Gaussian voting (Stage 4).

## 4. Pipeline

```
load 3DGS model + cameras.json
 -> render N all-around views
 -> run DINOv2 semantic segmentation per view  (per-pixel class id + confidence)
 -> lift per-view class labels to Gaussian votes
 -> accumulate votes across all views per Gaussian
 -> argmax vote with min-support thresholds  (else unlabeled)
 -> prune / instance-ify / retain labels + label_map.json
 -> publish one final semantic_point_cloud.ply
 -> validate
```

### Stage 1 - All-around multi-view rendering

Goal: cover the scene from as many directions as possible so voting is robust.

- Start from the scene `cameras.json`. Use every available camera, not an evenly
  spaced subset. More views = more robust votes.
- V1 does not synthesize virtual cameras. Unvalidated poses can introduce
  artifacts and do not have a calibrated quality score. Orbit/hemisphere views
  are a later extension.
- Render RGB at a fixed width (start at 960 px, matching the existing runs).
- Record, per view, a **view-quality weight**. Every real v1 camera has weight
  `1.0`; the field keeps the vote format extensible without inventing quality
  estimates.

Reuse `selected_camera_items` / `iter_camera_items` and
`render_flashsplat` for rendering. If real-camera coverage later proves
insufficient, evaluate an orbit-camera generator as a separately validated
extension rather than changing v1 inputs.

### Stage 2 - Per-view DINOv2 semantic segmentation

- Use the official DINOv2 ViT-L/14 backbone plus its released ADE20K linear
  segmentation head (150 classes). ViT-L/14 is the selected v1 model; the
  released full Mask2Former head is not used.
- Input: the rendered RGB view. Output per view:
  - `class_id` map: `H x W` int, ADE20K class per pixel.
  - `confidence` map: `H x W` float in `[0, 1]` (softmax max probability).
- Save each view's outputs under `stages/01_dinov2/seg/<view>.npz` and a color
  overlay under `stages/01_dinov2/overlays/` for QA.
- Maintain a fixed **ADE20K -> project ontology** table
  (`configs/ade20k_to_project.json`). V1 preserves every ADE20K identity:
  zero-based ADE ids `0..149` map to project ids `1..150`; project id `0` is
  reserved for abstention/unlabeled. Names are normalized, but classes are not
  merged and none are dropped by scene-specific policy. Thing/stuff flags follow
  the ADE20K panoptic metadata published with Mask2Former rather than a local
  scene heuristic.
- Pixels whose maximum softmax probability is below the global threshold
  `MIN_PIXEL_CONFIDENCE=0.5` become abstentions before lifting. The saved NPZ
  retains the raw ADE class id and float16 maximum probability.

### Stage 3 - Lift per-view labels to Gaussian votes

For each view, turn the per-pixel `class_id` map into per-Gaussian evidence with
the existing FlashSplat rasterizer.

Compact the project classes present in a view into one integer mask, keeping
local row zero for abstention. One FlashSplat call returns `used_count[k,g]`
for all local rows, including abstention. The current FlashSplat rasterizer
allocates one additional all-zero sentinel row; validate and remove that row
before voting. Define
`visibility[g] = sum_k used_count[k,g]`. A supported class row casts:

```
vote(v,g,c) = view_quality(v)
              * mean_confidence(v,c)
              * used_count[c,g] / visibility[g]
```

Only rows with raw `used_count > 0.05` are stored. The abstention row uses the
mean maximum-softmax confidence of the pixels rejected by the global threshold;
if a view has no rejected pixels, its abstention confidence is zero. This keeps
abstention as conservative uncertainty evidence without turning every rejected
pixel into a unit-confidence negative vote. The per-view value is recorded as
`abstain_mean_confidence` in the vote manifest. Each view is saved as sparse
`indices`, `class_ids`, and `weights` arrays.

Either way, the result of Stage 3 for one view is a set of
`(gaussian_index, project_class_id, weight)` votes.

### Stage 4 - Multi-view voting / fusion

Accumulate votes across all views into an exact per-class, per-Gaussian score
matrix and pick the winner. Fusion exposes three scene-neutral modes so
abstention behavior can be isolated without rerunning DINOv2 or FlashSplat:

- `joint` is the production-compatible default. Abstain row 0 participates in
  the argmax and agreement denominator.
- `semantic_only` is an aggressive diagnostic upper bound. The winner and
  agreement use only semantic rows 1-150; abstain is measured but does not
  reject a result.
- `separate_abstain` also chooses and measures agreement over semantic rows,
  then independently requires
  `semantic_mass / (semantic_mass + abstain_mass) >= MIN_SEMANTIC_EVIDENCE`
  (default 0.5). This separates class disagreement from uncertainty while
  retaining abstain as a conservative gate.

- `votes[g, c] += weight` for every vote from Stage 3.
- Track `views_supporting[g, c]` = number of distinct views that voted
  class `c` for Gaussian `g`.
- In `joint` mode, assign:
  - `label(g) = argmax_c votes[g, c]`, **only if**
    - `views_supporting[g, argmax] >= MIN_VIEWS` (default 2), and
    - `votes[g, argmax] / sum_c votes[g, c] >= MIN_AGREEMENT` (default 0.5),
      where the sum includes abstention.
  - Otherwise `label(g) = 0` (unlabeled).
- Exact winner ties stay unlabeled. No second-choice label is assigned after a
  threshold or later pruning failure.
- In both semantic modes, exact ties between semantic classes also stay
  unlabeled. `MIN_VIEWS` and `MIN_AGREEMENT` still apply; only
  `separate_abstain` additionally applies `MIN_SEMANTIC_EVIDENCE`.
- Log raw-argmax, thresholded, and final post-pruning unlabeled ratios.
- Save the mode-specific winner agreement and semantic-evidence fraction for
  every Gaussian so comparisons remain auditable.

**Memory note.** Fusion creates a temporary disk-backed float32 matrix of shape
`(151, N_gaussians)`, accumulates each sparse view exactly, processes winners in
chunks, and deletes the matrix. This avoids a RAM spike and avoids approximate
top-k state.

### Stage 5 - Prune, instance-ify, export

Reuse the existing post-processing so the deliverable matches the current D1
contract:

- Support-adaptive pruning of tiny/weak labels.
- Leave each stuff class as one semantic group.
- Split every thing class into 3D voxel-connected components using one global,
  Gaussian-scale-derived voxel rule. Remove components below the global
  `max(500 Gaussians, 1% of the largest class component)` rule, then create
  deterministic instance IDs.
- Apply only the existing scene-neutral support-adaptive size thresholds. A
  pruned component becomes unlabeled and is not reassigned.
- Build `label_map.json` and retain `gaussian_labels.npy` for downstream stages.
- Do not materialize a semantic PLY inside an intermediate fusion or merge
  stage. The selected final result alone writes
  `deliverables/semantic_point_cloud.ply` with the added `property int label`.

### Stage 6 - Validation

Run `validate_task1_outputs.py`: every Gaussian has a label, `label_map.json`
covers every nonzero id, and no nonzero class is a placeholder. Produce semantic
overlay contact sheets for visual QA, exactly as the current pipeline does.

## 5. Deliverables (unchanged D1 contract)

```
outputs/eyenavgs_task1/<scene>_dinov2/
  stages/
    01_real_camera_views/
      rgb_renders/
      dinov2_segments/     # per-view class_id + confidence
      dinov2_overlays/     # per-view color segmentation for QA
    02_flashsplat_votes/
      view_votes/          # sparse per-view votes
    03_exact_fusion/
      gaussian_labels.npy
      winner_agreements.npy
      winner_semantic_evidence.npy
      dinov2_vote_summary.json
  deliverables/
    semantic_point_cloud.ply
    label_map.json
  visualizations/
    semantic_point_cloud_supersplat_debug.ply
    grounding_extensions_supersplat_debug.ply  # hybrid mode with accepted groups
  validation/
    task1_validation.json
    visible_overlay_coverage.json
```

## 6. Configuration

New config files:

- `configs/ade20k_to_project.json` - ADE20K(150) -> project ontology mapping.
- Reuse the scene name conventions from the existing scripts.

Environment variables (mirror the existing sbatch style):

```
SCENE, MODEL_DIR, OUTPUT_NAME
RENDER_MAX_WIDTH        (default 960)
VIEW_COUNT              (default: all cameras.json cameras)
CAMERA_SOURCE           (must be real in v1)
DINOV2_HEAD             (must be linear in v1)
MIN_PIXEL_CONFIDENCE    (default 0.5)
MIN_VIEWS               (default 2)
MIN_AGREEMENT           (default 0.5)
FUSION_MODE             (joint, semantic_only, or separate_abstain; default joint)
MIN_SEMANTIC_EVIDENCE   (default 0.5; used only by separate_abstain)
BASELINE_LABELS         (optional accepted GroundingDINO gaussian_labels.npy)
ENABLE_GROUNDINGDINO    (default 0; set 1 for continuous extension branch)
GROUNDING_EXTENSION_CONFIG
GROUNDING_EXTENSION_CLASSES (optional QA override)
GROUNDING_GUARD_MAX_PROMPT_WORDS (default 180)
GROUNDING_BASE_VIEW_COUNT   (default 50)
GROUNDING_TARGET_VIEW_COUNT (default 20)
COLOR_MODE              (class by default; instance for debugging only)
```

When enabled, the scheduler reuses the all-camera RGB renders, selects 50
evenly spaced seed views, adds up to 20 projection-safe views using low DINOv2
coverage and pose diversity, runs the mature GroundingDINO/SAM/FlashSplat
cleanup branch, and merges only surviving configured extension groups. Before
detection it automatically builds a per-run vocabulary containing the DINOv2
classes that survived scene fusion plus the selected extension classes. When a
DINO class matches a richer scene prompt specification (for example,
`television_receiver` to `television`), it reuses those prompts; otherwise it
uses the normalized DINO class name. Thing guards are retained ahead of stuff
guards if the conservative prompt-word budget is reached. Extension prompts
are never dropped.

Both roles participate in GroundingDINO/SAM and evidence-based fusion, which
lets an existing DINO class compete with a custom false positive. The merger
still accepts only `GROUNDING_EXTENSION_CLASSES`; guard groups are retained as
debug evidence but cannot alter the DINO base. Guards are ordered ahead of
extensions for the fusion's exact-tie fallback, while non-tied ownership
continues to be decided by multi-view evidence. The generated config and source
hashes are written to
`stages/04_grounding_camera_selection/grounding_guard_config.json`. The
standalone GroundingDINO scheduler remains available for isolated debugging.

### Experimental same-ontology ADE refinement

`scripts/slurm/slurm_task1_ade_refinement_scene.sbatch` is a cached-DINO
experimental entry point for testing Grounded-SAM boundary refinement without
enabling custom vocabulary. It derives its Grounding vocabulary automatically
from ADE classes that survived DINO fusion with at least 500 Gaussians and two
source views. The vocabulary is scene-adaptive but contains no scene-specific
class list. Dataset spellings that are poor natural-language prompts use the
global aliases in `configs/ade20k_grounding_prompts.json`; aliases never change
class identity.

The refinement merge remains inside the ADE ontology and applies six global
guards:

1. Missing-vocabulary classes are disabled and cannot enter the output.
2. Grounding groups must already pass the shared multi-view fusion, signed
   evidence, size, and spatial cleanup.
3. A Gaussian is eligible only when exactly one prompted ADE class has at least
   two positive views and a positive-evidence ratio of at least 0.5. Competing
   robust class claims are left at the DINO label.
4. Each Grounding group must contain at least 500 uniquely claimed Gaussians
   from an existing same-class DINO instance and cover at least 10% of that
   anchor.
5. The uniquely claimed support is split into globally scaled 3D voxel
   components. A component can refine only when at least 10% of its Gaussians
   already belong to the selected same-class DINO anchor. This prevents a
   small valid anchor elsewhere in the scene from authorizing a disconnected
   Grounding region. Accepted components can relabel only their remaining
   uniquely claimed support.
6. Grounding groups for ADE `stuff` classes may fill abstentions or refine
   other `stuff`, but cannot overwrite an existing DINO `thing` instance.
   Precise Grounded-SAM `thing` groups may still correct either kind. This
   preserves object instances against broad connected ceiling, floor, road,
   and stair masks without naming any scene or class exception.

The cached eight-scene spatial thing-guard comparison is stored as
`<scene>_dinov2_ade_refinement_auto_spatial_thing_guard_v3`. Reviewed support
improved from 27.98% to 18.84% wrong bicycle on the Bicycle bench, from 17.03%
to 93.22% windowpane on the Dr. Johnson window, from 48.08% to 94.06%
television in Room, from 67.74% to 79.21% truck on the Truck body, and from
44.22% to 63.52% railing on the Treehill fence. The spatial guard also blocked
the first prototype's Train building spill. Treehill path remains essentially
unchanged and missing-vocabulary identities remain unavailable, so this is
still an ablation rather than the production hybrid default.

These rules treat Grounded-SAM as guarded local boundary evidence rather than
ground truth. The scheduler reuses the DINO camera renders and labels, writes a
separate `refinement_changes.npy`, and generates both full-scene and
changes-only overlay contact sheets. It is an ablation path, not yet the
production hybrid default.

The final visualization export always creates a stable-color full-scene
SuperSplat PLY. Hybrid runs with accepted extension groups also create a focused
SuperSplat PLY that leaves the extension classes colored and dims every other
Gaussian. These are the supported direct 3D debugging artifacts. Intermediate
stage semantic PLYs are intentionally omitted because they duplicate the full
3DGS payload; stage labels, maps, summaries, masks, and overlays retain the
actual diagnostic state without that storage duplication.

Room extension candidates live in
`configs/task1_hybrid_extensions.room.json`. Cached same-camera QA originally
selected `piano` and `speaker` as defaults. A later full continuous run exposed
an extension-only `piano` false positive over the television region; the
automatic guard vocabulary addresses that failure mode without permitting a
guard label to enter the final merge. `media_console` remains an explicit-only
candidate because its cached mask leaked broadly into wall and floor, while
`guitar` remains an explicit-only no-op candidate. Pass
`GROUNDING_EXTENSION_CLASSES` to run a single candidate during further QA.

Cached 70-view validation of the guarded branch retained an 8,470-Gaussian
`television_01` guard and removed the false television-shaped `piano_02` from
the merge. The final custom result contains one 9,201-Gaussian upright piano
and three visually coherent speaker groups totaling 14,443 Gaussians. It
changes 23,644 DINO labels (17,183 previously unlabeled) and preserves the DINO
array outside those four extension masks. The guard branch therefore keeps
the configured Room defaults while correcting the extension-only competition
failure.

### Room pilot commands

Prepare the external repository, separate segmentation environment, and
official checkpoints:

```bash
bash scripts/cluster/clone_external_repos.sh
bash scripts/setup/install_dinov2_segmentation.sh
bash scripts/setup/install_semantic_extensions.sh
```

First submit a five-camera smoke test from the project root:

```bash
SCENE=room \
OUTPUT_NAME=room_dinov2_smoke_v1 \
CAMERA_INDICES=0,78,155,232,310 \
BASELINE_LABELS=../../outputs/eyenavgs_task1/room_semantic_targeted_v2/stages/03_semantic_fusion/gaussian_labels.npy \
sbatch scripts/slurm/slurm_task1_dinov2_scene.sbatch
```

Inspect `dinov2_overlays/`, the vote summary, semantic overlay contact sheet,
and structural validation. If those pass, submit all 311 real Room cameras:

```bash
SCENE=room \
OUTPUT_NAME=room_dinov2_multiview_v1 \
VIEW_COUNT=0 \
BASELINE_LABELS=../../outputs/eyenavgs_task1/room_semantic_targeted_v2/stages/03_semantic_fusion/gaussian_labels.npy \
sbatch scripts/slurm/slurm_task1_dinov2_scene.sbatch
```

These are user submission checkpoints; repository automation never submits a
scheduled job itself.

## 7. Acceptance Criteria

A scene passes when:

- every Gaussian has a `label`; `label_map.json` covers every nonzero id;
- semantic overlays show plausible object boundaries from several viewpoints;
- raw, thresholded, and final unlabeled ratios are reported; the final ratio is
  compared with the current GroundingDINO baseline on the same scene when
  `BASELINE_LABELS` is supplied;
- obvious large objects are not merged into background labels;
- a short notes file records known weak labels.

Gaze-hit coverage is explicitly reported as `not_run` in v1. Task 2 has not yet
established gaze-to-scene coordinate alignment, and this repository has no
validated ray/point consumer to reuse. Add the gaze-relevant ratio only after
that alignment is accepted.

## 8. Known Risks and Mitigations

- **ADE20K is still a fixed 150-class set.** Unlike GroundingDINO (which leaves
  unknown objects *unlabeled*), the seg head will force an unknown object into
  the nearest of its 150 classes, producing **silent mislabels**. Mitigation:
  keep the global low-confidence -> `unlabeled` threshold, require multiview
  agreement, and spot-check overlays. V1 does not hide classes through an
  ambiguous or scene-specific ignore mapping.
- **View artifacts corrupt votes.** Weight votes by view quality; require
  `MIN_VIEWS`/`MIN_AGREEMENT`; artifacts rarely agree across many views.
- **Fine gaze targets (faces, text, object parts) are not ADE20K classes.**
  Out of scope for this front-end; see Extensions.
- **Memory blow-up on large scenes.** Map to the project ontology before
  accumulation and use the temporary disk-backed exact vote matrix.
- **FlashSplat binary portability.** An extension compiled only for the RTX
  8000's `sm_75` architecture is invalid on an L40 (`sm_89`) and can surface as
  a nonsensical CUDA allocation rather than a clear architecture error.
  `scripts/setup/install_semantic_extensions.sh` therefore builds native
  `sm_75`, native `sm_86`, and `sm_86` PTX by default; the PTX is forward-JIT
  compatible with the L40 while preserving RTX 8000 support.

## 9. Optional Extensions (not required for v1)

- **Long-tail / open-vocabulary naming.** For objects outside ADE20K, run CLIP
  on class-agnostic regions and name on demand - only for instances that receive
  gaze hits.
- **Feature distillation.** Instead of hard per-view label voting, distill
  DINOv2 (or CLIP) features into the Gaussians (Feature-3DGS / LangSplat style)
  and classify/query once in 3D. This is the more robust long-term form of the
  same idea and removes hard-label voting noise.

## 10. Implemented Components

1. `render_task1_views.py` renders real cameras and records a manifest.
2. `dinov2_segment_views.py` runs the official ViT-L/14 ADE20K linear model.
3. `lift_dinov2_view_votes.py` makes one FlashSplat call per view and writes
   sparse normalized votes.
4. `fuse_dinov2_multiview_votes.py` performs exact disk-backed fusion,
   thresholds, 3D thing splitting, adaptive pruning, and export.
5. `slurm_task1_dinov2_scene.sbatch` runs the route plus the existing overlay,
   SuperSplat, contact-sheet, coverage, and structural validation tools.
6. Room remains the pilot scene. First run 3-5 explicit cameras as a smoke
   test, inspect DINO overlays, then run every real Room camera.

## 11. References

- DINOv2: self-supervised ViT backbone with released ADE20K/VOC segmentation
  heads (Oquab et al., 2023).
- Current front-end and contract: `docs/TASK1_SEMANTIC_ANNOTATION.md`.
- Gaze-hit evaluation is deferred until Task 2 coordinate alignment is
  validated.
