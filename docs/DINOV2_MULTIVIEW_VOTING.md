# Task 1 Alternative Front-End: DINOv2 Multi-View Voting

This document specifies an alternative semantic-labeling front-end for the
EyeNavGS 3DGS scenes. It replaces the current hand-written-vocabulary detector
(GroundingDINO + SAM) with a **prompt-free, dense DINOv2 semantic segmentation**
front-end, then assigns one label per Gaussian by **multi-view voting**.

The output contract is unchanged: each Gaussian still receives one integer
`label`, and each scene still ships a `label_map.json`. This route only changes
how the 2D semantic evidence is produced and how it is fused into 3D. It is a
drop-in alternative to the `01_grounded_sam` and (optionally) `02_flashsplat`
stages, not a rewrite of the whole pipeline.

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
| 2D mask -> Gaussian support lifting | Reuse | `scripts/task1/run_flashsplat_mask_proposals.py` |
| Pruning / connected-component / instance IDs | Reuse | `scripts/task1/cluster_semantic_flashsplat_proposals.py` |
| `label_map.json` build + PLY writing | Reuse | `cluster_semantic_flashsplat_proposals.py`, `add_labels_from_npy.py` |
| Structural validation | Reuse | `scripts/task1/validate_task1_outputs.py` |
| **2D semantic evidence** | **Replace** | new: `dinov2_segment_views.py` |
| **3D fusion / voting** | **Replace or adapt** | new: `dinov2_multiview_vote.py` |

The `01_grounded_sam` stage is replaced by DINOv2 dense segmentation. The
FlashSplat lifting in `02_flashsplat` can be reused **unchanged** by feeding it
one binary mask per class per view (see Stage 3, Option A). The final fusion
stage is replaced by simple per-Gaussian voting (Stage 4).

## 4. Pipeline

```
load 3DGS model + cameras.json
 -> render N all-around views
 -> run DINOv2 semantic segmentation per view  (per-pixel class id + confidence)
 -> lift per-view class labels to Gaussian votes
 -> accumulate votes across all views per Gaussian
 -> argmax vote with min-support thresholds  (else unlabeled)
 -> prune / instance-ify / export semantic_point_cloud.ply + label_map.json
 -> validate
```

### Stage 1 - All-around multi-view rendering

Goal: cover the scene from as many directions as possible so voting is robust.

- Start from the scene `cameras.json`. Use every available camera, not an evenly
  spaced subset. More views = more robust votes.
- If the training cameras do not cover the scene from all directions, synthesize
  additional virtual cameras (e.g. an orbit / hemisphere around the scene
  center). 3DGS renders any pose cheaply, so dense coverage is affordable.
- Render RGB at a fixed width (start at 960 px, matching the existing runs).
- Record, per view, a **view-quality weight** in `[0, 1]` (see Stage 4). Novel
  or extreme poses produce splatting artifacts (floaters, blur) and should carry
  less voting weight.

Reuse `selected_camera_items` / `iter_camera_items` and
`render_flashsplat` for rendering. Add an orbit-camera generator only if
`cameras.json` coverage is insufficient.

### Stage 2 - Per-view DINOv2 semantic segmentation

- Use the official DINOv2 backbone plus its released **semantic segmentation
  head** (ADE20K, 150 classes). Prefer the Mask2Former head for cleaner
  boundaries; the linear head is acceptable for a first pass.
- Input: the rendered RGB view. Output per view:
  - `class_id` map: `H x W` int, ADE20K class per pixel.
  - `confidence` map: `H x W` float in `[0, 1]` (softmax max probability).
- Save each view's outputs under `stages/01_dinov2/seg/<view>.npz` and a color
  overlay under `stages/01_dinov2/overlays/` for QA.
- Maintain a fixed **ADE20K -> project ontology** mapping table
  (`configs/ade20k_to_project.json`). This maps the 150 ADE20K classes to the
  project label names/ids (many ADE20K classes may merge, e.g.
  `wall/building` -> `building`, `grass/tree/plant` -> `vegetation`). Classes
  the project does not care about map to `ignore`.

### Stage 3 - Lift per-view labels to Gaussian votes

For each view we must turn the per-pixel `class_id` map into per-Gaussian
evidence. Two options; **Option A is recommended** because it reuses tested code.

**Option A (recommended): reuse FlashSplat.** For each view, split the
`class_id` map into one binary mask per present class. Feed each class mask
through the existing FlashSplat lifting
(`run_flashsplat_mask_proposals.py` / `render_flashsplat`), which returns the
per-Gaussian support (`used_count`) for that mask. Multiply the support by the
mean DINOv2 confidence inside the mask and the view-quality weight to get the
vote weight this view casts for that class on those Gaussians.

**Option B (simpler, no FlashSplat): index rasterization.** Render a Gaussian
**index / depth buffer** for the view, so each pixel knows which front-most
Gaussian it belongs to. Then each pixel casts one weighted vote for its
`class_id` onto that Gaussian. This is lighter but coarser (only the front
surface votes) and needs a rasterizer that can emit the winning Gaussian id per
pixel.

Either way, the result of Stage 3 for one view is a set of
`(gaussian_index, project_class_id, weight)` votes.

### Stage 4 - Multi-view voting / fusion

Accumulate votes across all views into a per-Gaussian, per-class score matrix
and pick the winner.

- `votes[g, c] += weight` for every vote from Stage 3.
- Also track `views_supporting[g, c]` = number of distinct views that voted
  class `c` for Gaussian `g`.
- Assign:
  - `label(g) = argmax_c votes[g, c]`, **only if**
    - `views_supporting[g, argmax] >= MIN_VIEWS` (default 2), and
    - `votes[g, argmax] / sum_c votes[g, c] >= MIN_AGREEMENT` (default 0.5).
  - Otherwise `label(g) = 0` (unlabeled).
- `MIN_VIEWS` and `MIN_AGREEMENT` are the two knobs that trade coverage against
  precision. Log both the raw and thresholded unlabeled ratio.

**Memory note.** A dense `votes` matrix is `N_gaussians x N_classes` float32.
For 6M Gaussians x ~40 project classes that is ~1 GB; x150 ADE20K classes is
~3.6 GB. Map ADE20K -> project ontology **before** accumulation to shrink the
class axis, and/or keep only a per-Gaussian top-k running tally instead of a
dense matrix.

### Stage 5 - Prune, instance-ify, export

Reuse the existing post-processing so the deliverable matches the current D1
contract:

- Support-adaptive pruning of tiny/weak labels.
- Remove disconnected 3D islands from thing labels; leave stuff untouched.
- Create instance IDs (`bicycle_01`, `tree_02`) via connected components.
- Build `label_map.json` and write `semantic_point_cloud.ply` with the added
  `property int label` using `write_ply_with_labels`.

### Stage 6 - Validation

Run `validate_task1_outputs.py`: every Gaussian has a label, `label_map.json`
covers every nonzero id, and no nonzero class is a placeholder. Produce semantic
overlay contact sheets for visual QA, exactly as the current pipeline does.

## 5. Deliverables (unchanged D1 contract)

```
outputs/eyenavgs_task1/<scene>_dinov2/
  stages/
    01_dinov2/
      seg/                 # per-view class_id + confidence
      overlays/            # per-view color segmentation for QA
    03_semantic_fusion/
      gaussian_labels.npy
      vote_summary.json    # per-class vote mass, view support, thresholds used
  deliverables/
    semantic_point_cloud.ply
    label_map.json
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
ADD_ORBIT_VIEWS         (0/1, synthesize hemisphere views if coverage is poor)
DINOV2_HEAD             (mask2former | linear)
MIN_VIEWS               (default 2)
MIN_AGREEMENT           (default 0.5)
```

## 7. Acceptance Criteria

A scene passes when:

- every Gaussian has a `label`; `label_map.json` covers every nonzero id;
- semantic overlays show plausible object boundaries from several viewpoints;
- the thresholded unlabeled ratio is reported and is **lower** than the current
  GroundingDINO baseline on the same scene (this is the primary reason for the
  new front-end);
- obvious large objects are not merged into background labels;
- a short notes file records known weak labels.

Report the unlabeled ratio two ways: over all Gaussians, and over only the
Gaussians that are actually hit by gaze rays (the gaze-relevant number is what
matters downstream). See `visualize_gaze.py::march_hits` for the ray/point
intersection used to define gaze hits.

## 8. Known Risks and Mitigations

- **ADE20K is still a fixed 150-class set.** Unlike GroundingDINO (which leaves
  unknown objects *unlabeled*), the seg head will force an unknown object into
  the nearest of its 150 classes, producing **silent mislabels**. Mitigation:
  keep a low-confidence -> `unlabeled` threshold, and map ambiguous ADE20K
  classes to `ignore` in the ontology table. Spot-check overlays.
- **View artifacts corrupt votes.** Weight votes by view quality; require
  `MIN_VIEWS`/`MIN_AGREEMENT`; artifacts rarely agree across many views.
- **Fine gaze targets (faces, text, object parts) are not ADE20K classes.**
  Out of scope for this front-end; see Extensions.
- **Memory blow-up on large scenes.** Map to the project ontology before
  accumulation; use top-k tallies.

## 9. Optional Extensions (not required for v1)

- **Long-tail / open-vocabulary naming.** For objects outside ADE20K, run CLIP
  on class-agnostic regions and name on demand - only for instances that receive
  gaze hits.
- **Feature distillation.** Instead of hard per-view label voting, distill
  DINOv2 (or CLIP) features into the Gaussians (Feature-3DGS / LangSplat style)
  and classify/query once in 3D. This is the more robust long-term form of the
  same idea and removes hard-label voting noise.

## 10. Suggested Implementation Order

1. `dinov2_segment_views.py`: render views + run DINOv2 seg head + save
   `class_id`/`confidence` + overlays. Verify on 3-5 views first.
2. `configs/ade20k_to_project.json`: author and sanity-check the ontology map.
3. Stage 3 Option A: adapt `run_flashsplat_mask_proposals.py` to accept
   per-class binary masks from DINOv2 instead of GroundingDINO+SAM masks.
4. `dinov2_multiview_vote.py`: accumulate votes, apply thresholds, write
   `gaussian_labels.npy` + `vote_summary.json`.
5. Reuse Stage 5/6 for pruning, instance IDs, export, and validation.
6. Compare unlabeled ratio (all + gaze-hit) against the GroundingDINO baseline
   on one scene (start with `room`, the smallest matched model).

## 11. Implementation (v1, code complete - not yet run)

The route is implemented with a **pluggable 2D backend** instead of the
DINOv2 heads specified above. Rationale: the released DINOv2 linear head is
patch-coarse (ADE20K mIoU ~47, 14 px boundaries) and its Mask2Former variant
depends on legacy mmcv/mmsegmentation that conflicts with the
`gaussian_grouping_true` environment. The v1 backend is HuggingFace
Mask2Former-Swin-L (ADE20K semantic, mIoU ~56); DINOv3 ViT-7B/16 with the
official ADE20K M2F segmentor (mIoU ~63, needs a 40 GB-class GPU) is wired in
as a comparison backend.

Deliberate deviations from the specification above:

1. **Stage 2 backend**: `--backend mask2former | dinov3` replaces the DINOv2
   linear/M2F heads. Both emit the same per-view contract
   (`project_class` + `confidence`).
2. **Stage 3 lifting**: instead of one binary mask per class per view
   (Option A), the compact class map is passed directly as a FlashSplat
   multi-object index mask, so one rasterizer call per <=32-class batch
   lifts every class in the view at once.
3. **Stage 4 voting** adds a third knob `MIN_VISIBLE_RATIO`
   (`views_supporting / visible_views`), because absolute `MIN_VIEWS` alone
   is biased against Gaussians visible in few views.
4. **Fusion modes**: `--mode full` (pure voting, the route as specified) and
   `--mode fill` (votes only fill Gaussians the accepted GroundingDINO
   baseline left unlabeled; stuff-only by default so accepted thing
   instances stay atomic). `fill` is the production default; `full` is the
   ablation.
5. **Ontology mapping happens in Stage 2** (before storage), so votes are
   accumulated in the compact project class space (memory note in Stage 4).
   Confidence gating happens in Stage 3, so re-lifting with a different
   `MIN_CONFIDENCE` does not re-run the model.

Files:

- `configs/ade20k_to_project.json` - ADE20K(150) -> project ontology.
- `scripts/task1/ade20k_ontology.py` - ontology loader (numpy only).
- `scripts/task1/dense_seg_backends.py` - mask2former / dinov3 backends.
- `scripts/task1/segment_views_semantic.py` - Stage 1+2 (render + segment).
- `scripts/task1/lift_semantic_votes.py` - Stage 3 (index-mask lifting).
- `scripts/task1/fuse_semantic_votes.py` - Stage 4+5 (vote fusion, instances,
  pruning, export). Pure-numpy decision core, unit-tested.
- `scripts/slurm/slurm_task1_dense_semantic_scene.sbatch` - scheduler
  (env: `SCENE`, `MODE=fill|full`, `SEG_BACKEND=mask2former|dinov3`,
  `VIEW_COUNT=0` for all cameras, `FILL_SOURCE=<accepted dir>`).
- `tests/test_dense_semantic_vote.py` - ontology + vote-logic tests.

Known limitations recorded for QA: ADE20K has no railroad-track class (train
scene tracks must stay covered by the baseline in fill mode), no wheel class
(truck wheels only survive in fill mode), and the 150-class closed set still
force-classifies unknown objects - the `MIN_CONFIDENCE` gate plus `ignore`
mappings are the mitigation, exactly as Section 8 anticipated.

## 12. References

- DINOv2: self-supervised ViT backbone with released ADE20K/VOC segmentation
  heads (Oquab et al., 2023).
- Current front-end and contract: `docs/TASK1_SEMANTIC_ANNOTATION.md`.
- Gaze ray/point intersection consumer: `visualize_gaze.py::march_hits`.
