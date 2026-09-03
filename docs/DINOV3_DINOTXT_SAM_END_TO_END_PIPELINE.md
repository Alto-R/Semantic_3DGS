# DINOv3 dino.txt + SAM OOV semantic pipeline

This document explains the implementation and intermediate contracts for the
maintained pipeline. For the shortest complete command sequence, use the
[complete quickstart](DINOV3_DINOTXT_SAM_QUICKSTART.md).

The pipeline is inference-only. It does not train, fine-tune, delete
Gaussians, select cameras by hand, or select individual Gaussians by hand.

## 1. What the pipeline produces

The final result is a new, versioned output root containing:

- the accepted DINOv3 base labels as the fallback;
- an OOV class label only where the dino.txt/SAM evidence survives 3D gates;
- a semantic PLY with the original Gaussian geometry and a per-vertex `label`;
- a class-colored SuperSplat PLY for visual inspection;
- evidence arrays, manifests, summaries, and validation output.

The important distinction is:

```text
closed-set DINOv3 ADE20K base  ->  accepted labels/map (immutable fallback)
                                      |
RGB views -> dino.txt text scores -> SAM masks -> OOV mask manifest
                                      |
                 compose one multiclass map per camera
                                      |
                     FlashSplat weighted 3D votes
                                      |
                    multiview OOV gates / abstention
                                      |
                    new labels + semantic PLY + QA PLY
```

The DINOv3 ADE20K model and the DINOv3 dino.txt model are two pretrained
inference heads with different jobs. dino.txt does not replace the accepted
base; it proposes an OOV identity which must beat the base in multiple views.

## 2. Code map

The maintained implementation is split into four reusable pieces:

| Role | Entry point | Contract |
| --- | --- | --- |
| Render reconstruction cameras | `scripts/task1/dinov2/render_task1_views.py` | `view_manifest.json` plus `rgb_renders/` |
| Closed-set DINOv3 segmentation | `scripts/task1/dinov3/dinov3_segment_views.py` | `dinov3_manifest.json` plus per-view `class_id` maps |
| dino.txt + automatic SAM mask source | `scripts/task1/dinov3/dinotxt_sam_mask_pilot.py` | `grounded_sam_manifest.json` plus mask stacks |
| OOV composition, lifting, and fusion | `scripts/task1/dinov3/compose_oov_multiclass_votes.py` | semantic labels, maps, evidence, and semantic PLY |

The OOV fusion wrapper is
[run_oov_multiclass_scene.sh](../scripts/task1/dinov3/run_oov_multiclass_scene.sh).
It consumes an existing OOV mask manifest and DINOv3 base. The complete
quickstart runs base creation, dino.txt/SAM inference, this fusion wrapper, and
final render-back in one shell block.

The final wrapper also calls
[export_supersplat_label_colors.py](../scripts/task1/qa/export_supersplat_label_colors.py)
and
[validate_task1_outputs.py](../scripts/task1/qa/validate_task1_outputs.py).

## 3. Required inputs and invariants

All runtime roots must be derived from the checkout or supplied as environment
variables. Do not put personal usernames, hosts, or cluster mount paths in
tracked files.

For a scene, the workspace must provide:

```text
<workspace>/
  data/3dgs_models/graphdeco/<scene>/
    cameras.json
    point_cloud/iteration_30000/point_cloud.ply
  data/models/dinov3/
    dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
    dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth
    dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
    dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth
    bpe_simple_vocab_16e6.txt.gz
  external/dinov3/                 # pinned, clean Git checkout
  external/FlashSplat/
  external/segment-anything/
  outputs/eyenavgs_task1/
```

The dino.txt stage additionally needs a local Segment Anything checkout and a
SAM ViT-H checkpoint. Their paths are passed explicitly; the repository does
not assume a personal installation location.

Before running, verify:

1. `cameras.json`, the source PLY, and all RGB renders refer to the same scene
   and Gaussian iteration.
2. The accepted base `gaussian_labels.npy` length equals the source PLY vertex
   count, and its `label_map.json` contains id `0` and every used id.
3. The OOV mask manifest and the DINOv3 view manifest use the same
   `camera_index`/file names and render dimensions.
4. The dino.txt repository is at commit
   `6876159a11b4df116f30f667f8c9888617df0751` and has no tracked changes.
5. The two dino.txt checkpoint SHA-256 values are:
   - ViT-L/16 backbone:
     `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`
     (the expected prefix is `8aa4cbdd`);
   - dino.txt head/text encoder:
     `a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0`.
6. The BPE tokenizer SHA-256 is
   `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a`.

The pilot verifies these dino.txt and tokenizer hashes. The current SAM
loader verifies that the checkpoint exists and loads it as `vit_h`; its SAM
weights are not silently substituted.

## 4. Stage 0: create the base

The complete route creates a fresh DINOv3 ADE20K base through the maintained
recovery scheduler:

```bash
SCENE=<scene> \
OUTPUT_NAME=<scene>_dinov3_abstention_recovery_review_v1 \
bash scripts/slurm/slurm_task1_dinov3_end_to_end_recovery_scene.sbatch
```

The base output provides:

```text
<base-output>/gaussian_labels.npy
<base-output>/label_map.json
<base-output>/stages/01_real_camera_views/view_manifest.json
<base-output>/stages/01_real_camera_views/rgb_renders/
<base-output>/stages/01_real_camera_views/dinov3_manifest.json
<base-output>/semantic_point_cloud.ply
<base-output>/semantic_point_cloud_supersplat_debug.ply
<base-output>/summary.json
```

Keep this directory immutable. Every OOV experiment gets a new
`OUTPUT_NAME`; do not overwrite the accepted base.

## 5. Stage 1: render the real cameras

The view cache is generated from the reconstruction cameras, not arbitrary
orbit views. `--count 0` means every camera in `cameras.json`.

```bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd -P)"
SCENE=<scene>
MODEL_DIR="${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}"
VIEW_DIR="${WORKSPACE_ROOT}/outputs/eyenavgs_task1/${SCENE}_dinotxt_source_v1/stages/01_real_camera_views"

conda run --no-capture-output -n semantic_3dgs_renderer \
  python -m scripts.task1.dinov2.render_task1_views \
  --model-path "${MODEL_DIR}" \
  --output-dir "${VIEW_DIR}" \
  --flashsplat-root "${WORKSPACE_ROOT}/external/FlashSplat" \
  --iteration 30000 \
  --count 0 \
  --max-width 960
```

For the final OOV run, use all real cameras whenever compute permits. A small
evenly spaced subset is a 2D review pilot, not equivalent 3D evidence.

## 6. Stage 2: run closed-set DINOv3 on the same view cache

If the selected base already contains `dinov3_manifest.json` for this exact
view cache, reuse it. Otherwise run
`dinov3_segment_views.py` with the official ViT-7B ADE20K backbone and
Mask2Former head, or use the maintained DINOv3 scene scheduler. The segmenter
writes one `class_id` map per frame and a provenance-rich
`dinov3_manifest.json`; it does not by itself create accepted 3D labels.

The OOV wrapper can rerun this stage when `SKIP_DINOV3=0`, but the recommended
review workflow reuses an already accepted and audited segmentation cache with
`SKIP_DINOV3=1` so the base and OOV evidence share exactly the same cameras.

## 7. Stage 3: configure the dino.txt competitive vocabulary

Create a declarative config under `configs/`, for example
`configs/task1_dinotxt_sam.drjohnson_window_shutter.json`.

```json
{
  "target_class": "window_shutter",
  "classes": [
    {"class": "window_shutter", "prompts": ["window shutter", "wooden window shutters"]},
    {"class": "door", "prompts": ["interior door", "wooden room door"]},
    {"class": "window", "prompts": ["window", "windowpane"]},
    {"class": "wall", "prompts": ["interior wall", "painted wall"]}
  ],
  "selection": {
    "prompt_aggregation": "mean",
    "prompt_top_k": 2,
    "min_target_probability": 0.35,
    "min_competitor_margin": 0.05,
    "min_target_win_fraction": 0.50,
    "max_selected_masks": 8,
    "containment_threshold": 0.90,
    "nms_iou": 0.70
  }
}
```

The configured classes form a competitive vocabulary. The pilot emits masks
for `target_class`; competitors are present to reject lookalikes. Prompt
aliases are expanded automatically with seven photographic templates. Use
`topk_mean` when a class has several aliases and the strongest aliases should
carry more weight, but keep the vocabulary and prompt wording scene-neutral.

The current pilot is one target class per output. The 3D compositor supports
multiple OOV class IDs when a manifest contains per-mask class metadata, but
there is no safe automatic manifest merger in the repository. For several
targets, run and review each target separately or add a tested manifest-merging
stage before attempting a joint fusion.

## 8. Stage 4: run and review the 2D dino.txt + SAM pilot

The pilot is intentionally review-only. It never calls FlashSplat and never
writes semantic labels or a PLY.

For a small review set:

```bash
conda run --no-capture-output -n dinov3_semantic \
  python -m scripts.task1.dinov3.dinotxt_sam_mask_pilot \
  --view-dir "${VIEW_DIR}" \
  --view-count 50 \
  --config "${PROJECT_ROOT}/configs/task1_dinotxt_sam.<scene>_<target>.json" \
  --output-dir "${WORKSPACE_ROOT}/outputs/eyenavgs_task1/<scene>_dinotxt_<target>_2d_v1" \
  --dinov3-root "${WORKSPACE_ROOT}/external/dinov3" \
  --backbone-checkpoint "${WORKSPACE_ROOT}/data/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
  --dinotxt-checkpoint "${WORKSPACE_ROOT}/data/models/dinov3/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth" \
  --bpe-path "${WORKSPACE_ROOT}/data/models/dinov3/bpe_simple_vocab_16e6.txt.gz" \
  --dinotxt-hub-entry dinov3_vitl16_dinotxt_tet1280d20h24l \
  --segment-anything-root "${SEGMENT_ANYTHING_ROOT}" \
  --sam-checkpoint "${SAM_CHECKPOINT}" \
  --device cuda \
  --resize 512 --side 384 --stride 192 \
  --sam-points-per-side 32 \
  --sam-predicted-iou-threshold 0.88 \
  --sam-stability-threshold 0.92
```

For a final all-camera source, derive the indices from the manifest instead of
inventing an order:

```bash
CAMERA_INDICES="$(python -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); print(",".join(str(f["camera_index"]) for f in d["frames"]))' "${VIEW_DIR}/view_manifest.json")"
```

Pass that value as `--camera-indices "${CAMERA_INDICES}"` in place of
`--view-count`. The script preserves the requested camera order and rejects
missing indices.

### What happens inside the pilot

1. Each RGB image is short-side resized to 512 and normalized with ImageNet
   statistics.
2. DINOv3 ViT-L/16 image patch features are evaluated over 384-pixel crops
   with 192-pixel stride. Patch sizes are aligned before interpolation.
3. Text features are encoded for every alias/template. Per-class logits use
   the configured `mean`, `max`, or `topk_mean` reduction.
4. The learned positive `exp(logit_scale)` is applied before the class softmax.
   This is essential; omitting it produces near-uniform probabilities and no
   selected masks.
5. SAM ViT-H automatic masks are generated (`points_per_side=32`, no crop
   layers, predicted-IoU threshold `0.88`, stability threshold `0.92`).
6. Each complete SAM mask is scored by target probability, strongest
   competitor probability, target-vs-competitor margin, and target pixel-win
   fraction.
7. Masks pass the probability/margin/win-fraction gates, then smaller masks
   that are at least 90% contained in a larger passing mask are removed. IoU
   NMS and the per-frame cap finish selection.

The output root contains:

```text
probabilities/*.npy
mask_stacks/*.npz
selected_masks/*.png
selected_overlays/*.png
ranked_overlays/*.png
review_panels/*.png
dinotxt_sam_pilot_report.json
grounded_sam_manifest.json
```

Review `review_panels/`, `ranked_overlays/`, and `selected_overlays/` against
the original RGB images. Do not continue to 3D if the target masks are
fragmented, consistently select a competitor, or include obvious disjoint
objects. The historical filename `grounded_sam_manifest.json` is only the
generic mask-stack interface here; this route uses automatic SAM plus dino.txt,
not GroundingDINO.

## 9. Stage 5: compose a single multiclass map per camera

The OOV compositor receives the closed-set DINOv3 `class_id` map and the
reviewed dino.txt/SAM mask stack for each matching camera.

For each pixel:

1. convert the ADE20K `class_id` through `configs/ade20k_to_project.json`;
2. resolve overlapping OOV masks by their recorded confidence (strictly higher
   confidence wins; exact ties retain the earlier mask);
3. overwrite only pixels covered by an OOV mask;
4. leave all other pixels as the closed-set project class.

This creates one dense, multiclass semantic map per camera. It is important
that the map contains the base classes and the OOV class together: FlashSplat
then observes the candidate and its competitors in one render rather than in
separate class-specific passes.

## 10. Stage 6: lift maps with FlashSplat

For each composed map, the compositor renders the original Gaussian model once
with the map's local class IDs as `gt_mask`. FlashSplat returns sparse
per-Gaussian `used_count` values. These are converted to weighted records:

```text
view_votes/<frame>.npz:
  indices   : Gaussian indices visible in that camera
  class_ids : class associated with each sparse contribution
  weights   : positive alpha/transmittance contribution
```

No Gaussian is created or deleted. The original PLY is only read during this
stage. The compositor checks that its vertex count equals the base label count.

## 11. Stage 7: deterministic multiview OOV fusion

For each Gaussian and camera, `camera_winners()` reduces duplicate sparse
contributions to the largest weight. Exact weight ties are deterministic: the
lowest class ID wins.

Across cameras the compositor records:

- `visible_view_count.npy`;
- `incumbent_winner_view_count.npy` for the accepted base label;
- `oov_winner_view_count.npy` for each OOV class;
- `oov_positive_view_count.npy`, where a per-view OOV contribution reaches
  `oov_positive_mass_threshold` (default `0.50`);
- `oov_mass.npy` for the accumulated OOV contribution.

The best OOV class is selected by highest OOV winner-view count, then highest
mass. The candidate replaces the base only when all default, class-neutral
gates pass:

| Gate | Default |
| --- | ---: |
| visible views | `>= 3` |
| best OOV winner views | `>= 2` |
| best OOV winner share | `>= 0.50` of visible views |
| OOV mass share | `>= 0.35` using the compositor's visible-view denominator |
| positive OOV views | `>= 2` |
| positive per-view mass threshold | `0.50` |
| candidate advantage over incumbent | `>= 0.05` winner-share margin |

Otherwise the accepted base label remains unchanged, including label `0` for
an unlabeled Gaussian. Existing base labels already using an OOV ID are not
reaccepted by the default `already_target_is_accepted` guard.

The two-visible-view experiment changes only the first gate to `2`; it is an
ablation, not the default production policy. It should be compared visually
and reported as a separate versioned output.

## 12. Stage 8: materialize labels and PLYs

The compositor writes:

```text
stages/02_oov_multiclass_votes/
  gaussian_labels.npy
  gaussian_project_class_ids.npy
  visible_view_count.npy
  incumbent_winner_view_count.npy
  oov_winner_view_count.npy
  oov_positive_view_count.npy
  oov_mass.npy
  label_map.json
  semantic_point_cloud.ply
  composed_maps/*.npz
  view_votes/*.npz
  multiclass_fusion_summary.json
  vote_manifest.json
```

`semantic_point_cloud.ply` is made by copying the original binary Gaussian PLY
and appending an integer `label` property. Geometry, SH coefficients, and
Gaussian count remain unchanged. The output label map allocates noncolliding
OOV IDs automatically (or accepts explicit `class=id` assignments), records
the OOV source/contract, and reports per-class counts.

## 13. Stage 9: export a Supersplat visualization and validate

The raw semantic PLY carries labels but does not necessarily display semantic
colors. The wrapper bakes the label palette into `f_dc_0..2`, zeros all
`f_rest_*` coefficients, and writes:

```text
semantic_point_cloud_supersplat_debug.ply
semantic_color_legend.json
```

Open the `*_supersplat_debug.ply` in SuperSplat. Use the JSON legend to map
colors to class IDs; do not judge semantic colors from the raw PLY.

The standard validation command is:

```bash
conda run --no-capture-output -n semantic_3dgs_renderer \
  python -m scripts.task1.qa.validate_task1_outputs \
  --labels-npy "${FUSION_DIR}/gaussian_labels.npy" \
  --label-map "${FUSION_DIR}/label_map.json" \
  --semantic-ply "${FUSION_DIR}/semantic_point_cloud.ply"
```

Validation checks label-map consistency, used IDs, id `0`, and PLY vertex
count/label-property consistency. It does not establish visual correctness.
Visual acceptance requires original RGB views beside semantic overlays or the
class-colored SuperSplat PLY, especially at object boundaries and in views not
used for a small pilot.

## 14. Fusion-only command for existing base and masks

Once the pilot is visually accepted, reuse the accepted base's closed-set
DINOv3 cache and run the immutable OOV wrapper. The following is a template;
replace only placeholders and derive all absolute paths from the checkout or
workspace:

```bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd -P)"
SCENE=<scene>
BASE_OUTPUT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1/<accepted-dinov3-base>"
PILOT_OUTPUT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1/<accepted-dinotxt-pilot>"

env \
  SCENE="${SCENE}" \
  SOURCE_VIEW_DIR="${BASE_OUTPUT}/stages/01_real_camera_views" \
  DINO_INPUT_DIR="${BASE_OUTPUT}/stages/01_real_camera_views" \
  OOV_MANIFEST="${PILOT_OUTPUT}/grounded_sam_manifest.json" \
  OOV_MASK_DIR="${PILOT_OUTPUT}" \
  BASE_LABELS="${BASE_OUTPUT}/gaussian_labels.npy" \
  BASE_LABEL_MAP="${BASE_OUTPUT}/label_map.json" \
  OUTPUT_NAME="${SCENE}_dinov3_dinotxt_sam_oov_<target>_v1" \
  SKIP_DINOV3=1 \
  bash scripts/task1/dinov3/run_oov_multiclass_scene.sh
```

The wrapper refuses an existing output root. This is intentional: change the
output name for every threshold, view-count, prompt, or code revision.

If the base cache does not contain `dinov3_manifest.json`, set
`SKIP_DINOV3=0` and supply a complete `SOURCE_VIEW_DIR`, DINOv3 ViT-7B
checkpoints, and a clean pinned DINOv3 checkout. In that mode the wrapper
reruns closed-set segmentation before composing the OOV maps; it still expects
the accepted base labels/map as its fallback.

## 15. New scene/object checklist

For each new target:

1. Confirm the class is genuinely OOV or otherwise unreliable in the accepted
   ADE20K base.
2. Add a config with the target and plausible competitors; avoid scene
   coordinates or instance-specific wording.
3. Render the exact real cameras and run a small dino.txt/SAM review pilot.
4. Inspect ranked masks and selected overlays against the RGB views.
5. Run the full-camera pilot using the same `view_manifest.json`.
6. Run the OOV compositor against the immutable accepted base.
7. Inspect `multiclass_fusion_summary.json`, transitions, evidence arrays, and
   the Supersplat PLY.
8. Validate counts and PLY structure, then review multiple views visually.
9. Keep rejected runs as versioned diagnostics; never overwrite or fold them
   into the accepted base without a new review decision.

## 16. Known failure modes and correct responses

| Symptom | Likely boundary | Response |
| --- | --- | --- |
| Probabilities are near `1 / class_count`; no masks selected | dino.txt scaling or checkpoint mismatch | Confirm learned `logit_scale` is recorded (the corrected run is about `100`), verify hashes and pinned repo |
| Many rails/strips/handles from one object | redundant SAM masks | Keep containment suppression, IoU NMS, and the mask cap; inspect the selected mask stack |
| Good 2D panels but halos/streaks in 3D | 2D-to-3D projection/lifting | Inspect view votes and ownership evidence; do not loosen all gates from a single view |
| A nearby lookalike wins | target/competitor ambiguity | Improve the competitive vocabulary or abstain; do not add a manual Gaussian allowlist to this global route |
| OOV output is empty | insufficient visible/agreement/mass support | Inspect per-view masks and evidence arrays; a safe empty result is preferable to leakage |
| `Output exists` | immutable output policy | Choose a new versioned output name |
| PLY opens without semantic colors | raw semantic PLY was opened | Open `semantic_point_cloud_supersplat_debug.ply` instead |

GroundingDINO+SAM remains a separate proposal/correction workflow. It is not a
hidden dependency of this dino.txt route, and its reviewed source allowlists
must not be mixed into this automatic manifest without an explicit, tested
conversion.

## 17. Local checks and provenance

Before transferring a reviewed commit to a cluster checkout:

Run the Python checks inside the repository's configured semantic/DINOv3
environment; a minimal desktop Python without NumPy or PyTorch cannot import
these tests.

```bash
python -m py_compile \
  scripts/task1/dinov3/dinotxt_sam_mask_pilot.py \
  scripts/task1/dinov3/compose_oov_multiclass_votes.py
python -m unittest discover -s tests -p 'test_dinotxt_sam_mask_pilot.py'
python -m unittest discover -s tests -p 'test_dinov3_oov_multiclass.py'
bash -n scripts/task1/dinov3/run_oov_multiclass_scene.sh
git diff --check
```

Record the exact repository commit, DINOv3 repository commit, checkpoint
hashes, source PLY hash, base output name, view-manifest hash, config path,
thresholds, and output name in the run log/summary. The generated
`dinotxt_sam_pilot_report.json`, `multiclass_fusion_summary.json`, and
`vote_manifest.json` are the primary provenance records.

## 18. Long-running execution policy

The user owns Slurm submission and job monitoring. Do not submit or monitor
Slurm jobs from the assistant. For direct execution, use a detached session
with a log and status file, for example:

```bash
tmux new-session -d -s <session> \
  'env <variables> bash scripts/task1/dinov3/run_oov_multiclass_scene.sh \
   > <log-path> 2>&1; printf "%s\n" "$?" > <status-path>'
```

The completion signal is a status file containing `0`; a nonzero value means
the log must be inspected before retrying. Assistant-side observation is
periodic and stops after five minutes. After that, the user can check:

```bash
cat <status-path>
tail -n 80 <log-path>
```

Never delete a partial output blindly. First verify the exact versioned output
root and preserve its log for diagnosis.
