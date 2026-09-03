# DINOv3, dino.txt, and SAM: complete quickstart

This is the complete one-run path. It starts with a Graphdeco reconstruction,
creates fresh DINOv3 views and labels, runs dino.txt plus SAM on every real
camera, fuses the OOV masks back into the 3D Gaussians, and renders the final
labels back into per-camera PNGs.

No previously accepted semantic result is required. The DINOv3 cache used by
the fusion stage is created by the same run immediately before fusion.

The run produces:

- base DINOv3 semantic PLY and Supersplat PLY;
- dino.txt/SAM mask stacks, probabilities, and manifests;
- final OOV-fused semantic PLY and Supersplat PLY;
- final labels, label maps, legends, summaries, and vote manifests;
- per-camera class-colored FlashSplat render-back PNGs plus RGB PNGs.

Run this directly with bash from the repository root. Do not submit this
file to Slurm.

## Minimum requirements

- A Graphdeco model at:

  ~~~text
  <workspace>/data/3dgs_models/graphdeco/<scene>/cameras.json
  <workspace>/data/3dgs_models/graphdeco/<scene>/point_cloud/iteration_30000/point_cloud.ply
  ~~~

- The pinned DINOv3 checkout at commit
  6876159a11b4df116f30f667f8c9888617df0751.
- DINOv3 ViT-7B ADE20K checkpoints:

  ~~~text
  dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
  dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth
  ~~~

- dino.txt ViT-L files:

  ~~~text
  dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
  dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth
  bpe_simple_vocab_16e6.txt.gz
  ~~~

- FlashSplat, Segment Anything, CUDA, and conda.
- The dinov3_semantic and semantic_3dgs_renderer environments.
- A SAM ViT-H checkpoint.
- A target config in configs/ with target_class, its prompts, and any
  competing classes. The config filename used below is:
  configs/task1_dinotxt_sam.<scene>_<target>.json.

## One complete run

Set the scene and target, then paste the whole block. Every output name must
be new because the pipeline refuses to overwrite an existing run.

~~~bash
set -euo pipefail
cd <path-to-repository>

export PROJECT_ROOT="$(pwd -P)"
export WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd -P)"
export TASK1_ROOT="$WORKSPACE_ROOT/outputs/eyenavgs_task1"

export SCENE="drjohnson"
export TARGET_CLASS="window_shutter"
export ITERATION=30000
export VIEW_COUNT=0
export RENDER_MAX_WIDTH=960

printf -v TARGET_CONFIG '%s/configs/task1_dinotxt_sam.%s_%s.json' \
  "$PROJECT_ROOT" "$SCENE" "$TARGET_CLASS"
printf -v BASE_OUTPUT_NAME '%s_dinov3_abstention_recovery_review_v1' "$SCENE"
printf -v DINO_TXT_OUTPUT_NAME '%s_dinotxt_sam_%s_all_views_v1' \
  "$SCENE" "$TARGET_CLASS"
printf -v FINAL_OUTPUT_NAME '%s_dinov3_dinotxt_sam_oov_%s_v1' \
  "$SCENE" "$TARGET_CLASS"

export BASE_ROOT="$TASK1_ROOT/$BASE_OUTPUT_NAME"
export BASE_VIEW_DIR="$BASE_ROOT/stages/01_real_camera_views"
export BASE_LABELS="$BASE_ROOT/gaussian_labels.npy"
export BASE_LABEL_MAP="$BASE_ROOT/label_map.json"
export DINO_TXT_OUTPUT="$TASK1_ROOT/$DINO_TXT_OUTPUT_NAME"
export FINAL_ROOT="$TASK1_ROOT/$FINAL_OUTPUT_NAME"
export FINAL_FUSION_DIR="$FINAL_ROOT/stages/02_oov_multiclass_votes"

export DINOV3_ROOT="$WORKSPACE_ROOT/external/dinov3"
export FLASHSPLAT_ROOT="$WORKSPACE_ROOT/external/FlashSplat"
export SEGMENT_ANYTHING_ROOT="$WORKSPACE_ROOT/external/segment-anything"
export ONTOLOGY="$PROJECT_ROOT/configs/ade20k_to_project.json"
export SAM_CHECKPOINT="<path-to-sam-vit-h-checkpoint>"

export DINOV3_BACKBONE="$WORKSPACE_ROOT/data/models/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
export DINOV3_SEGMENTOR="$WORKSPACE_ROOT/data/models/dinov3/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"
export DINOTXT_BACKBONE="$WORKSPACE_ROOT/data/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
export DINOTXT_CHECKPOINT="$WORKSPACE_ROOT/data/models/dinov3/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
export BPE_PATH="$WORKSPACE_ROOT/data/models/dinov3/bpe_simple_vocab_16e6.txt.gz"

# 1. Graphdeco -> fresh RGB views -> DINOv3 labels -> base PLYs.
PROJECT_ROOT="$PROJECT_ROOT" \
SCENE="$SCENE" \
OUTPUT_NAME="$BASE_OUTPUT_NAME" \
VIEW_COUNT="$VIEW_COUNT" \
ITERATION="$ITERATION" \
RENDER_MAX_WIDTH="$RENDER_MAX_WIDTH" \
DINOV3_ROOT="$DINOV3_ROOT" \
FLASHSPLAT_ROOT="$FLASHSPLAT_ROOT" \
DINOV3_BACKBONE="$DINOV3_BACKBONE" \
DINOV3_SEGMENTOR="$DINOV3_SEGMENTOR" \
ONTOLOGY="$ONTOLOGY" \
bash scripts/slurm/slurm_task1_dinov3_end_to_end_recovery_scene.sbatch

# Use every camera rendered by the preceding stage.
export CAMERA_INDICES="$(python -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); print(",".join(str(f["camera_index"]) for f in d["frames"]))' "$BASE_VIEW_DIR/view_manifest.json")"

# 2. Fresh dino.txt classification + SAM masks for every real camera.
conda run --no-capture-output -n dinov3_semantic \
  python -m scripts.task1.dinov3.dinotxt_sam_mask_pilot \
  --view-dir "$BASE_VIEW_DIR" \
  --camera-indices "$CAMERA_INDICES" \
  --config "$TARGET_CONFIG" \
  --output-dir "$DINO_TXT_OUTPUT" \
  --dinov3-root "$DINOV3_ROOT" \
  --backbone-checkpoint "$DINOTXT_BACKBONE" \
  --dinotxt-checkpoint "$DINOTXT_CHECKPOINT" \
  --bpe-path "$BPE_PATH" \
  --dinotxt-hub-entry dinov3_vitl16_dinotxt_tet1280d20h24l \
  --segment-anything-root "$SEGMENT_ANYTHING_ROOT" \
  --sam-checkpoint "$SAM_CHECKPOINT" \
  --device cuda \
  --resize 512 \
  --side 384 \
  --stride 192

# 3. Fresh OOV lift/fusion using the DINOv3 cache made in step 1.
PROJECT_ROOT="$PROJECT_ROOT" \
SCENE="$SCENE" \
TASK1_ROOT="$TASK1_ROOT" \
MODEL_DIR="$WORKSPACE_ROOT/data/3dgs_models/graphdeco/$SCENE" \
ITERATION="$ITERATION" \
RENDER_MAX_WIDTH="$RENDER_MAX_WIDTH" \
ONTOLOGY="$ONTOLOGY" \
FLASHSPLAT_ROOT="$FLASHSPLAT_ROOT" \
DINO_INPUT_DIR="$BASE_VIEW_DIR" \
OOV_MANIFEST="$DINO_TXT_OUTPUT/grounded_sam_manifest.json" \
OOV_MASK_DIR="$DINO_TXT_OUTPUT" \
BASE_LABELS="$BASE_LABELS" \
BASE_LABEL_MAP="$BASE_LABEL_MAP" \
OUTPUT_NAME="$FINAL_OUTPUT_NAME" \
SKIP_DINOV3=1 \
bash scripts/task1/dinov3/run_oov_multiclass_scene.sh

# 4. Final labels -> per-camera class-colored FlashSplat PNGs.
conda run --no-capture-output -n semantic_3dgs_renderer \
  python -m scripts.task1.qa.render_auto_label_overlays \
  --model-path "$WORKSPACE_ROOT/data/3dgs_models/graphdeco/$SCENE" \
  --labels-npy "$FINAL_FUSION_DIR/gaussian_labels.npy" \
  --label-map "$FINAL_FUSION_DIR/label_map.json" \
  --output-dir "$FINAL_ROOT/visualizations/semantic_overlays" \
  --rgb-output-dir "$FINAL_ROOT/visualizations/rgb_renders" \
  --source-view-manifest "$BASE_VIEW_DIR/view_manifest.json" \
  --flashsplat-root "$FLASHSPLAT_ROOT" \
  --iteration "$ITERATION" \
  --camera-indices "$CAMERA_INDICES" \
  --count 0 \
  --max-width "$RENDER_MAX_WIDTH" \
  --max-labels 64 \
  --color-mode class \
  --overlay-alpha 1.0
~~~

SEGMENT_ANYTHING_ROOT must point to the checkout directory that contains
the segment_anything Python package. If that checkout is nested inside another
repository, set the variable to the nested directory.

## Output layout

The base DINOv3 stage writes:

~~~text
$BASE_ROOT/stages/01_real_camera_views/view_manifest.json
$BASE_ROOT/stages/01_real_camera_views/rgb_renders/
$BASE_ROOT/stages/01_real_camera_views/dinov3_manifest.json
$BASE_ROOT/stages/02_flashsplat_votes/
$BASE_ROOT/gaussian_labels.npy
$BASE_ROOT/label_map.json
$BASE_ROOT/semantic_point_cloud.ply
$BASE_ROOT/semantic_point_cloud_supersplat_debug.ply
$BASE_ROOT/summary.json
~~~

The dino.txt/SAM stage writes:

~~~text
$DINO_TXT_OUTPUT/grounded_sam_manifest.json
$DINO_TXT_OUTPUT/mask_stacks/
$DINO_TXT_OUTPUT/probabilities/
$DINO_TXT_OUTPUT/selected_masks/
$DINO_TXT_OUTPUT/selected_overlays/
$DINO_TXT_OUTPUT/ranked_overlays/
$DINO_TXT_OUTPUT/review_panels/
~~~

The final OOV fusion writes:

~~~text
$FINAL_FUSION_DIR/gaussian_labels.npy
$FINAL_FUSION_DIR/label_map.json
$FINAL_FUSION_DIR/semantic_point_cloud.ply
$FINAL_FUSION_DIR/semantic_point_cloud_supersplat_debug.ply
$FINAL_FUSION_DIR/semantic_color_legend.json
$FINAL_FUSION_DIR/multiclass_fusion_summary.json
$FINAL_FUSION_DIR/vote_manifest.json
~~~

The final per-camera render-back stage writes:

~~~text
$FINAL_ROOT/visualizations/semantic_overlays/*.png
$FINAL_ROOT/visualizations/rgb_renders/*.png
$FINAL_ROOT/visualizations/semantic_overlays/label_overlay_manifest.json
~~~

Open the final semantic_point_cloud_supersplat_debug.ply in Supersplat. The
base, dino.txt/SAM, and final fusion roots all stay under TASK1_ROOT, so the
complete run record is easy to find in one output directory.
