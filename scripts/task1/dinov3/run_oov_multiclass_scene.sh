#!/usr/bin/env bash
# Run the generic DINOv3 + OOV-mask multiclass fusion without Slurm.
#
# The caller supplies an existing all-camera RGB/view manifest and an OOV
# mask manifest.  The script creates an immutable output root, segments the
# matching views, composes/lifts one multiclass map per camera, and writes a
# labeled PLY plus a class-colored SuperSplat PLY.  Set SKIP_DINOV3=1 and
# DINO_INPUT_DIR to reuse a completed DINOv3 stage from another output root.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SCENE="${SCENE:?Set SCENE to the target scene}"
SOURCE_VIEW_DIR="${SOURCE_VIEW_DIR:-}"
OOV_MANIFEST="${OOV_MANIFEST:?Set OOV_MANIFEST to the validated OOV mask manifest}"
OOV_MASK_DIR="${OOV_MASK_DIR:?Set OOV_MASK_DIR to the OOV mask directory or its output root}"
BASE_LABELS="${BASE_LABELS:?Set BASE_LABELS to the accepted fallback labels}"
BASE_LABEL_MAP="${BASE_LABEL_MAP:?Set BASE_LABEL_MAP to the accepted fallback label map}"

TASK1_ROOT="${TASK1_ROOT:-${WORKSPACE_ROOT}/outputs/eyenavgs_task1}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_dinotxt_sam_oov_multiclass_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TASK1_ROOT}/${OUTPUT_NAME}}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
ITERATION="${ITERATION:-30000}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${WORKSPACE_ROOT}/external/FlashSplat}"
DINOV3_ROOT="${DINOV3_ROOT:-${WORKSPACE_ROOT}/external/dinov3}"
DINOV3_BACKBONE="${DINOV3_BACKBONE:-${WORKSPACE_ROOT}/data/models/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth}"
DINOV3_SEGMENTOR="${DINOV3_SEGMENTOR:-${WORKSPACE_ROOT}/data/models/dinov3/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth}"
DINOV3_ENV="${DINOV3_ENV:-dinov3_semantic}"
GAUSSIAN_ENV="${GAUSSIAN_ENV:-semantic_3dgs_renderer}"
DINOV3_PRECISION="${DINOV3_PRECISION:-bfloat16}"
DINOV3_CROP_SIZE="${DINOV3_CROP_SIZE:-896}"
DINOV3_STRIDE="${DINOV3_STRIDE:-596}"
DINOV3_CHECKPOINT_LOAD_MODE="${DINOV3_CHECKPOINT_LOAD_MODE:-local_mmap}"
DINOV3_MAX_CUDA_MEMORY_GIB="${DINOV3_MAX_CUDA_MEMORY_GIB:-42}"
OOV_CLASSES="${OOV_CLASSES:-}"
OOV_LABEL_IDS="${OOV_LABEL_IDS:-}"
MIN_VISIBLE_VIEWS="${MIN_VISIBLE_VIEWS:-3}"
MIN_OOV_WINNER_VIEWS="${MIN_OOV_WINNER_VIEWS:-2}"
MIN_OOV_WINNER_SHARE="${MIN_OOV_WINNER_SHARE:-0.50}"
MIN_OOV_MASS_SHARE="${MIN_OOV_MASS_SHARE:-0.35}"
MIN_OOV_POSITIVE_VIEWS="${MIN_OOV_POSITIVE_VIEWS:-2}"
OOV_POSITIVE_MASS_THRESHOLD="${OOV_POSITIVE_MASS_THRESHOLD:-0.50}"
WINNER_MARGIN="${WINNER_MARGIN:-0.05}"
SKIP_DINOV3="${SKIP_DINOV3:-0}"

VIEW_DIR="${OUTPUT_ROOT}/stages/01_real_camera_views"
FUSION_DIR="${OUTPUT_ROOT}/stages/02_oov_multiclass_votes"
DINO_INPUT_DIR="${DINO_INPUT_DIR:-${VIEW_DIR}}"
SEMANTIC_PLY="${FUSION_DIR}/semantic_point_cloud.ply"
SUPER_SPLAT_PLY="${FUSION_DIR}/semantic_point_cloud_supersplat_debug.ply"

case "${SKIP_DINOV3}" in
  0|1) ;;
  *)
    echo "SKIP_DINOV3 must be 0 or 1: ${SKIP_DINOV3}" >&2
    exit 2
    ;;
esac

for path in \
  "${OOV_MANIFEST}" \
  "${OOV_MASK_DIR}" \
  "${BASE_LABELS}" \
  "${BASE_LABEL_MAP}" \
  "${MODEL_DIR}/cameras.json" \
  "${MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply" \
  "${FLASHSPLAT_ROOT}" \
  "${ONTOLOGY}"; do
  test -e "${path}"
done
if [[ "${SKIP_DINOV3}" == "0" ]]; then
  : "${SOURCE_VIEW_DIR:?Set SOURCE_VIEW_DIR to an all-camera render directory}"
  for path in \
    "${SOURCE_VIEW_DIR}/view_manifest.json" \
    "${SOURCE_VIEW_DIR}/rgb_renders" \
    "${DINOV3_ROOT}" \
    "${DINOV3_BACKBONE}" \
    "${DINOV3_SEGMENTOR}"; do
    test -e "${path}"
  done
else
  test -d "${DINO_INPUT_DIR}"
  test -f "${DINO_INPUT_DIR}/dinov3_manifest.json"
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Output exists; choose a new OUTPUT_NAME: ${OUTPUT_ROOT}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"

if [[ "${SKIP_DINOV3}" == "0" ]]; then
  mkdir -p "${VIEW_DIR}"
  cp -- "${SOURCE_VIEW_DIR}/view_manifest.json" "${VIEW_DIR}/view_manifest.json"
  ln -s -- "${SOURCE_VIEW_DIR}/rgb_renders" "${VIEW_DIR}/rgb_renders"
  DINO_INPUT_DIR="${VIEW_DIR}"

  conda run --no-capture-output -n "${DINOV3_ENV}" \
    python -m scripts.task1.dinov3.dinov3_segment_views \
    --input-dir "${DINO_INPUT_DIR}" \
    --dinov3-root "${DINOV3_ROOT}" \
    --backbone-checkpoint "${DINOV3_BACKBONE}" \
    --segmentor-checkpoint "${DINOV3_SEGMENTOR}" \
    --hub-entry dinov3_vit7b16_ms \
    --expected-repo-commit 6876159a11b4df116f30f667f8c9888617df0751 \
    --expected-backbone-sha256 a955f4ea3bec4fcd666bf363630da4386383069b482c8a927e17a3e1154965b7 \
    --expected-segmentor-sha256 bf307cb1c2fd95046feb1bf9a8a13dae60a746bddd8f5297134da95525dbcb42 \
    --ontology "${ONTOLOGY}" \
    --min-pixel-confidence 0.0 \
    --device cuda \
    --precision "${DINOV3_PRECISION}" \
    --crop-size "${DINOV3_CROP_SIZE}" \
    --stride "${DINOV3_STRIDE}" \
    --checkpoint-load-mode "${DINOV3_CHECKPOINT_LOAD_MODE}" \
    --max-cuda-memory-gib "${DINOV3_MAX_CUDA_MEMORY_GIB}" \
    --overwrite
fi

CLASS_ARGS=()
if [[ -n "${OOV_CLASSES}" ]]; then
  CLASS_ARGS+=(--oov-classes "${OOV_CLASSES}")
fi
if [[ -n "${OOV_LABEL_IDS}" ]]; then
  CLASS_ARGS+=(--oov-label-ids "${OOV_LABEL_IDS}")
fi

conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.compose_oov_multiclass_votes \
  --scene "${SCENE}" \
  --model-path "${MODEL_DIR}" \
  --dense-input-dir "${DINO_INPUT_DIR}" \
  --segmentation-manifest "${DINO_INPUT_DIR}/dinov3_manifest.json" \
  --oov-manifest "${OOV_MANIFEST}" \
  --oov-mask-dir "${OOV_MASK_DIR}" \
  --base-labels "${BASE_LABELS}" \
  --base-label-map "${BASE_LABEL_MAP}" \
  --output-dir "${FUSION_DIR}" \
  --ontology "${ONTOLOGY}" \
  --flashsplat-root "${FLASHSPLAT_ROOT}" \
  --iteration "${ITERATION}" \
  --max-width "${RENDER_MAX_WIDTH}" \
  --min-visible-views "${MIN_VISIBLE_VIEWS}" \
  --min-oov-winner-views "${MIN_OOV_WINNER_VIEWS}" \
  --min-oov-winner-share "${MIN_OOV_WINNER_SHARE}" \
  --min-oov-mass-share "${MIN_OOV_MASS_SHARE}" \
  --min-oov-positive-views "${MIN_OOV_POSITIVE_VIEWS}" \
  --oov-positive-mass-threshold "${OOV_POSITIVE_MASS_THRESHOLD}" \
  --winner-margin "${WINNER_MARGIN}" \
  "${CLASS_ARGS[@]}"

conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.export_supersplat_label_colors \
  --input-ply "${SEMANTIC_PLY}" \
  --output-ply "${SUPER_SPLAT_PLY}" \
  --label-map "${FUSION_DIR}/label_map.json" \
  --metadata-json "${FUSION_DIR}/semantic_color_legend.json" \
  --overwrite

conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.validate_task1_outputs \
  --labels-npy "${FUSION_DIR}/gaussian_labels.npy" \
  --label-map "${FUSION_DIR}/label_map.json" \
  --semantic-ply "${SEMANTIC_PLY}"

printf 'output=%s\nsemantic_ply=%s\nsupersplat_ply=%s\n' \
  "${OUTPUT_ROOT}" "${SEMANTIC_PLY}" "${SUPER_SPLAT_PLY}"
