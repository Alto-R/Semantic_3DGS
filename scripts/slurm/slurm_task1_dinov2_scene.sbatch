#!/usr/bin/env bash
#SBATCH --job-name=task1_dinov2_scene
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
EXTERNAL_ROOT="${WORKSPACE_ROOT}/external"
FLASHSPLAT_ROOT="${EXTERNAL_ROOT}/FlashSplat"
DINOV2_ROOT="${EXTERNAL_ROOT}/dinov2"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:-room}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov2_multiview_v1}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
VIEW_DIR="${OUT_DIR}/stages/01_real_camera_views"
VOTE_DIR="${OUT_DIR}/stages/02_flashsplat_votes"
FUSION_DIR="${OUT_DIR}/stages/03_exact_fusion"
DELIVERABLE_DIR="${OUT_DIR}/deliverables"
VISUALIZATION_DIR="${OUT_DIR}/visualizations"
QA_RGB_DIR="${VISUALIZATION_DIR}/qa_rgb"
QA_OVERLAY_DIR="${VISUALIZATION_DIR}/semantic_overlays"
CONTACT_DIR="${VISUALIZATION_DIR}/contact_sheets"
VALIDATION_DIR="${OUT_DIR}/validation"
RUN_LOG_DIR="${OUT_DIR}/logs"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
DINOV2_ENV="${DINOV2_ENV:-dinov2_segmentation}"
DINOV2_CHECKPOINT_DIR="${DINOV2_CHECKPOINT_DIR:-${DINOV2_ROOT}/checkpoints}"
DINOV2_BACKBONE="${DINOV2_BACKBONE:-${DINOV2_CHECKPOINT_DIR}/dinov2_vitl14_pretrain.pth}"
DINOV2_HEAD_CONFIG="${DINOV2_HEAD_CONFIG:-${DINOV2_CHECKPOINT_DIR}/dinov2_vitl14_ade20k_linear_config.py}"
DINOV2_HEAD_CHECKPOINT="${DINOV2_HEAD_CHECKPOINT:-${DINOV2_CHECKPOINT_DIR}/dinov2_vitl14_ade20k_linear_head.pth}"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"

CAMERA_SOURCE="${CAMERA_SOURCE:-real}"
DINOV2_HEAD="${DINOV2_HEAD:-linear}"
CAMERA_INDICES="${CAMERA_INDICES:-}"
VIEW_COUNT="${VIEW_COUNT:-0}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
QA_VIEW_COUNT="${QA_VIEW_COUNT:-12}"
MIN_PIXEL_CONFIDENCE="${MIN_PIXEL_CONFIDENCE:-0.50}"
FLASHSPLAT_SUPPORT_THRESHOLD="${FLASHSPLAT_SUPPORT_THRESHOLD:-0.05}"
MIN_VIEWS="${MIN_VIEWS:-2}"
MIN_AGREEMENT="${MIN_AGREEMENT:-0.50}"
MIN_ASSIGNED_GAUSSIANS="${MIN_ASSIGNED_GAUSSIANS:-0}"
MIN_ASSIGNED_THING_GAUSSIANS="${MIN_ASSIGNED_THING_GAUSSIANS:-5000}"
MIN_ASSIGNED_STUFF_GAUSSIANS="${MIN_ASSIGNED_STUFF_GAUSSIANS:-10000}"
ADAPTIVE_STUFF_MIN_VIEW_RATIO="${ADAPTIVE_STUFF_MIN_VIEW_RATIO:-0.50}"
ADAPTIVE_STUFF_THRESHOLD_RATIO="${ADAPTIVE_STUFF_THRESHOLD_RATIO:-0.75}"
ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO="${ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO:-0.50}"
RESET_OUTPUT="${RESET_OUTPUT:-1}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"
BASELINE_LABELS="${BASELINE_LABELS:-}"

if [[ "${CAMERA_SOURCE}" != "real" ]]; then
  echo "v1 supports only real cameras from cameras.json; orbit views are deferred" >&2
  exit 2
fi
if [[ "${DINOV2_HEAD}" != "linear" ]]; then
  echo "v1 supports only the official ADE20K linear head" >&2
  exit 2
fi

test -d "${MODEL_DIR}"
test -f "${MODEL_DIR}/cameras.json"
test -d "${FLASHSPLAT_ROOT}"
test -d "${DINOV2_ROOT}"
test -f "${DINOV2_BACKBONE}"
test -f "${DINOV2_HEAD_CONFIG}"
test -f "${DINOV2_HEAD_CHECKPOINT}"
test -f "${ONTOLOGY}"
if [[ -n "${BASELINE_LABELS}" ]]; then
  test -f "${BASELINE_LABELS}"
fi

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nmodel=%s\noutput=%s\ncamera_source=%s\nview_count=%s\ndinov2_head=%s\n' \
    "${SCENE}" "${MODEL_DIR}" "${OUT_DIR}" "${CAMERA_SOURCE}" "${VIEW_COUNT}" "${DINOV2_HEAD}"
  exit 0
fi

if [[ "${RESET_OUTPUT}" = "1" && -d "${OUT_DIR}" ]]; then
  case "${OUT_DIR}" in
    "${TASK1_ROOT}"/*) rm -rf -- "${OUT_DIR}" ;;
    *) echo "Refusing to reset unexpected output path: ${OUT_DIR}" >&2; exit 2 ;;
  esac
fi

mkdir -p \
  "${VIEW_DIR}" "${VOTE_DIR}" "${FUSION_DIR}" "${DELIVERABLE_DIR}" \
  "${QA_RGB_DIR}" "${QA_OVERLAY_DIR}" "${CONTACT_DIR}" \
  "${VALIDATION_DIR}" "${RUN_LOG_DIR}" "${TASK1_ROOT}/logs"

cd "${PROJECT_ROOT}"

run_stage() {
  local stage_name="$1"
  local environment="$2"
  shift 2
  echo "===== ${stage_name} ====="
  conda run --no-capture-output -n "${environment}" "$@" 2>&1 | tee "${RUN_LOG_DIR}/${stage_name}.log"
}

run_report() {
  local stage_name="$1"
  local environment="$2"
  local report_path="$3"
  shift 3
  echo "===== ${stage_name} ====="
  conda run --no-capture-output -n "${environment}" "$@" \
    | tee "${report_path}" "${RUN_LOG_DIR}/${stage_name}.log"
}

{
  echo "job=${SLURM_JOB_ID:-local}"
  echo "host=$(hostname)"
  echo "scene=${SCENE}"
  echo "model=${MODEL_DIR}"
  echo "output=${OUT_DIR}"
  echo "camera_source=${CAMERA_SOURCE}"
  echo "camera_indices=${CAMERA_INDICES:-<all>}"
  echo "view_count=${VIEW_COUNT}"
  echo "render_max_width=${RENDER_MAX_WIDTH}"
  echo "dinov2_head=${DINOV2_HEAD}"
  echo "min_pixel_confidence=${MIN_PIXEL_CONFIDENCE}"
  echo "flashsplat_support_threshold=${FLASHSPLAT_SUPPORT_THRESHOLD}"
  echo "min_views=${MIN_VIEWS}"
  echo "min_agreement=${MIN_AGREEMENT}"
  echo "baseline_labels=${BASELINE_LABELS:-<not_provided>}"
  nvidia-smi
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

run_stage 01_render_real_camera_views "${GAUSSIAN_ENV}" \
  python scripts/task1/render_task1_views.py \
    --model-path "${MODEL_DIR}" \
    --output-dir "${VIEW_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --camera-indices "${CAMERA_INDICES}" \
    --count "${VIEW_COUNT}" \
    --overwrite

run_stage 02_dinov2_ade20k_linear "${DINOV2_ENV}" \
  python scripts/task1/dinov2_segment_views.py \
    --input-dir "${VIEW_DIR}" \
    --dinov2-root "${DINOV2_ROOT}" \
    --backbone-checkpoint "${DINOV2_BACKBONE}" \
    --head-config "${DINOV2_HEAD_CONFIG}" \
    --head-checkpoint "${DINOV2_HEAD_CHECKPOINT}" \
    --ontology "${ONTOLOGY}" \
    --min-pixel-confidence "${MIN_PIXEL_CONFIDENCE}" \
    --device cuda \
    --overwrite

run_stage 03_lift_per_view_votes "${GAUSSIAN_ENV}" \
  python scripts/task1/lift_dinov2_view_votes.py \
    --model-path "${MODEL_DIR}" \
    --input-dir "${VIEW_DIR}" \
    --output-dir "${VOTE_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --ontology "${ONTOLOGY}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --min-pixel-confidence "${MIN_PIXEL_CONFIDENCE}" \
    --support-threshold "${FLASHSPLAT_SUPPORT_THRESHOLD}" \
    --overwrite

FUSION_BASELINE_ARGS=()
if [[ -n "${BASELINE_LABELS}" ]]; then
  FUSION_BASELINE_ARGS=(--baseline-labels "${BASELINE_LABELS}")
fi

run_stage 04_exact_multiview_fusion "${GAUSSIAN_ENV}" \
  python scripts/task1/fuse_dinov2_multiview_votes.py \
    --model-path "${MODEL_DIR}" \
    --vote-dir "${VOTE_DIR}" \
    --output-dir "${FUSION_DIR}" \
    --ontology "${ONTOLOGY}" \
    --iteration 30000 \
    --scene "${SCENE}" \
    --min-views "${MIN_VIEWS}" \
    --min-agreement "${MIN_AGREEMENT}" \
    --voxel-scale-multiplier 4 \
    --min-voxel-size 0.01 \
    --max-voxel-size 0.20 \
    --min-component-gaussians 500 \
    --min-component-ratio 0.01 \
    --min-assigned-gaussians "${MIN_ASSIGNED_GAUSSIANS}" \
    --min-assigned-thing-gaussians "${MIN_ASSIGNED_THING_GAUSSIANS}" \
    --min-assigned-stuff-gaussians "${MIN_ASSIGNED_STUFF_GAUSSIANS}" \
    --adaptive-stuff-min-view-ratio "${ADAPTIVE_STUFF_MIN_VIEW_RATIO}" \
    --adaptive-stuff-threshold-ratio "${ADAPTIVE_STUFF_THRESHOLD_RATIO}" \
    --adaptive-thing-threshold-floor-ratio "${ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO}" \
    "${FUSION_BASELINE_ARGS[@]}" \
    --overwrite

cp "${FUSION_DIR}/semantic_point_cloud.ply" "${DELIVERABLE_DIR}/semantic_point_cloud.ply"
cp "${FUSION_DIR}/label_map.json" "${DELIVERABLE_DIR}/label_map.json"

run_stage 05_supersplat_debug_ply "${GAUSSIAN_ENV}" \
  python scripts/task1/export_supersplat_label_colors.py \
    --input-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-ply "${VISUALIZATION_DIR}/semantic_point_cloud_supersplat_debug.ply" \
    --metadata-json "${VISUALIZATION_DIR}/semantic_point_cloud_supersplat_debug.json" \
    --overwrite

run_stage 06_semantic_overlays "${GAUSSIAN_ENV}" \
  python scripts/task1/render_auto_label_overlays.py \
    --model-path "${MODEL_DIR}" \
    --labels-npy "${FUSION_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-dir "${QA_OVERLAY_DIR}" \
    --rgb-output-dir "${QA_RGB_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --max-labels 0 \
    --count "${QA_VIEW_COUNT}"

run_stage 07_visible_coverage "${GAUSSIAN_ENV}" \
  python scripts/task1/measure_overlay_coverage.py \
    --rgb-dir "${QA_RGB_DIR}" \
    --overlay-dir "${QA_OVERLAY_DIR}" \
    --output "${VALIDATION_DIR}/visible_overlay_coverage.json" \
    --difference-threshold 2 \
    --exclude-border 1

run_stage 08_contact_sheets "${GAUSSIAN_ENV}" \
  python scripts/task1/make_contact_sheet.py \
    --image-dir "${QA_OVERLAY_DIR}" \
    --output "${CONTACT_DIR}/semantic_labels.png" \
    --columns 4 \
    --thumb-width 360

run_report 09_task1_validation "${GAUSSIAN_ENV}" "${VALIDATION_DIR}/task1_validation.json" \
  python scripts/task1/validate_task1_outputs.py \
    --labels-npy "${FUSION_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --semantic-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --visible-coverage "${VALIDATION_DIR}/visible_overlay_coverage.json"

find "${OUT_DIR}" -maxdepth 4 -type f | sort
