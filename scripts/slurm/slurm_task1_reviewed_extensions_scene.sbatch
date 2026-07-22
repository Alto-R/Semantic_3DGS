#!/usr/bin/env bash
#SBATCH --job-name=task1_reviewed_ext
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"
FLASHSPLAT_ROOT="${WORKSPACE_ROOT}/external/FlashSplat"

SCENE="${SCENE:?Set SCENE to the target scene}"
BASE_OUTPUT_NAME="${BASE_OUTPUT_NAME:?Set BASE_OUTPUT_NAME to an existing v5 output}"
OUTPUT_NAME="${OUTPUT_NAME:-${BASE_OUTPUT_NAME}_reviewed_extensions_v1}"
SOURCE_NAME="${SOURCE_NAME:-${SCENE}_dinov2_separate_abstain_allviews_v1}"
BASE_DIR="${TASK1_ROOT}/${BASE_OUTPUT_NAME}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
SOURCE_VIEW_DIR="${TASK1_ROOT}/${SOURCE_NAME}/stages/01_real_camera_views"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"

BASE_LABELS="${BASE_DIR}/deliverables/gaussian_labels.npy"
BASE_LABEL_MAP="${BASE_DIR}/deliverables/label_map.json"
EXTENSION_CONFIG="${EXTENSION_CONFIG:?Set EXTENSION_CONFIG to a reviewed extension config}"
EXTENSION_LABELS="${EXTENSION_LABELS:?Set EXTENSION_LABELS to reviewed source labels}"
EXTENSION_LABEL_MAP="${EXTENSION_LABEL_MAP:?Set EXTENSION_LABEL_MAP to the source label map}"
EXTENSION_MANIFEST="${EXTENSION_MANIFEST:-}"
EXTENSION_CLASSES="${EXTENSION_CLASSES:-}"

MERGE_DIR="${OUT_DIR}/stages/01_reviewed_extension_merge"
DELIVERABLE_DIR="${OUT_DIR}/deliverables"
VISUALIZATION_DIR="${OUT_DIR}/visualizations"
QA_RGB_DIR="${VISUALIZATION_DIR}/qa_rgb"
QA_OVERLAY_DIR="${VISUALIZATION_DIR}/semantic_overlays"
CONTACT_DIR="${VISUALIZATION_DIR}/contact_sheets"
VALIDATION_DIR="${OUT_DIR}/validation"
RUN_LOG_DIR="${OUT_DIR}/logs"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
COLOR_MODE="${COLOR_MODE:-class}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
QA_VIEW_COUNT="${QA_VIEW_COUNT:-12}"
RESET_OUTPUT="${RESET_OUTPUT:-0}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

if [[ "${COLOR_MODE}" != "class" && "${COLOR_MODE}" != "instance" ]]; then
  echo "COLOR_MODE must be class or instance" >&2
  exit 2
fi
if [[ "${RESET_OUTPUT}" != "0" && "${RESET_OUTPUT}" != "1" ]]; then
  echo "RESET_OUTPUT must be 0 or 1" >&2
  exit 2
fi

test -d "${FLASHSPLAT_ROOT}"
test -f "${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
test -f "${SOURCE_VIEW_DIR}/view_manifest.json"
test -f "${BASE_LABELS}"
test -f "${BASE_LABEL_MAP}"
test -f "${EXTENSION_CONFIG}"
test -f "${EXTENSION_LABELS}"
test -f "${EXTENSION_LABEL_MAP}"
test -f "${ONTOLOGY}"
if [[ -n "${EXTENSION_MANIFEST}" ]]; then
  test -f "${EXTENSION_MANIFEST}"
fi

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nbase=%s\noutput=%s\nextension_config=%s\nextension_labels=%s\nextension_label_map=%s\nextension_manifest=%s\nextension_classes=%s\n' \
    "${SCENE}" "${BASE_DIR}" "${OUT_DIR}" "${EXTENSION_CONFIG}" \
    "${EXTENSION_LABELS}" "${EXTENSION_LABEL_MAP}" \
    "${EXTENSION_MANIFEST}" "${EXTENSION_CLASSES}"
  exit 0
fi

if [[ -e "${OUT_DIR}" ]]; then
  if [[ "${RESET_OUTPUT}" != "1" || ! -d "${OUT_DIR}" ]]; then
    echo "Output exists; choose a new OUTPUT_NAME or set RESET_OUTPUT=1: ${OUT_DIR}" >&2
    exit 2
  fi
  case "${OUT_DIR}" in
    "${TASK1_ROOT}"/*) rm -rf -- "${OUT_DIR}" ;;
    *) echo "Refusing to reset unexpected output path: ${OUT_DIR}" >&2; exit 2 ;;
  esac
fi

mkdir -p \
  "${MERGE_DIR}" "${DELIVERABLE_DIR}" \
  "${QA_RGB_DIR}" "${QA_OVERLAY_DIR}" "${CONTACT_DIR}" \
  "${VALIDATION_DIR}" "${RUN_LOG_DIR}" "${TASK1_ROOT}/logs"

cd "${PROJECT_ROOT}"

run_stage() {
  local stage_name="$1"
  shift
  echo "===== ${stage_name} ====="
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" "$@" 2>&1 \
    | tee "${RUN_LOG_DIR}/${stage_name}.log"
}

run_report() {
  local stage_name="$1"
  local report_path="$2"
  shift 2
  echo "===== ${stage_name} ====="
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" "$@" \
    | tee "${report_path}" "${RUN_LOG_DIR}/${stage_name}.log"
}

MANIFEST_ARGS=()
if [[ -n "${EXTENSION_MANIFEST}" ]]; then
  MANIFEST_ARGS=(--extension-manifest "${EXTENSION_MANIFEST}")
fi

run_stage 01_reviewed_extension_merge \
  python -m scripts.task1.merge.merge_semantic_extensions \
    --base-labels "${BASE_LABELS}" \
    --base-label-map "${BASE_LABEL_MAP}" \
    --extension-labels "${EXTENSION_LABELS}" \
    --extension-label-map "${EXTENSION_LABEL_MAP}" \
    "${MANIFEST_ARGS[@]}" \
    --extension-config "${EXTENSION_CONFIG}" \
    --ontology "${ONTOLOGY}" \
    --source-ply "${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply" \
    --output-dir "${MERGE_DIR}" \
    --semantic-ply-path "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --scene "${SCENE}" \
    --include-classes "${EXTENSION_CLASSES}" \
    --overwrite

cp "${MERGE_DIR}/gaussian_labels.npy" "${DELIVERABLE_DIR}/gaussian_labels.npy"
cp "${MERGE_DIR}/label_map.json" "${DELIVERABLE_DIR}/label_map.json"

run_stage 02_supersplat_debug_ply \
  python -m scripts.task1.qa.export_supersplat_label_colors \
    --input-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-ply "${VISUALIZATION_DIR}/semantic_point_cloud_supersplat_debug.ply" \
    --metadata-json "${VISUALIZATION_DIR}/semantic_point_cloud_supersplat_debug.json" \
    --color-mode "${COLOR_MODE}" \
    --overwrite

run_stage 03_semantic_overlays \
  python -m scripts.task1.qa.render_auto_label_overlays \
    --model-path "${MODEL_DIR}" \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-dir "${QA_OVERLAY_DIR}" \
    --rgb-output-dir "${QA_RGB_DIR}" \
    --source-view-manifest "${SOURCE_VIEW_DIR}/view_manifest.json" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --max-labels 0 \
    --count "${QA_VIEW_COUNT}" \
    --color-mode "${COLOR_MODE}"

run_stage 04_visible_coverage \
  python -m scripts.task1.qa.measure_overlay_coverage \
    --rgb-dir "${QA_RGB_DIR}" \
    --overlay-dir "${QA_OVERLAY_DIR}" \
    --output "${VALIDATION_DIR}/visible_overlay_coverage.json" \
    --difference-threshold 2 \
    --exclude-border 1

run_stage 05_contact_sheet \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${QA_OVERLAY_DIR}" \
    --output "${CONTACT_DIR}/semantic_labels.png" \
    --columns 4 \
    --thumb-width 360

run_stage 06_semantic_color_legend \
  python -m scripts.task1.qa.write_semantic_color_legend \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --output-json "${VISUALIZATION_DIR}/semantic_color_legend.json" \
    --output-png "${VISUALIZATION_DIR}/semantic_color_legend.png" \
    --color-mode "${COLOR_MODE}" \
    --overwrite

run_report 07_task1_validation "${VALIDATION_DIR}/task1_validation.json" \
  python -m scripts.task1.qa.validate_task1_outputs \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --semantic-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --visible-coverage "${VALIDATION_DIR}/visible_overlay_coverage.json"

printf 'final_mode=adaptive_instance_guard_v5_reviewed_extensions_v1\nscene=%s\nbase=%s\nextension_config=%s\nextension_labels=%s\nextension_label_map=%s\nextension_manifest=%s\nextension_classes=%s\n' \
  "${SCENE}" "${BASE_DIR}" "${EXTENSION_CONFIG}" \
  "${EXTENSION_LABELS}" "${EXTENSION_LABEL_MAP}" \
  "${EXTENSION_MANIFEST}" "${EXTENSION_CLASSES}" \
  > "${VALIDATION_DIR}/final_mode.txt"

find "${OUT_DIR}" -maxdepth 4 -type f | sort
