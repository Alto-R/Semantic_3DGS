#!/usr/bin/env bash
#SBATCH --job-name=task1_recolor_output
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
FLASHSPLAT_ROOT="${WORKSPACE_ROOT}/external/FlashSplat"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:-room}"
SOURCE_NAME="${SOURCE_NAME:?Set SOURCE_NAME to an existing Task 1 output name}"
SOURCE_DIR="${TASK1_ROOT}/${SOURCE_NAME}"
OUTPUT_NAME="${OUTPUT_NAME:-${SOURCE_NAME}_stable_class_colors}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
COLOR_MODE="${COLOR_MODE:-class}"
QA_VIEW_COUNT="${QA_VIEW_COUNT:-12}"
CAMERA_INDICES="${CAMERA_INDICES:-}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
RESET_OUTPUT="${RESET_OUTPUT:-1}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"
GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"

SOURCE_LABEL_MAP="${SOURCE_LABEL_MAP:-${SOURCE_DIR}/deliverables/label_map.json}"
SOURCE_SEMANTIC_PLY="${SOURCE_SEMANTIC_PLY:-${SOURCE_DIR}/deliverables/semantic_point_cloud.ply}"
SOURCE_LABELS="${SOURCE_LABELS:-}"
if [[ -z "${SOURCE_LABELS}" ]]; then
  for candidate in \
    "${SOURCE_DIR}/deliverables/gaussian_labels.npy" \
    "${SOURCE_DIR}/stages/08_hybrid_merge/gaussian_labels.npy" \
    "${SOURCE_DIR}/stages/03_exact_fusion/gaussian_labels.npy" \
    "${SOURCE_DIR}/stages/03_semantic_fusion/gaussian_labels.npy"; do
    if [[ -f "${candidate}" ]]; then
      SOURCE_LABELS="${candidate}"
      break
    fi
  done
fi

if [[ "${COLOR_MODE}" != "class" && "${COLOR_MODE}" != "instance" ]]; then
  echo "COLOR_MODE must be class or instance" >&2
  exit 2
fi
test -d "${MODEL_DIR}"
test -d "${FLASHSPLAT_ROOT}"
test -f "${SOURCE_LABELS}"
test -f "${SOURCE_LABEL_MAP}"
test -f "${SOURCE_SEMANTIC_PLY}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nsource=%s\noutput=%s\nlabels=%s\ncolor_mode=%s\n' \
    "${SCENE}" "${SOURCE_DIR}" "${OUT_DIR}" "${SOURCE_LABELS}" "${COLOR_MODE}"
  exit 0
fi

if [[ "${RESET_OUTPUT}" = "1" && -d "${OUT_DIR}" ]]; then
  case "${OUT_DIR}" in
    "${TASK1_ROOT}"/*) rm -rf -- "${OUT_DIR}" ;;
    *) echo "Refusing to reset unexpected output path: ${OUT_DIR}" >&2; exit 2 ;;
  esac
fi

OVERLAY_DIR="${OUT_DIR}/visualizations/semantic_overlays"
RGB_DIR="${OUT_DIR}/visualizations/qa_rgb"
CONTACT_DIR="${OUT_DIR}/visualizations/contact_sheets"
VALIDATION_DIR="${OUT_DIR}/validation"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OVERLAY_DIR}" "${RGB_DIR}" "${CONTACT_DIR}" "${VALIDATION_DIR}" "${LOG_DIR}" "${TASK1_ROOT}/logs"
cd "${PROJECT_ROOT}"

run_stage() {
  local stage_name="$1"
  shift
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" "$@" 2>&1 \
    | tee "${LOG_DIR}/${stage_name}.log"
}

run_stage 01_supersplat_debug \
  python -m scripts.task1.qa.export_supersplat_label_colors \
    --input-ply "${SOURCE_SEMANTIC_PLY}" \
    --label-map "${SOURCE_LABEL_MAP}" \
    --output-ply "${OUT_DIR}/visualizations/semantic_point_cloud_supersplat_debug.ply" \
    --metadata-json "${OUT_DIR}/visualizations/semantic_point_cloud_supersplat_debug.json" \
    --color-mode "${COLOR_MODE}" \
    --overwrite

run_stage 02_semantic_overlays \
  python -m scripts.task1.qa.render_auto_label_overlays \
    --model-path "${MODEL_DIR}" \
    --labels-npy "${SOURCE_LABELS}" \
    --label-map "${SOURCE_LABEL_MAP}" \
    --output-dir "${OVERLAY_DIR}" \
    --rgb-output-dir "${RGB_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --camera-indices "${CAMERA_INDICES}" \
    --count "${QA_VIEW_COUNT}" \
    --max-labels 0 \
    --color-mode "${COLOR_MODE}"

run_stage 03_visible_coverage \
  python -m scripts.task1.qa.measure_overlay_coverage \
    --rgb-dir "${RGB_DIR}" \
    --overlay-dir "${OVERLAY_DIR}" \
    --output "${VALIDATION_DIR}/visible_overlay_coverage.json" \
    --difference-threshold 2 \
    --exclude-border 1

run_stage 04_contact_sheet \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${OVERLAY_DIR}" \
    --output "${CONTACT_DIR}/semantic_labels.png" \
    --columns 4 \
    --thumb-width 360

run_stage 05_color_legend \
  python -m scripts.task1.qa.write_semantic_color_legend \
    --label-map "${SOURCE_LABEL_MAP}" \
    --labels-npy "${SOURCE_LABELS}" \
    --output-json "${OUT_DIR}/visualizations/semantic_color_legend.json" \
    --output-png "${OUT_DIR}/visualizations/semantic_color_legend.png" \
    --color-mode "${COLOR_MODE}" \
    --overwrite

printf 'source_output=%s\nsource_labels=%s\nsource_label_map=%s\npalette_mode=%s\n' \
  "${SOURCE_DIR}" "${SOURCE_LABELS}" "${SOURCE_LABEL_MAP}" "${COLOR_MODE}" \
  | tee "${VALIDATION_DIR}/recolor_provenance.txt"

find "${OUT_DIR}" -maxdepth 4 -type f | sort
