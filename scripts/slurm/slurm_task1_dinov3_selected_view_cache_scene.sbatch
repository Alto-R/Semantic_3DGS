#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_selected_cache
#SBATCH --account=gpulab02
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

# Automatic report-only DINOv3 cache for the first visibility-ranked prefix
# reaching 99% two-view coverage of globally multiview-capable Gaussians.
# It accepts no manual camera list and runs no 3D vote lifting or fusion.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
CAMERA_SELECTION_OUTPUT_NAME="${CAMERA_SELECTION_OUTPUT_NAME:?Set CAMERA_SELECTION_OUTPUT_NAME to the completed visibility selection output}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_selected_99pct_two_view_cache_v1}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
SELECTION_DIR="${TASK1_ROOT}/${CAMERA_SELECTION_OUTPUT_NAME}/selection"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
BASE_SCHEDULER="${PROJECT_ROOT}/scripts/slurm/slurm_task1_dinov3_scene.a100.sbatch"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
DINOV3_ENV="${DINOV3_ENV:-dinov3_semantic}"
TARGET_TWO_VIEW_PERCENT="${TARGET_TWO_VIEW_PERCENT:-99}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

if [[ "${CONFIG_ONLY}" != "0" && "${CONFIG_ONLY}" != "1" ]]; then
  echo "CONFIG_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ "${TARGET_TWO_VIEW_PERCENT}" != "99" ]]; then
  echo "This reviewed cache stage requires TARGET_TWO_VIEW_PERCENT=99" >&2
  exit 2
fi
if [[ -n "${CAMERA_INDICES:-}" || -n "${VIEW_COUNT:-}" ]]; then
  echo "Manual CAMERA_INDICES and VIEW_COUNT are not accepted by this automatic stage" >&2
  exit 2
fi
if ! [[ "${TARGET_TWO_VIEW_PERCENT}" =~ ^[1-9][0-9]?$|^100$ ]]; then
  echo "TARGET_TWO_VIEW_PERCENT must be an integer from 1 through 100" >&2
  exit 2
fi

test -f "${BASE_SCHEDULER}"
test -f "${MODEL_DIR}/cameras.json"
test -f "${SELECTION_DIR}/camera_selection_report.json"
test -f "${SELECTION_DIR}/greedy_camera_indices.npy"
test -f "${SELECTION_DIR}/coverage_curve.npz"

cd "${PROJECT_ROOT}"
mapfile -t SELECTION_VALUES < <(
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
    python -m scripts.task1.dinov3.selected_view_cache resolve \
      --selection-dir "${SELECTION_DIR}" \
      --model-path "${MODEL_DIR}" \
      --target-percent "${TARGET_TWO_VIEW_PERCENT}" \
      --source-max-width 960 \
      --format lines
)
if [[ "${#SELECTION_VALUES[@]}" -ne 2 ]]; then
  echo "Automatic camera-prefix resolver returned unexpected output" >&2
  exit 2
fi
SELECTED_VIEW_COUNT="${SELECTION_VALUES[0]}"
SELECTED_CAMERA_INDICES="${SELECTION_VALUES[1]}"
if ! [[ "${SELECTED_VIEW_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Automatic camera-prefix resolver returned an invalid count" >&2
  exit 2
fi
if [[ -z "${SELECTED_CAMERA_INDICES}" ]]; then
  echo "Automatic camera-prefix resolver returned an empty prefix" >&2
  exit 2
fi

printf 'scene=%s\nselection=%s\noutput=%s\nthreshold_group=two_view_coverage_of_multiview_capable\ntarget_percent=%s\nselected_view_count=%s\nmanual_camera_selection_used=0\nreport_only=1\nsemantic_vote_lifting_run=0\nsemantic_fusion_run=0\nsemantic_labels_written=0\nlabel_map_written=0\nsemantic_ply_written=0\n' \
  "${SCENE}" "${SELECTION_DIR}" "${OUT_DIR}" \
  "${TARGET_TWO_VIEW_PERCENT}" "${SELECTED_VIEW_COUNT}"

COMMON_SETTINGS=(
  PROJECT_ROOT="${PROJECT_ROOT}"
  SCENE="${SCENE}"
  MODEL_DIR="${MODEL_DIR}"
  OUTPUT_NAME="${OUTPUT_NAME}"
  GAUSSIAN_ENV="${GAUSSIAN_ENV}"
  DINOV3_ENV="${DINOV3_ENV}"
  CAMERA_INDICES="${SELECTED_CAMERA_INDICES}"
  VIEW_COUNT="${SELECTED_VIEW_COUNT}"
  RENDER_MAX_WIDTH=960
  REPORT_ONLY=1
  WRITE_SEMANTIC_PLY=0
  RESET_OUTPUT=0
  DINOV3_PRECISION=bfloat16
  DINOV3_CROP_SIZE=512
  DINOV3_STRIDE=384
  DINOV3_CHECKPOINT_LOAD_MODE=local_mmap
  DINOV3_MAX_CUDA_MEMORY_GIB=40
  DINOV3_MIN_HOST_AVAILABLE_GIB=
  DINOV3_MIN_GPU_FREE_GIB=
  DINOV3_RESOURCE_GPU_ID=
  MIN_PIXEL_CONFIDENCE=0.0
)

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  env "${COMMON_SETTINGS[@]}" CONFIG_ONLY=1 bash "${BASE_SCHEDULER}"
  exit 0
fi

env "${COMMON_SETTINGS[@]}" CONFIG_ONLY=0 bash "${BASE_SCHEDULER}"

conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.selected_view_cache finalize \
    --selection-dir "${SELECTION_DIR}" \
    --model-path "${MODEL_DIR}" \
    --output-dir "${OUT_DIR}" \
    --target-percent "${TARGET_TWO_VIEW_PERCENT}" \
    --render-max-width 960 \
    --crop-size 512 \
    --stride 384 \
    --precision bfloat16 \
    --checkpoint-load-mode local_mmap \
    2>&1 | tee "${OUT_DIR}/logs/05_selected_view_cache_validation.log"

cat >> "${OUT_DIR}/experiment_mode.txt" <<EOF
selection_mode=automatic_visibility_threshold_prefix
source_camera_selection_output=${CAMERA_SELECTION_OUTPUT_NAME}
threshold_group=two_view_coverage_of_multiview_capable
target_percent=${TARGET_TWO_VIEW_PERCENT}
selected_view_count=${SELECTED_VIEW_COUNT}
manual_camera_selection_used=0
semantic_vote_lifting_run=0
semantic_fusion_run=0
label_map_written=0
EOF

test -f "${OUT_DIR}/selection/selected_view_cache_report.json"
test -f "${OUT_DIR}/selection/selected_camera_indices.npy"
test ! -e "${OUT_DIR}/stages/02_flashsplat_votes/gaussian_labels.npy"
test ! -e "${OUT_DIR}/stages/03_exact_fusion/gaussian_labels.npy"
test ! -e "${OUT_DIR}/deliverables/label_map.json"
if find "${OUT_DIR}" -type f -name '*.ply' -print -quit | grep -q .; then
  echo "Report-only selected-view cache unexpectedly wrote a PLY" >&2
  exit 2
fi

find "${OUT_DIR}" -maxdepth 4 -type f | sort
