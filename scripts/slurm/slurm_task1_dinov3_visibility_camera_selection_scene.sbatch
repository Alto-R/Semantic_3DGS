#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_camera_select
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

# CPU-only, report-only deterministic camera ranking from an existing
# all-camera Gaussian visibility audit. It runs no renderer or DINO model and
# writes no semantic labels, label map, or PLY.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
VISIBILITY_OUTPUT_NAME="${VISIBILITY_OUTPUT_NAME:?Set VISIBILITY_OUTPUT_NAME to the completed all-camera visibility output}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_visibility_camera_selection_v1}"
SOURCE_DIR="${TASK1_ROOT}/${VISIBILITY_OUTPUT_NAME}/audit"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
SELECTION_DIR="${OUT_DIR}/selection"
RUN_LOG_DIR="${OUT_DIR}/logs"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

if [[ "${CONFIG_ONLY}" != "0" && "${CONFIG_ONLY}" != "1" ]]; then
  echo "CONFIG_ONLY must be 0 or 1" >&2
  exit 2
fi

for filename in \
  all_camera_visibility_report.json \
  camera_visibility_bits.npy \
  camera_indices.npy \
  visibility_view_count.npy; do
  test -f "${SOURCE_DIR}/${filename}"
done

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nsource=%s\noutput=%s\nmode=deterministic_greedy_visibility_multicover\nselection_policy=single_coverage_first_then_second_coverage\ntie_break_policy=source_camera_row_order\ncoverage_curve_written=1\nmanual_camera_selection_used=0\nsemantic_inference_used=0\nprior_semantic_labels_used=0\nrenderer_used=0\nreport_only=1\nsemantic_labels_written=0\nlabel_map_written=0\nsemantic_ply_written=0\n' \
    "${SCENE}" "${SOURCE_DIR}" "${OUT_DIR}"
  exit 0
fi

if [[ -e "${OUT_DIR}" ]]; then
  echo "Output exists; choose a fresh OUTPUT_NAME:" >&2
  echo "${OUT_DIR}" >&2
  exit 2
fi

mkdir -p "${RUN_LOG_DIR}" "${TASK1_ROOT}/logs"
cd "${PROJECT_ROOT}"

{
  echo "job=${SLURM_JOB_ID:-local}"
  echo "host=$(hostname)"
  echo "project=${PROJECT_ROOT}"
  echo "scene=${SCENE}"
  echo "source=${SOURCE_DIR}"
  echo "output=${OUT_DIR}"
  echo "mode=deterministic_greedy_visibility_multicover"
  echo "selection_policy=single_coverage_first_then_second_coverage"
  echo "tie_break_policy=source_camera_row_order"
  echo "manual_camera_selection_used=0"
  echo "semantic_inference_used=0"
  echo "prior_semantic_labels_used=0"
  echo "renderer_used=0"
  echo "report_only=1"
  echo "semantic_labels_written=0"
  echo "label_map_written=0"
  echo "semantic_ply_written=0"
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.select_visibility_cameras \
    --visibility-dir "${SOURCE_DIR}" \
    --output-dir "${SELECTION_DIR}" \
    2>&1 | tee "${RUN_LOG_DIR}/01_visibility_camera_selection.log"

for filename in \
  camera_selection_report.json \
  greedy_camera_rows.npy \
  greedy_camera_indices.npy \
  coverage_curve.npz; do
  test -f "${SELECTION_DIR}/${filename}"
done
test ! -e "${SELECTION_DIR}/gaussian_labels.npy"
test ! -e "${SELECTION_DIR}/label_map.json"
if find "${OUT_DIR}" -type f -name '*.ply' -print -quit | grep -q .; then
  echo "Report-only camera selection unexpectedly wrote a PLY" >&2
  exit 2
fi

cat > "${OUT_DIR}/experiment_mode.txt" <<EOF
mode=deterministic_greedy_visibility_multicover
source_visibility_output=${VISIBILITY_OUTPUT_NAME}
selection_policy=single_coverage_first_then_second_coverage
tie_break_policy=source_camera_row_order
manual_camera_selection_used=0
semantic_inference_used=0
prior_semantic_labels_used=0
renderer_used=0
report_only=1
semantic_labels_written=0
label_map_written=0
semantic_ply_written=0
EOF

find "${OUT_DIR}" -maxdepth 3 -type f | sort
