#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_visibility
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

# Report-only visibility audit over every real reconstruction camera.
# It runs no DINO model and writes no semantic labels, label map, or PLY.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${WORKSPACE_ROOT}/external/FlashSplat}"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_all_camera_visibility_v1}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
AUDIT_DIR="${OUT_DIR}/audit"
RUN_LOG_DIR="${OUT_DIR}/logs"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
ITERATION="${ITERATION:-30000}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
VISIBILITY_SUPPORT_THRESHOLD="${VISIBILITY_SUPPORT_THRESHOLD:-0.0}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

if [[ "${CONFIG_ONLY}" != "0" && "${CONFIG_ONLY}" != "1" ]]; then
  echo "CONFIG_ONLY must be 0 or 1" >&2
  exit 2
fi
if ! [[ "${ITERATION}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ITERATION must be a positive integer" >&2
  exit 2
fi
if ! [[ "${RENDER_MAX_WIDTH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RENDER_MAX_WIDTH must be a positive integer" >&2
  exit 2
fi
if ! [[ "${VISIBILITY_SUPPORT_THRESHOLD}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "VISIBILITY_SUPPORT_THRESHOLD must be a non-negative number" >&2
  exit 2
fi

test -d "${MODEL_DIR}"
test -f "${MODEL_DIR}/cameras.json"
test -f "${MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply"
test -d "${FLASHSPLAT_ROOT}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nmodel=%s\noutput=%s\nmode=all_reconstruction_camera_visibility_audit\ncamera_selection_policy=all_reconstruction_cameras_in_cameras_json_order\nrender_max_width=%s\nvisibility_support_threshold=%s\nsemantic_inference_used=0\nprior_semantic_labels_used=0\nreport_only=1\nsemantic_labels_written=0\nlabel_map_written=0\nsemantic_ply_written=0\n' \
    "${SCENE}" "${MODEL_DIR}" "${OUT_DIR}" "${RENDER_MAX_WIDTH}" \
    "${VISIBILITY_SUPPORT_THRESHOLD}"
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
  echo "model=${MODEL_DIR}"
  echo "output=${OUT_DIR}"
  echo "mode=all_reconstruction_camera_visibility_audit"
  echo "camera_selection_policy=all_reconstruction_cameras_in_cameras_json_order"
  echo "render_max_width=${RENDER_MAX_WIDTH}"
  echo "visibility_support_threshold=${VISIBILITY_SUPPORT_THRESHOLD}"
  echo "semantic_inference_used=0"
  echo "prior_semantic_labels_used=0"
  echo "report_only=1"
  echo "semantic_labels_written=0"
  echo "label_map_written=0"
  echo "semantic_ply_written=0"
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.audit_all_camera_visibility \
    --model-path "${MODEL_DIR}" \
    --output-dir "${AUDIT_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration "${ITERATION}" \
    --max-width "${RENDER_MAX_WIDTH}" \
    --support-threshold "${VISIBILITY_SUPPORT_THRESHOLD}" \
    2>&1 | tee "${RUN_LOG_DIR}/01_all_camera_visibility_audit.log"

for filename in \
  all_camera_visibility_report.json \
  camera_visibility_bits.npy \
  camera_indices.npy \
  visibility_view_count.npy \
  visibility_total_support.npy \
  visibility_max_support.npy \
  first_camera_index.npy \
  best_camera_index.npy \
  unobserved_gaussian_indices.npy \
  single_view_gaussian_indices.npy; do
  test -f "${AUDIT_DIR}/${filename}"
done
test ! -e "${AUDIT_DIR}/gaussian_labels.npy"
test ! -e "${AUDIT_DIR}/label_map.json"
if find "${OUT_DIR}" -type f -name '*.ply' -print -quit | grep -q .; then
  echo "Report-only visibility audit unexpectedly wrote a PLY" >&2
  exit 2
fi

cat > "${OUT_DIR}/experiment_mode.txt" <<EOF
mode=all_reconstruction_camera_visibility_audit
camera_selection_policy=all_reconstruction_cameras_in_cameras_json_order
visibility_definition=sum_flashsplat_used_count_over_all_mask_rows_gt_threshold
visibility_support_threshold=${VISIBILITY_SUPPORT_THRESHOLD}
semantic_inference_used=0
prior_semantic_labels_used=0
report_only=1
semantic_labels_written=0
label_map_written=0
semantic_ply_written=0
EOF

find "${OUT_DIR}" -maxdepth 3 -type f | sort
