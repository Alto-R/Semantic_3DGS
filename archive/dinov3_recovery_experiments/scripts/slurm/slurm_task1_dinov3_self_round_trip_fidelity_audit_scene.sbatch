#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_selfrt
#SBATCH --account=gpulab02
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

# Automatic report-only same-camera FlashSplat self-round-trip audit.  This
# consumes the validated selected-view DINOv3 cache and the complete dense
# FlashSplat vote cache from the leave-one-camera-out audit.  It reruns neither
# DINOv3 inference nor FlashSplat lifting and performs no cross-camera fusion.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
SOURCE_CACHE_OUTPUT_NAME="${SOURCE_CACHE_OUTPUT_NAME:?Set SOURCE_CACHE_OUTPUT_NAME to the completed automatic selected-view DINOv3 cache}"
SOURCE_VOTE_OUTPUT_NAME="${SOURCE_VOTE_OUTPUT_NAME:?Set SOURCE_VOTE_OUTPUT_NAME to the completed round-trip audit containing dense FlashSplat votes}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_self_round_trip_fidelity_audit_v1}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
SOURCE_CACHE_ROOT="${TASK1_ROOT}/${SOURCE_CACHE_OUTPUT_NAME}"
SOURCE_VIEW_DIR="${SOURCE_CACHE_ROOT}/stages/01_real_camera_views"
SOURCE_CACHE_REPORT="${SOURCE_CACHE_ROOT}/selection/selected_view_cache_report.json"
SOURCE_VOTE_ROOT="${TASK1_ROOT}/${SOURCE_VOTE_OUTPUT_NAME}/stages/01_flashsplat_votes"
VOTE_MANIFEST="${SOURCE_VOTE_ROOT}/vote_manifest.json"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
AUDIT_DIR="${OUT_DIR}/stages/01_self_round_trip_fidelity"
CONTACT_DIR="${OUT_DIR}/visualizations/contact_sheets"
LOG_DIR="${OUT_DIR}/logs"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${WORKSPACE_ROOT}/external/FlashSplat}"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

if [[ "${CONFIG_ONLY}" != "0" && "${CONFIG_ONLY}" != "1" ]]; then
  echo "CONFIG_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ -n "${CAMERA_INDICES:-}" || -n "${VIEW_COUNT:-}" ]]; then
  echo "Manual CAMERA_INDICES and VIEW_COUNT are not accepted by this automatic audit" >&2
  exit 2
fi
if [[ -n "${MIN_VIEWS:-}" || -n "${CAMERA_MAJORITY_THRESHOLD:-}" || -n "${MIN_PIXEL_CONFIDENCE:-}" ]]; then
  echo "Semantic and consensus thresholds are not accepted by this diagnostic contract" >&2
  exit 2
fi

test -f "${MODEL_DIR}/cameras.json"
test -f "${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
test -f "${SOURCE_VIEW_DIR}/view_manifest.json"
test -f "${SOURCE_VIEW_DIR}/dinov3_manifest.json"
test -f "${SOURCE_CACHE_REPORT}"
test -f "${VOTE_MANIFEST}"
test -f "${ONTOLOGY}"
test -d "${FLASHSPLAT_ROOT}"

printf 'scene=%s\nsource_cache=%s\nsource_votes=%s\noutput=%s\nreport_only=1\ndinov3_inference_rerun=0\nflashsplat_lifting_rerun=0\nflashsplat_vote_cache_reused=1\ncross_camera_fusion_used=0\nmanual_camera_selection_used=0\ncamera_vote_policy=one_unique_winner_or_abstain\nself_round_trip_policy=same_camera_only\naccepted_gaussian_labels_written=0\nlabel_map_written=0\nsemantic_ply_written=0\n' \
  "${SCENE}" "${SOURCE_CACHE_ROOT}" "${SOURCE_VOTE_ROOT}" "${OUT_DIR}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  exit 0
fi
if [[ -e "${OUT_DIR}" ]]; then
  echo "Output exists: ${OUT_DIR}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${CONTACT_DIR}"
cd "${PROJECT_ROOT}"

run_stage() {
  local stage="$1"
  shift
  echo "===== ${stage} ====="
  "$@" 2>&1 | tee "${LOG_DIR}/${stage}.log"
}

run_stage 01_same_camera_flashsplat_self_round_trip \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.self_round_trip_fidelity_audit \
    --model-path "${MODEL_DIR}" \
    --source-view-dir "${SOURCE_VIEW_DIR}" \
    --selected-cache-report "${SOURCE_CACHE_REPORT}" \
    --vote-manifest "${VOTE_MANIFEST}" \
    --ontology "${ONTOLOGY}" \
    --output-dir "${AUDIT_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000

run_stage 02_self_overlay_contact_sheet \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${AUDIT_DIR}/self_overlays" \
    --output "${CONTACT_DIR}/self_round_trip_overlays.png" \
    --columns 5 \
    --thumb-width 240

run_stage 03_self_disagreement_contact_sheet \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${AUDIT_DIR}/self_disagreement" \
    --output "${CONTACT_DIR}/self_round_trip_disagreement.png" \
    --columns 5 \
    --thumb-width 240

cat > "${OUT_DIR}/experiment_mode.txt" <<EOF
mode=automatic_cached_dinov3_same_camera_flashsplat_self_round_trip_fidelity_audit
source_cache_output=${SOURCE_CACHE_OUTPUT_NAME}
source_vote_output=${SOURCE_VOTE_OUTPUT_NAME}
report_only=1
dinov3_inference_rerun=0
flashsplat_lifting_rerun=0
flashsplat_vote_cache_reused=1
cross_camera_fusion_used=0
manual_camera_selection_used=0
manual_gaussian_selection_used=0
camera_vote_policy=one_unique_max_class_per_camera_and_gaussian_else_abstain
self_round_trip_policy=render_each_camera_only_from_its_own_cached_gaussian_identities
accepted_gaussian_labels_written=0
gaussian_project_class_array_written=0
label_map_written=0
semantic_ply_written=0
EOF

test -f "${AUDIT_DIR}/self_round_trip_fidelity_report.json"
test -f "${AUDIT_DIR}/self_confusion_matrix.npz"
test -f "${CONTACT_DIR}/self_round_trip_overlays.png"
test -f "${CONTACT_DIR}/self_round_trip_disagreement.png"
for forbidden in gaussian_labels.npy gaussian_project_class_ids.npy project_class_id.npy label_map.json semantic_point_cloud.ply; do
  if find "${OUT_DIR}" -type f -name "${forbidden}" -print -quit | grep -q .; then
    echo "Report-only audit unexpectedly wrote ${forbidden}" >&2
    exit 2
  fi
done
if find "${OUT_DIR}" -type f -name '*.ply' -print -quit | grep -q .; then
  echo "Report-only audit unexpectedly wrote a PLY" >&2
  exit 2
fi

find "${OUT_DIR}" -maxdepth 5 -type f | sort
