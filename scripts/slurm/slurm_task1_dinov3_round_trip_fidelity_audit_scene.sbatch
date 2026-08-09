#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_roundtrip
#SBATCH --account=gpulab02
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

# Automatic report-only 2D-to-3D-to-2D fidelity audit.  This consumes the
# validated threshold-selected DINOv3 cache, reruns only FlashSplat lifting,
# and evaluates every source camera by leave-one-camera-out consensus.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
SOURCE_CACHE_OUTPUT_NAME="${SOURCE_CACHE_OUTPUT_NAME:?Set SOURCE_CACHE_OUTPUT_NAME to the completed automatic selected-view DINOv3 cache}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_round_trip_fidelity_audit_v1}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
SOURCE_ROOT="${TASK1_ROOT}/${SOURCE_CACHE_OUTPUT_NAME}"
SOURCE_VIEW_DIR="${SOURCE_ROOT}/stages/01_real_camera_views"
SOURCE_CACHE_REPORT="${SOURCE_ROOT}/selection/selected_view_cache_report.json"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
VOTE_DIR="${OUT_DIR}/stages/01_flashsplat_votes"
AUDIT_DIR="${OUT_DIR}/stages/02_round_trip_fidelity"
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
if [[ -n "${MIN_VIEWS:-}" || -n "${CAMERA_MAJORITY_THRESHOLD:-}" ]]; then
  echo "Consensus thresholds are fixed by the report-only audit contract" >&2
  exit 2
fi

test -f "${MODEL_DIR}/cameras.json"
test -f "${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
test -f "${SOURCE_VIEW_DIR}/view_manifest.json"
test -f "${SOURCE_VIEW_DIR}/dinov3_manifest.json"
test -f "${SOURCE_CACHE_REPORT}"
test -f "${ONTOLOGY}"
test -d "${FLASHSPLAT_ROOT}"

printf 'scene=%s\nsource_cache=%s\noutput=%s\nreport_only=1\ndinov3_inference_rerun=0\nflashsplat_lifting_rerun=1\nmanual_camera_selection_used=0\ncamera_vote_policy=one_unique_winner_or_abstain\nconsensus_policy=at_least_two_remaining_cameras_and_unique_strict_majority\nheldout_policy=every_cached_camera_excluded_from_its_own_prediction\naccepted_gaussian_labels_written=0\nlabel_map_written=0\nsemantic_ply_written=0\n' \
  "${SCENE}" "${SOURCE_ROOT}" "${OUT_DIR}"

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

run_stage 01_flashsplat_dense_view_votes \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.lift_dense_view_votes \
    --model-path "${MODEL_DIR}" \
    --input-dir "${SOURCE_VIEW_DIR}" \
    --segmentation-manifest "${SOURCE_VIEW_DIR}/dinov3_manifest.json" \
    --output-dir "${VOTE_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --ontology "${ONTOLOGY}" \
    --iteration 30000 \
    --max-width 960

run_stage 02_leave_one_camera_out_round_trip \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.round_trip_fidelity_audit \
    --model-path "${MODEL_DIR}" \
    --source-view-dir "${SOURCE_VIEW_DIR}" \
    --selected-cache-report "${SOURCE_CACHE_REPORT}" \
    --vote-manifest "${VOTE_DIR}/vote_manifest.json" \
    --ontology "${ONTOLOGY}" \
    --output-dir "${AUDIT_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000

run_stage 03_overlay_contact_sheet \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${AUDIT_DIR}/heldout_overlays" \
    --output "${CONTACT_DIR}/heldout_round_trip_overlays.png" \
    --columns 5 \
    --thumb-width 240

run_stage 04_disagreement_contact_sheet \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${AUDIT_DIR}/heldout_disagreement" \
    --output "${CONTACT_DIR}/heldout_round_trip_disagreement.png" \
    --columns 5 \
    --thumb-width 240

cat > "${OUT_DIR}/experiment_mode.txt" <<EOF
mode=automatic_cached_dinov3_round_trip_fidelity_audit
source_cache_output=${SOURCE_CACHE_OUTPUT_NAME}
report_only=1
dinov3_inference_rerun=0
flashsplat_lifting_rerun=1
manual_camera_selection_used=0
manual_gaussian_selection_used=0
camera_vote_policy=one_unique_max_class_per_camera_and_gaussian_else_abstain
consensus_policy=at_least_two_cameras_unique_strict_majority_else_abstain
heldout_policy=exclude_target_camera_before_consensus
accepted_gaussian_labels_written=0
gaussian_project_class_array_written=0
label_map_written=0
semantic_ply_written=0
EOF

test -f "${AUDIT_DIR}/round_trip_fidelity_report.json"
test -f "${AUDIT_DIR}/gaussian_camera_agreement_diagnostics.npz"
test -f "${AUDIT_DIR}/heldout_confusion_matrix.npz"
test -f "${CONTACT_DIR}/heldout_round_trip_overlays.png"
test -f "${CONTACT_DIR}/heldout_round_trip_disagreement.png"
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
