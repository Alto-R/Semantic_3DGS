#!/usr/bin/env bash
# Automatic incremental DINOv3 evidence and report-only abstention recovery.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${WORKSPACE_ROOT}/external/FlashSplat}"

SCENE="${SCENE:?Set SCENE to playroom or drjohnson}"
SOURCE_HARD_AUDIT_OUTPUT_NAME="${SOURCE_HARD_AUDIT_OUTPUT_NAME:?Set the completed hard-vote audit output}"
SOURCE_VISIBILITY_OUTPUT_NAME="${SOURCE_VISIBILITY_OUTPUT_NAME:?Set the completed all-camera visibility output}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_detected_abstention_recovery_v1}"
ADDITIONAL_VIEW_OUTPUT_NAME="${ADDITIONAL_VIEW_OUTPUT_NAME:-${OUTPUT_NAME}_additional_views}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"

HARD_ROOT="${TASK1_ROOT}/${SOURCE_HARD_AUDIT_OUTPUT_NAME}"
BASELINE_VOTE_MANIFEST="${HARD_ROOT}/stages/01_flashsplat_votes/vote_manifest.json"
HARD_AUDIT_REPORT="${HARD_ROOT}/stages/02_round_trip_fidelity/round_trip_fidelity_report.json"
HARD_DIAGNOSTICS="${HARD_ROOT}/stages/02_round_trip_fidelity/gaussian_camera_agreement_diagnostics.npz"
VISIBILITY_DIR="${TASK1_ROOT}/${SOURCE_VISIBILITY_OUTPUT_NAME}/audit"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
SELECTION_DIR="${OUT_DIR}/selection"
ADDITIONAL_VIEW_ROOT="${TASK1_ROOT}/${ADDITIONAL_VIEW_OUTPUT_NAME}"
ADDITIONAL_VIEW_DIR="${ADDITIONAL_VIEW_ROOT}/stages/01_real_camera_views"
ADDITIONAL_VOTE_DIR="${OUT_DIR}/stages/01_additional_flashsplat_votes"
RECOVERY_DIR="${OUT_DIR}/stages/02_detected_abstention_recovery"
LOG_DIR="${OUT_DIR}/logs"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-semantic_3dgs_renderer}"
DINOV3_ENV="${DINOV3_ENV:-dinov3_semantic}"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
DINOV3_CROP_SIZE="${DINOV3_CROP_SIZE:-896}"
DINOV3_STRIDE="${DINOV3_STRIDE:-596}"
DINOV3_PRECISION="${DINOV3_PRECISION:-bfloat16}"
DINOV3_CHECKPOINT_LOAD_MODE="${DINOV3_CHECKPOINT_LOAD_MODE:-local_mmap}"
DINOV3_MAX_CUDA_MEMORY_GIB="${DINOV3_MAX_CUDA_MEMORY_GIB:-42}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

if [[ "${SCENE}" != "playroom" && "${SCENE}" != "drjohnson" ]]; then
  echo "This two-scene experiment accepts only playroom or drjohnson" >&2
  exit 2
fi
for boolean in CONFIG_ONLY; do
  value="${!boolean}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${boolean} must be 0 or 1" >&2
    exit 2
  fi
done
if [[ -n "${CAMERA_INDICES:-}" || -n "${VIEW_COUNT:-}" ]]; then
  echo "Manual CAMERA_INDICES and VIEW_COUNT are not accepted" >&2
  exit 2
fi
if [[ -n "${MIN_VIEWS:-}" || -n "${CAMERA_MAJORITY_THRESHOLD:-}" ]]; then
  echo "Consensus thresholds are fixed by the audit contract" >&2
  exit 2
fi

for path in \
  "${MODEL_DIR}/cameras.json" \
  "${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply" \
  "${BASELINE_VOTE_MANIFEST}" \
  "${HARD_AUDIT_REPORT}" \
  "${HARD_DIAGNOSTICS}" \
  "${VISIBILITY_DIR}/all_camera_visibility_report.json" \
  "${VISIBILITY_DIR}/camera_visibility_bits.npy" \
  "${VISIBILITY_DIR}/camera_indices.npy" \
  "${ONTOLOGY}"; do
  test -f "${path}"
done
test -d "${FLASHSPLAT_ROOT}"

printf 'scene=%s\nsource_hard_audit=%s\nsource_visibility=%s\noutput=%s\nadditional_view_output=%s\ncamera_selection=automatic_detected_abstention_visibility_and_pose_diversity\ncoverage_target=0.99\naccepted_hard_labels_immutable=1\nmanual_camera_selection_used=0\nmanual_gaussian_selection_used=0\nreport_only=1\nsemantic_ply_written=0\n' \
  "${SCENE}" "${HARD_ROOT}" "${VISIBILITY_DIR}" "${OUT_DIR}" \
  "${ADDITIONAL_VIEW_ROOT}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  exit 0
fi
for fresh in "${OUT_DIR}" "${ADDITIONAL_VIEW_ROOT}"; do
  if [[ -e "${fresh}" ]]; then
    echo "Output exists; choose fresh OUTPUT_NAME and ADDITIONAL_VIEW_OUTPUT_NAME values:" >&2
    echo "${fresh}" >&2
    exit 2
  fi
done

mkdir -p "${LOG_DIR}" "${TASK1_ROOT}/logs"
cd "${PROJECT_ROOT}"

run_stage() {
  local stage="$1"
  shift
  echo "===== ${stage} ====="
  "$@" 2>&1 | tee "${LOG_DIR}/${stage}.log"
}

{
  echo "job=${SLURM_JOB_ID:-local}"
  echo "host=$(hostname)"
  echo "project=${PROJECT_ROOT}"
  echo "scene=${SCENE}"
  echo "accepted_hard_labels_immutable=1"
  echo "manual_camera_selection_used=0"
  echo "manual_gaussian_selection_used=0"
  echo "v5_used=0"
  echo "dinov2_used=0"
  nvidia-smi
} | tee "${LOG_DIR}/00_run_configuration.log"

run_stage 01_select_additional_evidence_cameras \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.select_abstention_evidence_cameras \
    --visibility-dir "${VISIBILITY_DIR}" \
    --baseline-vote-manifest "${BASELINE_VOTE_MANIFEST}" \
    --hard-diagnostics "${HARD_DIAGNOSTICS}" \
    --cameras-json "${MODEL_DIR}/cameras.json" \
    --output-dir "${SELECTION_DIR}"

ADDITIONAL_CAMERA_INDICES="$(tr -d '\r\n' < "${SELECTION_DIR}/additional_camera_indices.txt")"
ADDITIONAL_VIEW_COUNT="$(awk -F, '{print NF}' "${SELECTION_DIR}/additional_camera_indices.txt")"
if [[ -z "${ADDITIONAL_CAMERA_INDICES}" || "${ADDITIONAL_VIEW_COUNT}" -lt 1 ]]; then
  echo "Automatic selector returned no additional cameras" >&2
  exit 2
fi

run_stage 02_incremental_dinov3_views \
  env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    SCENE="${SCENE}" \
    MODEL_DIR="${MODEL_DIR}" \
    OUTPUT_NAME="${ADDITIONAL_VIEW_OUTPUT_NAME}" \
    FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT}" \
    GAUSSIAN_ENV="${GAUSSIAN_ENV}" \
    DINOV3_ENV="${DINOV3_ENV}" \
    CAMERA_INDICES="${ADDITIONAL_CAMERA_INDICES}" \
    VIEW_COUNT="${ADDITIONAL_VIEW_COUNT}" \
    RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH}" \
    DINOV3_CROP_SIZE="${DINOV3_CROP_SIZE}" \
    DINOV3_STRIDE="${DINOV3_STRIDE}" \
    DINOV3_PRECISION="${DINOV3_PRECISION}" \
    DINOV3_CHECKPOINT_LOAD_MODE="${DINOV3_CHECKPOINT_LOAD_MODE}" \
    DINOV3_MAX_CUDA_MEMORY_GIB="${DINOV3_MAX_CUDA_MEMORY_GIB}" \
    REPORT_ONLY=1 \
    WRITE_SEMANTIC_PLY=0 \
    RESET_OUTPUT=0 \
    CONFIG_ONLY=0 \
    bash "${PROJECT_ROOT}/scripts/slurm/slurm_task1_dinov3_scene.a100.sbatch"

run_stage 03_lift_additional_hard_evidence \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.lift_dense_view_votes \
    --model-path "${MODEL_DIR}" \
    --input-dir "${ADDITIONAL_VIEW_DIR}" \
    --segmentation-manifest "${ADDITIONAL_VIEW_DIR}/dinov3_manifest.json" \
    --output-dir "${ADDITIONAL_VOTE_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --ontology "${ONTOLOGY}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}"

run_stage 04_report_detected_abstention_recovery \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.recover_detected_abstentions \
    --scene "${SCENE}" \
    --baseline-vote-manifest "${BASELINE_VOTE_MANIFEST}" \
    --additional-vote-manifest "${ADDITIONAL_VOTE_DIR}/vote_manifest.json" \
    --hard-audit-report "${HARD_AUDIT_REPORT}" \
    --hard-diagnostics "${HARD_DIAGNOSTICS}" \
    --selection-report "${SELECTION_DIR}/additional_camera_selection_report.json" \
    --ontology "${ONTOLOGY}" \
    --output-dir "${RECOVERY_DIR}"

cat > "${OUT_DIR}/experiment_mode.txt" <<EOF
mode=automatic_detected_abstention_recovery_audit
scene=${SCENE}
source_hard_audit=${SOURCE_HARD_AUDIT_OUTPUT_NAME}
source_visibility=${SOURCE_VISIBILITY_OUTPUT_NAME}
additional_view_output=${ADDITIONAL_VIEW_OUTPUT_NAME}
camera_selection=automatic_visibility_gain_pose_novelty
coverage_target=0.99
accepted_hard_labels_immutable=1
additional_evidence_first=equal_camera_strict_majority
secondary_evidence=wilson_camera_reliability_times_within_camera_winning_mass
secondary_acceptance=at_least_two_cameras_unique_weighted_strict_majority
manual_camera_selection_used=0
manual_gaussian_selection_used=0
v5_used=0
dinov2_used=0
accepted_gaussian_labels_written=0
label_map_written=0
semantic_ply_written=0
EOF

test -f "${SELECTION_DIR}/additional_camera_selection_report.json"
test -f "${ADDITIONAL_VOTE_DIR}/vote_manifest.json"
test -f "${RECOVERY_DIR}/detected_abstention_recovery_report.json"
test -f "${RECOVERY_DIR}/diagnostic_candidate_project_class_ids.npy"
test ! -e "${OUT_DIR}/gaussian_labels.npy"
test ! -e "${OUT_DIR}/label_map.json"
if find "${OUT_DIR}" -type f -name '*.ply' -print -quit | grep -q .; then
  echo "Report-only recovery unexpectedly wrote a PLY" >&2
  exit 2
fi

find "${OUT_DIR}" -maxdepth 5 -type f | sort
