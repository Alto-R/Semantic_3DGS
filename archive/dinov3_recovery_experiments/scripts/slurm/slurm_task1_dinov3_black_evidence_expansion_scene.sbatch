#!/usr/bin/env bash
# Automatic black-evidence camera expansion, recovery, and component audit.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${WORKSPACE_ROOT}/external/FlashSplat}"

SCENE="${SCENE:?Set SCENE to playroom or drjohnson}"
SOURCE_HARD_AUDIT_OUTPUT_NAME="${SOURCE_HARD_AUDIT_OUTPUT_NAME:?Set the completed hard-vote audit output}"
SOURCE_VISIBILITY_OUTPUT_NAME="${SOURCE_VISIBILITY_OUTPUT_NAME:?Set the completed all-camera visibility output}"
SOURCE_RECOVERY_OUTPUT_NAME="${SOURCE_RECOVERY_OUTPUT_NAME:?Set the completed abstention recovery output}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_black_evidence_expansion_v1}"
ADDITIONAL_VIEW_OUTPUT_NAME="${ADDITIONAL_VIEW_OUTPUT_NAME:-${OUTPUT_NAME}_additional_views}"
NEW_RECOVERY_OUTPUT_NAME="${NEW_RECOVERY_OUTPUT_NAME:-${SCENE}_dinov3_black_evidence_recovery_v1}"
NEW_COMPONENT_AUDIT_OUTPUT_NAME="${NEW_COMPONENT_AUDIT_OUTPUT_NAME:-${SCENE}_dinov3_observed_black_component_graph_audit_v2}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"

HARD_ROOT="${TASK1_ROOT}/${SOURCE_HARD_AUDIT_OUTPUT_NAME}"
BASELINE_VOTE_MANIFEST="${HARD_ROOT}/stages/01_flashsplat_votes/vote_manifest.json"
HARD_AUDIT_REPORT="${HARD_ROOT}/stages/02_round_trip_fidelity/round_trip_fidelity_report.json"
HARD_DIAGNOSTICS="${HARD_ROOT}/stages/02_round_trip_fidelity/gaussian_camera_agreement_diagnostics.npz"
HARD_CONFUSION="${HARD_ROOT}/stages/02_round_trip_fidelity/heldout_confusion_matrix.npz"
VISIBILITY_DIR="${TASK1_ROOT}/${SOURCE_VISIBILITY_OUTPUT_NAME}/audit"
RECOVERY_ROOT="${TASK1_ROOT}/${SOURCE_RECOVERY_OUTPUT_NAME}"
EXISTING_ADDITIONAL_VOTE_DIR="${RECOVERY_ROOT}/stages/01_additional_flashsplat_votes"
RECOVERY_CANDIDATE_LABELS="${RECOVERY_ROOT}/stages/02_detected_abstention_recovery/diagnostic_candidate_project_class_ids.npy"
EXISTING_SELECTION_REPORT="${RECOVERY_ROOT}/selection/additional_camera_selection_report.json"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
SELECTION_DIR="${OUT_DIR}/selection"
ADDITIONAL_VIEW_ROOT="${TASK1_ROOT}/${ADDITIONAL_VIEW_OUTPUT_NAME}"
ADDITIONAL_VIEW_DIR="${ADDITIONAL_VIEW_ROOT}/stages/01_real_camera_views"
NEW_RECOVERY_ROOT="${TASK1_ROOT}/${NEW_RECOVERY_OUTPUT_NAME}"
NEW_ADDITIONAL_VOTE_DIR="${NEW_RECOVERY_ROOT}/stages/01_additional_flashsplat_votes"
NEW_RECOVERY_DIR="${NEW_RECOVERY_ROOT}/stages/02_detected_abstention_recovery"
NEW_COMPONENT_AUDIT_ROOT="${TASK1_ROOT}/${NEW_COMPONENT_AUDIT_OUTPUT_NAME}"
NEW_COMPONENT_AUDIT_DIR="${NEW_COMPONENT_AUDIT_ROOT}/stages/01_observed_black_component_graph_audit"
LOG_DIR="${OUT_DIR}/logs"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-semantic_3dgs_renderer}"
DINOV3_ENV="${DINOV3_ENV:-dinov3_semantic}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
DINOV3_CROP_SIZE="${DINOV3_CROP_SIZE:-896}"
DINOV3_STRIDE="${DINOV3_STRIDE:-596}"
DINOV3_PRECISION="${DINOV3_PRECISION:-bfloat16}"
DINOV3_CHECKPOINT_LOAD_MODE="${DINOV3_CHECKPOINT_LOAD_MODE:-local_mmap}"
DINOV3_MAX_CUDA_MEMORY_GIB="${DINOV3_MAX_CUDA_MEMORY_GIB:-42}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

if [[ "${SCENE}" != "playroom" && "${SCENE}" != "drjohnson" ]]; then
  echo "SCENE must be playroom or drjohnson" >&2
  exit 2
fi
if [[ "${CONFIG_ONLY}" != "0" && "${CONFIG_ONLY}" != "1" ]]; then
  echo "CONFIG_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ -n "${CAMERA_INDICES:-}" || -n "${VIEW_COUNT:-}" ]]; then
  echo "Manual camera selection is not accepted" >&2
  exit 2
fi
if [[ -n "${RECOVERY_CLASSES:-}" || -n "${RECOVERY_GAUSSIAN_INDICES:-}" ]]; then
  echo "Manual class or Gaussian selection is not accepted" >&2
  exit 2
fi

for path in \
  "${MODEL_DIR}/cameras.json" \
  "${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply" \
  "${BASELINE_VOTE_MANIFEST}" \
  "${HARD_AUDIT_REPORT}" \
  "${HARD_DIAGNOSTICS}" \
  "${HARD_CONFUSION}" \
  "${VISIBILITY_DIR}/all_camera_visibility_report.json" \
  "${VISIBILITY_DIR}/camera_visibility_bits.npy" \
  "${VISIBILITY_DIR}/camera_indices.npy" \
  "${EXISTING_ADDITIONAL_VOTE_DIR}/vote_manifest.json" \
  "${RECOVERY_CANDIDATE_LABELS}" \
  "${EXISTING_SELECTION_REPORT}" \
  "${ONTOLOGY}"; do
  test -f "${path}"
done
test -d "${FLASHSPLAT_ROOT}"

printf 'scene=%s\nsource_hard_audit=%s\nsource_visibility=%s\nsource_recovery=%s\noutput=%s\nadditional_view_output=%s\nnew_recovery_output=%s\nnew_component_audit_output=%s\ncamera_selection=automatic_current_black_visibility_and_pose_diversity\ncoverage_target=0.99\naccepted_hard_labels_immutable=1\nmanual_camera_selection_used=0\nmanual_gaussian_selection_used=0\nreport_only=1\naccepted_gaussian_labels_written=0\nsemantic_ply_written=0\n' \
  "${SCENE}" "${HARD_ROOT}" "${VISIBILITY_DIR}" "${RECOVERY_ROOT}" "${OUT_DIR}" \
  "${ADDITIONAL_VIEW_ROOT}" "${NEW_RECOVERY_ROOT}" "${NEW_COMPONENT_AUDIT_ROOT}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  exit 0
fi
for fresh in "${OUT_DIR}" "${ADDITIONAL_VIEW_ROOT}" "${NEW_RECOVERY_ROOT}" "${NEW_COMPONENT_AUDIT_ROOT}"; do
  if [[ -e "${fresh}" ]]; then
    echo "Output exists; choose fresh output names:" >&2
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

run_stage 01_select_black_evidence_cameras \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.select_black_evidence_cameras \
    --visibility-dir "${VISIBILITY_DIR}" \
    --baseline-vote-manifest "${BASELINE_VOTE_MANIFEST}" \
    --additional-vote-manifest "${EXISTING_ADDITIONAL_VOTE_DIR}/vote_manifest.json" \
    --recovery-candidate-labels "${RECOVERY_CANDIDATE_LABELS}" \
    --cameras-json "${MODEL_DIR}/cameras.json" \
    --ontology "${ONTOLOGY}" \
    --output-dir "${SELECTION_DIR}"

ADDITIONAL_CAMERA_INDICES="$(tr -d '\r\n' < "${SELECTION_DIR}/additional_camera_indices.txt")"
ADDITIONAL_VIEW_COUNT="$(awk -F, '{print NF}' "${SELECTION_DIR}/additional_camera_indices.txt")"
if [[ -z "${ADDITIONAL_CAMERA_INDICES}" || "${ADDITIONAL_VIEW_COUNT}" -lt 1 ]]; then
  echo "Automatic black-evidence selector returned no additional cameras" >&2
  exit 2
fi

run_stage 02_black_evidence_dinov3_views \
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

run_stage 03_lift_black_evidence_hard_votes \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.lift_dense_view_votes \
    --model-path "${MODEL_DIR}" \
    --input-dir "${ADDITIONAL_VIEW_DIR}" \
    --segmentation-manifest "${ADDITIONAL_VIEW_DIR}/dinov3_manifest.json" \
    --output-dir "${NEW_RECOVERY_ROOT}/stages/00_new_additional_votes" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --ontology "${ONTOLOGY}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}"

run_stage 04_merge_additional_vote_manifests \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.merge_additional_vote_manifests \
    --existing-vote-dir "${EXISTING_ADDITIONAL_VOTE_DIR}" \
    --new-vote-dir "${NEW_RECOVERY_ROOT}/stages/00_new_additional_votes" \
    --existing-selection-report "${EXISTING_SELECTION_REPORT}" \
    --new-selection-report "${SELECTION_DIR}/black_evidence_camera_selection_report.json" \
    --output-dir "${NEW_ADDITIONAL_VOTE_DIR}"

run_stage 05_report_black_evidence_recovery \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.recover_detected_abstentions \
    --scene "${SCENE}" \
    --baseline-vote-manifest "${BASELINE_VOTE_MANIFEST}" \
    --additional-vote-manifest "${NEW_ADDITIONAL_VOTE_DIR}/vote_manifest.json" \
    --hard-audit-report "${HARD_AUDIT_REPORT}" \
    --hard-diagnostics "${HARD_DIAGNOSTICS}" \
    --selection-report "${NEW_ADDITIONAL_VOTE_DIR}/additional_camera_selection_report.json" \
    --ontology "${ONTOLOGY}" \
    --output-dir "${NEW_RECOVERY_DIR}"

run_stage 06_rebuild_component_graph_audit \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.observed_black_component_graph_audit \
    --scene "${SCENE}" \
    --baseline-vote-manifest "${BASELINE_VOTE_MANIFEST}" \
    --additional-vote-manifest "${NEW_ADDITIONAL_VOTE_DIR}/vote_manifest.json" \
    --hard-audit-report "${HARD_AUDIT_REPORT}" \
    --hard-diagnostics "${HARD_DIAGNOSTICS}" \
    --hard-confusion "${HARD_CONFUSION}" \
    --recovery-report "${NEW_RECOVERY_DIR}/detected_abstention_recovery_report.json" \
    --recovery-candidate-labels "${NEW_RECOVERY_DIR}/diagnostic_candidate_project_class_ids.npy" \
    --recovery-source-codes "${NEW_RECOVERY_DIR}/diagnostic_recovery_source_codes.npy" \
    --source-ply "${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply" \
    --ontology "${ONTOLOGY}" \
    --output-dir "${NEW_COMPONENT_AUDIT_DIR}"

test -f "${SELECTION_DIR}/black_evidence_camera_selection_report.json"
test -f "${NEW_ADDITIONAL_VOTE_DIR}/vote_manifest.json"
test -f "${NEW_RECOVERY_DIR}/detected_abstention_recovery_report.json"
test -f "${NEW_COMPONENT_AUDIT_DIR}/observed_black_component_graph_audit.json"
test -f "${NEW_COMPONENT_AUDIT_DIR}/observed_black_component_graph_diagnostics.npz"
if find "${OUT_DIR}" "${NEW_RECOVERY_ROOT}" "${NEW_COMPONENT_AUDIT_ROOT}" -type f -name '*.ply' -print -quit | grep -q .; then
  echo "Black-evidence expansion unexpectedly wrote a PLY" >&2
  exit 2
fi
if find "${OUT_DIR}" "${NEW_RECOVERY_ROOT}" "${NEW_COMPONENT_AUDIT_ROOT}" -type f \( -name 'gaussian_labels.npy' -o -name 'label_map.json' \) -print -quit | grep -q .; then
  echo "Black-evidence expansion unexpectedly wrote accepted labels" >&2
  exit 2
fi

find "${OUT_DIR}" "${NEW_RECOVERY_ROOT}" "${NEW_COMPONENT_AUDIT_ROOT}" -maxdepth 4 -type f | sort
