#!/usr/bin/env bash
# Report-only held-out validation for automatic abstention recovery.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to playroom or drjohnson}"
SOURCE_CACHE_OUTPUT_NAME="${SOURCE_CACHE_OUTPUT_NAME:?Set the selected DINOv3 cache output}"
SOURCE_HARD_AUDIT_OUTPUT_NAME="${SOURCE_HARD_AUDIT_OUTPUT_NAME:?Set the completed hard-vote audit output}"
SOURCE_RECOVERY_OUTPUT_NAME="${SOURCE_RECOVERY_OUTPUT_NAME:?Set the completed recovery output}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_abstention_round_trip_validation_v1}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"

SOURCE_CACHE_ROOT="${TASK1_ROOT}/${SOURCE_CACHE_OUTPUT_NAME}"
SOURCE_VIEW_DIR="${SOURCE_CACHE_ROOT}/stages/01_real_camera_views"
SOURCE_CACHE_REPORT="${SOURCE_CACHE_ROOT}/selection/selected_view_cache_report.json"
HARD_ROOT="${TASK1_ROOT}/${SOURCE_HARD_AUDIT_OUTPUT_NAME}"
BASELINE_VOTE_MANIFEST="${HARD_ROOT}/stages/01_flashsplat_votes/vote_manifest.json"
HARD_AUDIT_REPORT="${HARD_ROOT}/stages/02_round_trip_fidelity/round_trip_fidelity_report.json"
HARD_DIAGNOSTICS="${HARD_ROOT}/stages/02_round_trip_fidelity/gaussian_camera_agreement_diagnostics.npz"
RECOVERY_ROOT="${TASK1_ROOT}/${SOURCE_RECOVERY_OUTPUT_NAME}"
ADDITIONAL_VOTE_MANIFEST="${RECOVERY_ROOT}/stages/01_additional_flashsplat_votes/vote_manifest.json"
SELECTION_REPORT="${RECOVERY_ROOT}/selection/additional_camera_selection_report.json"
RECOVERY_REPORT="${RECOVERY_ROOT}/stages/02_detected_abstention_recovery/detected_abstention_recovery_report.json"
CANDIDATE_LABELS="${RECOVERY_ROOT}/stages/02_detected_abstention_recovery/diagnostic_candidate_project_class_ids.npy"
RECOVERY_SOURCE_CODES="${RECOVERY_ROOT}/stages/02_detected_abstention_recovery/diagnostic_recovery_source_codes.npy"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
AUDIT_DIR="${OUT_DIR}/stages/01_abstention_round_trip_validation"
LOG_DIR="${OUT_DIR}/logs"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-semantic_3dgs_renderer}"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${WORKSPACE_ROOT}/external/FlashSplat}"
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

for path in \
  "${MODEL_DIR}/cameras.json" \
  "${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply" \
  "${SOURCE_VIEW_DIR}/view_manifest.json" \
  "${SOURCE_VIEW_DIR}/dinov3_manifest.json" \
  "${SOURCE_CACHE_REPORT}" \
  "${BASELINE_VOTE_MANIFEST}" \
  "${HARD_AUDIT_REPORT}" \
  "${HARD_DIAGNOSTICS}" \
  "${ADDITIONAL_VOTE_MANIFEST}" \
  "${SELECTION_REPORT}" \
  "${RECOVERY_REPORT}" \
  "${CANDIDATE_LABELS}" \
  "${RECOVERY_SOURCE_CODES}" \
  "${ONTOLOGY}"; do
  test -f "${path}"
done
test -d "${FLASHSPLAT_ROOT}"

printf 'scene=%s\nsource_cache=%s\nhard_audit=%s\nrecovery=%s\noutput=%s\nheldout_policy=exclude_each_original_baseline_camera_from_baseline_and_recovery_evidence\nheldout_calibration_policy=recompute_all_camera_reliabilities_after_excluding_the_heldout_camera\nreport_only=1\naccepted_gaussian_labels_written=0\nsemantic_ply_written=0\n' \
  "${SCENE}" "${SOURCE_CACHE_ROOT}" "${HARD_ROOT}" "${RECOVERY_ROOT}" "${OUT_DIR}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  exit 0
fi
if [[ -e "${OUT_DIR}" ]]; then
  echo "Output exists; choose a fresh OUTPUT_NAME: ${OUT_DIR}" >&2
  exit 2
fi

run_stage() {
  local stage_name="$1"
  shift
  "$@" 2>&1 | tee "${LOG_DIR}/${stage_name}.log"
}

mkdir -p "${LOG_DIR}" "${TASK1_ROOT}/logs"
cd "${PROJECT_ROOT}"
run_stage 01_abstention_round_trip_validation \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.abstention_round_trip_validation \
    --scene "${SCENE}" \
    --model-path "${MODEL_DIR}" \
    --source-view-dir "${SOURCE_VIEW_DIR}" \
    --selected-cache-report "${SOURCE_CACHE_REPORT}" \
    --baseline-vote-manifest "${BASELINE_VOTE_MANIFEST}" \
    --additional-vote-manifest "${ADDITIONAL_VOTE_MANIFEST}" \
    --hard-audit-report "${HARD_AUDIT_REPORT}" \
    --hard-diagnostics "${HARD_DIAGNOSTICS}" \
    --selection-report "${SELECTION_REPORT}" \
    --recovery-report "${RECOVERY_REPORT}" \
    --candidate-labels "${CANDIDATE_LABELS}" \
    --recovery-source-codes "${RECOVERY_SOURCE_CODES}" \
    --ontology "${ONTOLOGY}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --output-dir "${AUDIT_DIR}"

test -f "${AUDIT_DIR}/recovery_round_trip_validation_report.json"
test -f "${AUDIT_DIR}/experiment_mode.txt"

CONTACT_DIR="${OUT_DIR}/visualizations/contact_sheets"
mkdir -p "${CONTACT_DIR}"
run_stage 02_candidate_overlay_contact_sheet \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${AUDIT_DIR}/candidate_heldout_overlays" \
    --output "${CONTACT_DIR}/candidate_heldout_overlays.png" \
    --columns 5 \
    --thumb-width 240

run_stage 03_candidate_disagreement_contact_sheet \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${AUDIT_DIR}/candidate_heldout_disagreement" \
    --output "${CONTACT_DIR}/candidate_heldout_disagreement.png" \
    --columns 5 \
    --thumb-width 240

test -f "${CONTACT_DIR}/candidate_heldout_overlays.png"
test -f "${CONTACT_DIR}/candidate_heldout_disagreement.png"
if find "${OUT_DIR}" -type f -name '*.ply' -print -quit | grep -q .; then
  echo "Recovery validation unexpectedly wrote a PLY" >&2
  exit 2
fi
if find "${OUT_DIR}" -type f \( -name 'gaussian_labels.npy' -o -name 'label_map.json' \) -print -quit | grep -q .; then
  echo "Recovery validation unexpectedly wrote accepted labels" >&2
  exit 2
fi

find "${OUT_DIR}" -maxdepth 3 -type f | sort
