#!/usr/bin/env bash
# Report-only held-out validation for component-graph black recovery.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to playroom or drjohnson}"
SOURCE_CACHE_OUTPUT_NAME="${SOURCE_CACHE_OUTPUT_NAME:?Set the selected DINOv3 cache output}"
SOURCE_HARD_AUDIT_OUTPUT_NAME="${SOURCE_HARD_AUDIT_OUTPUT_NAME:?Set the completed hard-vote audit output}"
SOURCE_RECOVERY_OUTPUT_NAME="${SOURCE_RECOVERY_OUTPUT_NAME:?Set the completed recovery output}"
SOURCE_COMPONENT_AUDIT_OUTPUT_NAME="${SOURCE_COMPONENT_AUDIT_OUTPUT_NAME:?Set the completed component-graph audit output}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_observed_black_component_graph_round_trip_validation_v1}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"

SOURCE_CACHE_ROOT="${TASK1_ROOT}/${SOURCE_CACHE_OUTPUT_NAME}"
SOURCE_VIEW_DIR="${SOURCE_CACHE_ROOT}/stages/01_real_camera_views"
SOURCE_CACHE_REPORT="${SOURCE_CACHE_ROOT}/selection/selected_view_cache_report.json"
HARD_ROOT="${TASK1_ROOT}/${SOURCE_HARD_AUDIT_OUTPUT_NAME}"
BASELINE_VOTE_MANIFEST="${HARD_ROOT}/stages/01_flashsplat_votes/vote_manifest.json"
HARD_AUDIT_REPORT="${HARD_ROOT}/stages/02_round_trip_fidelity/round_trip_fidelity_report.json"
HARD_DIAGNOSTICS="${HARD_ROOT}/stages/02_round_trip_fidelity/gaussian_camera_agreement_diagnostics.npz"
HARD_CONFUSION="${HARD_ROOT}/stages/02_round_trip_fidelity/heldout_confusion_matrix.npz"
RECOVERY_ROOT="${TASK1_ROOT}/${SOURCE_RECOVERY_OUTPUT_NAME}"
ADDITIONAL_VOTE_MANIFEST="${RECOVERY_ROOT}/stages/01_additional_flashsplat_votes/vote_manifest.json"
RECOVERY_REPORT="${RECOVERY_ROOT}/stages/02_detected_abstention_recovery/detected_abstention_recovery_report.json"
CANDIDATE_LABELS="${RECOVERY_ROOT}/stages/02_detected_abstention_recovery/diagnostic_candidate_project_class_ids.npy"
RECOVERY_SOURCE_CODES="${RECOVERY_ROOT}/stages/02_detected_abstention_recovery/diagnostic_recovery_source_codes.npy"
COMPONENT_AUDIT_ROOT="${TASK1_ROOT}/${SOURCE_COMPONENT_AUDIT_OUTPUT_NAME}"
COMPONENT_AUDIT_DIR="${COMPONENT_AUDIT_ROOT}/stages/01_observed_black_component_graph_audit"
COMPONENT_AUDIT_REPORT="${COMPONENT_AUDIT_DIR}/observed_black_component_graph_audit.json"
COMPONENT_DIAGNOSTICS="${COMPONENT_AUDIT_DIR}/observed_black_component_graph_diagnostics.npz"
SOURCE_PLY="${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
AUDIT_DIR="${OUT_DIR}/stages/01_component_graph_round_trip_validation"
LOG_DIR="${OUT_DIR}/logs"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-semantic_3dgs_renderer}"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${WORKSPACE_ROOT}/external/FlashSplat}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"
DINOV2_VOTE_MANIFEST="${DINOV2_VOTE_MANIFEST:-}"

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
if [[ -n "${GRAPH_THRESHOLD:-}" || -n "${COMPONENT_SCORE_THRESHOLD:-}" ]]; then
  echo "Graph and component thresholds are calibrated automatically" >&2
  exit 2
fi

for path in \
  "${MODEL_DIR}/cameras.json" \
  "${SOURCE_PLY}" \
  "${SOURCE_VIEW_DIR}/view_manifest.json" \
  "${SOURCE_VIEW_DIR}/dinov3_manifest.json" \
  "${SOURCE_CACHE_REPORT}" \
  "${BASELINE_VOTE_MANIFEST}" \
  "${HARD_AUDIT_REPORT}" \
  "${HARD_DIAGNOSTICS}" \
  "${HARD_CONFUSION}" \
  "${ADDITIONAL_VOTE_MANIFEST}" \
  "${RECOVERY_REPORT}" \
  "${CANDIDATE_LABELS}" \
  "${RECOVERY_SOURCE_CODES}" \
  "${COMPONENT_AUDIT_REPORT}" \
  "${COMPONENT_DIAGNOSTICS}" \
  "${ONTOLOGY}"; do
  test -f "${path}"
done
if [[ -n "${DINOV2_VOTE_MANIFEST}" ]]; then
  test -f "${DINOV2_VOTE_MANIFEST}"
fi
test -d "${FLASHSPLAT_ROOT}"

printf 'scene=%s\nsource_cache=%s\nhard_audit=%s\nrecovery=%s\ncomponent_audit=%s\noutput=%s\nheldout_policy=exclude_each_original_baseline_camera_from_baseline_and_component_graph_evidence\nheldout_calibration_policy=recompute_all_camera_reliabilities_and_rebuild_the_component_graph_after_excluding_the_heldout_camera\nreport_only=1\naccepted_gaussian_labels_written=0\nsemantic_ply_written=0\n' \
  "${SCENE}" "${SOURCE_CACHE_ROOT}" "${HARD_ROOT}" "${RECOVERY_ROOT}" "${COMPONENT_AUDIT_ROOT}" "${OUT_DIR}"
printf 'dinov2_agreement_gate=%s\n' "${DINOV2_VOTE_MANIFEST:+1}"

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
DINOV2_ARG=()
if [[ -n "${DINOV2_VOTE_MANIFEST}" ]]; then
  DINOV2_ARG=(--dinov2-vote-manifest "${DINOV2_VOTE_MANIFEST}")
fi
run_stage 01_component_graph_round_trip_validation \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.observed_black_component_graph_round_trip_validation \
    --scene "${SCENE}" \
    --model-path "${MODEL_DIR}" \
    --source-view-dir "${SOURCE_VIEW_DIR}" \
    --selected-cache-report "${SOURCE_CACHE_REPORT}" \
    --baseline-vote-manifest "${BASELINE_VOTE_MANIFEST}" \
    --additional-vote-manifest "${ADDITIONAL_VOTE_MANIFEST}" \
    --hard-audit-report "${HARD_AUDIT_REPORT}" \
    --hard-diagnostics "${HARD_DIAGNOSTICS}" \
    --hard-confusion "${HARD_CONFUSION}" \
    --recovery-report "${RECOVERY_REPORT}" \
    --candidate-labels "${CANDIDATE_LABELS}" \
    --recovery-source-codes "${RECOVERY_SOURCE_CODES}" \
    "${DINOV2_ARG[@]}" \
    --component-audit-report "${COMPONENT_AUDIT_REPORT}" \
    --component-diagnostics "${COMPONENT_DIAGNOSTICS}" \
    --source-ply "${SOURCE_PLY}" \
    --ontology "${ONTOLOGY}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --output-dir "${AUDIT_DIR}"

test -f "${AUDIT_DIR}/component_graph_round_trip_validation_report.json"
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
  echo "Component-graph validation unexpectedly wrote a PLY" >&2
  exit 2
fi
if find "${OUT_DIR}" -type f \( -name 'gaussian_labels.npy' -o -name 'label_map.json' \) -print -quit | grep -q .; then
  echo "Component-graph validation unexpectedly wrote accepted labels" >&2
  exit 2
fi

find "${OUT_DIR}" -maxdepth 3 -type f | sort
