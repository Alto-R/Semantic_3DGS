#!/usr/bin/env bash
# Cache-only report for conservatively recovering camera-observed black Gaussians.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to playroom or drjohnson}"
SOURCE_HARD_AUDIT_OUTPUT_NAME="${SOURCE_HARD_AUDIT_OUTPUT_NAME:?Set the completed hard-vote audit output}"
SOURCE_RECOVERY_OUTPUT_NAME="${SOURCE_RECOVERY_OUTPUT_NAME:?Set the completed additional-view recovery output}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_observed_black_calibrated_spatial_audit_v1}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"

HARD_ROOT="${TASK1_ROOT}/${SOURCE_HARD_AUDIT_OUTPUT_NAME}"
BASELINE_VOTE_MANIFEST="${HARD_ROOT}/stages/01_flashsplat_votes/vote_manifest.json"
HARD_AUDIT_DIR="${HARD_ROOT}/stages/02_round_trip_fidelity"
HARD_AUDIT_REPORT="${HARD_AUDIT_DIR}/round_trip_fidelity_report.json"
HARD_DIAGNOSTICS="${HARD_AUDIT_DIR}/gaussian_camera_agreement_diagnostics.npz"
HARD_CONFUSION="${HARD_AUDIT_DIR}/heldout_confusion_matrix.npz"

RECOVERY_ROOT="${TASK1_ROOT}/${SOURCE_RECOVERY_OUTPUT_NAME}"
ADDITIONAL_VOTE_MANIFEST="${RECOVERY_ROOT}/stages/01_additional_flashsplat_votes/vote_manifest.json"
RECOVERY_DIR="${RECOVERY_ROOT}/stages/02_detected_abstention_recovery"
RECOVERY_REPORT="${RECOVERY_DIR}/detected_abstention_recovery_report.json"
RECOVERY_CANDIDATE_LABELS="${RECOVERY_DIR}/diagnostic_candidate_project_class_ids.npy"
RECOVERY_SOURCE_CODES="${RECOVERY_DIR}/diagnostic_recovery_source_codes.npy"

SOURCE_PLY="${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
AUDIT_DIR="${OUT_DIR}/stages/01_observed_black_calibrated_spatial_audit"
LOG_DIR="${OUT_DIR}/logs"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-semantic_3dgs_renderer}"
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
if [[ -n "${MIN_SEMANTIC_CAMERAS:-}" || -n "${MIN_SPATIAL_NEIGHBORS:-}" ]]; then
  echo "Evidence and spatial thresholds are fixed by the common policy" >&2
  exit 2
fi

for path in \
  "${SOURCE_PLY}" \
  "${BASELINE_VOTE_MANIFEST}" \
  "${HARD_AUDIT_REPORT}" \
  "${HARD_DIAGNOSTICS}" \
  "${HARD_CONFUSION}" \
  "${ADDITIONAL_VOTE_MANIFEST}" \
  "${RECOVERY_REPORT}" \
  "${RECOVERY_CANDIDATE_LABELS}" \
  "${RECOVERY_SOURCE_CODES}" \
  "${ONTOLOGY}"; do
  test -f "${path}"
done

printf 'scene=%s\nsource_hard_audit=%s\nsource_recovery=%s\noutput=%s\ncache_only=1\ndinov3_inference_rerun=0\nflashsplat_lifting_rerun=0\nzero_camera_gaussians_forced_black=1\nsingle_camera_gaussians_forced_black=1\nsemantic_class_selected_before_spatial_corroboration=1\nspatial_class_selection=0\nbroad_surface_spatial_relaxation=0\nreport_only=1\naccepted_gaussian_labels_written=0\nsemantic_ply_written=0\n' \
  "${SCENE}" "${HARD_ROOT}" "${RECOVERY_ROOT}" "${OUT_DIR}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  exit 0
fi
if [[ -e "${OUT_DIR}" ]]; then
  echo "Output exists; choose a fresh OUTPUT_NAME: ${OUT_DIR}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${TASK1_ROOT}/logs"
cd "${PROJECT_ROOT}"
conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.observed_black_calibrated_spatial_audit \
    --scene "${SCENE}" \
    --baseline-vote-manifest "${BASELINE_VOTE_MANIFEST}" \
    --additional-vote-manifest "${ADDITIONAL_VOTE_MANIFEST}" \
    --hard-audit-report "${HARD_AUDIT_REPORT}" \
    --hard-diagnostics "${HARD_DIAGNOSTICS}" \
    --hard-confusion "${HARD_CONFUSION}" \
    --recovery-report "${RECOVERY_REPORT}" \
    --recovery-candidate-labels "${RECOVERY_CANDIDATE_LABELS}" \
    --recovery-source-codes "${RECOVERY_SOURCE_CODES}" \
    --source-ply "${SOURCE_PLY}" \
    --ontology "${ONTOLOGY}" \
    --output-dir "${AUDIT_DIR}" \
  2>&1 | tee "${LOG_DIR}/01_observed_black_calibrated_spatial_audit.log"

test -f "${AUDIT_DIR}/observed_black_calibrated_spatial_audit.json"
test -f "${AUDIT_DIR}/observed_black_diagnostics.npz"
test -f "${AUDIT_DIR}/experiment_mode.txt"
if find "${OUT_DIR}" -type f -name '*.ply' -print -quit | grep -q .; then
  echo "Observed-black audit unexpectedly wrote a PLY" >&2
  exit 2
fi
if find "${OUT_DIR}" -type f \( -name 'gaussian_labels.npy' -o -name 'label_map.json' \) -print -quit | grep -q .; then
  echo "Observed-black audit unexpectedly wrote accepted labels" >&2
  exit 2
fi

find "${OUT_DIR}" -maxdepth 4 -type f | sort
