#!/usr/bin/env bash
# Automatic report-only calibrated DINOv3 fusion.
# Select mode tunes one global policy; validate mode applies it unchanged.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
SOURCE_SOFT_OUTPUT_NAME="${SOURCE_SOFT_OUTPUT_NAME:?Set SOURCE_SOFT_OUTPUT_NAME to the completed soft-vote cache output}"
POLICY_MODE="${POLICY_MODE:-select}"
POLICY_SOURCE_OUTPUT_NAME="${POLICY_SOURCE_OUTPUT_NAME:-}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_calibrated_probability_${POLICY_MODE}_v1}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
SOURCE_ROOT="${TASK1_ROOT}/${SOURCE_SOFT_OUTPUT_NAME}"
SOURCE_VIEW_DIR="${SOURCE_ROOT}/stages/01_real_camera_views"
SOFT_VOTE_DIR="${SOURCE_ROOT}/stages/02_soft_probability_votes"
SOURCE_SELECTION_REPORT="${SOURCE_ROOT}/selection/selected_view_cache_report.json"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
POLICY_DIR="${OUT_DIR}/policy"
AUDIT_DIR="${OUT_DIR}/stages/01_calibrated_probability_round_trip"
CONTACT_DIR="${OUT_DIR}/visualizations/contact_sheets"
LOG_DIR="${OUT_DIR}/logs"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-semantic_3dgs_renderer}"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${WORKSPACE_ROOT}/external/FlashSplat}"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

if [[ "${CONFIG_ONLY}" != "0" && "${CONFIG_ONLY}" != "1" ]]; then
  echo "CONFIG_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ "${POLICY_MODE}" != "select" && "${POLICY_MODE}" != "validate" ]]; then
  echo "POLICY_MODE must be select or validate" >&2
  exit 2
fi
if [[ -n "${TOP_K:-}" || -n "${TEMPERATURE:-}" || -n "${CONFIDENCE_WEIGHT:-}" || -n "${CANDIDATE_POLICIES:-}" ]]; then
  echo "Manual calibration policies are not accepted" >&2
  exit 2
fi
if [[ "${POLICY_MODE}" = "select" && -n "${POLICY_SOURCE_OUTPUT_NAME}" ]]; then
  echo "Select mode does not accept POLICY_SOURCE_OUTPUT_NAME" >&2
  exit 2
fi
if [[ "${POLICY_MODE}" = "validate" && -z "${POLICY_SOURCE_OUTPUT_NAME}" ]]; then
  echo "Validate mode requires POLICY_SOURCE_OUTPUT_NAME" >&2
  exit 2
fi

test -f "${MODEL_DIR}/cameras.json"
test -f "${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
test -f "${SOURCE_VIEW_DIR}/view_manifest.json"
test -f "${SOURCE_VIEW_DIR}/dinov3_manifest.json"
test -f "${SOFT_VOTE_DIR}/soft_vote_manifest.json"
test -f "${SOURCE_SELECTION_REPORT}"
test -f "${ONTOLOGY}"
test -d "${FLASHSPLAT_ROOT}"

POLICY_REPORT="${POLICY_DIR}/calibrated_probability_policy_report.json"
if [[ "${POLICY_MODE}" = "validate" ]]; then
  SOURCE_POLICY_REPORT="${TASK1_ROOT}/${POLICY_SOURCE_OUTPUT_NAME}/policy/calibrated_probability_policy_report.json"
  test -f "${SOURCE_POLICY_REPORT}"
  POLICY_REPORT="${POLICY_DIR}/locked_calibrated_probability_policy_report.json"
fi

printf 'scene=%s\nsource_soft_output=%s\noutput=%s\npolicy_mode=%s\npolicy_source_output=%s\npolicy_report=%s\nmanual_candidate_selection_used=0\nmanual_camera_selection_used=0\nmanual_gaussian_selection_used=0\nreport_only=1\naccepted_gaussian_labels_written=0\nlabel_map_written=0\nsemantic_ply_written=0\n' \
  "${SCENE}" "${SOURCE_ROOT}" "${OUT_DIR}" "${POLICY_MODE}" "${POLICY_SOURCE_OUTPUT_NAME:-<none>}" "${POLICY_REPORT}"

if [[ -e "${OUT_DIR}" ]]; then
  echo "Output exists: ${OUT_DIR}" >&2
  exit 2
fi
if [[ "${CONFIG_ONLY}" = "1" ]]; then
  exit 0
fi

mkdir -p "${LOG_DIR}" "${CONTACT_DIR}"
cd "${PROJECT_ROOT}"

run_stage() {
  local stage="$1"
  shift
  echo "===== ${stage} ====="
  "$@" 2>&1 | tee "${LOG_DIR}/${stage}.log"
}

if [[ "${POLICY_MODE}" = "select" ]]; then
  run_stage 01_select_global_calibrated_probability_policy \
    conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
    python -m scripts.task1.dinov3.calibrated_probability_fusion \
      --soft-vote-manifest "${SOFT_VOTE_DIR}/soft_vote_manifest.json" \
      --selected-cache-report "${SOURCE_SELECTION_REPORT}" \
      --ontology "${ONTOLOGY}" \
      --output-dir "${POLICY_DIR}"
else
  mkdir -p "${POLICY_DIR}"
  cp -- "${SOURCE_POLICY_REPORT}" "${POLICY_REPORT}"
fi

run_stage 02_calibrated_leave_one_camera_out_round_trip \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.soft_probability_round_trip_audit \
    --model-path "${MODEL_DIR}" \
    --source-view-dir "${SOURCE_VIEW_DIR}" \
    --selected-cache-report "${SOURCE_SELECTION_REPORT}" \
    --soft-vote-manifest "${SOFT_VOTE_DIR}/soft_vote_manifest.json" \
    --calibration-policy-report "${POLICY_REPORT}" \
    --ontology "${ONTOLOGY}" \
    --output-dir "${AUDIT_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000

run_stage 03_calibrated_overlay_contact_sheet \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${AUDIT_DIR}/heldout_overlays" \
    --output "${CONTACT_DIR}/calibrated_heldout_overlays.png" \
    --columns 5 --thumb-width 240

run_stage 04_calibrated_disagreement_contact_sheet \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${AUDIT_DIR}/heldout_disagreement" \
    --output "${CONTACT_DIR}/calibrated_heldout_disagreement.png" \
    --columns 5 --thumb-width 240

cat > "${OUT_DIR}/experiment_mode.txt" <<EOF
mode=automatic_dinov3_calibrated_probability_fusion
scene=${SCENE}
source_soft_output=${SOURCE_SOFT_OUTPUT_NAME}
policy_mode=${POLICY_MODE}
policy_source_output=${POLICY_SOURCE_OUTPUT_NAME:-<none>}
policy_report=${POLICY_REPORT}
selection_rule=global_leave_one_camera_out_gaussian_accuracy_with_abstention_penalty
external_validation=lock_selected_policy_and_apply_unchanged_to_another_scene
report_only=1
manual_candidate_selection_used=0
manual_camera_selection_used=0
manual_gaussian_selection_used=0
scene_specific_rules=0
class_specific_thresholds=0
accepted_gaussian_labels_written=0
gaussian_project_class_array_written=0
label_map_written=0
semantic_ply_written=0
EOF

test -f "${POLICY_REPORT}"
test -f "${AUDIT_DIR}/calibrated_probability_round_trip_report.json"
test -f "${AUDIT_DIR}/heldout_confusion_matrix.npz"
test -f "${CONTACT_DIR}/calibrated_heldout_overlays.png"
test -f "${CONTACT_DIR}/calibrated_heldout_disagreement.png"
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
