#!/usr/bin/env bash
# Materialize recovery labels only after the paired held-out gate passes.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to playroom or drjohnson}"
SOURCE_HARD_AUDIT_OUTPUT_NAME="${SOURCE_HARD_AUDIT_OUTPUT_NAME:?Set the hard-vote audit output}"
SOURCE_RECOVERY_OUTPUT_NAME="${SOURCE_RECOVERY_OUTPUT_NAME:?Set the recovery output}"
SOURCE_VALIDATION_OUTPUT_NAME="${SOURCE_VALIDATION_OUTPUT_NAME:?Set the held-out validation output}"
TWO_SCENE_GATE_PATH="${TWO_SCENE_GATE_PATH:?Set the paired held-out gate JSON path}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_abstention_recovery_materialized_v1}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"

HARD_ROOT="${TASK1_ROOT}/${SOURCE_HARD_AUDIT_OUTPUT_NAME}"
HARD_AUDIT_REPORT="${HARD_ROOT}/stages/02_round_trip_fidelity/round_trip_fidelity_report.json"
HARD_DIAGNOSTICS="${HARD_ROOT}/stages/02_round_trip_fidelity/gaussian_camera_agreement_diagnostics.npz"
BASELINE_VOTE_MANIFEST="${HARD_ROOT}/stages/01_flashsplat_votes/vote_manifest.json"
SOURCE_PLY="${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
RECOVERY_ROOT="${TASK1_ROOT}/${SOURCE_RECOVERY_OUTPUT_NAME}"
RECOVERY_REPORT="${RECOVERY_ROOT}/stages/02_detected_abstention_recovery/detected_abstention_recovery_report.json"
VALIDATION_ROOT="${TASK1_ROOT}/${SOURCE_VALIDATION_OUTPUT_NAME}"
VALIDATION_REPORT="${VALIDATION_ROOT}/stages/01_abstention_round_trip_validation/recovery_round_trip_validation_report.json"
CANDIDATE_LABELS="${RECOVERY_ROOT}/stages/02_detected_abstention_recovery/diagnostic_candidate_project_class_ids.npy"
RECOVERY_SOURCE_CODES="${RECOVERY_ROOT}/stages/02_detected_abstention_recovery/diagnostic_recovery_source_codes.npy"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
DELIVERABLE_DIR="${OUT_DIR}/deliverables"
VISUALIZATION_DIR="${OUT_DIR}/visualizations"
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
if [[ -n "${CAMERA_INDICES:-}" || -n "${VIEW_COUNT:-}" || -n "${MIN_VIEWS:-}" ]]; then
  echo "Manual camera selection and consensus overrides are not accepted" >&2
  exit 2
fi

for path in \
  "${SOURCE_PLY}" \
  "${HARD_AUDIT_REPORT}" \
  "${HARD_DIAGNOSTICS}" \
  "${BASELINE_VOTE_MANIFEST}" \
  "${RECOVERY_REPORT}" \
  "${VALIDATION_REPORT}" \
  "${CANDIDATE_LABELS}" \
  "${RECOVERY_SOURCE_CODES}" \
  "${TWO_SCENE_GATE_PATH}" \
  "${ONTOLOGY}"; do
  test -f "${path}"
done

printf 'scene=%s\nhard_audit=%s\nrecovery=%s\nvalidation=%s\ntwo_scene_gate=%s\noutput=%s\nreport_only_gate_required=1\nmanual_camera_selection_used=0\nmanual_gaussian_selection_used=0\nzero_camera_fill_used=0\npropagation_used=0\nsemantic_ply_written=1\n' \
  "${SCENE}" "${HARD_ROOT}" "${RECOVERY_ROOT}" "${VALIDATION_ROOT}" "${TWO_SCENE_GATE_PATH}" "${OUT_DIR}"

if [[ -e "${OUT_DIR}" ]]; then
  echo "Output exists: ${OUT_DIR}" >&2
  exit 2
fi
if [[ "${CONFIG_ONLY}" = "1" ]]; then
  exit 0
fi

mkdir -p "${LOG_DIR}" "${VISUALIZATION_DIR}"
cd "${PROJECT_ROOT}"

run_stage() {
  local stage="$1"
  shift
  echo "===== ${stage} ====="
  "$@" 2>&1 | tee "${LOG_DIR}/${stage}.log"
}

run_stage 01_materialize_audited_abstention_recovery \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.materialize_abstention_recovery \
    --scene "${SCENE}" \
    --source-ply "${SOURCE_PLY}" \
    --hard-audit-report "${HARD_AUDIT_REPORT}" \
    --hard-diagnostics "${HARD_DIAGNOSTICS}" \
    --baseline-vote-manifest "${BASELINE_VOTE_MANIFEST}" \
    --recovery-report "${RECOVERY_REPORT}" \
    --validation-report "${VALIDATION_REPORT}" \
    --candidate-labels "${CANDIDATE_LABELS}" \
    --recovery-source-codes "${RECOVERY_SOURCE_CODES}" \
    --two-scene-gate "${TWO_SCENE_GATE_PATH}" \
    --ontology "${ONTOLOGY}" \
    --output-dir "${DELIVERABLE_DIR}"

run_stage 02_validate_semantic_ply \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.validate_task1_outputs \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --semantic-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply"

run_stage 03_export_colored_supersplat_ply \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.export_supersplat_label_colors \
    --input-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --output-ply "${VISUALIZATION_DIR}/semantic_point_cloud_colored.ply" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --metadata-json "${VISUALIZATION_DIR}/semantic_point_cloud_colored.json" \
    --color-mode class

run_stage 04_inspect_colored_ply \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.inspect_ply \
    "${VISUALIZATION_DIR}/semantic_point_cloud_colored.ply"

run_stage 05_write_color_legend \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.write_semantic_color_legend \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --output-json "${VISUALIZATION_DIR}/semantic_color_legend.json" \
    --output-png "${VISUALIZATION_DIR}/semantic_color_legend.png" \
    --color-mode class

cat > "${OUT_DIR}/experiment_mode.txt" <<EOF
mode=audited_abstention_recovery_materialization
scene=${SCENE}
source_hard_audit=${SOURCE_HARD_AUDIT_OUTPUT_NAME}
source_recovery=${SOURCE_RECOVERY_OUTPUT_NAME}
source_validation=${SOURCE_VALIDATION_OUTPUT_NAME}
two_scene_gate=${TWO_SCENE_GATE_PATH}
report_only_gate_required=1
manual_camera_selection_used=0
manual_gaussian_selection_used=0
zero_camera_fill_used=0
propagation_used=0
semantic_ply=${DELIVERABLE_DIR}/semantic_point_cloud.ply
colored_ply=${VISUALIZATION_DIR}/semantic_point_cloud_colored.ply
EOF

for path in \
  "${DELIVERABLE_DIR}/gaussian_labels.npy" \
  "${DELIVERABLE_DIR}/gaussian_project_class_ids.npy" \
  "${DELIVERABLE_DIR}/recovery_source_codes.npy" \
  "${DELIVERABLE_DIR}/label_map.json" \
  "${DELIVERABLE_DIR}/abstention_recovery_materialization_summary.json" \
  "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
  "${VISUALIZATION_DIR}/semantic_point_cloud_colored.ply" \
  "${VISUALIZATION_DIR}/semantic_point_cloud_colored.json" \
  "${VISUALIZATION_DIR}/semantic_color_legend.json" \
  "${VISUALIZATION_DIR}/semantic_color_legend.png"; do
  test -f "${path}"
done

find "${OUT_DIR}" -maxdepth 4 -type f | sort
