#!/usr/bin/env bash
# Reproduce an audited hard-camera DINOv3 consensus and export inspection PLYs.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
SOURCE_HARD_AUDIT_OUTPUT_NAME="${SOURCE_HARD_AUDIT_OUTPUT_NAME:?Set SOURCE_HARD_AUDIT_OUTPUT_NAME to the completed hard-vote audit output}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_hard_vote_materialized_v1}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
SOURCE_ROOT="${TASK1_ROOT}/${SOURCE_HARD_AUDIT_OUTPUT_NAME}"
VOTE_MANIFEST="${SOURCE_ROOT}/stages/01_flashsplat_votes/vote_manifest.json"
AUDIT_REPORT="${SOURCE_ROOT}/stages/02_round_trip_fidelity/round_trip_fidelity_report.json"
SOURCE_PLY="${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
DELIVERABLE_DIR="${OUT_DIR}/deliverables"
VISUALIZATION_DIR="${OUT_DIR}/visualizations"
LOG_DIR="${OUT_DIR}/logs"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-semantic_3dgs_renderer}"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

if [[ "${CONFIG_ONLY}" != "0" && "${CONFIG_ONLY}" != "1" ]]; then
  echo "CONFIG_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ -n "${CAMERA_INDICES:-}" || -n "${VIEW_COUNT:-}" ]]; then
  echo "Manual CAMERA_INDICES and VIEW_COUNT are not accepted" >&2
  exit 2
fi
if [[ -n "${MIN_VIEWS:-}" || -n "${CAMERA_MAJORITY_THRESHOLD:-}" ]]; then
  echo "Consensus thresholds are fixed by the audited policy" >&2
  exit 2
fi

test -f "${SOURCE_PLY}"
test -f "${VOTE_MANIFEST}"
test -f "${AUDIT_REPORT}"
test -f "${ONTOLOGY}"

printf 'scene=%s\nsource_hard_audit=%s\noutput=%s\ncamera_vote_policy=one_unique_max_class_per_camera_and_gaussian_else_abstain\nconsensus_policy=at_least_two_cameras_unique_strict_majority_else_abstain\nunaccepted_policy=label_zero_without_fill_or_propagation\nmanual_camera_selection_used=0\nmanual_gaussian_selection_used=0\nsemantic_ply_written=1\ncolored_ply_written=1\n' \
  "${SCENE}" "${SOURCE_ROOT}" "${OUT_DIR}"

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

run_stage 01_materialize_audited_hard_vote_consensus \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.materialize_hard_vote_consensus \
    --scene "${SCENE}" \
    --source-ply "${SOURCE_PLY}" \
    --audit-report "${AUDIT_REPORT}" \
    --vote-manifest "${VOTE_MANIFEST}" \
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
mode=audited_hard_vote_consensus_materialization
scene=${SCENE}
source_hard_audit=${SOURCE_HARD_AUDIT_OUTPUT_NAME}
camera_vote_policy=one_unique_max_class_per_camera_and_gaussian_else_abstain
consensus_policy=at_least_two_cameras_unique_strict_majority_else_abstain
unaccepted_policy=label_zero_without_fill_or_propagation
manual_camera_selection_used=0
manual_gaussian_selection_used=0
scene_specific_rules=0
class_specific_thresholds=0
semantic_ply=${DELIVERABLE_DIR}/semantic_point_cloud.ply
colored_ply=${VISUALIZATION_DIR}/semantic_point_cloud_colored.ply
EOF

test -f "${DELIVERABLE_DIR}/gaussian_labels.npy"
test -f "${DELIVERABLE_DIR}/gaussian_project_class_ids.npy"
test -f "${DELIVERABLE_DIR}/abstain_reason_codes.npy"
test -f "${DELIVERABLE_DIR}/label_map.json"
test -f "${DELIVERABLE_DIR}/hard_vote_materialization_summary.json"
test -f "${DELIVERABLE_DIR}/semantic_point_cloud.ply"
test -f "${VISUALIZATION_DIR}/semantic_point_cloud_colored.ply"
test -f "${VISUALIZATION_DIR}/semantic_point_cloud_colored.json"
test -f "${VISUALIZATION_DIR}/semantic_color_legend.json"
test -f "${VISUALIZATION_DIR}/semantic_color_legend.png"

find "${OUT_DIR}" -maxdepth 4 -type f | sort
