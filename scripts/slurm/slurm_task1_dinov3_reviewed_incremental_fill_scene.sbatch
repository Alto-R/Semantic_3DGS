#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_review_fill
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

# Fresh candidate v3:
# immutable preferred v2 + explicitly reviewed incremental-fill components ->
# semantic deliverables, SuperSplat debug PLY, and matched 70-view QA.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${WORKSPACE_ROOT}/external/FlashSplat}"

SCENE="${SCENE:?Set SCENE to the target scene}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
SOURCE_OUTPUT_NAME="${SOURCE_OUTPUT_NAME:-${SCENE}_dinov3_3d_first_report_targeted_50plus20_v1}"
PREFERRED_OUTPUT_NAME="${PREFERRED_OUTPUT_NAME:-${SCENE}_dinov3_camera_owned_dense_propagation_v2}"
INCREMENTAL_AUDIT_OUTPUT_NAME="${INCREMENTAL_AUDIT_OUTPUT_NAME:-${SCENE}_dinov3_incremental_spatial_core_fill_audit_v1}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_reviewed_incremental_fill_candidate_v3}"

SOURCE_ROOT="${TASK1_ROOT}/${SOURCE_OUTPUT_NAME}"
PREFERRED_ROOT="${TASK1_ROOT}/${PREFERRED_OUTPUT_NAME}"
INCREMENTAL_AUDIT_ROOT="${TASK1_ROOT}/${INCREMENTAL_AUDIT_OUTPUT_NAME}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
DELIVERABLE_DIR="${OUT_DIR}/deliverables"
VISUALIZATION_DIR="${OUT_DIR}/visualizations"
OVERLAY_DIR="${VISUALIZATION_DIR}/semantic_overlays"
RGB_DIR="${VISUALIZATION_DIR}/qa_rgb"
CONTACT_SHEET="${VISUALIZATION_DIR}/semantic_labels.png"
VALIDATION_DIR="${OUT_DIR}/validation"
RUN_LOG_DIR="${OUT_DIR}/logs"

PREFERRED_LABELS="${PREFERRED_ROOT}/deliverables/gaussian_labels.npy"
PREFERRED_LABEL_MAP="${PREFERRED_ROOT}/deliverables/label_map.json"
INCREMENTAL_FILL_REPORT="${INCREMENTAL_AUDIT_ROOT}/audit/incremental_spatial_core_fill_audit.json"
INCREMENTAL_FILL_SUPPORTS="${INCREMENTAL_AUDIT_ROOT}/audit/incremental_fill_supports.npz"
SOURCE_VIEW_MANIFEST="${SOURCE_ROOT}/stages/01_dinov3_query_evidence/view_manifest.json"
SOURCE_PLY="${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
REVIEW_CONFIG="${REVIEW_CONFIG:-${PROJECT_ROOT}/configs/dinov3_incremental_fill_review.${SCENE}.json}"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"
RESET_OUTPUT="${RESET_OUTPUT:-0}"

for value_name in CONFIG_ONLY RESET_OUTPUT; do
  value="${!value_name}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${value_name} must be 0 or 1" >&2
    exit 2
  fi
done
if [[ "${RESET_OUTPUT}" != "0" ]]; then
  echo "This candidate requires a fresh output and refuses RESET_OUTPUT=1" >&2
  exit 2
fi

test -d "${MODEL_DIR}"
test -d "${FLASHSPLAT_ROOT}"
test -f "${SOURCE_PLY}"
test -f "${PREFERRED_LABELS}"
test -f "${PREFERRED_LABEL_MAP}"
test -f "${INCREMENTAL_FILL_REPORT}"
test -f "${INCREMENTAL_FILL_SUPPORTS}"
test -f "${SOURCE_VIEW_MANIFEST}"
test -f "${ONTOLOGY}"
test -f "${REVIEW_CONFIG}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\npreferred_v2=%s\nincremental_audit=%s\nreview_config=%s\noutput=%s\nmode=dinov3_reviewed_incremental_fill_candidate_v3\ncandidate_only=1\npreferred_labels_read_only=1\nassignment_policy=fill_only_preferred_zero_gaussians\nreview_policy=explicit_exact_projection_component_exclusions\nmatched_source_view_manifest=%s\ninference_rerun=0\nflashsplat_rerun=0\nspatial_core_rerun=0\nsemantic_labels_written=1\nsemantic_project_class_arrays_written=1\nlabel_map_written=1\nsemantic_ply_written=1\n' \
    "${SCENE}" "${PREFERRED_ROOT}" "${INCREMENTAL_AUDIT_ROOT}" \
    "${REVIEW_CONFIG}" "${OUT_DIR}" "${SOURCE_VIEW_MANIFEST}"
  exit 0
fi

if [[ -e "${OUT_DIR}" ]]; then
  echo "Output exists; choose a fresh OUTPUT_NAME:" >&2
  echo "${OUT_DIR}" >&2
  exit 2
fi

mkdir -p "${RUN_LOG_DIR}" "${VALIDATION_DIR}" "${TASK1_ROOT}/logs"
cd "${PROJECT_ROOT}"

run_stage() {
  local stage_name="$1"
  shift
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" "$@" 2>&1 \
    | tee "${RUN_LOG_DIR}/${stage_name}.log"
}

{
  echo "job=${SLURM_JOB_ID:-local}"
  echo "host=$(hostname)"
  echo "project=${PROJECT_ROOT}"
  echo "scene=${SCENE}"
  echo "preferred_v2=${PREFERRED_ROOT}"
  echo "incremental_audit=${INCREMENTAL_AUDIT_ROOT}"
  echo "review_config=${REVIEW_CONFIG}"
  echo "output=${OUT_DIR}"
  echo "mode=dinov3_reviewed_incremental_fill_candidate_v3"
  echo "candidate_only=1"
  echo "preferred_labels_read_only=1"
  echo "assignment_policy=fill_only_preferred_zero_gaussians"
  echo "review_policy=explicit_exact_projection_component_exclusions"
  echo "inference_rerun=0"
  echo "flashsplat_rerun=0"
  echo "spatial_core_rerun=0"
  nvidia-smi
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

run_stage 01_materialize_reviewed_incremental_fill \
  python -m scripts.task1.dinov3.materialize_reviewed_incremental_fill \
    --preferred-labels "${PREFERRED_LABELS}" \
    --preferred-label-map "${PREFERRED_LABEL_MAP}" \
    --incremental-fill-report "${INCREMENTAL_FILL_REPORT}" \
    --incremental-fill-supports "${INCREMENTAL_FILL_SUPPORTS}" \
    --review-config "${REVIEW_CONFIG}" \
    --ontology "${ONTOLOGY}" \
    --source-ply "${SOURCE_PLY}" \
    --output-dir "${DELIVERABLE_DIR}" \
    --scene "${SCENE}"

run_stage 02_validate_candidate \
  python -m scripts.task1.qa.validate_task1_outputs \
    --labels "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --semantic-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    2>&1 | tee "${VALIDATION_DIR}/validate_task1_outputs.log"

run_stage 03_supersplat_debug \
  python -m scripts.task1.qa.export_supersplat_label_colors \
    --input-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-ply "${VISUALIZATION_DIR}/semantic_point_cloud_supersplat_debug.ply" \
    --metadata-json "${VISUALIZATION_DIR}/semantic_point_cloud_supersplat_debug.json" \
    --color-mode class \
    --overwrite

QA_CAMERA_INDICES="$(jq -r '[.frames[].camera_index] | join(",")' "${SOURCE_VIEW_MANIFEST}")"
QA_VIEW_COUNT="$(jq -r '.frames | length' "${SOURCE_VIEW_MANIFEST}")"
test -n "${QA_CAMERA_INDICES}"
if ! [[ "${QA_VIEW_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Source view manifest has no QA frames" >&2
  exit 2
fi

run_stage 04_semantic_overlays \
  python -m scripts.task1.qa.render_auto_label_overlays \
    --model-path "${MODEL_DIR}" \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-dir "${OVERLAY_DIR}" \
    --rgb-output-dir "${RGB_DIR}" \
    --source-view-manifest "${SOURCE_VIEW_MANIFEST}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --max-labels 0 \
    --camera-indices "${QA_CAMERA_INDICES}" \
    --count "${QA_VIEW_COUNT}" \
    --color-mode class

run_stage 05_visible_coverage \
  python -m scripts.task1.qa.measure_overlay_coverage \
    --rgb-dir "${RGB_DIR}" \
    --overlay-dir "${OVERLAY_DIR}" \
    --output "${VALIDATION_DIR}/visible_overlay_coverage.json" \
    --difference-threshold 2 \
    --exclude-border 1

run_stage 06_contact_sheet \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${OVERLAY_DIR}" \
    --output "${CONTACT_SHEET}" \
    --columns 4 \
    --thumb-width 300

run_stage 07_color_legend \
  python -m scripts.task1.qa.write_semantic_color_legend \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --output-json "${VISUALIZATION_DIR}/semantic_color_legend.json" \
    --output-png "${VISUALIZATION_DIR}/semantic_color_legend.png" \
    --color-mode class \
    --overwrite

test -f "${DELIVERABLE_DIR}/gaussian_labels.npy"
test -f "${DELIVERABLE_DIR}/gaussian_project_class_ids.npy"
test -f "${DELIVERABLE_DIR}/incremental_fill_mask.npy"
test -f "${DELIVERABLE_DIR}/excluded_review_component_mask.npy"
test -f "${DELIVERABLE_DIR}/incremental_fill_component_ids.npy"
test -f "${DELIVERABLE_DIR}/reviewed_incremental_fill_supports.npz"
test -f "${DELIVERABLE_DIR}/label_map.json"
test -f "${DELIVERABLE_DIR}/semantic_point_cloud.ply"
test -f "${DELIVERABLE_DIR}/reviewed_incremental_fill_summary.json"
test -f "${VISUALIZATION_DIR}/semantic_point_cloud_supersplat_debug.ply"
test -f "${OVERLAY_DIR}/label_overlay_manifest.json"
test -f "${VALIDATION_DIR}/validate_task1_outputs.log"
test -f "${VALIDATION_DIR}/visible_overlay_coverage.json"
test -f "${CONTACT_SHEET}"
test -f "${VISUALIZATION_DIR}/semantic_color_legend.json"
test -f "${VISUALIZATION_DIR}/semantic_color_legend.png"

{
  echo "mode=dinov3_reviewed_incremental_fill_candidate_v3"
  echo "candidate_only=1"
  echo "preferred_v2=${PREFERRED_ROOT}"
  echo "incremental_audit=${INCREMENTAL_AUDIT_ROOT}"
  echo "review_config=${REVIEW_CONFIG}"
  echo "preferred_labels_read_only=1"
  echo "assignment_policy=fill_only_preferred_zero_gaussians"
  echo "review_policy=explicit_exact_projection_component_exclusions"
  echo "inference_rerun=0"
  echo "flashsplat_rerun=0"
  echo "spatial_core_rerun=0"
  echo "semantic_labels_written=1"
  echo "semantic_project_class_arrays_written=1"
  echo "label_map_written=1"
  echo "semantic_ply_written=1"
} > "${OUT_DIR}/experiment_mode.txt"

find "${OUT_DIR}" -maxdepth 4 -type f | sort
