#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_core_fill
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

# Cached report-only incremental-fill audit:
# resolved multiview spatial cores - immutable preferred labels -> adaptive
# residual components -> exact matched-view 3D support projections.
# This scheduler writes no semantic labels, project-class arrays, label map,
# or PLY.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${WORKSPACE_ROOT}/external/FlashSplat}"

SCENE="${SCENE:?Set SCENE to the target scene}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
SOURCE_OUTPUT_NAME="${SOURCE_OUTPUT_NAME:-${SCENE}_dinov3_3d_first_report_targeted_50plus20_v1}"
SPATIAL_CORE_OUTPUT_NAME="${SPATIAL_CORE_OUTPUT_NAME:-${SCENE}_dinov3_multiview_spatial_core_audit_v1}"
PREFERRED_OUTPUT_NAME="${PREFERRED_OUTPUT_NAME:-${SCENE}_dinov3_camera_owned_dense_propagation_v2}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_incremental_spatial_core_fill_audit_v1}"

SOURCE_ROOT="${TASK1_ROOT}/${SOURCE_OUTPUT_NAME}"
SPATIAL_CORE_ROOT="${TASK1_ROOT}/${SPATIAL_CORE_OUTPUT_NAME}"
PREFERRED_ROOT="${TASK1_ROOT}/${PREFERRED_OUTPUT_NAME}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
AUDIT_DIR="${OUT_DIR}/audit"
OVERLAY_DIR="${OUT_DIR}/visualizations/exact_incremental_fill_overlays"
MASK_RENDER_DIR="${OUT_DIR}/visualizations/exact_incremental_fill_masks"
OVERLAY_CONTACT_SHEET="${OUT_DIR}/visualizations/exact_incremental_fill_overlays.png"
MASK_CONTACT_SHEET="${OUT_DIR}/visualizations/exact_incremental_fill_masks.png"
RUN_LOG_DIR="${OUT_DIR}/logs"

SPATIAL_CORE_REPORT="${SPATIAL_CORE_ROOT}/audit/multiview_spatial_core_audit.json"
SPATIAL_CORE_SUPPORTS="${SPATIAL_CORE_ROOT}/audit/proposed_core_supports.npz"
SOURCE_PROPOSAL_MANIFEST="${SOURCE_ROOT}/stages/03_flashsplat_region_supports/proposal_manifest.json"
SOURCE_VIEW_MANIFEST="${SOURCE_ROOT}/stages/01_dinov3_query_evidence/view_manifest.json"
PREFERRED_LABELS="${PREFERRED_ROOT}/deliverables/gaussian_labels.npy"
SOURCE_PLY="${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
MIN_SPATIAL_COMPONENT_GAUSSIANS="${MIN_SPATIAL_COMPONENT_GAUSSIANS:-500}"
MIN_SPATIAL_COMPONENT_CAMERAS="${MIN_SPATIAL_COMPONENT_CAMERAS:-2}"
VOXEL_SCALE_MULTIPLIER="${VOXEL_SCALE_MULTIPLIER:-4.0}"
MIN_VOXEL_SIZE="${MIN_VOXEL_SIZE:-0.01}"
MAX_VOXEL_SIZE="${MAX_VOXEL_SIZE:-0.20}"
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
  echo "This audit requires a fresh output and refuses RESET_OUTPUT=1" >&2
  exit 2
fi

test -d "${MODEL_DIR}"
test -d "${FLASHSPLAT_ROOT}"
test -f "${SOURCE_PLY}"
test -f "${SPATIAL_CORE_REPORT}"
test -f "${SPATIAL_CORE_SUPPORTS}"
test -f "${SOURCE_PROPOSAL_MANIFEST}"
test -f "${SOURCE_VIEW_MANIFEST}"
test -f "${PREFERRED_LABELS}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nsource=%s\nspatial_core_source=%s\npreferred_label_source=%s\noutput=%s\nmode=dinov3_incremental_spatial_core_fill_audit\nreport_only=1\npreferred_labels_read_only=1\nresidual_policy=resolved_spatial_core_minus_all_nonzero_preferred_labels\ngeometry_policy=adaptive_26_neighbor_voxel_components\nacceptance_policy=global_minimum_residual_size_and_independent_camera_count\nvisualization_scope=exact_projection_of_accepted_unlabeled_residual_3d_support\nmatched_source_view_manifest=%s\nmin_spatial_component_gaussians=%s\nmin_spatial_component_cameras=%s\nvoxel_scale_multiplier=%s\nmin_voxel_size=%s\nmax_voxel_size=%s\nscene_specific_rules=0\nclass_specific_thresholds=0\noffline_threshold_sweep_used=0\ninference_rerun=0\nflashsplat_rerun=0\nspatial_core_rerun=0\nsemantic_labels_written=0\nsemantic_project_class_arrays_written=0\nlabel_map_written=0\nsemantic_ply_written=0\n' \
    "${SCENE}" "${SOURCE_ROOT}" "${SPATIAL_CORE_ROOT}" \
    "${PREFERRED_ROOT}" "${OUT_DIR}" "${SOURCE_VIEW_MANIFEST}" \
    "${MIN_SPATIAL_COMPONENT_GAUSSIANS}" \
    "${MIN_SPATIAL_COMPONENT_CAMERAS}" "${VOXEL_SCALE_MULTIPLIER}" \
    "${MIN_VOXEL_SIZE}" "${MAX_VOXEL_SIZE}"
  exit 0
fi

if [[ -e "${OUT_DIR}" ]]; then
  echo "Output exists; choose a fresh OUTPUT_NAME:" >&2
  echo "${OUT_DIR}" >&2
  exit 2
fi

mkdir -p "${RUN_LOG_DIR}" "${TASK1_ROOT}/logs"
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
  echo "source=${SOURCE_ROOT}"
  echo "spatial_core_source=${SPATIAL_CORE_ROOT}"
  echo "preferred_label_source=${PREFERRED_ROOT}"
  echo "output=${OUT_DIR}"
  echo "mode=dinov3_incremental_spatial_core_fill_audit"
  echo "report_only=1"
  echo "preferred_labels_read_only=1"
  echo "inference_rerun=0"
  echo "flashsplat_rerun=0"
  echo "spatial_core_rerun=0"
  echo "semantic_labels_written=0"
  echo "semantic_project_class_arrays_written=0"
  echo "label_map_written=0"
  echo "semantic_ply_written=0"
  nvidia-smi
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

run_stage 01_incremental_spatial_core_fill_audit \
  python -m scripts.task1.dinov3.audit_incremental_spatial_core_fill \
    --spatial-core-report "${SPATIAL_CORE_REPORT}" \
    --spatial-core-supports "${SPATIAL_CORE_SUPPORTS}" \
    --proposal-manifest "${SOURCE_PROPOSAL_MANIFEST}" \
    --preferred-labels "${PREFERRED_LABELS}" \
    --source-ply "${SOURCE_PLY}" \
    --output-dir "${AUDIT_DIR}" \
    --scene "${SCENE}" \
    --min-spatial-component-gaussians "${MIN_SPATIAL_COMPONENT_GAUSSIANS}" \
    --min-spatial-component-cameras "${MIN_SPATIAL_COMPONENT_CAMERAS}" \
    --voxel-scale-multiplier "${VOXEL_SCALE_MULTIPLIER}" \
    --min-voxel-size "${MIN_VOXEL_SIZE}" \
    --max-voxel-size "${MAX_VOXEL_SIZE}"

run_stage 02_render_exact_incremental_fill \
  python -m scripts.task1.dinov3.render_incremental_spatial_core_fill \
    --audit-report "${AUDIT_DIR}/incremental_spatial_core_fill_audit.json" \
    --support-npz "${AUDIT_DIR}/incremental_fill_supports.npz" \
    --model-path "${MODEL_DIR}" \
    --source-view-manifest "${SOURCE_VIEW_MANIFEST}" \
    --overlay-output-dir "${OVERLAY_DIR}" \
    --mask-output-dir "${MASK_RENDER_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}"

run_stage 03_exact_overlay_contact_sheet \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${OVERLAY_DIR}" \
    --output "${OVERLAY_CONTACT_SHEET}" \
    --columns 4 \
    --thumb-width 300

run_stage 04_exact_mask_contact_sheet \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${MASK_RENDER_DIR}" \
    --output "${MASK_CONTACT_SHEET}" \
    --columns 4 \
    --thumb-width 300

test -f "${AUDIT_DIR}/incremental_spatial_core_fill_audit.json"
test -f "${AUDIT_DIR}/incremental_fill_mask.npy"
test -f "${AUDIT_DIR}/unlabeled_residual_candidate_mask.npy"
test -f "${AUDIT_DIR}/excluded_preferred_label_mask.npy"
test -f "${AUDIT_DIR}/incremental_fill_supports.npz"
test -f "${OVERLAY_DIR}/exact_residual_overlay_manifest.json"
test -f "${OVERLAY_CONTACT_SHEET}"
test -f "${MASK_CONTACT_SHEET}"
if find "${OUT_DIR}" -type f \( \
  -name 'gaussian_labels.npy' -o \
  -name 'gaussian_project_class_ids.npy' -o \
  -name 'label_map.json' \
\) -print -quit | grep -q .; then
  echo "Report-only audit unexpectedly wrote semantic label artifacts" >&2
  exit 2
fi
if find "${OUT_DIR}" -type f -name '*.ply' -print -quit | grep -q .; then
  echo "Report-only audit unexpectedly wrote a PLY" >&2
  exit 2
fi

{
  echo "mode=dinov3_incremental_spatial_core_fill_audit"
  echo "report_only=1"
  echo "source=${SOURCE_ROOT}"
  echo "spatial_core_source=${SPATIAL_CORE_ROOT}"
  echo "preferred_label_source=${PREFERRED_ROOT}"
  echo "preferred_labels_read_only=1"
  echo "residual_policy=resolved_spatial_core_minus_all_nonzero_preferred_labels"
  echo "geometry_policy=adaptive_26_neighbor_voxel_components"
  echo "acceptance_policy=global_minimum_residual_size_and_independent_camera_count"
  echo "visualization_scope=exact_projection_of_accepted_unlabeled_residual_3d_support"
  echo "scene_specific_rules=0"
  echo "class_specific_thresholds=0"
  echo "offline_threshold_sweep_used=0"
  echo "inference_rerun=0"
  echo "flashsplat_rerun=0"
  echo "spatial_core_rerun=0"
  echo "semantic_labels_written=0"
  echo "semantic_project_class_arrays_written=0"
  echo "label_map_written=0"
  echo "semantic_ply_written=0"
} > "${OUT_DIR}/experiment_mode.txt"

find "${OUT_DIR}" -maxdepth 4 -type f | sort
