#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_spatial_core
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

# Cached DINOv3-only report:
# targeted 70-view components and proposals -> consensus-filtered multiview
# spatial cores, overlap diagnostics, and matched review overlays.
# This scheduler writes no semantic labels, project-class arrays, label map,
# or PLY.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
SOURCE_OUTPUT_NAME="${SOURCE_OUTPUT_NAME:-${SCENE}_dinov3_3d_first_report_targeted_50plus20_v1}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_multiview_spatial_core_audit_v1}"

SOURCE_ROOT="${TASK1_ROOT}/${SOURCE_OUTPUT_NAME}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
AUDIT_DIR="${OUT_DIR}/audit"
OVERLAY_DIR="${OUT_DIR}/visualizations/multiview_spatial_core_overlays"
CONTACT_SHEET="${OUT_DIR}/visualizations/multiview_spatial_core_components.png"
RUN_LOG_DIR="${OUT_DIR}/logs"

SOURCE_COMPONENT_REPORT="${SOURCE_ROOT}/stages/04_3d_components/component_report.json"
SOURCE_PROPOSAL_MANIFEST="${SOURCE_ROOT}/stages/03_flashsplat_region_supports/proposal_manifest.json"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"

MIN_GAUSSIAN_CAMERA_SUPPORT="${MIN_GAUSSIAN_CAMERA_SUPPORT:-2}"
MIN_SPATIAL_COMPONENT_GAUSSIANS="${MIN_SPATIAL_COMPONENT_GAUSSIANS:-500}"
MIN_SPATIAL_COMPONENT_CAMERAS="${MIN_SPATIAL_COMPONENT_CAMERAS:-2}"
VOXEL_SCALE_MULTIPLIER="${VOXEL_SCALE_MULTIPLIER:-4.0}"
MIN_VOXEL_SIZE="${MIN_VOXEL_SIZE:-0.01}"
MAX_VOXEL_SIZE="${MAX_VOXEL_SIZE:-0.20}"
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

test -f "${ONTOLOGY}"
test -f "${SOURCE_COMPONENT_REPORT}"
test -f "${SOURCE_PROPOSAL_MANIFEST}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nsource=%s\noutput=%s\nmode=dinov3_multiview_spatial_core_audit\nreport_only=1\nidentity_policy=strict_or_exactly_one_outlier_leave_one_out_stable\noutlier_policy=exclude_single_dissenting_camera\ngaussian_support_policy=one_presence_vote_per_agreeing_camera\ngeometry_policy=adaptive_26_neighbor_voxel_components\ncross_class_overlap_policy=unique_maximum_supporting_camera_count_else_abstain\nmin_gaussian_camera_support=%s\nmin_spatial_component_gaussians=%s\nmin_spatial_component_cameras=%s\nvoxel_scale_multiplier=%s\nmin_voxel_size=%s\nmax_voxel_size=%s\nscene_specific_rules=0\nclass_specific_thresholds=0\noffline_threshold_sweep_used=0\nv5_used=0\nv5_labels_used=0\nv5_method_adapted=1\ndinov2_used=0\ninference_rerun=0\nflashsplat_rerun=0\nsemantic_labels_written=0\nsemantic_project_class_arrays_written=0\nlabel_map_written=0\nsemantic_ply_written=0\n' \
    "${SCENE}" "${SOURCE_ROOT}" "${OUT_DIR}" \
    "${MIN_GAUSSIAN_CAMERA_SUPPORT}" "${MIN_SPATIAL_COMPONENT_GAUSSIANS}" \
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
  echo "output=${OUT_DIR}"
  echo "mode=dinov3_multiview_spatial_core_audit"
  echo "report_only=1"
  echo "v5_used=0"
  echo "v5_labels_used=0"
  echo "v5_method_adapted=1"
  echo "dinov2_used=0"
  echo "inference_rerun=0"
  echo "flashsplat_rerun=0"
  echo "semantic_labels_written=0"
  echo "semantic_project_class_arrays_written=0"
  echo "label_map_written=0"
  echo "semantic_ply_written=0"
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

run_stage 01_multiview_spatial_core_audit \
  python -m scripts.task1.dinov3.audit_multiview_spatial_core \
    --component-report "${SOURCE_COMPONENT_REPORT}" \
    --proposal-manifest "${SOURCE_PROPOSAL_MANIFEST}" \
    --ontology "${ONTOLOGY}" \
    --output-dir "${AUDIT_DIR}" \
    --scene "${SCENE}" \
    --min-gaussian-camera-support "${MIN_GAUSSIAN_CAMERA_SUPPORT}" \
    --min-spatial-component-gaussians "${MIN_SPATIAL_COMPONENT_GAUSSIANS}" \
    --min-spatial-component-cameras "${MIN_SPATIAL_COMPONENT_CAMERAS}" \
    --voxel-scale-multiplier "${VOXEL_SCALE_MULTIPLIER}" \
    --min-voxel-size "${MIN_VOXEL_SIZE}" \
    --max-voxel-size "${MAX_VOXEL_SIZE}"

run_stage 02_render_multiview_spatial_core_overlays \
  python -m scripts.task1.dinov3.render_3d_component_overlays \
    --association-report "${AUDIT_DIR}/multiview_spatial_core_audit.json" \
    --proposal-manifest "${SOURCE_PROPOSAL_MANIFEST}" \
    --output-dir "${OVERLAY_DIR}"

run_stage 03_multiview_spatial_core_contact_sheet \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${OVERLAY_DIR}" \
    --output "${CONTACT_SHEET}" \
    --columns 4 \
    --thumb-width 300

test -f "${AUDIT_DIR}/multiview_spatial_core_audit.json"
test -f "${AUDIT_DIR}/proposed_core_mask.npy"
test -f "${AUDIT_DIR}/proposed_core_owner_count.npy"
test -f "${AUDIT_DIR}/proposed_core_overlap_mask.npy"
test -f "${AUDIT_DIR}/proposed_core_cross_class_overlap_mask.npy"
test -f "${AUDIT_DIR}/proposed_core_cross_class_tie_mask.npy"
test -f "${AUDIT_DIR}/proposed_core_supports.npz"
test -f "${OVERLAY_DIR}/component_overlay_manifest.json"
test -f "${CONTACT_SHEET}"
test ! -e "${AUDIT_DIR}/gaussian_labels.npy"
test ! -e "${AUDIT_DIR}/gaussian_project_class_ids.npy"
test ! -e "${AUDIT_DIR}/label_map.json"
if find "${OUT_DIR}" -type f -name '*.ply' -print -quit | grep -q .; then
  echo "Report-only audit unexpectedly wrote a PLY" >&2
  exit 2
fi

{
  echo "mode=dinov3_multiview_spatial_core_audit"
  echo "report_only=1"
  echo "source=${SOURCE_ROOT}"
  echo "identity_policy=strict_or_exactly_one_outlier_leave_one_out_stable"
  echo "outlier_policy=exclude_single_dissenting_camera"
  echo "gaussian_support_policy=one_presence_vote_per_agreeing_camera"
  echo "geometry_policy=adaptive_26_neighbor_voxel_components"
  echo "cross_class_overlap_policy=unique_maximum_supporting_camera_count_else_abstain"
  echo "scene_specific_rules=0"
  echo "class_specific_thresholds=0"
  echo "offline_threshold_sweep_used=0"
  echo "v5_used=0"
  echo "v5_labels_used=0"
  echo "v5_method_adapted=1"
  echo "dinov2_used=0"
  echo "inference_rerun=0"
  echo "flashsplat_rerun=0"
  echo "semantic_labels_written=0"
  echo "semantic_project_class_arrays_written=0"
  echo "label_map_written=0"
  echo "semantic_ply_written=0"
} > "${OUT_DIR}/experiment_mode.txt"

find "${OUT_DIR}" -maxdepth 4 -type f | sort
