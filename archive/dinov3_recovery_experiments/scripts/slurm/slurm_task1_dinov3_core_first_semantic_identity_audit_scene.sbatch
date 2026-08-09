#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_core_first
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

# Fully automatic cached-evidence diagnostic:
# every cached class-agnostic proposal graph, including mixed graphs
# -> multiview support -> adaptive spatial cores -> per-core DINOv3 identity
# -> exact report-only projections.
# No scene/class/component decisions, labels, label maps, or PLY are written.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${WORKSPACE_ROOT}/external/FlashSplat}"

SCENE="${SCENE:?Set SCENE to the target scene}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
SOURCE_OUTPUT_NAME="${SOURCE_OUTPUT_NAME:-${SCENE}_dinov3_3d_first_report_targeted_50plus20_v1}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_core_first_semantic_identity_audit_v1}"

SOURCE_ROOT="${TASK1_ROOT}/${SOURCE_OUTPUT_NAME}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
AUDIT_DIR="${OUT_DIR}/audit"
VISUALIZATION_DIR="${OUT_DIR}/visualizations"
OVERLAY_DIR="${VISUALIZATION_DIR}/exact_overlays"
MASK_DIR="${VISUALIZATION_DIR}/exact_masks"
RUN_LOG_DIR="${OUT_DIR}/logs"

SOURCE_COMPONENT_REPORT="${SOURCE_ROOT}/stages/04_3d_components/component_report.json"
SOURCE_PROPOSAL_MANIFEST="${SOURCE_ROOT}/stages/03_flashsplat_region_supports/proposal_manifest.json"
SOURCE_VIEW_MANIFEST="${SOURCE_ROOT}/stages/01_dinov3_query_evidence/view_manifest.json"
SOURCE_PLY="${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
MIN_GAUSSIAN_CAMERA_SUPPORT="${MIN_GAUSSIAN_CAMERA_SUPPORT:-2}"
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
test -f "${SOURCE_COMPONENT_REPORT}"
test -f "${SOURCE_PROPOSAL_MANIFEST}"
test -f "${SOURCE_VIEW_MANIFEST}"
test -f "${ONTOLOGY}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nsource=%s\noutput=%s\nmode=dinov3_core_first_semantic_identity_audit_v1\nreport_only=1\norder_of_operations=all_graphs_to_multiview_spatial_cores_then_per_core_identity\nsource_mixed_components_included=1\nidentity_policy=per_core_strict_or_exactly_one_outlier_leave_one_out_stable\noutlier_policy=exclude_dissenting_camera_then_recompute_and_resplit\nmin_gaussian_camera_support=%s\nmin_spatial_component_gaussians=%s\nmin_spatial_component_cameras=%s\nvoxel_scale_multiplier=%s\nmin_voxel_size=%s\nmax_voxel_size=%s\nscene_specific_rules=0\nclass_specific_thresholds=0\nmanual_component_decisions=0\ninference_rerun=0\nflashsplat_rerun=0\nsemantic_labels_written=0\nsemantic_project_class_arrays_written=0\nlabel_map_written=0\nsemantic_ply_written=0\n' \
    "${SCENE}" "${SOURCE_ROOT}" "${OUT_DIR}" \
    "${MIN_GAUSSIAN_CAMERA_SUPPORT}" \
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
  echo "output=${OUT_DIR}"
  echo "mode=dinov3_core_first_semantic_identity_audit_v1"
  echo "report_only=1"
  echo "order_of_operations=all_graphs_to_multiview_spatial_cores_then_per_core_identity"
  echo "source_mixed_components_included=1"
  echo "scene_specific_rules=0"
  echo "class_specific_thresholds=0"
  echo "manual_component_decisions=0"
  echo "inference_rerun=0"
  echo "flashsplat_rerun=0"
  echo "semantic_labels_written=0"
  echo "semantic_project_class_arrays_written=0"
  echo "label_map_written=0"
  echo "semantic_ply_written=0"
  nvidia-smi
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

run_stage 01_core_first_semantic_identity_audit \
  python -m scripts.task1.dinov3.audit_core_first_semantic_identity \
    --component-report "${SOURCE_COMPONENT_REPORT}" \
    --proposal-manifest "${SOURCE_PROPOSAL_MANIFEST}" \
    --ontology "${ONTOLOGY}" \
    --output-dir "${AUDIT_DIR}" \
    --scene "${SCENE}" \
    --min-gaussian-camera-support "${MIN_GAUSSIAN_CAMERA_SUPPORT}" \
    --min-spatial-component-gaussians \
      "${MIN_SPATIAL_COMPONENT_GAUSSIANS}" \
    --min-spatial-component-cameras "${MIN_SPATIAL_COMPONENT_CAMERAS}" \
    --voxel-scale-multiplier "${VOXEL_SCALE_MULTIPLIER}" \
    --min-voxel-size "${MIN_VOXEL_SIZE}" \
    --max-voxel-size "${MAX_VOXEL_SIZE}"

run_stage 02_render_exact_core_first_support \
  python -m scripts.task1.dinov3.render_incremental_spatial_core_fill \
    --audit-report "${AUDIT_DIR}/core_first_semantic_identity_audit.json" \
    --support-npz "${AUDIT_DIR}/core_first_supports.npz" \
    --model-path "${MODEL_DIR}" \
    --source-view-manifest "${SOURCE_VIEW_MANIFEST}" \
    --overlay-output-dir "${OVERLAY_DIR}" \
    --mask-output-dir "${MASK_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}"

run_stage 03_exact_overlay_contact_sheet \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${OVERLAY_DIR}" \
    --output "${VISUALIZATION_DIR}/exact_overlays.png" \
    --columns 4 \
    --thumb-width 300

run_stage 04_exact_mask_contact_sheet \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${MASK_DIR}" \
    --output "${VISUALIZATION_DIR}/exact_masks.png" \
    --columns 4 \
    --thumb-width 300

test -f "${AUDIT_DIR}/core_first_semantic_identity_audit.json"
test -f "${AUDIT_DIR}/core_first_mask.npy"
test -f "${AUDIT_DIR}/core_first_owner_count.npy"
test -f "${AUDIT_DIR}/core_first_overlap_mask.npy"
test -f "${AUDIT_DIR}/core_first_cross_class_overlap_mask.npy"
test -f "${AUDIT_DIR}/core_first_cross_class_tie_mask.npy"
test -f "${AUDIT_DIR}/core_first_supports.npz"
test -f "${OVERLAY_DIR}/exact_residual_overlay_manifest.json"
test -f "${VISUALIZATION_DIR}/exact_overlays.png"
test -f "${VISUALIZATION_DIR}/exact_masks.png"
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
  echo "mode=dinov3_core_first_semantic_identity_audit_v1"
  echo "report_only=1"
  echo "order_of_operations=all_graphs_to_multiview_spatial_cores_then_per_core_identity"
  echo "source_mixed_components_included=1"
  echo "identity_policy=per_core_strict_or_exactly_one_outlier_leave_one_out_stable"
  echo "outlier_policy=exclude_dissenting_camera_then_recompute_and_resplit"
  echo "scene_specific_rules=0"
  echo "class_specific_thresholds=0"
  echo "manual_component_decisions=0"
  echo "inference_rerun=0"
  echo "flashsplat_rerun=0"
  echo "semantic_labels_written=0"
  echo "semantic_project_class_arrays_written=0"
  echo "label_map_written=0"
  echo "semantic_ply_written=0"
} > "${OUT_DIR}/experiment_mode.txt"

find "${OUT_DIR}" -maxdepth 5 -type f | sort
