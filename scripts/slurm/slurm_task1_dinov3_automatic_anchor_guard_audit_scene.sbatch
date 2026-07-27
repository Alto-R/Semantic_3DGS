#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_anchor_guard
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
# incremental residuals + immutable preferred-v2 3D anchors -> global
# class-consistent neighborhood profiles -> exact report-only projections.
# No scene/component decisions, labels, label maps, or PLY are written.

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
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_automatic_anchor_guard_audit_v1}"

SOURCE_ROOT="${TASK1_ROOT}/${SOURCE_OUTPUT_NAME}"
PREFERRED_ROOT="${TASK1_ROOT}/${PREFERRED_OUTPUT_NAME}"
INCREMENTAL_ROOT="${TASK1_ROOT}/${INCREMENTAL_AUDIT_OUTPUT_NAME}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
AUDIT_DIR="${OUT_DIR}/audit"
VISUALIZATION_DIR="${OUT_DIR}/visualizations"
RUN_LOG_DIR="${OUT_DIR}/logs"

INCREMENTAL_REPORT="${INCREMENTAL_ROOT}/audit/incremental_spatial_core_fill_audit.json"
INCREMENTAL_SUPPORTS="${INCREMENTAL_ROOT}/audit/incremental_fill_supports.npz"
SOURCE_PROPOSAL_MANIFEST="${SOURCE_ROOT}/stages/03_flashsplat_region_supports/proposal_manifest.json"
SOURCE_VIEW_MANIFEST="${SOURCE_ROOT}/stages/01_dinov3_query_evidence/view_manifest.json"
PREFERRED_LABELS="${PREFERRED_ROOT}/deliverables/gaussian_labels.npy"
SOURCE_PLY="${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
NEIGHBOR_COUNT="${NEIGHBOR_COUNT:-16}"
MIN_SAME_CLASS_NEIGHBORS="${MIN_SAME_CLASS_NEIGHBORS:-4}"
MIN_SAME_CLASS_FRACTION="${MIN_SAME_CLASS_FRACTION:-0.75}"
MAX_SAME_TO_COMPETING_DISTANCE_RATIO="${MAX_SAME_TO_COMPETING_DISTANCE_RATIO:-1.0}"
MIN_SPATIAL_COMPONENT_GAUSSIANS="${MIN_SPATIAL_COMPONENT_GAUSSIANS:-500}"
MIN_SPATIAL_COMPONENT_CAMERAS="${MIN_SPATIAL_COMPONENT_CAMERAS:-2}"
QUERY_WORKERS="${QUERY_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"
RESET_OUTPUT="${RESET_OUTPUT:-0}"
PROFILES=(radius_1x radius_2x radius_4x)

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
test -f "${INCREMENTAL_REPORT}"
test -f "${INCREMENTAL_SUPPORTS}"
test -f "${SOURCE_PROPOSAL_MANIFEST}"
test -f "${SOURCE_VIEW_MANIFEST}"
test -f "${PREFERRED_LABELS}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nsource=%s\npreferred_v2=%s\nincremental_audit=%s\noutput=%s\nmode=dinov3_automatic_anchor_guard_audit_v1\nreport_only=1\npreferred_labels_read_only=1\nanchor_policy=local_same_class_neighbors_fraction_and_competing_distance\nprofiles=radius_1x,radius_2x,radius_4x\nneighbor_count=%s\nmin_same_class_neighbors=%s\nmin_same_class_fraction=%s\nmax_same_to_competing_distance_ratio=%s\nmin_spatial_component_gaussians=%s\nmin_spatial_component_cameras=%s\nscene_specific_rules=0\nclass_specific_thresholds=0\nmanual_component_decisions=0\nsemantic_labels_written=0\nsemantic_project_class_arrays_written=0\nlabel_map_written=0\nsemantic_ply_written=0\n' \
    "${SCENE}" "${SOURCE_ROOT}" "${PREFERRED_ROOT}" "${INCREMENTAL_ROOT}" \
    "${OUT_DIR}" "${NEIGHBOR_COUNT}" "${MIN_SAME_CLASS_NEIGHBORS}" \
    "${MIN_SAME_CLASS_FRACTION}" \
    "${MAX_SAME_TO_COMPETING_DISTANCE_RATIO}" \
    "${MIN_SPATIAL_COMPONENT_GAUSSIANS}" \
    "${MIN_SPATIAL_COMPONENT_CAMERAS}"
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
  echo "preferred_v2=${PREFERRED_ROOT}"
  echo "incremental_audit=${INCREMENTAL_ROOT}"
  echo "output=${OUT_DIR}"
  echo "mode=dinov3_automatic_anchor_guard_audit_v1"
  echo "report_only=1"
  echo "preferred_labels_read_only=1"
  echo "scene_specific_rules=0"
  echo "class_specific_thresholds=0"
  echo "manual_component_decisions=0"
  echo "semantic_labels_written=0"
  echo "semantic_project_class_arrays_written=0"
  echo "label_map_written=0"
  echo "semantic_ply_written=0"
  nvidia-smi
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

run_stage 01_automatic_anchor_guard_audit \
  python -m scripts.task1.dinov3.audit_automatic_anchor_guard_fill \
    --incremental-fill-report "${INCREMENTAL_REPORT}" \
    --incremental-fill-supports "${INCREMENTAL_SUPPORTS}" \
    --proposal-manifest "${SOURCE_PROPOSAL_MANIFEST}" \
    --preferred-labels "${PREFERRED_LABELS}" \
    --source-ply "${SOURCE_PLY}" \
    --output-dir "${AUDIT_DIR}" \
    --scene "${SCENE}" \
    --neighbor-count "${NEIGHBOR_COUNT}" \
    --min-same-class-neighbors "${MIN_SAME_CLASS_NEIGHBORS}" \
    --min-same-class-fraction "${MIN_SAME_CLASS_FRACTION}" \
    --max-same-to-competing-distance-ratio \
      "${MAX_SAME_TO_COMPETING_DISTANCE_RATIO}" \
    --min-spatial-component-gaussians \
      "${MIN_SPATIAL_COMPONENT_GAUSSIANS}" \
    --min-spatial-component-cameras "${MIN_SPATIAL_COMPONENT_CAMERAS}" \
    --query-workers "${QUERY_WORKERS}"

stage_number=2
for profile in "${PROFILES[@]}"; do
  profile_root="${AUDIT_DIR}/${profile}"
  overlay_dir="${VISUALIZATION_DIR}/${profile}/exact_overlays"
  mask_dir="${VISUALIZATION_DIR}/${profile}/exact_masks"
  run_stage "$(printf '%02d' "${stage_number}")_render_${profile}" \
    python -m scripts.task1.dinov3.render_incremental_spatial_core_fill \
      --audit-report "${profile_root}/automatic_anchor_guard_audit.json" \
      --support-npz "${profile_root}/automatic_fill_supports.npz" \
      --model-path "${MODEL_DIR}" \
      --source-view-manifest "${SOURCE_VIEW_MANIFEST}" \
      --overlay-output-dir "${overlay_dir}" \
      --mask-output-dir "${mask_dir}" \
      --flashsplat-root "${FLASHSPLAT_ROOT}" \
      --iteration 30000 \
      --max-width "${RENDER_MAX_WIDTH}"
  stage_number=$((stage_number + 1))
  run_stage "$(printf '%02d' "${stage_number}")_overlay_sheet_${profile}" \
    python -m scripts.task1.qa.make_contact_sheet \
      --image-dir "${overlay_dir}" \
      --output "${VISUALIZATION_DIR}/${profile}/exact_overlays.png" \
      --columns 4 \
      --thumb-width 300
  stage_number=$((stage_number + 1))
  run_stage "$(printf '%02d' "${stage_number}")_mask_sheet_${profile}" \
    python -m scripts.task1.qa.make_contact_sheet \
      --image-dir "${mask_dir}" \
      --output "${VISUALIZATION_DIR}/${profile}/exact_masks.png" \
      --columns 4 \
      --thumb-width 300
  stage_number=$((stage_number + 1))
done

test -f "${AUDIT_DIR}/automatic_anchor_guard_summary.json"
for profile in "${PROFILES[@]}"; do
  test -f "${AUDIT_DIR}/${profile}/automatic_anchor_guard_audit.json"
  test -f "${AUDIT_DIR}/${profile}/automatic_fill_mask.npy"
  test -f "${AUDIT_DIR}/${profile}/automatic_fill_supports.npz"
  test -f "${VISUALIZATION_DIR}/${profile}/exact_overlays/exact_residual_overlay_manifest.json"
  test -f "${VISUALIZATION_DIR}/${profile}/exact_overlays.png"
  test -f "${VISUALIZATION_DIR}/${profile}/exact_masks.png"
done
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
  echo "mode=dinov3_automatic_anchor_guard_audit_v1"
  echo "report_only=1"
  echo "preferred_labels_read_only=1"
  echo "anchor_policy=local_same_class_neighbors_fraction_and_competing_distance"
  echo "profiles=radius_1x,radius_2x,radius_4x"
  echo "scene_specific_rules=0"
  echo "class_specific_thresholds=0"
  echo "manual_component_decisions=0"
  echo "semantic_labels_written=0"
  echo "semantic_project_class_arrays_written=0"
  echo "label_map_written=0"
  echo "semantic_ply_written=0"
} > "${OUT_DIR}/experiment_mode.txt"

find "${OUT_DIR}" -maxdepth 5 -type f | sort
