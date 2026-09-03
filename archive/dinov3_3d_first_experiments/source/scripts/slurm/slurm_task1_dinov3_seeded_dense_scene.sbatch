#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_seedprop
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

# Cached DINOv3-only finalization:
# immutable direct-region seeds + dense candidate labels -> seed-guarded 3D
# connected-component propagation -> validation, SuperSplat PLY, and visual QA.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${WORKSPACE_ROOT}/external/FlashSplat}"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
EVIDENCE_OUTPUT_NAME="${EVIDENCE_OUTPUT_NAME:-${SCENE}_dinov3_3d_first_report_v1}"
REGION_OUTPUT_NAME="${REGION_OUTPUT_NAME:-${SCENE}_dinov3_region_vote_majority_v1}"
DENSE_OUTPUT_NAME="${DENSE_OUTPUT_NAME:-${SCENE}_dinov3_dense_vote_complete_v1}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_seeded_dense_propagation_v1}"

EVIDENCE_ROOT="${TASK1_ROOT}/${EVIDENCE_OUTPUT_NAME}"
SOURCE_VIEW_DIR="${EVIDENCE_ROOT}/stages/01_dinov3_query_evidence"
REGION_DIR="${TASK1_ROOT}/${REGION_OUTPUT_NAME}/deliverables"
DENSE_DIR="${TASK1_ROOT}/${DENSE_OUTPUT_NAME}/deliverables"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
DELIVERABLE_DIR="${OUT_DIR}/deliverables"
VISUALIZATION_DIR="${OUT_DIR}/visualizations"
OVERLAY_DIR="${VISUALIZATION_DIR}/semantic_overlays"
RGB_DIR="${VISUALIZATION_DIR}/qa_rgb"
CONTACT_DIR="${VISUALIZATION_DIR}/contact_sheets"
VALIDATION_DIR="${OUT_DIR}/validation"
RUN_LOG_DIR="${OUT_DIR}/logs"

ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
SOURCE_PLY="${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
VOXEL_SCALE_MULTIPLIER="${VOXEL_SCALE_MULTIPLIER:-4.0}"
MIN_VOXEL_SIZE="${MIN_VOXEL_SIZE:-0.01}"
MAX_VOXEL_SIZE="${MAX_VOXEL_SIZE:-0.20}"
QA_VIEW_COUNT="${QA_VIEW_COUNT:-24}"
WRITE_SEMANTIC_PLY="${WRITE_SEMANTIC_PLY:-1}"
RESET_OUTPUT="${RESET_OUTPUT:-0}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

for value_name in WRITE_SEMANTIC_PLY RESET_OUTPUT CONFIG_ONLY; do
  value="${!value_name}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${value_name} must be 0 or 1" >&2
    exit 2
  fi
done

test -d "${MODEL_DIR}"
test -d "${FLASHSPLAT_ROOT}"
test -f "${SOURCE_PLY}"
test -f "${SOURCE_VIEW_DIR}/view_manifest.json"
test -f "${REGION_DIR}/gaussian_project_class_ids.npy"
test -f "${REGION_DIR}/region_vote_summary.json"
test -f "${DENSE_DIR}/gaussian_project_class_ids.npy"
test -f "${DENSE_DIR}/dense_vote_summary.json"
test -f "${ONTOLOGY}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nevidence=%s\nregion_seeds=%s\ndense_candidates=%s\noutput=%s\nmode=dinov3_seed_guarded_dense_3d_propagation\nvoxel_scale_multiplier=%s\nmin_voxel_size=%s\nmax_voxel_size=%s\nminimum_component_size_used=0\nscene_specific_rules=0\nclass_specific_thresholds=0\noffline_threshold_sweep_used=0\nv5_used=0\ndinov2_used=0\nsemantic_labels_written=1\nsemantic_ply_written=%s\nqa_view_count=%s\n' \
    "${SCENE}" "${EVIDENCE_ROOT}" "${REGION_DIR}" "${DENSE_DIR}" \
    "${OUT_DIR}" "${VOXEL_SCALE_MULTIPLIER}" "${MIN_VOXEL_SIZE}" \
    "${MAX_VOXEL_SIZE}" "${WRITE_SEMANTIC_PLY}" "${QA_VIEW_COUNT}"
  exit 0
fi

if [[ -d "${OUT_DIR}" ]]; then
  if [[ "${RESET_OUTPUT}" = "1" ]]; then
    case "${OUT_DIR}" in
      "${TASK1_ROOT}"/*) rm -rf -- "${OUT_DIR}" ;;
      *) echo "Refusing to reset unexpected output path: ${OUT_DIR}" >&2; exit 2 ;;
    esac
  else
    echo "Output exists; choose a fresh OUTPUT_NAME or explicitly set RESET_OUTPUT=1:" >&2
    echo "${OUT_DIR}" >&2
    exit 2
  fi
fi

mkdir -p \
  "${OVERLAY_DIR}" "${RGB_DIR}" "${CONTACT_DIR}" \
  "${VALIDATION_DIR}" "${RUN_LOG_DIR}" "${TASK1_ROOT}/logs"
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
  echo "evidence=${EVIDENCE_ROOT}"
  echo "region_seeds=${REGION_DIR}"
  echo "dense_candidates=${DENSE_DIR}"
  echo "output=${OUT_DIR}"
  echo "mode=dinov3_seed_guarded_dense_3d_propagation"
  echo "minimum_component_size_used=0"
  echo "scene_specific_rules=0"
  echo "class_specific_thresholds=0"
  echo "offline_threshold_sweep_used=0"
  echo "v5_used=0"
  echo "dinov2_used=0"
  nvidia-smi
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

propagation=(
  python -m scripts.task1.dinov3.propagate_dense_labels_from_region_seeds
  --seed-labels "${REGION_DIR}/gaussian_project_class_ids.npy"
  --seed-summary "${REGION_DIR}/region_vote_summary.json"
  --dense-labels "${DENSE_DIR}/gaussian_project_class_ids.npy"
  --dense-summary "${DENSE_DIR}/dense_vote_summary.json"
  --ontology "${ONTOLOGY}"
  --source-ply "${SOURCE_PLY}"
  --output-dir "${DELIVERABLE_DIR}"
  --scene "${SCENE}"
  --voxel-scale-multiplier "${VOXEL_SCALE_MULTIPLIER}"
  --min-voxel-size "${MIN_VOXEL_SIZE}"
  --max-voxel-size "${MAX_VOXEL_SIZE}"
)
if [[ "${WRITE_SEMANTIC_PLY}" = "0" ]]; then
  propagation+=(--no-semantic-ply)
fi
run_stage 01_seed_guarded_dense_propagation "${propagation[@]}"

validation=(
  python -m scripts.task1.qa.validate_task1_outputs
  --labels "${DELIVERABLE_DIR}/gaussian_labels.npy"
  --label-map "${DELIVERABLE_DIR}/label_map.json"
)
if [[ "${WRITE_SEMANTIC_PLY}" = "1" ]]; then
  validation+=(--semantic-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply")
fi
run_stage 02_validate_outputs "${validation[@]}" \
  2>&1 | tee "${VALIDATION_DIR}/validate_task1_outputs.log"

if [[ "${WRITE_SEMANTIC_PLY}" = "1" ]]; then
  run_stage 03_supersplat_debug \
    python -m scripts.task1.qa.export_supersplat_label_colors \
      --input-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
      --label-map "${DELIVERABLE_DIR}/label_map.json" \
      --output-ply "${VISUALIZATION_DIR}/semantic_point_cloud_supersplat_debug.ply" \
      --metadata-json "${VISUALIZATION_DIR}/semantic_point_cloud_supersplat_debug.json" \
      --color-mode class \
      --overwrite
fi

run_stage 04_semantic_overlays \
  python -m scripts.task1.qa.render_auto_label_overlays \
    --model-path "${MODEL_DIR}" \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-dir "${OVERLAY_DIR}" \
    --rgb-output-dir "${RGB_DIR}" \
    --source-view-manifest "${SOURCE_VIEW_DIR}/view_manifest.json" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --max-labels 0 \
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
    --output "${CONTACT_DIR}/semantic_labels.png" \
    --columns 4 \
    --thumb-width 360

run_stage 07_color_legend \
  python -m scripts.task1.qa.write_semantic_color_legend \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --output-json "${VISUALIZATION_DIR}/semantic_color_legend.json" \
    --output-png "${VISUALIZATION_DIR}/semantic_color_legend.png" \
    --color-mode class \
    --overwrite

{
  echo "mode=dinov3_seed_guarded_dense_3d_propagation"
  echo "evidence_report=${EVIDENCE_ROOT}"
  echo "region_seeds=${REGION_DIR}"
  echo "dense_candidates=${DENSE_DIR}"
  echo "minimum_component_size_used=0"
  echo "scene_specific_rules=0"
  echo "class_specific_thresholds=0"
  echo "offline_threshold_sweep_used=0"
  echo "v5_used=0"
  echo "dinov2_used=0"
  echo "semantic_labels_written=1"
  echo "semantic_ply_written=${WRITE_SEMANTIC_PLY}"
} > "${OUT_DIR}/experiment_mode.txt"

test -f "${DELIVERABLE_DIR}/gaussian_labels.npy"
test -f "${DELIVERABLE_DIR}/gaussian_project_class_ids.npy"
test -f "${DELIVERABLE_DIR}/region_seed_mask.npy"
test -f "${DELIVERABLE_DIR}/propagated_fill_mask.npy"
test -f "${DELIVERABLE_DIR}/label_source_codes.npy"
test -f "${DELIVERABLE_DIR}/dense_component_ids.npy"
test -f "${DELIVERABLE_DIR}/dense_component_status_codes.npy"
test -f "${DELIVERABLE_DIR}/label_map.json"
test -f "${DELIVERABLE_DIR}/propagation_summary.json"
test -f "${CONTACT_DIR}/semantic_labels.png"
if [[ "${WRITE_SEMANTIC_PLY}" = "1" ]]; then
  test -f "${DELIVERABLE_DIR}/semantic_point_cloud.ply"
  test -f "${VISUALIZATION_DIR}/semantic_point_cloud_supersplat_debug.ply"
fi

find "${OUT_DIR}" -maxdepth 4 -type f | sort
