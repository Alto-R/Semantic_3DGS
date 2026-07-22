#!/usr/bin/env bash
#SBATCH --job-name=task1_ade_replay
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"
FLASHSPLAT_ROOT="${WORKSPACE_ROOT}/external/FlashSplat"

SCENE="${SCENE:-room}"
SOURCE_NAME="${SOURCE_NAME:-${SCENE}_dinov2_separate_abstain_allviews_v1}"
CACHE_NAME="${CACHE_NAME:-${SCENE}_dinov2_ade_refinement_auto_spatial_thing_guard_v3}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov2_ade_refinement_adaptive_instance_guard_v5}"
SOURCE_DIR="${TASK1_ROOT}/${SOURCE_NAME}"
CACHE_DIR="${TASK1_ROOT}/${CACHE_NAME}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
SOURCE_VIEW_DIR="${SOURCE_DIR}/stages/01_real_camera_views"
BASE_LABELS="${SOURCE_DIR}/deliverables/gaussian_labels.npy"
BASE_LABEL_MAP="${SOURCE_DIR}/deliverables/label_map.json"
CACHE_CONFIG="${CACHE_DIR}/stages/04_grounding_camera_selection/automatic_ade_refinement_config.json"
CACHE_PROPOSALS="${CACHE_DIR}/stages/06_grounding_flashsplat"
CACHE_FUSION="${CACHE_DIR}/stages/07_grounding_fusion"
REFINEMENT_DIR="${OUT_DIR}/stages/08_ade_refinement"
EXTENSION_DIR="${OUT_DIR}/stages/09_missing_vocabulary_merge"
DELIVERABLE_DIR="${OUT_DIR}/deliverables"
VISUALIZATION_DIR="${OUT_DIR}/visualizations"
QA_RGB_DIR="${VISUALIZATION_DIR}/qa_rgb"
QA_OVERLAY_DIR="${VISUALIZATION_DIR}/semantic_overlays"
CONTACT_DIR="${VISUALIZATION_DIR}/contact_sheets"
VALIDATION_DIR="${OUT_DIR}/validation"
RUN_LOG_DIR="${OUT_DIR}/logs"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
COLOR_MODE="${COLOR_MODE:-class}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
QA_VIEW_COUNT="${QA_VIEW_COUNT:-12}"
RESET_OUTPUT="${RESET_OUTPUT:-1}"

EXTENSION_CONFIG="${EXTENSION_CONFIG:-}"
EXTENSION_LABELS="${EXTENSION_LABELS:-}"
EXTENSION_LABEL_MAP="${EXTENSION_LABEL_MAP:-}"
EXTENSION_MANIFEST="${EXTENSION_MANIFEST:-}"
EXTENSION_CLASSES="${EXTENSION_CLASSES:-}"

test -f "${BASE_LABELS}"
test -f "${BASE_LABEL_MAP}"
test -f "${CACHE_CONFIG}"
test -f "${CACHE_PROPOSALS}/class_evidence/class_evidence_manifest.json"
test -f "${CACHE_FUSION}/gaussian_labels.npy"
test -f "${CACHE_FUSION}/label_map.json"
test -f "${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
test -f "${SOURCE_VIEW_DIR}/view_manifest.json"
test -f "${ONTOLOGY}"

MISSING_VOCABULARY_ENABLED=0
if [[ -n "${EXTENSION_CONFIG}" || -n "${EXTENSION_LABELS}" || -n "${EXTENSION_LABEL_MAP}" ]]; then
  test -f "${EXTENSION_CONFIG}"
  test -f "${EXTENSION_LABELS}"
  test -f "${EXTENSION_LABEL_MAP}"
  if [[ -n "${EXTENSION_MANIFEST}" ]]; then
    test -f "${EXTENSION_MANIFEST}"
  fi
  MISSING_VOCABULARY_ENABLED=1
fi

if [[ "${RESET_OUTPUT}" = "1" && -d "${OUT_DIR}" ]]; then
  case "${OUT_DIR}" in
    "${TASK1_ROOT}"/*) rm -rf -- "${OUT_DIR}" ;;
    *) echo "Refusing to reset unexpected output path: ${OUT_DIR}" >&2; exit 2 ;;
  esac
fi

mkdir -p \
  "${REFINEMENT_DIR}" "${EXTENSION_DIR}" "${DELIVERABLE_DIR}" \
  "${QA_RGB_DIR}" "${QA_OVERLAY_DIR}" "${CONTACT_DIR}" \
  "${VALIDATION_DIR}" "${RUN_LOG_DIR}" "${TASK1_ROOT}/logs"

cd "${PROJECT_ROOT}"

run_stage() {
  local stage_name="$1"
  shift
  echo "===== ${stage_name} ====="
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" "$@" 2>&1 \
    | tee "${RUN_LOG_DIR}/${stage_name}.log"
}

run_report() {
  local stage_name="$1"
  local report_path="$2"
  shift 2
  echo "===== ${stage_name} ====="
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" "$@" \
    | tee "${report_path}" "${RUN_LOG_DIR}/${stage_name}.log"
}

run_stage 08_guarded_ade_refinement_replay \
  python -m scripts.task1.merge.merge_ade_refinement \
    --base-labels "${BASE_LABELS}" \
    --base-label-map "${BASE_LABEL_MAP}" \
    --grounding-labels "${CACHE_FUSION}/gaussian_labels.npy" \
    --grounding-label-map "${CACHE_FUSION}/label_map.json" \
    --class-evidence-dir "${CACHE_PROPOSALS}/class_evidence" \
    --refinement-config "${CACHE_CONFIG}" \
    --ontology "${ONTOLOGY}" \
    --source-ply "${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply" \
    --output-dir "${REFINEMENT_DIR}" \
    --semantic-ply-path "${REFINEMENT_DIR}/semantic_point_cloud.ply" \
    --scene "${SCENE}" \
    --min-anchor-gaussians 500 \
    --min-anchor-coverage 0.10 \
    --min-anchor-precision 0.10 \
    --min-group-proposals 2 \
    --min-group-source-views 2 \
    --evidence-min-positive-views 2 \
    --evidence-min-ratio 0.50 \
    --spatial-voxel-scale-multiplier 4.0 \
    --spatial-min-voxel-size 0.01 \
    --spatial-max-voxel-size 0.20 \
    --anchor-envelope-quantile 0.01 \
    --anchor-envelope-margin-ratio 0.05 \
    --ambiguous-thing-min-group-anchor-ratio 0.75 \
    --thing-halo-min-anchor-precision 0.75 \
    --nested-thing-min-containment 0.80 \
    --nested-thing-envelope-margin-ratio 0.25 \
    --nested-thing-min-parent-source-ratio 2.00 \
    --nested-thing-min-parent-overlap-coverage 0.10 \
    --prototype-min-robust-fraction 0.50 \
    --prototype-min-source-views 3 \
    --prototype-strong-source-views 5 \
    --prototype-max-dominant-competing-thing-fraction 0.50 \
    --prototype-max-partial-competing-thing-risk 0.40 \
    --prototype-ambiguous-min-dominant-competing-thing-fraction 0.90 \
    --prototype-max-shape-ratio 2.50 \
    --prototype-max-size-ratio 3.00 \
    --partial-anchor-extension-min-anchor-coverage 0.80 \
    --partial-anchor-extension-max-anchor-precision 0.25 \
    --partial-anchor-extension-min-robust-fraction 0.90 \
    --partial-anchor-extension-min-source-views 5 \
    --partial-anchor-extension-max-competing-thing-fraction 0.05 \
    --partial-anchor-extension-min-spatial-keep-fraction 0.95 \
    --overwrite

FINAL_LABELS="${REFINEMENT_DIR}/gaussian_labels.npy"
FINAL_LABEL_MAP="${REFINEMENT_DIR}/label_map.json"
FINAL_PLY="${REFINEMENT_DIR}/semantic_point_cloud.ply"
FINAL_MODE="dinov2_groundingdino_adaptive_instance_guard_v5"

if [[ "${MISSING_VOCABULARY_ENABLED}" = "1" ]]; then
  MANIFEST_ARGS=()
  if [[ -n "${EXTENSION_MANIFEST}" ]]; then
    MANIFEST_ARGS=(--extension-manifest "${EXTENSION_MANIFEST}")
  fi
  run_stage 09_missing_vocabulary_merge \
    python -m scripts.task1.merge.merge_semantic_extensions \
      --base-labels "${FINAL_LABELS}" \
      --base-label-map "${FINAL_LABEL_MAP}" \
      --extension-labels "${EXTENSION_LABELS}" \
      --extension-label-map "${EXTENSION_LABEL_MAP}" \
      "${MANIFEST_ARGS[@]}" \
      --extension-config "${EXTENSION_CONFIG}" \
      --ontology "${ONTOLOGY}" \
      --source-ply "${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply" \
      --output-dir "${EXTENSION_DIR}" \
      --semantic-ply-path "${EXTENSION_DIR}/semantic_point_cloud.ply" \
      --scene "${SCENE}" \
      --include-classes "${EXTENSION_CLASSES}" \
      --overwrite
  FINAL_LABELS="${EXTENSION_DIR}/gaussian_labels.npy"
  FINAL_LABEL_MAP="${EXTENSION_DIR}/label_map.json"
  FINAL_PLY="${EXTENSION_DIR}/semantic_point_cloud.ply"
fi

cp "${FINAL_LABELS}" "${DELIVERABLE_DIR}/gaussian_labels.npy"
cp "${FINAL_LABEL_MAP}" "${DELIVERABLE_DIR}/label_map.json"
cp "${FINAL_PLY}" "${DELIVERABLE_DIR}/semantic_point_cloud.ply"

run_stage 10_supersplat_debug_ply \
  python -m scripts.task1.qa.export_supersplat_label_colors \
    --input-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-ply "${VISUALIZATION_DIR}/semantic_point_cloud_supersplat_debug.ply" \
    --metadata-json "${VISUALIZATION_DIR}/semantic_point_cloud_supersplat_debug.json" \
    --color-mode "${COLOR_MODE}" \
    --overwrite

run_stage 11_semantic_overlays \
  python -m scripts.task1.qa.render_auto_label_overlays \
    --model-path "${MODEL_DIR}" \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-dir "${QA_OVERLAY_DIR}" \
    --rgb-output-dir "${QA_RGB_DIR}" \
    --source-view-manifest "${SOURCE_VIEW_DIR}/view_manifest.json" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --max-labels 0 \
    --count "${QA_VIEW_COUNT}" \
    --color-mode "${COLOR_MODE}"

run_stage 12_visible_coverage \
  python -m scripts.task1.qa.measure_overlay_coverage \
    --rgb-dir "${QA_RGB_DIR}" \
    --overlay-dir "${QA_OVERLAY_DIR}" \
    --output "${VALIDATION_DIR}/visible_overlay_coverage.json" \
    --difference-threshold 2 \
    --exclude-border 1

run_stage 13_contact_sheet \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${QA_OVERLAY_DIR}" \
    --output "${CONTACT_DIR}/semantic_labels.png" \
    --columns 4 \
    --thumb-width 360

run_stage 14_semantic_color_legend \
  python -m scripts.task1.qa.write_semantic_color_legend \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --output-json "${VISUALIZATION_DIR}/semantic_color_legend.json" \
    --output-png "${VISUALIZATION_DIR}/semantic_color_legend.png" \
    --color-mode "${COLOR_MODE}" \
    --overwrite

run_report 15_task1_validation "${VALIDATION_DIR}/task1_validation.json" \
  python -m scripts.task1.qa.validate_task1_outputs \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --semantic-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --visible-coverage "${VALIDATION_DIR}/visible_overlay_coverage.json"

printf 'final_mode=%s\nsource=%s\ncache=%s\nmissing_vocabulary_enabled=%s\n' \
  "${FINAL_MODE}" "${SOURCE_DIR}" "${CACHE_DIR}" \
  "${MISSING_VOCABULARY_ENABLED}" \
  > "${VALIDATION_DIR}/final_mode.txt"

find "${OUT_DIR}" -maxdepth 4 -type f | sort
