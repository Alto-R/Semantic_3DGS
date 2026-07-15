#!/usr/bin/env bash
#SBATCH --job-name=task1_semantic_scene
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)"
ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
FLASHSPLAT_ROOT="${ROOT}/external/FlashSplat"
TASK1_ROOT="${ROOT}/outputs/eyenavgs_task1"
SCENE="${SCENE:-bicycle}"
MODEL_DIR="${MODEL_DIR:-${ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_semantic}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
STAGE_ROOT="${OUT_DIR}/stages"
TARGET_GROUND_OUT_DIR="${STAGE_ROOT}/01_grounded_sam"
TARGET_PROPOSAL_DIR="${STAGE_ROOT}/02_flashsplat"
REUSE_SOURCE_OUT="${REUSE_SOURCE_OUT:-}"
if [ -n "${REUSE_SOURCE_OUT}" ]; then
  GROUND_OUT_DIR="${REUSE_SOURCE_OUT}/stages/01_grounded_sam"
  PROPOSAL_DIR="${REUSE_SOURCE_OUT}/stages/02_flashsplat"
else
  GROUND_OUT_DIR="${TARGET_GROUND_OUT_DIR}"
  PROPOSAL_DIR="${TARGET_PROPOSAL_DIR}"
fi
FUSION_DIR="${STAGE_ROOT}/03_semantic_fusion"
DELIVERABLE_DIR="${OUT_DIR}/deliverables"
VISUALIZATION_DIR="${OUT_DIR}/visualizations"
PLY_VIS_DIR="${VISUALIZATION_DIR}/ply"
SEMANTIC_OVERLAY_DIR="${VISUALIZATION_DIR}/overlays/semantic_labels"
FOCUS_NAME="${FOCUS_NAME:-focus_classes}"
FOCUS_CLASSES="${FOCUS_CLASSES:-}"
FOCUS_OVERLAY_DIR="${VISUALIZATION_DIR}/overlays/${FOCUS_NAME}"
CONTACT_DIR="${VISUALIZATION_DIR}/contact_sheets"
VALIDATION_DIR="${OUT_DIR}/validation"
RUN_LOG_DIR="${OUT_DIR}/logs"
TARGET_SELECTION_DIR="${VALIDATION_DIR}/camera_selection"

SCENE_CLASS_CONFIG="${PROJECT_ROOT}/configs/task1_semantic_classes.${SCENE}.json"
if [ -f "${SCENE_CLASS_CONFIG}" ]; then
  DEFAULT_CLASS_CONFIG="${SCENE_CLASS_CONFIG}"
else
  DEFAULT_CLASS_CONFIG="${PROJECT_ROOT}/configs/task1_semantic_classes.example.json"
fi
CLASS_CONFIG="${CLASS_CONFIG:-${DEFAULT_CLASS_CONFIG}}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
# An empty camera list enables evenly spaced sampling by SEMANTIC_VIEW_COUNT.
SEMANTIC_CAMERA_INDICES="${SEMANTIC_CAMERA_INDICES:-}"
SEMANTIC_VIEW_COUNT="${SEMANTIC_VIEW_COUNT:-50}"
AUTO_TARGET_VIEW_COUNT="${AUTO_TARGET_VIEW_COUNT:-0}"
TARGET_SELECTION_SOURCE="${TARGET_SELECTION_SOURCE:-}"
TARGET_CANDIDATE_COUNT="${TARGET_CANDIDATE_COUNT:-100}"
TARGET_PROJECTION_SAMPLE_COUNT="${TARGET_PROJECTION_SAMPLE_COUNT:-50000}"
TARGET_PROJECTION_MARGIN="${TARGET_PROJECTION_MARGIN:-1.5}"
TARGET_LOW_COVERAGE_QUANTILE="${TARGET_LOW_COVERAGE_QUANTILE:-0.35}"
MAX_DETECTIONS_PER_VIEW="${MAX_DETECTIONS_PER_VIEW:-32}"
MASK_BATCH_SIZE="${MASK_BATCH_SIZE:-4}"
FLASHSPLAT_SUPPORT_THRESHOLD="${FLASHSPLAT_SUPPORT_THRESHOLD:-0.05}"
CONTACT_THUMB_WIDTH="${CONTACT_THUMB_WIDTH:-360}"
MIN_ASSIGNED_GAUSSIANS="${MIN_ASSIGNED_GAUSSIANS:-0}"
MIN_ASSIGNED_THING_GAUSSIANS="${MIN_ASSIGNED_THING_GAUSSIANS:-5000}"
MIN_ASSIGNED_STUFF_GAUSSIANS="${MIN_ASSIGNED_STUFF_GAUSSIANS:-10000}"
ADAPTIVE_STUFF_MIN_VIEW_RATIO="${ADAPTIVE_STUFF_MIN_VIEW_RATIO:-0.50}"
ADAPTIVE_STUFF_THRESHOLD_RATIO="${ADAPTIVE_STUFF_THRESHOLD_RATIO:-0.75}"
ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO="${ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO:-0.50}"
MIN_LABEL_SCORE="${MIN_LABEL_SCORE:-0.0}"
COLOR_MODE="${COLOR_MODE:-class}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.30}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"
RESET_OUTPUT="${RESET_OUTPUT:-1}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-${ROOT}/InvRGBL_modif/pretrained/sam_vit_h_4b8939.pth}"
GROUNDED_SAM_ROOT="${GROUNDED_SAM_ROOT:-${ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/Grounded-Segment-Anything}"
GROUNDINGDINO_CONFIG="${GROUNDINGDINO_CONFIG:-${ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/saves/GroundingDINO_SwinT_OGC.py}"
GROUNDINGDINO_CHECKPOINT="${GROUNDINGDINO_CHECKPOINT:-${ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/saves/groundingdino_swint_ogc.pth}"
SAM_ROOT="${SAM_ROOT:-${GROUNDED_SAM_ROOT}/segment_anything}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TOKENIZERS_PARALLELISM

if [ "${COLOR_MODE}" != "class" ] && [ "${COLOR_MODE}" != "instance" ]; then
  echo "COLOR_MODE must be class or instance" >&2
  exit 2
fi

test -d "${MODEL_DIR}"
test -f "${MODEL_DIR}/cameras.json"
test -f "${CLASS_CONFIG}"

if [ -n "${REUSE_SOURCE_OUT}" ] && [ -z "${SEMANTIC_CAMERA_INDICES}" ]; then
  REUSE_GROUND_MANIFEST="${GROUND_OUT_DIR}/grounded_sam_manifest.json"
  test -f "${REUSE_GROUND_MANIFEST}"
  command -v jq >/dev/null 2>&1
  SEMANTIC_CAMERA_INDICES="$(jq -r '[.frames[].camera_index] | join(",")' "${REUSE_GROUND_MANIFEST}")"
  SEMANTIC_VIEW_COUNT="$(jq -r '.frames | length' "${REUSE_GROUND_MANIFEST}")"
  if [ -z "${SEMANTIC_CAMERA_INDICES}" ] || [ "${SEMANTIC_VIEW_COUNT}" -le 0 ]; then
    echo "Could not infer reuse cameras from ${REUSE_GROUND_MANIFEST}" >&2
    exit 2
  fi
fi

if [ -n "${REUSE_SOURCE_OUT}" ] && [ "${REUSE_SOURCE_OUT}" = "${OUT_DIR}" ]; then
  echo "Refusing to reuse stages from the output directory that will be reset: ${OUT_DIR}" >&2
  exit 2
fi

if [ "${AUTO_TARGET_VIEW_COUNT}" -gt 0 ] && [ -n "${REUSE_SOURCE_OUT}" ]; then
  echo "Automatic targeted views require newly generated masks and proposals; REUSE_SOURCE_OUT must be empty." >&2
  exit 2
fi

if [ "${AUTO_TARGET_VIEW_COUNT}" -gt 0 ] && [ -z "${TARGET_SELECTION_SOURCE}" ]; then
  echo "Automatic targeted views require TARGET_SELECTION_SOURCE to name an accepted baseline." >&2
  exit 2
fi

if [ "${CONFIG_ONLY}" = "1" ]; then
  printf 'scene=%s\nmodel=%s\noutput=%s\nclass_config=%s\nfocus_classes=%s\nview_count=%s\ncolor_mode=%s\n' \
    "${SCENE}" \
    "${MODEL_DIR}" \
    "${OUT_DIR}" \
    "${CLASS_CONFIG}" \
    "${FOCUS_CLASSES:-<none>}" \
    "${SEMANTIC_VIEW_COUNT}" \
    "${COLOR_MODE}"
  exit 0
fi

if [ "${RESET_OUTPUT}" = "1" ] && [ -d "${OUT_DIR}" ]; then
  case "${OUT_DIR}" in
    "${TASK1_ROOT}"/*) rm -rf -- "${OUT_DIR}" ;;
    *) echo "Refusing to reset unexpected output path: ${OUT_DIR}" >&2; exit 2 ;;
  esac
fi

mkdir -p \
  "${STAGE_ROOT}" \
  "${FUSION_DIR}" \
  "${DELIVERABLE_DIR}" \
  "${PLY_VIS_DIR}" \
  "${SEMANTIC_OVERLAY_DIR}" \
  "${FOCUS_OVERLAY_DIR}" \
  "${CONTACT_DIR}" \
  "${VALIDATION_DIR}" \
  "${TARGET_SELECTION_DIR}" \
  "${RUN_LOG_DIR}" \
  "${TASK1_ROOT}/logs"

if [ -n "${REUSE_SOURCE_OUT}" ]; then
  test -f "${GROUND_OUT_DIR}/grounded_sam_manifest.json"
  test -f "${PROPOSAL_DIR}/proposal_manifest.json"
  test -f "${PROPOSAL_DIR}/class_evidence/class_evidence_manifest.json"
  ln -s "${GROUND_OUT_DIR}" "${TARGET_GROUND_OUT_DIR}"
  ln -s "${PROPOSAL_DIR}" "${TARGET_PROPOSAL_DIR}"
else
  mkdir -p "${GROUND_OUT_DIR}" "${PROPOSAL_DIR}"
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

conda activate gaussian_grouping_true
cd "${PROJECT_ROOT}"

run_stage() {
  local stage_name="$1"
  shift
  echo "===== ${stage_name} ====="
  "$@" 2>&1 | tee "${RUN_LOG_DIR}/${stage_name}.log"
}

run_report() {
  local output_path="$1"
  local log_name="$2"
  shift 2
  "$@" | tee "${output_path}" "${RUN_LOG_DIR}/${log_name}.log"
}

if [ "${AUTO_TARGET_VIEW_COUNT}" -gt 0 ]; then
  BASELINE_MANIFEST="${TARGET_SELECTION_SOURCE}/stages/01_grounded_sam/grounded_sam_manifest.json"
  BASELINE_LABELS="${TARGET_SELECTION_SOURCE}/stages/03_semantic_fusion/gaussian_labels.npy"
  BASELINE_LABEL_MAP="${TARGET_SELECTION_SOURCE}/deliverables/label_map.json"
  test -f "${BASELINE_MANIFEST}"
  test -f "${TARGET_SELECTION_SOURCE}/validation/visible_overlay_coverage.json"
  test -f "${BASELINE_LABELS}"
  test -f "${BASELINE_LABEL_MAP}"

  run_stage 00a_screen_target_candidates \
    python scripts/task1/select_targeted_cameras.py screen \
      --model-path "${MODEL_DIR}" \
      --baseline-manifest "${BASELINE_MANIFEST}" \
      --output "${TARGET_SELECTION_DIR}/candidate_screen.json" \
      --indices-output "${TARGET_SELECTION_DIR}/candidate_indices.txt" \
      --iteration 30000 \
      --sample-count "${TARGET_PROJECTION_SAMPLE_COUNT}" \
      --candidate-count "${TARGET_CANDIDATE_COUNT}" \
      --projection-margin "${TARGET_PROJECTION_MARGIN}"

  CANDIDATE_CAMERA_INDICES="$(tr -d '\r\n' < "${TARGET_SELECTION_DIR}/candidate_indices.txt")"
  test -n "${CANDIDATE_CAMERA_INDICES}"

  run_stage 00b_render_target_candidates \
    python scripts/task1/render_auto_label_overlays.py \
      --model-path "${MODEL_DIR}" \
      --labels-npy "${BASELINE_LABELS}" \
      --label-map "${BASELINE_LABEL_MAP}" \
      --output-dir "${TARGET_SELECTION_DIR}/candidate_overlays" \
      --rgb-output-dir "${TARGET_SELECTION_DIR}/candidate_rgb" \
      --flashsplat-root "${FLASHSPLAT_ROOT}" \
      --iteration 30000 \
      --max-width "${RENDER_MAX_WIDTH}" \
      --camera-indices "${CANDIDATE_CAMERA_INDICES}" \
      --count "${TARGET_CANDIDATE_COUNT}"

  run_stage 00c_measure_target_candidates \
    python scripts/task1/measure_overlay_coverage.py \
      --rgb-dir "${TARGET_SELECTION_DIR}/candidate_rgb" \
      --overlay-dir "${TARGET_SELECTION_DIR}/candidate_overlays" \
      --output "${TARGET_SELECTION_DIR}/candidate_visible_coverage.json" \
      --difference-threshold 2 \
      --exclude-border 1

  run_stage 00d_select_targeted_cameras \
    python scripts/task1/select_targeted_cameras.py select \
      --model-path "${MODEL_DIR}" \
      --baseline-manifest "${BASELINE_MANIFEST}" \
      --candidate-manifest "${TARGET_SELECTION_DIR}/candidate_overlays/label_overlay_manifest.json" \
      --coverage-report "${TARGET_SELECTION_DIR}/candidate_visible_coverage.json" \
      --output "${TARGET_SELECTION_DIR}/targeted_camera_selection.json" \
      --indices-output "${TARGET_SELECTION_DIR}/final_camera_indices.txt" \
      --additional-count "${AUTO_TARGET_VIEW_COUNT}" \
      --low-coverage-quantile "${TARGET_LOW_COVERAGE_QUANTILE}" \
      --coverage-weight 0.8 \
      --novelty-weight 0.2

  SEMANTIC_CAMERA_INDICES="$(tr -d '\r\n' < "${TARGET_SELECTION_DIR}/final_camera_indices.txt")"
  SEMANTIC_VIEW_COUNT="$(awk -F, '{print NF}' "${TARGET_SELECTION_DIR}/final_camera_indices.txt")"
fi

{
  echo "job=${SLURM_JOB_ID:-local}"
  echo "host=$(hostname)"
  echo "project=${PROJECT_ROOT}"
  echo "scene=${SCENE}"
  echo "model=${MODEL_DIR}"
  echo "output=${OUT_DIR}"
  echo "class_config=${CLASS_CONFIG}"
  echo "focus_classes=${FOCUS_CLASSES:-<none>}"
  echo "reuse_source_output=${REUSE_SOURCE_OUT:-<none>}"
  echo "render_max_width=${RENDER_MAX_WIDTH}"
  echo "semantic_camera_indices=${SEMANTIC_CAMERA_INDICES:-<evenly-spaced>}"
  echo "semantic_view_count=${SEMANTIC_VIEW_COUNT}"
  echo "auto_target_view_count=${AUTO_TARGET_VIEW_COUNT}"
  echo "target_selection_source=${TARGET_SELECTION_SOURCE}"
  echo "target_candidate_count=${TARGET_CANDIDATE_COUNT}"
  echo "target_projection_sample_count=${TARGET_PROJECTION_SAMPLE_COUNT}"
  echo "target_projection_margin=${TARGET_PROJECTION_MARGIN}"
  echo "target_low_coverage_quantile=${TARGET_LOW_COVERAGE_QUANTILE}"
  echo "max_detections_per_view=${MAX_DETECTIONS_PER_VIEW}"
  echo "mask_batch_size=${MASK_BATCH_SIZE}"
  echo "flashsplat_support_threshold=${FLASHSPLAT_SUPPORT_THRESHOLD}"
  echo "min_assigned_gaussians=${MIN_ASSIGNED_GAUSSIANS}"
  echo "min_assigned_thing_gaussians=${MIN_ASSIGNED_THING_GAUSSIANS}"
  echo "min_assigned_stuff_gaussians=${MIN_ASSIGNED_STUFF_GAUSSIANS}"
  echo "adaptive_stuff_min_view_ratio=${ADAPTIVE_STUFF_MIN_VIEW_RATIO}"
  echo "adaptive_stuff_threshold_ratio=${ADAPTIVE_STUFF_THRESHOLD_RATIO}"
  echo "adaptive_thing_threshold_floor_ratio=${ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO}"
  echo "signed_class_evidence_min_positive_views=2"
  echo "signed_class_evidence_min_ratio=0.50"
  echo "consolidate_thing_instances=1"
  echo "color_mode=${COLOR_MODE}"
  nvidia-smi
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

BRANCH_REUSE_ARGS=()
if [ -n "${REUSE_SOURCE_OUT}" ]; then
  echo "reused ${GROUND_OUT_DIR}" | tee "${RUN_LOG_DIR}/01_grounded_sam.log"
  echo "reused ${PROPOSAL_DIR}" | tee "${RUN_LOG_DIR}/02_flashsplat.log"
  BRANCH_REUSE_ARGS=(--skip-mask-generation --skip-proposal-generation)
fi

run_stage 01_03_grounding_semantic_branch \
  python scripts/task1/run_grounding_semantic_branch.py \
    --model-path "${MODEL_DIR}" \
    --ground-output-dir "${GROUND_OUT_DIR}" \
    --proposal-dir "${PROPOSAL_DIR}" \
    --fusion-dir "${FUSION_DIR}" \
    --label-map-path "${DELIVERABLE_DIR}/label_map.json" \
    --semantic-ply-path "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --log-dir "${RUN_LOG_DIR}" \
    --class-config "${CLASS_CONFIG}" \
    --camera-indices "${SEMANTIC_CAMERA_INDICES}" \
    --view-count "${SEMANTIC_VIEW_COUNT}" \
    --scene "${SCENE}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --groundingdino-root "${GROUNDED_SAM_ROOT}" \
    --groundingdino-config "${GROUNDINGDINO_CONFIG}" \
    --groundingdino-checkpoint "${GROUNDINGDINO_CHECKPOINT}" \
    --segment-anything-root "${SAM_ROOT}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --render-max-width "${RENDER_MAX_WIDTH}" \
    --max-detections-per-view "${MAX_DETECTIONS_PER_VIEW}" \
    --mask-batch-size "${MASK_BATCH_SIZE}" \
    --flashsplat-support-threshold "${FLASHSPLAT_SUPPORT_THRESHOLD}" \
    --box-threshold "${BOX_THRESHOLD}" \
    --text-threshold "${TEXT_THRESHOLD}" \
    --min-assigned-gaussians "${MIN_ASSIGNED_GAUSSIANS}" \
    --min-assigned-thing-gaussians "${MIN_ASSIGNED_THING_GAUSSIANS}" \
    --min-assigned-stuff-gaussians "${MIN_ASSIGNED_STUFF_GAUSSIANS}" \
    --adaptive-stuff-min-view-ratio "${ADAPTIVE_STUFF_MIN_VIEW_RATIO}" \
    --adaptive-stuff-threshold-ratio "${ADAPTIVE_STUFF_THRESHOLD_RATIO}" \
    --adaptive-thing-threshold-floor-ratio "${ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO}" \
    --min-label-score "${MIN_LABEL_SCORE}" \
    "${BRANCH_REUSE_ARGS[@]}" \
    --overwrite

cp "${FUSION_DIR}/gaussian_labels.npy" "${DELIVERABLE_DIR}/gaussian_labels.npy"

run_report \
  "${VALIDATION_DIR}/semantic_point_cloud_inspection.json" \
  04_semantic_ply_inspection \
  python scripts/task1/inspect_ply.py "${DELIVERABLE_DIR}/semantic_point_cloud.ply" --json

run_stage 06_supersplat_debug_ply \
  python scripts/task1/export_supersplat_label_colors.py \
    --input-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-ply "${PLY_VIS_DIR}/semantic_point_cloud_supersplat_debug.ply" \
    --metadata-json "${PLY_VIS_DIR}/semantic_point_cloud_supersplat_debug.json" \
    --color-mode "${COLOR_MODE}" \
    --overwrite

if [ -n "${FOCUS_CLASSES}" ]; then
  run_stage 07_focus_debug_ply \
    python scripts/task1/export_supersplat_label_colors.py \
      --input-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
      --label-map "${DELIVERABLE_DIR}/label_map.json" \
      --output-ply "${PLY_VIS_DIR}/${FOCUS_NAME}_supersplat_debug.ply" \
      --metadata-json "${PLY_VIS_DIR}/${FOCUS_NAME}_supersplat_debug.json" \
      --focus-classes "${FOCUS_CLASSES}" \
      --color-mode "${COLOR_MODE}" \
      --overwrite
else
  echo "focus classes disabled" | tee "${RUN_LOG_DIR}/07_focus_debug_ply.log"
fi

run_report \
  "${VALIDATION_DIR}/semantic_point_cloud_supersplat_debug_inspection.json" \
  07_supersplat_debug_ply_inspection \
  python scripts/task1/inspect_ply.py "${PLY_VIS_DIR}/semantic_point_cloud_supersplat_debug.ply" --json

run_stage 08_semantic_overlays \
  python scripts/task1/render_auto_label_overlays.py \
    --model-path "${MODEL_DIR}" \
    --labels-npy "${FUSION_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-dir "${SEMANTIC_OVERLAY_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --camera-indices "${SEMANTIC_CAMERA_INDICES}" \
    --count "${SEMANTIC_VIEW_COUNT}" \
    --color-mode "${COLOR_MODE}"

if [ -n "${FOCUS_CLASSES}" ]; then
  run_stage 09_focus_overlays \
    python scripts/task1/render_auto_label_overlays.py \
      --model-path "${MODEL_DIR}" \
      --labels-npy "${FUSION_DIR}/gaussian_labels.npy" \
      --label-map "${DELIVERABLE_DIR}/label_map.json" \
      --output-dir "${FOCUS_OVERLAY_DIR}" \
      --flashsplat-root "${FLASHSPLAT_ROOT}" \
      --iteration 30000 \
      --max-width "${RENDER_MAX_WIDTH}" \
      --camera-indices "${SEMANTIC_CAMERA_INDICES}" \
      --count "${SEMANTIC_VIEW_COUNT}" \
      --focus-classes "${FOCUS_CLASSES}" \
      --color-mode "${COLOR_MODE}"
else
  echo "focus classes disabled" | tee "${RUN_LOG_DIR}/09_focus_overlays.log"
fi

run_stage 10_visible_coverage \
  python scripts/task1/measure_overlay_coverage.py \
    --rgb-dir "${GROUND_OUT_DIR}/rgb_renders" \
    --overlay-dir "${SEMANTIC_OVERLAY_DIR}" \
    --output "${VALIDATION_DIR}/visible_overlay_coverage.json" \
    --difference-threshold 2 \
    --exclude-border 1

run_stage 11_contact_sheets \
  python scripts/task1/make_contact_sheet.py \
    --image-dir "${GROUND_OUT_DIR}/overlays" \
    --output "${CONTACT_DIR}/grounded_sam.png" \
    --columns 5 \
    --thumb-width "${CONTACT_THUMB_WIDTH}"

run_stage 12_semantic_contact_sheet \
  python scripts/task1/make_contact_sheet.py \
    --image-dir "${SEMANTIC_OVERLAY_DIR}" \
    --output "${CONTACT_DIR}/semantic_labels.png" \
    --columns 5 \
    --thumb-width "${CONTACT_THUMB_WIDTH}"

if [ -n "${FOCUS_CLASSES}" ]; then
  run_stage 13_focus_contact_sheet \
    python scripts/task1/make_contact_sheet.py \
      --image-dir "${FOCUS_OVERLAY_DIR}" \
      --output "${CONTACT_DIR}/${FOCUS_NAME}.png" \
      --columns 5 \
      --thumb-width "${CONTACT_THUMB_WIDTH}"
else
  echo "focus classes disabled" | tee "${RUN_LOG_DIR}/13_focus_contact_sheet.log"
fi

run_stage 14_semantic_color_legend \
  python scripts/task1/write_semantic_color_legend.py \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --output-json "${VISUALIZATION_DIR}/semantic_color_legend.json" \
    --output-png "${VISUALIZATION_DIR}/semantic_color_legend.png" \
    --color-mode "${COLOR_MODE}" \
    --overwrite

run_report \
  "${VALIDATION_DIR}/task1_validation.json" \
  15_task1_validation \
  python scripts/task1/validate_task1_outputs.py \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --semantic-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --visible-coverage "${VALIDATION_DIR}/visible_overlay_coverage.json"

run_stage 16_pipeline_summary \
  python scripts/task1/summarize_task1_semantic_run.py \
    --output-dir "${OUT_DIR}" \
    --summary-path "${VALIDATION_DIR}/pipeline_run_summary.json" \
    --focus-name "${FOCUS_NAME}"

find "${OUT_DIR}" -maxdepth 4 -type f | sort
