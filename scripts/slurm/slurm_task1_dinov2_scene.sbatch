#!/usr/bin/env bash
#SBATCH --job-name=task1_dinov2_scene
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
EXTERNAL_ROOT="${WORKSPACE_ROOT}/external"
FLASHSPLAT_ROOT="${EXTERNAL_ROOT}/FlashSplat"
DINOV2_ROOT="${EXTERNAL_ROOT}/dinov2"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:-room}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov2_multiview_v1}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
VIEW_DIR="${OUT_DIR}/stages/01_real_camera_views"
VOTE_DIR="${OUT_DIR}/stages/02_flashsplat_votes"
FUSION_DIR="${OUT_DIR}/stages/03_exact_fusion"
GROUNDING_SELECTION_DIR="${OUT_DIR}/stages/04_grounding_camera_selection"
GROUNDING_GUARD_CONFIG="${GROUNDING_SELECTION_DIR}/grounding_guard_config.json"
GROUNDING_MASK_DIR="${OUT_DIR}/stages/05_grounded_sam"
GROUNDING_PROPOSAL_DIR="${OUT_DIR}/stages/06_grounding_flashsplat"
GROUNDING_FUSION_DIR="${OUT_DIR}/stages/07_grounding_fusion"
HYBRID_DIR="${OUT_DIR}/stages/08_hybrid_merge"
DELIVERABLE_DIR="${OUT_DIR}/deliverables"
VISUALIZATION_DIR="${OUT_DIR}/visualizations"
QA_RGB_DIR="${VISUALIZATION_DIR}/qa_rgb"
QA_OVERLAY_DIR="${VISUALIZATION_DIR}/semantic_overlays"
CONTACT_DIR="${VISUALIZATION_DIR}/contact_sheets"
FOCUS_OVERLAY_DIR="${VISUALIZATION_DIR}/grounding_extension_overlays"
VALIDATION_DIR="${OUT_DIR}/validation"
RUN_LOG_DIR="${OUT_DIR}/logs"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
DINOV2_ENV="${DINOV2_ENV:-dinov2_segmentation}"
GROUNDING_ENV="${GROUNDING_ENV:-${GAUSSIAN_ENV}}"
DINOV2_CHECKPOINT_DIR="${DINOV2_CHECKPOINT_DIR:-${DINOV2_ROOT}/checkpoints}"
DINOV2_BACKBONE="${DINOV2_BACKBONE:-${DINOV2_CHECKPOINT_DIR}/dinov2_vitl14_pretrain.pth}"
DINOV2_HEAD_CONFIG="${DINOV2_HEAD_CONFIG:-${DINOV2_CHECKPOINT_DIR}/dinov2_vitl14_ade20k_linear_config.py}"
DINOV2_HEAD_CHECKPOINT="${DINOV2_HEAD_CHECKPOINT:-${DINOV2_CHECKPOINT_DIR}/dinov2_vitl14_ade20k_linear_head.pth}"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"

ENABLE_GROUNDINGDINO="${ENABLE_GROUNDINGDINO:-0}"
COLOR_MODE="${COLOR_MODE:-class}"
GROUNDING_CLASS_CONFIG="${GROUNDING_CLASS_CONFIG:-${PROJECT_ROOT}/configs/task1_semantic_classes.${SCENE}.json}"
GROUNDING_EXTENSION_CONFIG="${GROUNDING_EXTENSION_CONFIG:-${PROJECT_ROOT}/configs/task1_hybrid_extensions.${SCENE}.json}"
GROUNDING_EXTENSION_CLASSES="${GROUNDING_EXTENSION_CLASSES:-}"
GROUNDING_GUARD_MAX_PROMPT_WORDS="${GROUNDING_GUARD_MAX_PROMPT_WORDS:-180}"
GROUNDING_BASE_VIEW_COUNT="${GROUNDING_BASE_VIEW_COUNT:-50}"
GROUNDING_TARGET_VIEW_COUNT="${GROUNDING_TARGET_VIEW_COUNT:-20}"
GROUNDING_TARGET_CANDIDATE_COUNT="${GROUNDING_TARGET_CANDIDATE_COUNT:-100}"
GROUNDING_PROJECTION_SAMPLE_COUNT="${GROUNDING_PROJECTION_SAMPLE_COUNT:-50000}"
GROUNDING_PROJECTION_MARGIN="${GROUNDING_PROJECTION_MARGIN:-1.5}"
GROUNDING_LOW_COVERAGE_QUANTILE="${GROUNDING_LOW_COVERAGE_QUANTILE:-0.35}"
GROUNDING_MAX_DETECTIONS_PER_VIEW="${GROUNDING_MAX_DETECTIONS_PER_VIEW:-32}"
GROUNDING_MASK_BATCH_SIZE="${GROUNDING_MASK_BATCH_SIZE:-4}"
GROUNDING_FLASHSPLAT_SUPPORT_THRESHOLD="${GROUNDING_FLASHSPLAT_SUPPORT_THRESHOLD:-0.05}"
GROUNDING_MIN_ASSIGNED_GAUSSIANS="${GROUNDING_MIN_ASSIGNED_GAUSSIANS:-0}"
GROUNDING_MIN_ASSIGNED_THING_GAUSSIANS="${GROUNDING_MIN_ASSIGNED_THING_GAUSSIANS:-5000}"
GROUNDING_MIN_ASSIGNED_STUFF_GAUSSIANS="${GROUNDING_MIN_ASSIGNED_STUFF_GAUSSIANS:-10000}"
GROUNDING_ADAPTIVE_STUFF_MIN_VIEW_RATIO="${GROUNDING_ADAPTIVE_STUFF_MIN_VIEW_RATIO:-0.50}"
GROUNDING_ADAPTIVE_STUFF_THRESHOLD_RATIO="${GROUNDING_ADAPTIVE_STUFF_THRESHOLD_RATIO:-0.75}"
GROUNDING_ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO="${GROUNDING_ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO:-0.50}"
GROUNDING_MIN_LABEL_SCORE="${GROUNDING_MIN_LABEL_SCORE:-0.0}"
GROUNDING_BOX_THRESHOLD="${GROUNDING_BOX_THRESHOLD:-0.30}"
GROUNDING_TEXT_THRESHOLD="${GROUNDING_TEXT_THRESHOLD:-0.25}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-${WORKSPACE_ROOT}/InvRGBL_modif/pretrained/sam_vit_h_4b8939.pth}"
GROUNDED_SAM_ROOT="${GROUNDED_SAM_ROOT:-${WORKSPACE_ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/Grounded-Segment-Anything}"
GROUNDINGDINO_CONFIG="${GROUNDINGDINO_CONFIG:-${WORKSPACE_ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/saves/GroundingDINO_SwinT_OGC.py}"
GROUNDINGDINO_CHECKPOINT="${GROUNDINGDINO_CHECKPOINT:-${WORKSPACE_ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/saves/groundingdino_swint_ogc.pth}"
SAM_ROOT="${SAM_ROOT:-${GROUNDED_SAM_ROOT}/segment_anything}"

CAMERA_SOURCE="${CAMERA_SOURCE:-real}"
DINOV2_HEAD="${DINOV2_HEAD:-linear}"
CAMERA_INDICES="${CAMERA_INDICES:-}"
VIEW_COUNT="${VIEW_COUNT:-0}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
QA_VIEW_COUNT="${QA_VIEW_COUNT:-12}"
MIN_PIXEL_CONFIDENCE="${MIN_PIXEL_CONFIDENCE:-0.50}"
FLASHSPLAT_SUPPORT_THRESHOLD="${FLASHSPLAT_SUPPORT_THRESHOLD:-0.05}"
MIN_VIEWS="${MIN_VIEWS:-2}"
MIN_AGREEMENT="${MIN_AGREEMENT:-0.50}"
MIN_ASSIGNED_GAUSSIANS="${MIN_ASSIGNED_GAUSSIANS:-0}"
MIN_ASSIGNED_THING_GAUSSIANS="${MIN_ASSIGNED_THING_GAUSSIANS:-5000}"
MIN_ASSIGNED_STUFF_GAUSSIANS="${MIN_ASSIGNED_STUFF_GAUSSIANS:-10000}"
ADAPTIVE_STUFF_MIN_VIEW_RATIO="${ADAPTIVE_STUFF_MIN_VIEW_RATIO:-0.50}"
ADAPTIVE_STUFF_THRESHOLD_RATIO="${ADAPTIVE_STUFF_THRESHOLD_RATIO:-0.75}"
ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO="${ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO:-0.50}"
RESET_OUTPUT="${RESET_OUTPUT:-1}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"
BASELINE_LABELS="${BASELINE_LABELS:-}"

if [[ "${CAMERA_SOURCE}" != "real" ]]; then
  echo "v1 supports only real cameras from cameras.json; orbit views are deferred" >&2
  exit 2
fi
if [[ "${DINOV2_HEAD}" != "linear" ]]; then
  echo "v1 supports only the official ADE20K linear head" >&2
  exit 2
fi
if [[ "${ENABLE_GROUNDINGDINO}" != "0" && "${ENABLE_GROUNDINGDINO}" != "1" ]]; then
  echo "ENABLE_GROUNDINGDINO must be 0 or 1" >&2
  exit 2
fi
if [[ "${COLOR_MODE}" != "class" && "${COLOR_MODE}" != "instance" ]]; then
  echo "COLOR_MODE must be class or instance" >&2
  exit 2
fi

test -d "${MODEL_DIR}"
test -f "${MODEL_DIR}/cameras.json"
test -d "${FLASHSPLAT_ROOT}"
test -d "${DINOV2_ROOT}"
test -f "${DINOV2_BACKBONE}"
test -f "${DINOV2_HEAD_CONFIG}"
test -f "${DINOV2_HEAD_CHECKPOINT}"
test -f "${ONTOLOGY}"
if [[ -n "${BASELINE_LABELS}" ]]; then
  test -f "${BASELINE_LABELS}"
fi

GROUNDING_EXTENSION_CLASSES_RESOLVED=""
if [[ "${ENABLE_GROUNDINGDINO}" = "1" ]]; then
  if ! [[ "${GROUNDING_GUARD_MAX_PROMPT_WORDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GROUNDING_GUARD_MAX_PROMPT_WORDS must be a positive integer" >&2
    exit 2
  fi
  test -f "${GROUNDING_CLASS_CONFIG}"
  test -f "${GROUNDING_EXTENSION_CONFIG}"
  test -d "${GROUNDED_SAM_ROOT}"
  test -f "${GROUNDINGDINO_CONFIG}"
  test -f "${GROUNDINGDINO_CHECKPOINT}"
  test -d "${SAM_ROOT}"
  test -f "${SAM_CHECKPOINT}"
  command -v jq >/dev/null 2>&1
  if [[ -n "${GROUNDING_EXTENSION_CLASSES}" ]]; then
    GROUNDING_EXTENSION_CLASSES_RESOLVED="${GROUNDING_EXTENSION_CLASSES}"
  else
    GROUNDING_EXTENSION_CLASSES_RESOLVED="$(jq -r '.default_enabled_classes // [] | join(",")' "${GROUNDING_EXTENSION_CONFIG}")"
  fi
  if [[ -z "${GROUNDING_EXTENSION_CLASSES_RESOLVED}" ]]; then
    echo "GroundingDINO is enabled but no extension classes are selected." >&2
    echo "Set GROUNDING_EXTENSION_CLASSES during QA or populate default_enabled_classes." >&2
    exit 2
  fi
  IFS=',' read -r -a GROUNDING_EXTENSION_CLASS_ITEMS <<< "${GROUNDING_EXTENSION_CLASSES_RESOLVED}"
  for extension_class in "${GROUNDING_EXTENSION_CLASS_ITEMS[@]}"; do
    extension_class="${extension_class// /_}"
    if ! jq -e --arg class "${extension_class}" \
      '.candidate_classes | index($class) != null' "${GROUNDING_EXTENSION_CONFIG}" >/dev/null; then
      echo "Extension class is not a configured candidate: ${extension_class}" >&2
      exit 2
    fi
    if ! jq -e --arg class "${extension_class}" \
      '[.classes[].class] | index($class) != null' "${GROUNDING_CLASS_CONFIG}" >/dev/null; then
      echo "Extension class has no GroundingDINO prompts: ${extension_class}" >&2
      exit 2
    fi
    if ! jq -e --arg class "${extension_class}" \
      '[.classes[].project_class] | index($class) == null' "${ONTOLOGY}" >/dev/null; then
      echo "Extension class collides with the DINOv2 ontology: ${extension_class}" >&2
      exit 2
    fi
  done
fi

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nmodel=%s\noutput=%s\ncamera_source=%s\nview_count=%s\ndinov2_head=%s\nenable_groundingdino=%s\ngrounding_extension_classes=%s\ngrounding_guard_max_prompt_words=%s\ncolor_mode=%s\n' \
    "${SCENE}" "${MODEL_DIR}" "${OUT_DIR}" "${CAMERA_SOURCE}" "${VIEW_COUNT}" \
    "${DINOV2_HEAD}" "${ENABLE_GROUNDINGDINO}" \
    "${GROUNDING_EXTENSION_CLASSES_RESOLVED:-<disabled>}" \
    "${GROUNDING_GUARD_MAX_PROMPT_WORDS}" "${COLOR_MODE}"
  exit 0
fi

if [[ "${RESET_OUTPUT}" = "1" && -d "${OUT_DIR}" ]]; then
  case "${OUT_DIR}" in
    "${TASK1_ROOT}"/*) rm -rf -- "${OUT_DIR}" ;;
    *) echo "Refusing to reset unexpected output path: ${OUT_DIR}" >&2; exit 2 ;;
  esac
fi

mkdir -p \
  "${VIEW_DIR}" "${VOTE_DIR}" "${FUSION_DIR}" "${DELIVERABLE_DIR}" \
  "${GROUNDING_SELECTION_DIR}" "${GROUNDING_MASK_DIR}" \
  "${GROUNDING_PROPOSAL_DIR}" "${GROUNDING_FUSION_DIR}" "${HYBRID_DIR}" \
  "${QA_RGB_DIR}" "${QA_OVERLAY_DIR}" "${FOCUS_OVERLAY_DIR}" "${CONTACT_DIR}" \
  "${VALIDATION_DIR}" "${RUN_LOG_DIR}" "${TASK1_ROOT}/logs"

cd "${PROJECT_ROOT}"

run_stage() {
  local stage_name="$1"
  local environment="$2"
  shift 2
  echo "===== ${stage_name} ====="
  conda run --no-capture-output -n "${environment}" "$@" 2>&1 | tee "${RUN_LOG_DIR}/${stage_name}.log"
}

run_report() {
  local stage_name="$1"
  local environment="$2"
  local report_path="$3"
  shift 3
  echo "===== ${stage_name} ====="
  conda run --no-capture-output -n "${environment}" "$@" \
    | tee "${report_path}" "${RUN_LOG_DIR}/${stage_name}.log"
}

{
  echo "job=${SLURM_JOB_ID:-local}"
  echo "host=$(hostname)"
  echo "scene=${SCENE}"
  echo "model=${MODEL_DIR}"
  echo "output=${OUT_DIR}"
  echo "camera_source=${CAMERA_SOURCE}"
  echo "camera_indices=${CAMERA_INDICES:-<all>}"
  echo "view_count=${VIEW_COUNT}"
  echo "render_max_width=${RENDER_MAX_WIDTH}"
  echo "dinov2_head=${DINOV2_HEAD}"
  echo "enable_groundingdino=${ENABLE_GROUNDINGDINO}"
  echo "grounding_extension_classes=${GROUNDING_EXTENSION_CLASSES_RESOLVED:-<disabled>}"
  echo "grounding_guard_max_prompt_words=${GROUNDING_GUARD_MAX_PROMPT_WORDS}"
  echo "grounding_base_view_count=${GROUNDING_BASE_VIEW_COUNT}"
  echo "grounding_target_view_count=${GROUNDING_TARGET_VIEW_COUNT}"
  echo "color_mode=${COLOR_MODE}"
  echo "min_pixel_confidence=${MIN_PIXEL_CONFIDENCE}"
  echo "flashsplat_support_threshold=${FLASHSPLAT_SUPPORT_THRESHOLD}"
  echo "min_views=${MIN_VIEWS}"
  echo "min_agreement=${MIN_AGREEMENT}"
  echo "baseline_labels=${BASELINE_LABELS:-<not_provided>}"
  nvidia-smi
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

run_stage 01_render_real_camera_views "${GAUSSIAN_ENV}" \
  python scripts/task1/render_task1_views.py \
    --model-path "${MODEL_DIR}" \
    --output-dir "${VIEW_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --camera-indices "${CAMERA_INDICES}" \
    --count "${VIEW_COUNT}" \
    --overwrite

run_stage 02_dinov2_ade20k_linear "${DINOV2_ENV}" \
  python scripts/task1/dinov2_segment_views.py \
    --input-dir "${VIEW_DIR}" \
    --dinov2-root "${DINOV2_ROOT}" \
    --backbone-checkpoint "${DINOV2_BACKBONE}" \
    --head-config "${DINOV2_HEAD_CONFIG}" \
    --head-checkpoint "${DINOV2_HEAD_CHECKPOINT}" \
    --ontology "${ONTOLOGY}" \
    --min-pixel-confidence "${MIN_PIXEL_CONFIDENCE}" \
    --device cuda \
    --overwrite

run_stage 03_lift_per_view_votes "${GAUSSIAN_ENV}" \
  python scripts/task1/lift_dinov2_view_votes.py \
    --model-path "${MODEL_DIR}" \
    --input-dir "${VIEW_DIR}" \
    --output-dir "${VOTE_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --ontology "${ONTOLOGY}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --min-pixel-confidence "${MIN_PIXEL_CONFIDENCE}" \
    --support-threshold "${FLASHSPLAT_SUPPORT_THRESHOLD}" \
    --overwrite

FUSION_BASELINE_ARGS=()
if [[ -n "${BASELINE_LABELS}" ]]; then
  FUSION_BASELINE_ARGS=(--baseline-labels "${BASELINE_LABELS}")
fi

run_stage 04_exact_multiview_fusion "${GAUSSIAN_ENV}" \
  python scripts/task1/fuse_dinov2_multiview_votes.py \
    --model-path "${MODEL_DIR}" \
    --vote-dir "${VOTE_DIR}" \
    --output-dir "${FUSION_DIR}" \
    --ontology "${ONTOLOGY}" \
    --iteration 30000 \
    --scene "${SCENE}" \
    --min-views "${MIN_VIEWS}" \
    --min-agreement "${MIN_AGREEMENT}" \
    --voxel-scale-multiplier 4 \
    --min-voxel-size 0.01 \
    --max-voxel-size 0.20 \
    --min-component-gaussians 500 \
    --min-component-ratio 0.01 \
    --min-assigned-gaussians "${MIN_ASSIGNED_GAUSSIANS}" \
    --min-assigned-thing-gaussians "${MIN_ASSIGNED_THING_GAUSSIANS}" \
    --min-assigned-stuff-gaussians "${MIN_ASSIGNED_STUFF_GAUSSIANS}" \
    --adaptive-stuff-min-view-ratio "${ADAPTIVE_STUFF_MIN_VIEW_RATIO}" \
    --adaptive-stuff-threshold-ratio "${ADAPTIVE_STUFF_THRESHOLD_RATIO}" \
    --adaptive-thing-threshold-floor-ratio "${ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO}" \
    "${FUSION_BASELINE_ARGS[@]}" \
    --overwrite

FINAL_SOURCE_DIR="${FUSION_DIR}"
FINAL_MODE="dinov2"
MERGED_GROUNDING_GROUP_COUNT=0

if [[ "${ENABLE_GROUNDINGDINO}" = "1" ]]; then
  run_stage 05a_grounding_guard_config "${GAUSSIAN_ENV}" \
    python scripts/task1/build_grounding_guard_config.py \
      --dinov2-label-map "${FUSION_DIR}/label_map.json" \
      --scene-class-config "${GROUNDING_CLASS_CONFIG}" \
      --extension-config "${GROUNDING_EXTENSION_CONFIG}" \
      --include-classes "${GROUNDING_EXTENSION_CLASSES_RESOLVED}" \
      --output "${GROUNDING_GUARD_CONFIG}" \
      --max-prompt-words "${GROUNDING_GUARD_MAX_PROMPT_WORDS}" \
      --overwrite

  run_stage 05b_grounding_seed_views "${GAUSSIAN_ENV}" \
    python scripts/task1/select_targeted_cameras.py seed \
      --source-manifest "${VIEW_DIR}/view_manifest.json" \
      --output "${GROUNDING_SELECTION_DIR}/seed_manifest.json" \
      --indices-output "${GROUNDING_SELECTION_DIR}/seed_indices.txt" \
      --count "${GROUNDING_BASE_VIEW_COUNT}"

  run_stage 05c_grounding_candidate_screen "${GAUSSIAN_ENV}" \
    python scripts/task1/select_targeted_cameras.py screen \
      --model-path "${MODEL_DIR}" \
      --baseline-manifest "${GROUNDING_SELECTION_DIR}/seed_manifest.json" \
      --allowed-manifest "${VIEW_DIR}/view_manifest.json" \
      --output "${GROUNDING_SELECTION_DIR}/candidate_screen.json" \
      --indices-output "${GROUNDING_SELECTION_DIR}/candidate_indices.txt" \
      --iteration 30000 \
      --sample-count "${GROUNDING_PROJECTION_SAMPLE_COUNT}" \
      --candidate-count "${GROUNDING_TARGET_CANDIDATE_COUNT}" \
      --projection-margin "${GROUNDING_PROJECTION_MARGIN}"

  AVAILABLE_GROUNDING_CANDIDATES="$(jq -r '.selected_candidate_count' "${GROUNDING_SELECTION_DIR}/candidate_screen.json")"
  REQUESTED_GROUNDING_ADDITIONS="${GROUNDING_TARGET_VIEW_COUNT}"
  if (( REQUESTED_GROUNDING_ADDITIONS > AVAILABLE_GROUNDING_CANDIDATES )); then
    REQUESTED_GROUNDING_ADDITIONS="${AVAILABLE_GROUNDING_CANDIDATES}"
  fi

  if (( REQUESTED_GROUNDING_ADDITIONS > 0 )); then
    CANDIDATE_CAMERA_INDICES="$(tr -d '\r\n' < "${GROUNDING_SELECTION_DIR}/candidate_indices.txt")"
    test -n "${CANDIDATE_CAMERA_INDICES}"

    run_stage 05d_grounding_candidate_overlays "${GAUSSIAN_ENV}" \
      python scripts/task1/render_auto_label_overlays.py \
        --model-path "${MODEL_DIR}" \
        --labels-npy "${FUSION_DIR}/gaussian_labels.npy" \
        --label-map "${FUSION_DIR}/label_map.json" \
        --output-dir "${GROUNDING_SELECTION_DIR}/candidate_overlays" \
        --rgb-output-dir "${GROUNDING_SELECTION_DIR}/candidate_rgb" \
        --source-view-manifest "${VIEW_DIR}/view_manifest.json" \
        --flashsplat-root "${FLASHSPLAT_ROOT}" \
        --iteration 30000 \
        --max-width "${RENDER_MAX_WIDTH}" \
        --camera-indices "${CANDIDATE_CAMERA_INDICES}" \
        --count "${AVAILABLE_GROUNDING_CANDIDATES}" \
        --max-labels 0 \
        --color-mode "${COLOR_MODE}"

    run_stage 05e_grounding_candidate_coverage "${GAUSSIAN_ENV}" \
      python scripts/task1/measure_overlay_coverage.py \
        --rgb-dir "${GROUNDING_SELECTION_DIR}/candidate_rgb" \
        --overlay-dir "${GROUNDING_SELECTION_DIR}/candidate_overlays" \
        --output "${GROUNDING_SELECTION_DIR}/candidate_visible_coverage.json" \
        --difference-threshold 2 \
        --exclude-border 1

    run_stage 05f_grounding_target_selection "${GAUSSIAN_ENV}" \
      python scripts/task1/select_targeted_cameras.py select \
        --model-path "${MODEL_DIR}" \
        --baseline-manifest "${GROUNDING_SELECTION_DIR}/seed_manifest.json" \
        --candidate-manifest "${GROUNDING_SELECTION_DIR}/candidate_overlays/label_overlay_manifest.json" \
        --coverage-report "${GROUNDING_SELECTION_DIR}/candidate_visible_coverage.json" \
        --output "${GROUNDING_SELECTION_DIR}/targeted_camera_selection.json" \
        --indices-output "${GROUNDING_SELECTION_DIR}/final_camera_indices.txt" \
        --additional-count "${REQUESTED_GROUNDING_ADDITIONS}" \
        --low-coverage-quantile "${GROUNDING_LOW_COVERAGE_QUANTILE}" \
        --coverage-weight 0.8 \
        --novelty-weight 0.2
  else
    cp "${GROUNDING_SELECTION_DIR}/seed_indices.txt" "${GROUNDING_SELECTION_DIR}/final_camera_indices.txt"
  fi

  GROUNDING_CAMERA_INDICES="$(tr -d '\r\n' < "${GROUNDING_SELECTION_DIR}/final_camera_indices.txt")"
  GROUNDING_VIEW_COUNT="$(awk -F, '{print NF}' "${GROUNDING_SELECTION_DIR}/final_camera_indices.txt")"
  test -n "${GROUNDING_CAMERA_INDICES}"

  run_stage 06_08_grounding_extension_branch "${GROUNDING_ENV}" \
    python scripts/task1/run_grounding_semantic_branch.py \
      --model-path "${MODEL_DIR}" \
      --ground-output-dir "${GROUNDING_MASK_DIR}" \
      --proposal-dir "${GROUNDING_PROPOSAL_DIR}" \
      --fusion-dir "${GROUNDING_FUSION_DIR}" \
      --label-map-path "${GROUNDING_FUSION_DIR}/label_map.json" \
      --semantic-ply-path "${GROUNDING_FUSION_DIR}/semantic_point_cloud.ply" \
      --log-dir "${RUN_LOG_DIR}" \
      --class-config "${GROUNDING_GUARD_CONFIG}" \
      --source-view-manifest "${VIEW_DIR}/view_manifest.json" \
      --camera-indices "${GROUNDING_CAMERA_INDICES}" \
      --view-count "${GROUNDING_VIEW_COUNT}" \
      --scene "${SCENE}" \
      --flashsplat-root "${FLASHSPLAT_ROOT}" \
      --groundingdino-root "${GROUNDED_SAM_ROOT}" \
      --groundingdino-config "${GROUNDINGDINO_CONFIG}" \
      --groundingdino-checkpoint "${GROUNDINGDINO_CHECKPOINT}" \
      --segment-anything-root "${SAM_ROOT}" \
      --sam-checkpoint "${SAM_CHECKPOINT}" \
      --render-max-width "${RENDER_MAX_WIDTH}" \
      --max-detections-per-view "${GROUNDING_MAX_DETECTIONS_PER_VIEW}" \
      --mask-batch-size "${GROUNDING_MASK_BATCH_SIZE}" \
      --flashsplat-support-threshold "${GROUNDING_FLASHSPLAT_SUPPORT_THRESHOLD}" \
      --box-threshold "${GROUNDING_BOX_THRESHOLD}" \
      --text-threshold "${GROUNDING_TEXT_THRESHOLD}" \
      --min-assigned-gaussians "${GROUNDING_MIN_ASSIGNED_GAUSSIANS}" \
      --min-assigned-thing-gaussians "${GROUNDING_MIN_ASSIGNED_THING_GAUSSIANS}" \
      --min-assigned-stuff-gaussians "${GROUNDING_MIN_ASSIGNED_STUFF_GAUSSIANS}" \
      --adaptive-stuff-min-view-ratio "${GROUNDING_ADAPTIVE_STUFF_MIN_VIEW_RATIO}" \
      --adaptive-stuff-threshold-ratio "${GROUNDING_ADAPTIVE_STUFF_THRESHOLD_RATIO}" \
      --adaptive-thing-threshold-floor-ratio "${GROUNDING_ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO}" \
      --min-label-score "${GROUNDING_MIN_LABEL_SCORE}" \
      --overwrite

  run_stage 09_hybrid_merge "${GAUSSIAN_ENV}" \
    python scripts/task1/merge_semantic_extensions.py \
      --base-labels "${FUSION_DIR}/gaussian_labels.npy" \
      --base-label-map "${FUSION_DIR}/label_map.json" \
      --extension-labels "${GROUNDING_FUSION_DIR}/gaussian_labels.npy" \
      --extension-label-map "${GROUNDING_FUSION_DIR}/label_map.json" \
      --extension-manifest "${GROUNDING_MASK_DIR}/grounded_sam_manifest.json" \
      --extension-config "${GROUNDING_EXTENSION_CONFIG}" \
      --ontology "${ONTOLOGY}" \
      --source-ply "${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply" \
      --output-dir "${HYBRID_DIR}" \
      --scene "${SCENE}" \
      --include-classes "${GROUNDING_EXTENSION_CLASSES_RESOLVED}" \
      --overwrite

  FINAL_SOURCE_DIR="${HYBRID_DIR}"
  FINAL_MODE="dinov2_groundingdino_hybrid"
  MERGED_GROUNDING_GROUP_COUNT="$(jq -r '.merge.merged_group_count' "${HYBRID_DIR}/hybrid_merge_summary.json")"
fi

cp "${FINAL_SOURCE_DIR}/gaussian_labels.npy" "${DELIVERABLE_DIR}/gaussian_labels.npy"
cp "${FINAL_SOURCE_DIR}/semantic_point_cloud.ply" "${DELIVERABLE_DIR}/semantic_point_cloud.ply"
cp "${FINAL_SOURCE_DIR}/label_map.json" "${DELIVERABLE_DIR}/label_map.json"

run_stage 10_supersplat_debug_ply "${GAUSSIAN_ENV}" \
  python scripts/task1/export_supersplat_label_colors.py \
    --input-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-ply "${VISUALIZATION_DIR}/semantic_point_cloud_supersplat_debug.ply" \
    --metadata-json "${VISUALIZATION_DIR}/semantic_point_cloud_supersplat_debug.json" \
    --color-mode "${COLOR_MODE}" \
    --overwrite

run_stage 11_semantic_overlays "${GAUSSIAN_ENV}" \
  python scripts/task1/render_auto_label_overlays.py \
    --model-path "${MODEL_DIR}" \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-dir "${QA_OVERLAY_DIR}" \
    --rgb-output-dir "${QA_RGB_DIR}" \
    --source-view-manifest "${VIEW_DIR}/view_manifest.json" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --max-labels 0 \
    --count "${QA_VIEW_COUNT}" \
    --color-mode "${COLOR_MODE}"

if [[ "${ENABLE_GROUNDINGDINO}" = "1" && "${MERGED_GROUNDING_GROUP_COUNT}" -gt 0 ]]; then
  run_stage 12_grounding_extension_overlays "${GAUSSIAN_ENV}" \
    python scripts/task1/render_auto_label_overlays.py \
      --model-path "${MODEL_DIR}" \
      --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
      --label-map "${DELIVERABLE_DIR}/label_map.json" \
      --output-dir "${FOCUS_OVERLAY_DIR}" \
      --source-view-manifest "${VIEW_DIR}/view_manifest.json" \
      --flashsplat-root "${FLASHSPLAT_ROOT}" \
      --iteration 30000 \
      --max-width "${RENDER_MAX_WIDTH}" \
      --max-labels 0 \
      --count "${QA_VIEW_COUNT}" \
      --focus-classes "${GROUNDING_EXTENSION_CLASSES_RESOLVED}" \
      --color-mode "${COLOR_MODE}"
fi

run_stage 13_visible_coverage "${GAUSSIAN_ENV}" \
  python scripts/task1/measure_overlay_coverage.py \
    --rgb-dir "${QA_RGB_DIR}" \
    --overlay-dir "${QA_OVERLAY_DIR}" \
    --output "${VALIDATION_DIR}/visible_overlay_coverage.json" \
    --difference-threshold 2 \
    --exclude-border 1

run_stage 14_contact_sheets "${GAUSSIAN_ENV}" \
  python scripts/task1/make_contact_sheet.py \
    --image-dir "${QA_OVERLAY_DIR}" \
    --output "${CONTACT_DIR}/semantic_labels.png" \
    --columns 4 \
    --thumb-width 360

if [[ "${ENABLE_GROUNDINGDINO}" = "1" && "${MERGED_GROUNDING_GROUP_COUNT}" -gt 0 ]]; then
  run_stage 15_grounding_extension_contact_sheet "${GAUSSIAN_ENV}" \
    python scripts/task1/make_contact_sheet.py \
      --image-dir "${FOCUS_OVERLAY_DIR}" \
      --output "${CONTACT_DIR}/grounding_extensions.png" \
      --columns 4 \
      --thumb-width 360
fi

run_stage 16_semantic_color_legend "${GAUSSIAN_ENV}" \
  python scripts/task1/write_semantic_color_legend.py \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --output-json "${VISUALIZATION_DIR}/semantic_color_legend.json" \
    --output-png "${VISUALIZATION_DIR}/semantic_color_legend.png" \
    --color-mode "${COLOR_MODE}" \
    --overwrite

run_report 17_task1_validation "${GAUSSIAN_ENV}" "${VALIDATION_DIR}/task1_validation.json" \
  python scripts/task1/validate_task1_outputs.py \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --semantic-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --visible-coverage "${VALIDATION_DIR}/visible_overlay_coverage.json"

printf 'final_mode=%s\nfinal_source=%s\nmerged_grounding_groups=%s\n' \
  "${FINAL_MODE}" "${FINAL_SOURCE_DIR}" "${MERGED_GROUNDING_GROUP_COUNT}" \
  | tee "${VALIDATION_DIR}/final_mode.txt"

find "${OUT_DIR}" -maxdepth 4 -type f | sort
