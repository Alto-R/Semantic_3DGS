#!/usr/bin/env bash
#SBATCH --job-name=task1_ade_refine
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

PROJECT_ROOT="$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"
FLASHSPLAT_ROOT="${WORKSPACE_ROOT}/external/FlashSplat"

SCENE="${SCENE:-room}"
SOURCE_NAME="${SOURCE_NAME:-${SCENE}_dinov2_separate_abstain_allviews_v1}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov2_ade_refinement_auto_adaptive_instance_guard_v5}"
SOURCE_DIR="${TASK1_ROOT}/${SOURCE_NAME}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
SOURCE_VIEW_DIR="${SOURCE_DIR}/stages/01_real_camera_views"
BASE_LABELS="${SOURCE_DIR}/deliverables/gaussian_labels.npy"
BASE_LABEL_MAP="${SOURCE_DIR}/deliverables/label_map.json"
SELECTION_DIR="${OUT_DIR}/stages/04_grounding_camera_selection"
REFINEMENT_CONFIG="${SELECTION_DIR}/automatic_ade_refinement_config.json"
GROUNDING_MASK_DIR="${OUT_DIR}/stages/05_grounded_sam"
GROUNDING_PROPOSAL_DIR="${OUT_DIR}/stages/06_grounding_flashsplat"
GROUNDING_FUSION_DIR="${OUT_DIR}/stages/07_grounding_fusion"
REFINEMENT_DIR="${OUT_DIR}/stages/08_ade_refinement"
DELIVERABLE_DIR="${OUT_DIR}/deliverables"
VISUALIZATION_DIR="${OUT_DIR}/visualizations"
QA_RGB_DIR="${VISUALIZATION_DIR}/qa_rgb"
QA_OVERLAY_DIR="${VISUALIZATION_DIR}/semantic_overlays"
CHANGE_OVERLAY_DIR="${VISUALIZATION_DIR}/ade_refinement_change_overlays"
CONTACT_DIR="${VISUALIZATION_DIR}/contact_sheets"
VALIDATION_DIR="${OUT_DIR}/validation"
RUN_LOG_DIR="${OUT_DIR}/logs"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
GROUNDING_ENV="${GROUNDING_ENV:-${GAUSSIAN_ENV}}"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
ADE_PROMPT_ALIASES="${ADE_PROMPT_ALIASES:-${PROJECT_ROOT}/configs/ade20k_grounding_prompts.json}"
GROUNDED_SAM_ROOT="${GROUNDED_SAM_ROOT:-${WORKSPACE_ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/Grounded-Segment-Anything}"
GROUNDINGDINO_CONFIG="${GROUNDINGDINO_CONFIG:-${WORKSPACE_ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/saves/GroundingDINO_SwinT_OGC.py}"
GROUNDINGDINO_CHECKPOINT="${GROUNDINGDINO_CHECKPOINT:-${WORKSPACE_ROOT}/gaussian-grouping/Tracking-Anything-with-DEVA/saves/groundingdino_swint_ogc.pth}"
SAM_ROOT="${SAM_ROOT:-${GROUNDED_SAM_ROOT}/segment_anything}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-${WORKSPACE_ROOT}/InvRGBL_modif/pretrained/sam_vit_h_4b8939.pth}"

RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
QA_VIEW_COUNT="${QA_VIEW_COUNT:-12}"
COLOR_MODE="${COLOR_MODE:-class}"
GROUNDING_BASE_VIEW_COUNT="${GROUNDING_BASE_VIEW_COUNT:-50}"
GROUNDING_TARGET_VIEW_COUNT="${GROUNDING_TARGET_VIEW_COUNT:-20}"
GROUNDING_TARGET_CANDIDATE_COUNT="${GROUNDING_TARGET_CANDIDATE_COUNT:-100}"
GROUNDING_PROJECTION_SAMPLE_COUNT="${GROUNDING_PROJECTION_SAMPLE_COUNT:-50000}"
GROUNDING_PROJECTION_MARGIN="${GROUNDING_PROJECTION_MARGIN:-1.5}"
GROUNDING_LOW_COVERAGE_QUANTILE="${GROUNDING_LOW_COVERAGE_QUANTILE:-0.35}"
GROUNDING_MAX_DETECTIONS_PER_VIEW="${GROUNDING_MAX_DETECTIONS_PER_VIEW:-32}"
GROUNDING_MASK_BATCH_SIZE="${GROUNDING_MASK_BATCH_SIZE:-4}"
GROUNDING_FLASHSPLAT_SUPPORT_THRESHOLD="${GROUNDING_FLASHSPLAT_SUPPORT_THRESHOLD:-0.05}"
GROUNDING_BOX_THRESHOLD="${GROUNDING_BOX_THRESHOLD:-0.30}"
GROUNDING_TEXT_THRESHOLD="${GROUNDING_TEXT_THRESHOLD:-0.25}"
GROUNDING_MIN_ASSIGNED_GAUSSIANS="${GROUNDING_MIN_ASSIGNED_GAUSSIANS:-0}"
GROUNDING_MIN_ASSIGNED_THING_GAUSSIANS="${GROUNDING_MIN_ASSIGNED_THING_GAUSSIANS:-5000}"
GROUNDING_MIN_ASSIGNED_STUFF_GAUSSIANS="${GROUNDING_MIN_ASSIGNED_STUFF_GAUSSIANS:-10000}"
GROUNDING_ADAPTIVE_STUFF_MIN_VIEW_RATIO="${GROUNDING_ADAPTIVE_STUFF_MIN_VIEW_RATIO:-0.50}"
GROUNDING_ADAPTIVE_STUFF_THRESHOLD_RATIO="${GROUNDING_ADAPTIVE_STUFF_THRESHOLD_RATIO:-0.75}"
GROUNDING_ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO="${GROUNDING_ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO:-0.50}"
GROUNDING_MIN_LABEL_SCORE="${GROUNDING_MIN_LABEL_SCORE:-0.0}"
ADE_MIN_ANCHOR_GAUSSIANS="${ADE_MIN_ANCHOR_GAUSSIANS:-500}"
ADE_MIN_ANCHOR_COVERAGE="${ADE_MIN_ANCHOR_COVERAGE:-0.10}"
ADE_MIN_ANCHOR_PRECISION="${ADE_MIN_ANCHOR_PRECISION:-0.10}"
ADE_MIN_SOURCE_VIEWS="${ADE_MIN_SOURCE_VIEWS:-2}"
ADE_MAX_PROMPT_WORDS="${ADE_MAX_PROMPT_WORDS:-180}"
ADE_EVIDENCE_MIN_POSITIVE_VIEWS="${ADE_EVIDENCE_MIN_POSITIVE_VIEWS:-2}"
ADE_EVIDENCE_MIN_RATIO="${ADE_EVIDENCE_MIN_RATIO:-0.50}"
ADE_SPATIAL_VOXEL_SCALE_MULTIPLIER="${ADE_SPATIAL_VOXEL_SCALE_MULTIPLIER:-4.0}"
ADE_SPATIAL_MIN_VOXEL_SIZE="${ADE_SPATIAL_MIN_VOXEL_SIZE:-0.01}"
ADE_SPATIAL_MAX_VOXEL_SIZE="${ADE_SPATIAL_MAX_VOXEL_SIZE:-0.20}"
ADE_PROTOTYPE_MAX_PARTIAL_COMPETING_THING_RISK="${ADE_PROTOTYPE_MAX_PARTIAL_COMPETING_THING_RISK:-0.40}"
ADE_PARTIAL_ANCHOR_EXTENSION_MIN_COVERAGE="${ADE_PARTIAL_ANCHOR_EXTENSION_MIN_COVERAGE:-0.80}"
ADE_PARTIAL_ANCHOR_EXTENSION_MAX_PRECISION="${ADE_PARTIAL_ANCHOR_EXTENSION_MAX_PRECISION:-0.25}"
ADE_PARTIAL_ANCHOR_EXTENSION_MIN_ROBUST_FRACTION="${ADE_PARTIAL_ANCHOR_EXTENSION_MIN_ROBUST_FRACTION:-0.90}"
ADE_PARTIAL_ANCHOR_EXTENSION_MIN_SOURCE_VIEWS="${ADE_PARTIAL_ANCHOR_EXTENSION_MIN_SOURCE_VIEWS:-5}"
ADE_PARTIAL_ANCHOR_EXTENSION_MAX_COMPETING_THING_FRACTION="${ADE_PARTIAL_ANCHOR_EXTENSION_MAX_COMPETING_THING_FRACTION:-0.05}"
ADE_PARTIAL_ANCHOR_EXTENSION_MIN_SPATIAL_KEEP_FRACTION="${ADE_PARTIAL_ANCHOR_EXTENSION_MIN_SPATIAL_KEEP_FRACTION:-0.95}"
RESET_OUTPUT="${RESET_OUTPUT:-1}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

test -d "${MODEL_DIR}"
test -f "${MODEL_DIR}/cameras.json"
test -f "${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
test -f "${SOURCE_VIEW_DIR}/view_manifest.json"
test -f "${BASE_LABELS}"
test -f "${BASE_LABEL_MAP}"
test -f "${ONTOLOGY}"
test -f "${ADE_PROMPT_ALIASES}"
test -d "${FLASHSPLAT_ROOT}"
test -d "${GROUNDED_SAM_ROOT}"
test -f "${GROUNDINGDINO_CONFIG}"
test -f "${GROUNDINGDINO_CHECKPOINT}"
test -d "${SAM_ROOT}"
test -f "${SAM_CHECKPOINT}"
command -v jq >/dev/null 2>&1

if [[ "${COLOR_MODE}" != "class" && "${COLOR_MODE}" != "instance" ]]; then
  echo "COLOR_MODE must be class or instance" >&2
  exit 2
fi
if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nsource=%s\noutput=%s\nmissing_vocabulary_enabled=0\nmin_anchor_gaussians=%s\nmin_anchor_coverage=%s\nmin_anchor_precision=%s\nmin_source_views=%s\nmax_prompt_words=%s\nprototype_max_partial_competing_thing_risk=%s\npartial_anchor_extension_min_coverage=%s\npartial_anchor_extension_max_precision=%s\n' \
    "${SCENE}" "${SOURCE_DIR}" "${OUT_DIR}" "${ADE_MIN_ANCHOR_GAUSSIANS}" \
    "${ADE_MIN_ANCHOR_COVERAGE}" "${ADE_MIN_ANCHOR_PRECISION}" \
    "${ADE_MIN_SOURCE_VIEWS}" \
    "${ADE_MAX_PROMPT_WORDS}" \
    "${ADE_PROTOTYPE_MAX_PARTIAL_COMPETING_THING_RISK}" \
    "${ADE_PARTIAL_ANCHOR_EXTENSION_MIN_COVERAGE}" \
    "${ADE_PARTIAL_ANCHOR_EXTENSION_MAX_PRECISION}"
  exit 0
fi

if [[ "${RESET_OUTPUT}" = "1" && -d "${OUT_DIR}" ]]; then
  case "${OUT_DIR}" in
    "${TASK1_ROOT}"/*) rm -rf -- "${OUT_DIR}" ;;
    *) echo "Refusing to reset unexpected output path: ${OUT_DIR}" >&2; exit 2 ;;
  esac
fi

mkdir -p \
  "${OUT_DIR}/stages" "${SELECTION_DIR}" "${GROUNDING_MASK_DIR}" \
  "${GROUNDING_PROPOSAL_DIR}" "${GROUNDING_FUSION_DIR}" \
  "${REFINEMENT_DIR}" "${DELIVERABLE_DIR}" "${QA_RGB_DIR}" \
  "${QA_OVERLAY_DIR}" "${CHANGE_OVERLAY_DIR}" "${CONTACT_DIR}" \
  "${VALIDATION_DIR}" "${RUN_LOG_DIR}" "${TASK1_ROOT}/logs"
ln -s "${SOURCE_VIEW_DIR}" "${OUT_DIR}/stages/01_real_camera_views"
if [[ -d "${SOURCE_DIR}/stages/02_flashsplat_votes" ]]; then
  ln -s "${SOURCE_DIR}/stages/02_flashsplat_votes" \
    "${OUT_DIR}/stages/02_flashsplat_votes"
fi

cd "${PROJECT_ROOT}"

run_stage() {
  local stage_name="$1"
  local environment="$2"
  shift 2
  echo "===== ${stage_name} ====="
  conda run --no-capture-output -n "${environment}" "$@" 2>&1 \
    | tee "${RUN_LOG_DIR}/${stage_name}.log"
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
  echo "scene=${SCENE}"
  echo "source=${SOURCE_DIR}"
  echo "output=${OUT_DIR}"
  echo "missing_vocabulary_enabled=0"
  echo "ade_min_anchor_gaussians=${ADE_MIN_ANCHOR_GAUSSIANS}"
  echo "ade_min_anchor_coverage=${ADE_MIN_ANCHOR_COVERAGE}"
  echo "ade_min_anchor_precision=${ADE_MIN_ANCHOR_PRECISION}"
  echo "ade_min_source_views=${ADE_MIN_SOURCE_VIEWS}"
  echo "ade_max_prompt_words=${ADE_MAX_PROMPT_WORDS}"
  echo "ade_evidence_min_positive_views=${ADE_EVIDENCE_MIN_POSITIVE_VIEWS}"
  echo "ade_evidence_min_ratio=${ADE_EVIDENCE_MIN_RATIO}"
  nvidia-smi
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

run_stage 04a_automatic_ade_refinement_config "${GAUSSIAN_ENV}" \
  python -m scripts.task1.grounding.build_ade_refinement_config \
    --dinov2-label-map "${BASE_LABEL_MAP}" \
    --ontology "${ONTOLOGY}" \
    --prompt-aliases "${ADE_PROMPT_ALIASES}" \
    --output "${REFINEMENT_CONFIG}" \
    --min-anchor-gaussians "${ADE_MIN_ANCHOR_GAUSSIANS}" \
    --min-source-views "${ADE_MIN_SOURCE_VIEWS}" \
    --max-prompt-words "${ADE_MAX_PROMPT_WORDS}" \
    --overwrite

run_stage 04b_grounding_seed_views "${GAUSSIAN_ENV}" \
  python -m scripts.task1.grounding.select_targeted_cameras seed \
    --source-manifest "${SOURCE_VIEW_DIR}/view_manifest.json" \
    --output "${SELECTION_DIR}/seed_manifest.json" \
    --indices-output "${SELECTION_DIR}/seed_indices.txt" \
    --count "${GROUNDING_BASE_VIEW_COUNT}"

run_stage 04c_grounding_candidate_screen "${GAUSSIAN_ENV}" \
  python -m scripts.task1.grounding.select_targeted_cameras screen \
    --model-path "${MODEL_DIR}" \
    --baseline-manifest "${SELECTION_DIR}/seed_manifest.json" \
    --allowed-manifest "${SOURCE_VIEW_DIR}/view_manifest.json" \
    --output "${SELECTION_DIR}/candidate_screen.json" \
    --indices-output "${SELECTION_DIR}/candidate_indices.txt" \
    --iteration 30000 \
    --sample-count "${GROUNDING_PROJECTION_SAMPLE_COUNT}" \
    --candidate-count "${GROUNDING_TARGET_CANDIDATE_COUNT}" \
    --projection-margin "${GROUNDING_PROJECTION_MARGIN}"

AVAILABLE_CANDIDATES="$(jq -r '.selected_candidate_count' "${SELECTION_DIR}/candidate_screen.json")"
REQUESTED_ADDITIONS="${GROUNDING_TARGET_VIEW_COUNT}"
if (( REQUESTED_ADDITIONS > AVAILABLE_CANDIDATES )); then
  REQUESTED_ADDITIONS="${AVAILABLE_CANDIDATES}"
fi

if (( REQUESTED_ADDITIONS > 0 )); then
  CANDIDATE_CAMERA_INDICES="$(tr -d '\r\n' < "${SELECTION_DIR}/candidate_indices.txt")"
  test -n "${CANDIDATE_CAMERA_INDICES}"

  run_stage 04d_grounding_candidate_overlays "${GAUSSIAN_ENV}" \
    python -m scripts.task1.qa.render_auto_label_overlays \
      --model-path "${MODEL_DIR}" \
      --labels-npy "${BASE_LABELS}" \
      --label-map "${BASE_LABEL_MAP}" \
      --output-dir "${SELECTION_DIR}/candidate_overlays" \
      --rgb-output-dir "${SELECTION_DIR}/candidate_rgb" \
      --source-view-manifest "${SOURCE_VIEW_DIR}/view_manifest.json" \
      --flashsplat-root "${FLASHSPLAT_ROOT}" \
      --iteration 30000 \
      --max-width "${RENDER_MAX_WIDTH}" \
      --camera-indices "${CANDIDATE_CAMERA_INDICES}" \
      --count "${AVAILABLE_CANDIDATES}" \
      --max-labels 0 \
      --color-mode "${COLOR_MODE}"

  run_stage 04e_grounding_candidate_coverage "${GAUSSIAN_ENV}" \
    python -m scripts.task1.qa.measure_overlay_coverage \
      --rgb-dir "${SELECTION_DIR}/candidate_rgb" \
      --overlay-dir "${SELECTION_DIR}/candidate_overlays" \
      --output "${SELECTION_DIR}/candidate_visible_coverage.json" \
      --difference-threshold 2 \
      --exclude-border 1

  run_stage 04f_grounding_target_selection "${GAUSSIAN_ENV}" \
    python -m scripts.task1.grounding.select_targeted_cameras select \
      --model-path "${MODEL_DIR}" \
      --baseline-manifest "${SELECTION_DIR}/seed_manifest.json" \
      --candidate-manifest "${SELECTION_DIR}/candidate_overlays/label_overlay_manifest.json" \
      --coverage-report "${SELECTION_DIR}/candidate_visible_coverage.json" \
      --output "${SELECTION_DIR}/targeted_camera_selection.json" \
      --indices-output "${SELECTION_DIR}/final_camera_indices.txt" \
      --additional-count "${REQUESTED_ADDITIONS}" \
      --low-coverage-quantile "${GROUNDING_LOW_COVERAGE_QUANTILE}" \
      --coverage-weight 0.8 \
      --novelty-weight 0.2
else
  cp "${SELECTION_DIR}/seed_indices.txt" "${SELECTION_DIR}/final_camera_indices.txt"
fi

GROUNDING_CAMERA_INDICES="$(tr -d '\r\n' < "${SELECTION_DIR}/final_camera_indices.txt")"
GROUNDING_VIEW_COUNT="$(awk -F, '{print NF}' "${SELECTION_DIR}/final_camera_indices.txt")"
test -n "${GROUNDING_CAMERA_INDICES}"

run_stage 05_07_grounding_ade_branch "${GROUNDING_ENV}" \
  python -m scripts.task1.grounding.run_grounding_semantic_branch \
    --model-path "${MODEL_DIR}" \
    --ground-output-dir "${GROUNDING_MASK_DIR}" \
    --proposal-dir "${GROUNDING_PROPOSAL_DIR}" \
    --fusion-dir "${GROUNDING_FUSION_DIR}" \
    --label-map-path "${GROUNDING_FUSION_DIR}/label_map.json" \
    --no-semantic-ply \
    --log-dir "${RUN_LOG_DIR}" \
    --class-config "${REFINEMENT_CONFIG}" \
    --source-view-manifest "${SOURCE_VIEW_DIR}/view_manifest.json" \
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

run_stage 08_guarded_ade_refinement_merge "${GAUSSIAN_ENV}" \
  python -m scripts.task1.merge.merge_ade_refinement \
    --base-labels "${BASE_LABELS}" \
    --base-label-map "${BASE_LABEL_MAP}" \
    --grounding-labels "${GROUNDING_FUSION_DIR}/gaussian_labels.npy" \
    --grounding-label-map "${GROUNDING_FUSION_DIR}/label_map.json" \
    --class-evidence-dir "${GROUNDING_PROPOSAL_DIR}/class_evidence" \
    --refinement-config "${REFINEMENT_CONFIG}" \
    --ontology "${ONTOLOGY}" \
    --source-ply "${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply" \
    --output-dir "${REFINEMENT_DIR}" \
    --semantic-ply-path "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --scene "${SCENE}" \
    --min-anchor-gaussians "${ADE_MIN_ANCHOR_GAUSSIANS}" \
    --min-anchor-coverage "${ADE_MIN_ANCHOR_COVERAGE}" \
    --min-anchor-precision "${ADE_MIN_ANCHOR_PRECISION}" \
    --min-group-proposals 2 \
    --min-group-source-views 2 \
    --evidence-min-positive-views "${ADE_EVIDENCE_MIN_POSITIVE_VIEWS}" \
    --evidence-min-ratio "${ADE_EVIDENCE_MIN_RATIO}" \
    --spatial-voxel-scale-multiplier "${ADE_SPATIAL_VOXEL_SCALE_MULTIPLIER}" \
    --spatial-min-voxel-size "${ADE_SPATIAL_MIN_VOXEL_SIZE}" \
    --spatial-max-voxel-size "${ADE_SPATIAL_MAX_VOXEL_SIZE}" \
    --prototype-max-partial-competing-thing-risk "${ADE_PROTOTYPE_MAX_PARTIAL_COMPETING_THING_RISK}" \
    --partial-anchor-extension-min-anchor-coverage "${ADE_PARTIAL_ANCHOR_EXTENSION_MIN_COVERAGE}" \
    --partial-anchor-extension-max-anchor-precision "${ADE_PARTIAL_ANCHOR_EXTENSION_MAX_PRECISION}" \
    --partial-anchor-extension-min-robust-fraction "${ADE_PARTIAL_ANCHOR_EXTENSION_MIN_ROBUST_FRACTION}" \
    --partial-anchor-extension-min-source-views "${ADE_PARTIAL_ANCHOR_EXTENSION_MIN_SOURCE_VIEWS}" \
    --partial-anchor-extension-max-competing-thing-fraction "${ADE_PARTIAL_ANCHOR_EXTENSION_MAX_COMPETING_THING_FRACTION}" \
    --partial-anchor-extension-min-spatial-keep-fraction "${ADE_PARTIAL_ANCHOR_EXTENSION_MIN_SPATIAL_KEEP_FRACTION}" \
    --overwrite

cp "${REFINEMENT_DIR}/gaussian_labels.npy" "${DELIVERABLE_DIR}/gaussian_labels.npy"
cp "${REFINEMENT_DIR}/label_map.json" "${DELIVERABLE_DIR}/label_map.json"

run_stage 10_supersplat_debug_ply "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.export_supersplat_label_colors \
    --input-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-ply "${VISUALIZATION_DIR}/semantic_point_cloud_supersplat_debug.ply" \
    --metadata-json "${VISUALIZATION_DIR}/semantic_point_cloud_supersplat_debug.json" \
    --color-mode "${COLOR_MODE}" \
    --overwrite

run_stage 11_semantic_overlays "${GAUSSIAN_ENV}" \
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

run_stage 12_ade_change_overlays "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.render_auto_label_overlays \
    --model-path "${MODEL_DIR}" \
    --labels-npy "${REFINEMENT_DIR}/refinement_changes.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-dir "${CHANGE_OVERLAY_DIR}" \
    --source-view-manifest "${SOURCE_VIEW_DIR}/view_manifest.json" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --max-labels 0 \
    --count "${QA_VIEW_COUNT}" \
    --color-mode "${COLOR_MODE}"

run_stage 13_visible_coverage "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.measure_overlay_coverage \
    --rgb-dir "${QA_RGB_DIR}" \
    --overlay-dir "${QA_OVERLAY_DIR}" \
    --output "${VALIDATION_DIR}/visible_overlay_coverage.json" \
    --difference-threshold 2 \
    --exclude-border 1

run_stage 14_contact_sheets "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${QA_OVERLAY_DIR}" \
    --output "${CONTACT_DIR}/semantic_labels.png" \
    --columns 4 \
    --thumb-width 360

run_stage 15_change_contact_sheet "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${CHANGE_OVERLAY_DIR}" \
    --output "${CONTACT_DIR}/ade_refinement_changes.png" \
    --columns 4 \
    --thumb-width 360

run_stage 16_semantic_color_legend "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.write_semantic_color_legend \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --output-json "${VISUALIZATION_DIR}/semantic_color_legend.json" \
    --output-png "${VISUALIZATION_DIR}/semantic_color_legend.png" \
    --color-mode "${COLOR_MODE}" \
    --overwrite

run_report 17_task1_validation "${GAUSSIAN_ENV}" \
  "${VALIDATION_DIR}/task1_validation.json" \
  python -m scripts.task1.qa.validate_task1_outputs \
    --labels-npy "${DELIVERABLE_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --semantic-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --visible-coverage "${VALIDATION_DIR}/visible_overlay_coverage.json"

printf 'final_mode=dinov2_groundingdino_adaptive_instance_guard_v5\nsource=%s\nmissing_vocabulary_enabled=0\n' \
  "${SOURCE_DIR}" \
  > "${VALIDATION_DIR}/final_mode.txt"

find "${OUT_DIR}" -maxdepth 4 -type f | sort
