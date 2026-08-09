#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_soft_rt
#SBATCH --account=gpulab02
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# Automatic report-only probability-preserving DINOv3 round-trip audit.
# The selected cameras are derived from the saved 99% two-view visibility
# threshold.  The stage writes no accepted Gaussian labels, label map, or PLY.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
CAMERA_SELECTION_OUTPUT_NAME="${CAMERA_SELECTION_OUTPUT_NAME:?Set CAMERA_SELECTION_OUTPUT_NAME to the completed automatic visibility selection}"
SOURCE_CACHE_OUTPUT_NAME="${SOURCE_CACHE_OUTPUT_NAME:?Set SOURCE_CACHE_OUTPUT_NAME to the completed automatic 57-view DINOv3 cache}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_soft_probability_round_trip_v1}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
SELECTION_DIR="${TASK1_ROOT}/${CAMERA_SELECTION_OUTPUT_NAME}/selection"
SOURCE_CACHE_ROOT="${TASK1_ROOT}/${SOURCE_CACHE_OUTPUT_NAME}"
SOURCE_VIEW_DIR="${SOURCE_CACHE_ROOT}/stages/01_real_camera_views"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
VIEW_DIR="${OUT_DIR}/stages/01_real_camera_views"
SOFT_VOTE_DIR="${OUT_DIR}/stages/02_soft_probability_votes"
PRIMARY_AUDIT_DIR="${OUT_DIR}/stages/03_soft_probability_round_trip"
AUDIT_RETRY_DIR="${OUT_DIR}/stages/03_soft_probability_round_trip_retry_v1"
CONTACT_DIR="${OUT_DIR}/visualizations/contact_sheets"
LOG_DIR="${OUT_DIR}/logs"
BASE_SCHEDULER="${PROJECT_ROOT}/scripts/slurm/slurm_task1_dinov3_scene.a100.sbatch"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-semantic_3dgs_renderer}"
DINOV3_ENV="${DINOV3_ENV:-dinov3_semantic}"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${WORKSPACE_ROOT}/external/FlashSplat}"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"
TARGET_TWO_VIEW_PERCENT="${TARGET_TWO_VIEW_PERCENT:-99}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"
RESUME_FROM_SOFT_LIFT="${RESUME_FROM_SOFT_LIFT:-0}"
RESUME_FROM_SOFT_AUDIT="${RESUME_FROM_SOFT_AUDIT:-0}"
STOP_AFTER_SOFT_LIFT="${STOP_AFTER_SOFT_LIFT:-0}"
AUDIT_DIR="${PRIMARY_AUDIT_DIR}"
if [[ "${RESUME_FROM_SOFT_AUDIT}" = "1" ]]; then
  AUDIT_DIR="${AUDIT_RETRY_DIR}"
fi

if [[ "${CONFIG_ONLY}" != "0" && "${CONFIG_ONLY}" != "1" ]]; then
  echo "CONFIG_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ "${RESUME_FROM_SOFT_LIFT}" != "0" && "${RESUME_FROM_SOFT_LIFT}" != "1" ]]; then
  echo "RESUME_FROM_SOFT_LIFT must be 0 or 1" >&2
  exit 2
fi
if [[ "${RESUME_FROM_SOFT_AUDIT}" != "0" && "${RESUME_FROM_SOFT_AUDIT}" != "1" ]]; then
  echo "RESUME_FROM_SOFT_AUDIT must be 0 or 1" >&2
  exit 2
fi
if [[ "${STOP_AFTER_SOFT_LIFT}" != "0" && "${STOP_AFTER_SOFT_LIFT}" != "1" ]]; then
  echo "STOP_AFTER_SOFT_LIFT must be 0 or 1" >&2
  exit 2
fi
if [[ "${RESUME_FROM_SOFT_LIFT}" = "1" && "${RESUME_FROM_SOFT_AUDIT}" = "1" ]]; then
  echo "Only one resume mode may be enabled" >&2
  exit 2
fi
if [[ "${STOP_AFTER_SOFT_LIFT}" = "1" && "${RESUME_FROM_SOFT_AUDIT}" = "1" ]]; then
  echo "STOP_AFTER_SOFT_LIFT cannot be combined with stage-8 resume" >&2
  exit 2
fi
if [[ "${TARGET_TWO_VIEW_PERCENT}" != "99" ]]; then
  echo "This reviewed audit requires TARGET_TWO_VIEW_PERCENT=99" >&2
  exit 2
fi
if [[ -n "${CAMERA_INDICES:-}" || -n "${VIEW_COUNT:-}" ]]; then
  echo "Manual CAMERA_INDICES and VIEW_COUNT are not accepted" >&2
  exit 2
fi
if [[ -n "${SOFT_SAMPLE_COUNT:-}" || -n "${MIN_VIEWS:-}" || -n "${CAMERA_MAJORITY_THRESHOLD:-}" || -n "${MIN_PIXEL_CONFIDENCE:-}" ]]; then
  echo "Estimator and fusion thresholds are fixed by the audit contract" >&2
  exit 2
fi

test -f "${BASE_SCHEDULER}"
test -f "${MODEL_DIR}/cameras.json"
test -f "${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
test -f "${SELECTION_DIR}/camera_selection_report.json"
test -f "${SELECTION_DIR}/greedy_camera_indices.npy"
test -f "${SELECTION_DIR}/coverage_curve.npz"
test -f "${SOURCE_VIEW_DIR}/dinov3_manifest.json"
test -f "${SOURCE_CACHE_ROOT}/selection/selected_view_cache_report.json"
test -d "${FLASHSPLAT_ROOT}"
test -f "${ONTOLOGY}"

cd "${PROJECT_ROOT}"
mapfile -t SELECTION_VALUES < <(
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
    python -m scripts.task1.dinov3.selected_view_cache resolve \
      --selection-dir "${SELECTION_DIR}" \
      --model-path "${MODEL_DIR}" \
      --target-percent 99 \
      --source-max-width 960 \
      --format lines
)
if [[ "${#SELECTION_VALUES[@]}" -ne 2 ]]; then
  echo "Automatic camera-prefix resolver returned unexpected output" >&2
  exit 2
fi
SELECTED_VIEW_COUNT="${SELECTION_VALUES[0]}"
SELECTED_CAMERA_INDICES="${SELECTION_VALUES[1]}"
if ! [[ "${SELECTED_VIEW_COUNT}" =~ ^[1-9][0-9]*$ ]] || [[ -z "${SELECTED_CAMERA_INDICES}" ]]; then
  echo "Automatic camera-prefix resolver returned an invalid prefix" >&2
  exit 2
fi

printf 'scene=%s\nselection=%s\nsource_hard_cache=%s\noutput=%s\nselected_view_count=%s\nresume_from_soft_lift=%s\nresume_from_soft_audit=%s\nstop_after_soft_lift=%s\naudit_output=%s\nmanual_camera_selection_used=0\nprobability_storage=float16_150_class_npy\nsoft_lift=deterministic_8_stratum_categorical\nfusion=equal_camera_probability_sum_then_argmax\nreport_only=1\naccepted_gaussian_labels_written=0\nlabel_map_written=0\nsemantic_ply_written=0\n' \
  "${SCENE}" "${SELECTION_DIR}" "${SOURCE_CACHE_ROOT}" "${OUT_DIR}" "${SELECTED_VIEW_COUNT}" "${RESUME_FROM_SOFT_LIFT}" "${RESUME_FROM_SOFT_AUDIT}" "${STOP_AFTER_SOFT_LIFT}" "${AUDIT_DIR}"

COMMON_SETTINGS=(
  PROJECT_ROOT="${PROJECT_ROOT}"
  SCENE="${SCENE}"
  MODEL_DIR="${MODEL_DIR}"
  OUTPUT_NAME="${OUTPUT_NAME}"
  GAUSSIAN_ENV="${GAUSSIAN_ENV}"
  DINOV3_ENV="${DINOV3_ENV}"
  CAMERA_INDICES="${SELECTED_CAMERA_INDICES}"
  VIEW_COUNT="${SELECTED_VIEW_COUNT}"
  RENDER_MAX_WIDTH=960
  REPORT_ONLY=1
  WRITE_SEMANTIC_PLY=0
  RESET_OUTPUT=0
  DINOV3_PRECISION=bfloat16
  DINOV3_CROP_SIZE=512
  DINOV3_STRIDE=384
  DINOV3_CHECKPOINT_LOAD_MODE=local_mmap
  DINOV3_MAX_CUDA_MEMORY_GIB=40
  DINOV3_SAVE_PROBABILITIES=1
  DINOV3_MIN_HOST_AVAILABLE_GIB=
  DINOV3_MIN_GPU_FREE_GIB=
  DINOV3_RESOURCE_GPU_ID=
  MIN_PIXEL_CONFIDENCE=0.0
)

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  if [[ "${RESUME_FROM_SOFT_AUDIT}" = "1" ]]; then
    test -f "${OUT_DIR}/selection/selected_view_cache_report.json"
    test -f "${OUT_DIR}/selection/hard_cache_reproducibility.json"
    test -f "${SOFT_VOTE_DIR}/soft_vote_manifest.json"
    if [[ -e "${AUDIT_RETRY_DIR}" ]]; then
      echo "Stage-8 retry output already exists: ${AUDIT_RETRY_DIR}" >&2
      exit 2
    fi
    if [[ -e "${CONTACT_DIR}/soft_heldout_overlays.png" || -e "${CONTACT_DIR}/soft_heldout_disagreement.png" ]]; then
      echo "Stage-8 retry refuses existing soft contact sheets" >&2
      exit 2
    fi
  elif [[ "${RESUME_FROM_SOFT_LIFT}" = "1" ]]; then
    test -f "${OUT_DIR}/selection/selected_view_cache_report.json"
    test -f "${OUT_DIR}/selection/selected_camera_indices.npy"
    test -f "${OUT_DIR}/selection/hard_cache_reproducibility.json"
    if [[ -e "${SOFT_VOTE_DIR}" || -e "${AUDIT_DIR}" ]]; then
      echo "Resume requires stages 7 and 8 to be absent: ${OUT_DIR}" >&2
      exit 2
    fi
  else
    env "${COMMON_SETTINGS[@]}" CONFIG_ONLY=1 bash "${BASE_SCHEDULER}"
  fi
  exit 0
fi

run_stage() {
  local stage="$1"
  shift
  echo "===== ${stage} ====="
  "$@" 2>&1 | tee "${LOG_DIR}/${stage}.log"
}

if [[ "${RESUME_FROM_SOFT_AUDIT}" = "1" ]]; then
  if [[ ! -d "${OUT_DIR}" ]]; then
    echo "Stage-8 resume output does not exist: ${OUT_DIR}" >&2
    exit 2
  fi
  test -f "${OUT_DIR}/selection/selected_view_cache_report.json"
  test -f "${OUT_DIR}/selection/hard_cache_reproducibility.json"
  test -f "${SOFT_VOTE_DIR}/soft_vote_manifest.json"
  if [[ -e "${AUDIT_RETRY_DIR}" ]]; then
    echo "Stage-8 retry output already exists: ${AUDIT_RETRY_DIR}" >&2
    exit 2
  fi
  if [[ -e "${CONTACT_DIR}/soft_heldout_overlays.png" || -e "${CONTACT_DIR}/soft_heldout_disagreement.png" ]]; then
    echo "Stage-8 retry refuses existing soft contact sheets" >&2
    exit 2
  fi
elif [[ "${RESUME_FROM_SOFT_LIFT}" = "1" ]]; then
  if [[ ! -d "${OUT_DIR}" ]]; then
    echo "Resume output does not exist: ${OUT_DIR}" >&2
    exit 2
  fi
  if [[ -e "${SOFT_VOTE_DIR}" || -e "${AUDIT_DIR}" ]]; then
    echo "Resume refuses partial or completed stages 7-8: ${OUT_DIR}" >&2
    exit 2
  fi
  run_stage 06_resume_validate_completed_probability_cache \
    conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
    python -m scripts.task1.dinov3.selected_view_cache validate-resume \
      --selection-dir "${SELECTION_DIR}" \
      --model-path "${MODEL_DIR}" \
      --output-dir "${OUT_DIR}" \
      --reference-view-dir "${SOURCE_VIEW_DIR}"
else
  if [[ -e "${OUT_DIR}" ]]; then
    echo "Output exists: ${OUT_DIR}" >&2
    exit 2
  fi

  env "${COMMON_SETTINGS[@]}" CONFIG_ONLY=0 bash "${BASE_SCHEDULER}"

  run_stage 05_validate_probability_cache \
    conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
    python -m scripts.task1.dinov3.selected_view_cache finalize \
      --selection-dir "${SELECTION_DIR}" \
      --model-path "${MODEL_DIR}" \
      --output-dir "${OUT_DIR}" \
      --target-percent 99 \
      --render-max-width 960 \
      --crop-size 512 \
      --stride 384 \
      --precision bfloat16 \
      --checkpoint-load-mode local_mmap \
      --require-probabilities

  run_stage 06_compare_prior_hard_cache \
    conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
    python -m scripts.task1.dinov3.selected_view_cache compare-hard-cache \
      --reference-view-dir "${SOURCE_VIEW_DIR}" \
      --candidate-view-dir "${VIEW_DIR}" \
      --output "${OUT_DIR}/selection/hard_cache_reproducibility.json"
fi

if [[ "${RESUME_FROM_SOFT_AUDIT}" != "1" ]]; then
  run_stage 07_soft_probability_flashsplat_lift \
    conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
    python -m scripts.task1.dinov3.lift_soft_probability_view_votes \
      --model-path "${MODEL_DIR}" \
      --input-dir "${VIEW_DIR}" \
      --probability-manifest "${VIEW_DIR}/dinov3_manifest.json" \
      --selected-cache-report "${OUT_DIR}/selection/selected_view_cache_report.json" \
      --output-dir "${SOFT_VOTE_DIR}" \
      --flashsplat-root "${FLASHSPLAT_ROOT}" \
      --ontology "${ONTOLOGY}" \
      --iteration 30000 \
      --max-width 960 \
      --sample-count 8
fi

if [[ "${STOP_AFTER_SOFT_LIFT}" = "1" ]]; then
  cat >> "${OUT_DIR}/experiment_mode.txt" <<EOF
mode=automatic_dinov3_soft_probability_cache_only
selection_mode=automatic_visibility_threshold_prefix
source_camera_selection_output=${CAMERA_SELECTION_OUTPUT_NAME}
source_hard_cache_output=${SOURCE_CACHE_OUTPUT_NAME}
threshold_group=two_view_coverage_of_multiview_capable
target_percent=99
selected_view_count=${SELECTED_VIEW_COUNT}
resume_from_soft_lift=${RESUME_FROM_SOFT_LIFT}
stop_after_soft_lift=1
probability_storage=float16_150_class_npy
soft_lift_estimator=deterministic_circular_stratified_categorical_8_samples
report_only=1
manual_camera_selection_used=0
manual_gaussian_selection_used=0
accepted_gaussian_labels_written=0
gaussian_project_class_array_written=0
label_map_written=0
semantic_ply_written=0
EOF
  test -f "${SOFT_VOTE_DIR}/soft_vote_manifest.json"
  test ! -e "${PRIMARY_AUDIT_DIR}"
  test ! -e "${AUDIT_RETRY_DIR}"
  for forbidden in gaussian_labels.npy gaussian_project_class_ids.npy project_class_id.npy label_map.json semantic_point_cloud.ply; do
    if find "${OUT_DIR}" -type f -name "${forbidden}" -print -quit | grep -q .; then
      echo "Report-only cache unexpectedly wrote ${forbidden}" >&2
      exit 2
    fi
  done
  if find "${OUT_DIR}" -type f -name '*.ply' -print -quit | grep -q .; then
    echo "Report-only cache unexpectedly wrote a PLY" >&2
    exit 2
  fi
  find "${OUT_DIR}" -maxdepth 5 -type f | sort
  exit 0
fi

AUDIT_STAGE_NAME="08_soft_leave_one_camera_out_round_trip"
if [[ "${RESUME_FROM_SOFT_AUDIT}" = "1" ]]; then
  AUDIT_STAGE_NAME="08_retry_soft_leave_one_camera_out_round_trip"
fi
run_stage "${AUDIT_STAGE_NAME}" \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.soft_probability_round_trip_audit \
    --model-path "${MODEL_DIR}" \
    --source-view-dir "${VIEW_DIR}" \
    --selected-cache-report "${OUT_DIR}/selection/selected_view_cache_report.json" \
    --soft-vote-manifest "${SOFT_VOTE_DIR}/soft_vote_manifest.json" \
    --ontology "${ONTOLOGY}" \
    --output-dir "${AUDIT_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000

run_stage 09_soft_overlay_contact_sheet \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${AUDIT_DIR}/heldout_overlays" \
    --output "${CONTACT_DIR}/soft_heldout_overlays.png" \
    --columns 5 --thumb-width 240

run_stage 10_soft_disagreement_contact_sheet \
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${AUDIT_DIR}/heldout_disagreement" \
    --output "${CONTACT_DIR}/soft_heldout_disagreement.png" \
    --columns 5 --thumb-width 240

cat >> "${OUT_DIR}/experiment_mode.txt" <<EOF
mode=automatic_dinov3_soft_probability_round_trip_audit
selection_mode=automatic_visibility_threshold_prefix
source_camera_selection_output=${CAMERA_SELECTION_OUTPUT_NAME}
source_hard_cache_output=${SOURCE_CACHE_OUTPUT_NAME}
threshold_group=two_view_coverage_of_multiview_capable
target_percent=99
selected_view_count=${SELECTED_VIEW_COUNT}
resume_from_soft_lift=${RESUME_FROM_SOFT_LIFT}
resume_from_soft_audit=${RESUME_FROM_SOFT_AUDIT}
stop_after_soft_lift=${STOP_AFTER_SOFT_LIFT}
soft_audit_output=${AUDIT_DIR}
probability_storage=float16_150_class_npy
soft_lift_estimator=deterministic_circular_stratified_categorical_8_samples
fusion_policy=equal_camera_full_probability_sum_then_argmax
heldout_policy=exclude_target_camera_before_fusion
report_only=1
manual_camera_selection_used=0
manual_gaussian_selection_used=0
accepted_gaussian_labels_written=0
gaussian_project_class_array_written=0
label_map_written=0
semantic_ply_written=0
EOF

test -f "${AUDIT_DIR}/soft_probability_round_trip_report.json"
test -f "${OUT_DIR}/selection/hard_cache_reproducibility.json"
test -f "${AUDIT_DIR}/heldout_confusion_matrix.npz"
test -f "${CONTACT_DIR}/soft_heldout_overlays.png"
test -f "${CONTACT_DIR}/soft_heldout_disagreement.png"
for forbidden in gaussian_labels.npy gaussian_project_class_ids.npy project_class_id.npy label_map.json semantic_point_cloud.ply; do
  if find "${OUT_DIR}" -type f -name "${forbidden}" -print -quit | grep -q .; then
    echo "Report-only audit unexpectedly wrote ${forbidden}" >&2
    exit 2
  fi
done
if find "${OUT_DIR}" -type f -name '*.ply' -print -quit | grep -q .; then
  echo "Report-only audit unexpectedly wrote a PLY" >&2
  exit 2
fi

find "${OUT_DIR}" -maxdepth 5 -type f | sort
