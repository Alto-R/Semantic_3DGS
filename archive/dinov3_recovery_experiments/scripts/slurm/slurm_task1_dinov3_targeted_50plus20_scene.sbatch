#!/usr/bin/env bash
#SBATCH --job-name=task1_dv3_targeted
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

# Select 50 evenly spaced cameras plus 20 low-coverage, pose-diverse cameras,
# then run the independent report-only DINOv3 3D-first pipeline on the exact
# selected list. Prior labels guide camera selection only; they are never
# consumed by DINOv3 inference, proposal lifting, association, or identity.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
EXTERNAL_ROOT="${WORKSPACE_ROOT}/external"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${EXTERNAL_ROOT}/FlashSplat}"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
ALLOWED_VIEW_OUTPUT_NAME="${ALLOWED_VIEW_OUTPUT_NAME:?Set ALLOWED_VIEW_OUTPUT_NAME to a completed all-camera DINOv3 report}"
SELECTION_LABEL_OUTPUT_NAME="${SELECTION_LABEL_OUTPUT_NAME:?Set SELECTION_LABEL_OUTPUT_NAME to the reviewed DINOv3 label source}"
SELECTION_OUTPUT_NAME="${SELECTION_OUTPUT_NAME:?Set a fresh SELECTION_OUTPUT_NAME}"
OUTPUT_NAME="${OUTPUT_NAME:?Set a fresh report-only DINOv3 OUTPUT_NAME}"

ALLOWED_VIEW_ROOT="${TASK1_ROOT}/${ALLOWED_VIEW_OUTPUT_NAME}"
ALLOWED_VIEW_MANIFEST="${ALLOWED_VIEW_ROOT}/stages/01_dinov3_query_evidence/view_manifest.json"
SELECTION_LABEL_ROOT="${TASK1_ROOT}/${SELECTION_LABEL_OUTPUT_NAME}"
SELECTION_LABELS="${SELECTION_LABEL_ROOT}/deliverables/gaussian_labels.npy"
SELECTION_LABEL_MAP="${SELECTION_LABEL_ROOT}/deliverables/label_map.json"
SELECTION_DIR="${TASK1_ROOT}/${SELECTION_OUTPUT_NAME}"
REPORT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
RUN_LOG_DIR="${SELECTION_DIR}/logs"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
BASE_VIEW_COUNT="${BASE_VIEW_COUNT:-50}"
ADDITIONAL_VIEW_COUNT="${ADDITIONAL_VIEW_COUNT:-20}"
CANDIDATE_VIEW_COUNT="${CANDIDATE_VIEW_COUNT:-100}"
PROJECTION_SAMPLE_COUNT="${PROJECTION_SAMPLE_COUNT:-50000}"
PROJECTION_MARGIN="${PROJECTION_MARGIN:-1.5}"
LOW_COVERAGE_QUANTILE="${LOW_COVERAGE_QUANTILE:-0.35}"
COVERAGE_WEIGHT="${COVERAGE_WEIGHT:-0.8}"
POSE_NOVELTY_WEIGHT="${POSE_NOVELTY_WEIGHT:-0.2}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
COLOR_MODE="${COLOR_MODE:-class}"
RESET_OUTPUT="${RESET_OUTPUT:-0}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

if [[ "${RESET_OUTPUT}" != "0" ]]; then
  echo "This experiment requires fresh outputs; RESET_OUTPUT must remain 0" >&2
  exit 2
fi
for positive_count in \
  "${BASE_VIEW_COUNT}" \
  "${ADDITIONAL_VIEW_COUNT}" \
  "${CANDIDATE_VIEW_COUNT}" \
  "${PROJECTION_SAMPLE_COUNT}"; do
  if ! [[ "${positive_count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Camera and geometry counts must be positive integers" >&2
    exit 2
  fi
done

test -d "${MODEL_DIR}"
test -f "${MODEL_DIR}/cameras.json"
test -d "${FLASHSPLAT_ROOT}"
test -f "${ALLOWED_VIEW_MANIFEST}"
test -f "${SELECTION_LABELS}"
test -f "${SELECTION_LABEL_MAP}"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nmodel=%s\nselection_output=%s\ndinov3_report_output=%s\nallowed_view_manifest=%s\nselection_label_source=%s\nbase_views=%s\nadditional_views=%s\ncandidate_pool=%s\nprojection_samples=%s\nprojection_margin=%s\nlow_coverage_quantile=%s\ncoverage_weight=%s\npose_novelty_weight=%s\ncamera_selection_used_prior_labels=1\nsemantic_inference_used_prior_labels=0\nv5_used=0\ndinov2_used=0\nreport_only=1\nsemantic_labels_written=0\nsemantic_ply_written=0\n' \
    "${SCENE}" "${MODEL_DIR}" "${SELECTION_DIR}" "${REPORT_DIR}" \
    "${ALLOWED_VIEW_MANIFEST}" "${SELECTION_LABEL_ROOT}" \
    "${BASE_VIEW_COUNT}" "${ADDITIONAL_VIEW_COUNT}" "${CANDIDATE_VIEW_COUNT}" \
    "${PROJECTION_SAMPLE_COUNT}" "${PROJECTION_MARGIN}" \
    "${LOW_COVERAGE_QUANTILE}" "${COVERAGE_WEIGHT}" "${POSE_NOVELTY_WEIGHT}"
  PROJECT_ROOT="${PROJECT_ROOT}" \
  SCENE="${SCENE}" \
  MODEL_DIR="${MODEL_DIR}" \
  OUTPUT_NAME="${OUTPUT_NAME}" \
  FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT}" \
  CAMERA_INDICES="<generated-50plus20>" \
  VIEW_COUNT="$((BASE_VIEW_COUNT + ADDITIONAL_VIEW_COUNT))" \
  RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH}" \
  CAMERA_SELECTION_POLICY="50_even_plus_20_low_coverage_pose_diverse" \
  CAMERA_SELECTION_USED_PRIOR_LABELS=1 \
  CONFIG_ONLY=1 \
  RESET_OUTPUT=0 \
  bash "${PROJECT_ROOT}/scripts/slurm/slurm_task1_dinov3_3d_first_scene.sbatch"
  exit 0
fi

for fresh_output in "${SELECTION_DIR}" "${REPORT_DIR}"; do
  if [[ -e "${fresh_output}" ]]; then
    echo "Output exists; choose a fresh output name:" >&2
    echo "${fresh_output}" >&2
    exit 2
  fi
done

mkdir -p "${SELECTION_DIR}" "${RUN_LOG_DIR}" "${TASK1_ROOT}/logs"
cd "${PROJECT_ROOT}"

run_stage() {
  local stage_name="$1"
  shift
  echo "===== ${stage_name} ====="
  conda run --no-capture-output -n "${GAUSSIAN_ENV}" "$@" \
    2>&1 | tee "${RUN_LOG_DIR}/${stage_name}.log"
}

{
  echo "job=${SLURM_JOB_ID:-local}"
  echo "host=$(hostname)"
  echo "project=${PROJECT_ROOT}"
  echo "scene=${SCENE}"
  echo "selection_output=${SELECTION_DIR}"
  echo "dinov3_report_output=${REPORT_DIR}"
  echo "selection_policy=50_even_plus_20_low_coverage_pose_diverse"
  echo "camera_selection_used_prior_labels=1"
  echo "semantic_inference_used_prior_labels=0"
  echo "selection_label_source=${SELECTION_LABEL_ROOT}"
  echo "v5_used=0"
  echo "dinov2_used=0"
  nvidia-smi
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

run_stage 01_select_even_baseline \
  python -m scripts.task1.grounding.select_targeted_cameras seed \
    --source-manifest "${ALLOWED_VIEW_MANIFEST}" \
    --output "${SELECTION_DIR}/seed_manifest.json" \
    --indices-output "${SELECTION_DIR}/seed_indices.txt" \
    --count "${BASE_VIEW_COUNT}"

run_stage 02_screen_projection_safe_candidates \
  python -m scripts.task1.grounding.select_targeted_cameras screen \
    --model-path "${MODEL_DIR}" \
    --baseline-manifest "${SELECTION_DIR}/seed_manifest.json" \
    --allowed-manifest "${ALLOWED_VIEW_MANIFEST}" \
    --output "${SELECTION_DIR}/candidate_screen.json" \
    --indices-output "${SELECTION_DIR}/candidate_indices.txt" \
    --iteration 30000 \
    --sample-count "${PROJECTION_SAMPLE_COUNT}" \
    --candidate-count "${CANDIDATE_VIEW_COUNT}" \
    --projection-margin "${PROJECTION_MARGIN}"

AVAILABLE_CANDIDATES="$(jq -r '.selected_candidate_count' "${SELECTION_DIR}/candidate_screen.json")"
if (( AVAILABLE_CANDIDATES < ADDITIONAL_VIEW_COUNT )); then
  echo "Only ${AVAILABLE_CANDIDATES} safe candidates remain; ${ADDITIONAL_VIEW_COUNT} are required" >&2
  exit 2
fi
CANDIDATE_CAMERA_INDICES="$(tr -d '\r\n' < "${SELECTION_DIR}/candidate_indices.txt")"
test -n "${CANDIDATE_CAMERA_INDICES}"

run_stage 03_render_reviewed_label_coverage \
  python -m scripts.task1.qa.render_auto_label_overlays \
    --model-path "${MODEL_DIR}" \
    --labels-npy "${SELECTION_LABELS}" \
    --label-map "${SELECTION_LABEL_MAP}" \
    --output-dir "${SELECTION_DIR}/candidate_overlays" \
    --rgb-output-dir "${SELECTION_DIR}/candidate_rgb" \
    --source-view-manifest "${ALLOWED_VIEW_MANIFEST}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --camera-indices "${CANDIDATE_CAMERA_INDICES}" \
    --count "${AVAILABLE_CANDIDATES}" \
    --max-labels 0 \
    --color-mode "${COLOR_MODE}"

run_stage 04_measure_visible_label_coverage \
  python -m scripts.task1.qa.measure_overlay_coverage \
    --rgb-dir "${SELECTION_DIR}/candidate_rgb" \
    --overlay-dir "${SELECTION_DIR}/candidate_overlays" \
    --output "${SELECTION_DIR}/candidate_visible_coverage.json" \
    --difference-threshold 2 \
    --exclude-border 1

run_stage 05_select_low_coverage_pose_diverse_additions \
  python -m scripts.task1.grounding.select_targeted_cameras select \
    --model-path "${MODEL_DIR}" \
    --baseline-manifest "${SELECTION_DIR}/seed_manifest.json" \
    --candidate-manifest "${SELECTION_DIR}/candidate_overlays/label_overlay_manifest.json" \
    --coverage-report "${SELECTION_DIR}/candidate_visible_coverage.json" \
    --output "${SELECTION_DIR}/targeted_camera_selection.json" \
    --indices-output "${SELECTION_DIR}/final_camera_indices.txt" \
    --additional-count "${ADDITIONAL_VIEW_COUNT}" \
    --low-coverage-quantile "${LOW_COVERAGE_QUANTILE}" \
    --coverage-weight "${COVERAGE_WEIGHT}" \
    --novelty-weight "${POSE_NOVELTY_WEIGHT}"

FINAL_CAMERA_INDICES="$(tr -d '\r\n' < "${SELECTION_DIR}/final_camera_indices.txt")"
FINAL_VIEW_COUNT="$(awk -F, '{print NF}' "${SELECTION_DIR}/final_camera_indices.txt")"
EXPECTED_VIEW_COUNT="$((BASE_VIEW_COUNT + ADDITIONAL_VIEW_COUNT))"
if [[ "${FINAL_VIEW_COUNT}" != "${EXPECTED_VIEW_COUNT}" ]]; then
  echo "Expected ${EXPECTED_VIEW_COUNT} final cameras but selected ${FINAL_VIEW_COUNT}" >&2
  exit 2
fi

{
  echo "mode=dinov3_targeted_camera_selection"
  echo "selection_policy=50_even_plus_20_low_coverage_pose_diverse"
  echo "camera_selection_used_prior_labels=1"
  echo "semantic_inference_used_prior_labels=0"
  echo "selection_label_source=${SELECTION_LABEL_ROOT}"
  echo "v5_used=0"
  echo "dinov2_used=0"
  echo "semantic_labels_written=0"
  echo "semantic_ply_written=0"
} > "${SELECTION_DIR}/experiment_mode.txt"

PROJECT_ROOT="${PROJECT_ROOT}" \
SCENE="${SCENE}" \
MODEL_DIR="${MODEL_DIR}" \
OUTPUT_NAME="${OUTPUT_NAME}" \
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT}" \
CAMERA_INDICES="${FINAL_CAMERA_INDICES}" \
VIEW_COUNT="${FINAL_VIEW_COUNT}" \
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH}" \
CAMERA_SELECTION_POLICY="50_even_plus_20_low_coverage_pose_diverse" \
CAMERA_SELECTION_USED_PRIOR_LABELS=1 \
CONFIG_ONLY=0 \
RESET_OUTPUT=0 \
bash "${PROJECT_ROOT}/scripts/slurm/slurm_task1_dinov3_3d_first_scene.sbatch"

cp "${SELECTION_DIR}/targeted_camera_selection.json" \
  "${REPORT_DIR}/targeted_camera_selection.json"
cp "${SELECTION_DIR}/experiment_mode.txt" \
  "${REPORT_DIR}/camera_selection_experiment_mode.txt"

test ! -e "${SELECTION_DIR}/gaussian_labels.npy"
if find "${SELECTION_DIR}" -type f -name '*.ply' -print -quit | grep -q .; then
  echo "Camera-selection stage unexpectedly wrote a PLY" >&2
  exit 2
fi

find "${SELECTION_DIR}" -maxdepth 3 -type f | sort
