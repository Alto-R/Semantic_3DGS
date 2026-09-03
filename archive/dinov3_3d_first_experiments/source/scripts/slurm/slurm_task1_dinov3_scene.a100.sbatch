#!/usr/bin/env bash
#SBATCH --job-name=task1_dinov3_scene
#SBATCH --account=gpulab02
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=../../outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=../../outputs/eyenavgs_task1/logs/%x-%j.err

# Maintained Task 1 route with DINOv3 replacing only the 2D ADE20K producer:
#   FlashSplat render -> DINOv3 ADE20K M2F -> exact vote lift -> exact fusion.
# REPORT_ONLY=1 is the safe default and exits after 2D overlays/contact sheets.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
EXTERNAL_ROOT="${WORKSPACE_ROOT}/external"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${EXTERNAL_ROOT}/FlashSplat}"
DINOV3_ROOT="${DINOV3_ROOT:-${EXTERNAL_ROOT}/dinov3}"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:-room}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_ade20k_audit_v1}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
VIEW_DIR="${OUT_DIR}/stages/01_real_camera_views"
VOTE_DIR="${OUT_DIR}/stages/02_flashsplat_votes"
FUSION_DIR="${OUT_DIR}/stages/03_exact_fusion"
DELIVERABLE_DIR="${OUT_DIR}/deliverables"
VISUALIZATION_DIR="${OUT_DIR}/visualizations"
CONTACT_DIR="${VISUALIZATION_DIR}/contact_sheets"
QA_RGB_DIR="${VISUALIZATION_DIR}/qa_rgb"
QA_OVERLAY_DIR="${VISUALIZATION_DIR}/semantic_overlays"
VALIDATION_DIR="${OUT_DIR}/validation"
RUN_LOG_DIR="${OUT_DIR}/logs"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"
DINOV3_ENV="${DINOV3_ENV:-dinov3_semantic}"
DINOV3_BACKBONE="${DINOV3_BACKBONE:-${WORKSPACE_ROOT}/data/models/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth}"
DINOV3_SEGMENTOR="${DINOV3_SEGMENTOR:-${WORKSPACE_ROOT}/data/models/dinov3/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth}"
DINOV3_HUB_ENTRY="${DINOV3_HUB_ENTRY:-dinov3_vit7b16_ms}"
DINOV3_EXPECTED_REPO_COMMIT="${DINOV3_EXPECTED_REPO_COMMIT:-6876159a11b4df116f30f667f8c9888617df0751}"
DINOV3_EXPECTED_BACKBONE_SHA256="${DINOV3_EXPECTED_BACKBONE_SHA256:-a955f4ea3bec4fcd666bf363630da4386383069b482c8a927e17a3e1154965b7}"
DINOV3_EXPECTED_SEGMENTOR_SHA256="${DINOV3_EXPECTED_SEGMENTOR_SHA256:-bf307cb1c2fd95046feb1bf9a8a13dae60a746bddd8f5297134da95525dbcb42}"
DINOV3_PRECISION="${DINOV3_PRECISION:-bfloat16}"
DINOV3_CROP_SIZE="${DINOV3_CROP_SIZE:-896}"
DINOV3_STRIDE="${DINOV3_STRIDE:-596}"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"

CAMERA_INDICES="${CAMERA_INDICES:-}"
VIEW_COUNT="${VIEW_COUNT:-1}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
QA_VIEW_COUNT="${QA_VIEW_COUNT:-12}"
MIN_PIXEL_CONFIDENCE="${MIN_PIXEL_CONFIDENCE:-0.0}"
FLASHSPLAT_SUPPORT_THRESHOLD="${FLASHSPLAT_SUPPORT_THRESHOLD:-0.05}"
MIN_VIEWS="${MIN_VIEWS:-2}"
MIN_AGREEMENT="${MIN_AGREEMENT:-0.50}"
FUSION_MODE="${FUSION_MODE:-separate_abstain}"
MIN_SEMANTIC_EVIDENCE="${MIN_SEMANTIC_EVIDENCE:-0.50}"
MIN_ASSIGNED_GAUSSIANS="${MIN_ASSIGNED_GAUSSIANS:-0}"
MIN_ASSIGNED_THING_GAUSSIANS="${MIN_ASSIGNED_THING_GAUSSIANS:-5000}"
MIN_ASSIGNED_STUFF_GAUSSIANS="${MIN_ASSIGNED_STUFF_GAUSSIANS:-10000}"
ADAPTIVE_STUFF_MIN_VIEW_RATIO="${ADAPTIVE_STUFF_MIN_VIEW_RATIO:-0.50}"
ADAPTIVE_STUFF_THRESHOLD_RATIO="${ADAPTIVE_STUFF_THRESHOLD_RATIO:-0.75}"
ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO="${ADAPTIVE_THING_THRESHOLD_FLOOR_RATIO:-0.50}"
BASELINE_LABELS="${BASELINE_LABELS:-}"
REPORT_ONLY="${REPORT_ONLY:-1}"
WRITE_SEMANTIC_PLY="${WRITE_SEMANTIC_PLY:-0}"
RESET_OUTPUT="${RESET_OUTPUT:-0}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

if [[ "${REPORT_ONLY}" != "0" && "${REPORT_ONLY}" != "1" ]]; then
  echo "REPORT_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ "${WRITE_SEMANTIC_PLY}" != "0" && "${WRITE_SEMANTIC_PLY}" != "1" ]]; then
  echo "WRITE_SEMANTIC_PLY must be 0 or 1" >&2
  exit 2
fi
if [[ "${RESET_OUTPUT}" != "0" && "${RESET_OUTPUT}" != "1" ]]; then
  echo "RESET_OUTPUT must be 0 or 1" >&2
  exit 2
fi
if [[ "${DINOV3_PRECISION}" != "bfloat16" && "${DINOV3_PRECISION}" != "float32" ]]; then
  echo "DINOV3_PRECISION must be bfloat16 or float32" >&2
  exit 2
fi
if [[ "${FUSION_MODE}" != "joint" \
   && "${FUSION_MODE}" != "semantic_only" \
   && "${FUSION_MODE}" != "separate_abstain" ]]; then
  echo "FUSION_MODE must be joint, semantic_only, or separate_abstain" >&2
  exit 2
fi
if ! [[ "${DINOV3_EXPECTED_REPO_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "DINOV3_EXPECTED_REPO_COMMIT must be a full 40-character commit" >&2
  exit 2
fi
for expected_sha in \
  "${DINOV3_EXPECTED_BACKBONE_SHA256}" \
  "${DINOV3_EXPECTED_SEGMENTOR_SHA256}"; do
  if [[ -n "${expected_sha}" && ! "${expected_sha}" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "Expected checkpoint SHA256 values must contain 64 hexadecimal characters" >&2
    exit 2
  fi
done

test -d "${MODEL_DIR}"
test -f "${MODEL_DIR}/cameras.json"
test -d "${FLASHSPLAT_ROOT}"
test -d "${DINOV3_ROOT}"
test -f "${DINOV3_BACKBONE}"
test -f "${DINOV3_SEGMENTOR}"
test -f "${ONTOLOGY}"
test "$(git -C "${DINOV3_ROOT}" rev-parse HEAD)" = "${DINOV3_EXPECTED_REPO_COMMIT}"
test -z "$(git -C "${DINOV3_ROOT}" status --porcelain --untracked-files=no)"
if [[ -n "${BASELINE_LABELS}" ]]; then
  test -f "${BASELINE_LABELS}"
fi

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nmodel=%s\noutput=%s\nreport_only=%s\nwrite_semantic_ply=%s\nview_count=%s\nontology=%s\ndinov3_root=%s\ndinov3_commit=%s\ndinov3_backbone=%s\ndinov3_segmentor=%s\ndinov3_precision=%s\ndinov3_crop_size=%s\ndinov3_stride=%s\n' \
    "${SCENE}" "${MODEL_DIR}" "${OUT_DIR}" "${REPORT_ONLY}" \
    "${WRITE_SEMANTIC_PLY}" "${VIEW_COUNT}" "${ONTOLOGY}" \
    "${DINOV3_ROOT}" "${DINOV3_EXPECTED_REPO_COMMIT}" \
    "${DINOV3_BACKBONE}" "${DINOV3_SEGMENTOR}" "${DINOV3_PRECISION}" \
    "${DINOV3_CROP_SIZE}" "${DINOV3_STRIDE}"
  exit 0
fi

if [[ -d "${OUT_DIR}" ]]; then
  if [[ "${RESET_OUTPUT}" = "1" ]]; then
    case "${OUT_DIR}" in
      "${TASK1_ROOT}"/*) rm -rf -- "${OUT_DIR}" ;;
      *) echo "Refusing to reset unexpected output path: ${OUT_DIR}" >&2; exit 2 ;;
    esac
  else
    echo "Output exists; choose a new OUTPUT_NAME or explicitly set RESET_OUTPUT=1:" >&2
    echo "${OUT_DIR}" >&2
    exit 2
  fi
fi

mkdir -p \
  "${VIEW_DIR}" "${VOTE_DIR}" "${FUSION_DIR}" "${DELIVERABLE_DIR}" \
  "${CONTACT_DIR}" "${QA_RGB_DIR}" "${QA_OVERLAY_DIR}" \
  "${VALIDATION_DIR}" "${RUN_LOG_DIR}" "${TASK1_ROOT}/logs"

cd "${PROJECT_ROOT}"

run_stage() {
  local stage_name="$1"
  local environment="$2"
  shift 2
  echo "===== ${stage_name} ====="
  conda run --no-capture-output -n "${environment}" "$@" \
    2>&1 | tee "${RUN_LOG_DIR}/${stage_name}.log"
}

{
  echo "job=${SLURM_JOB_ID:-local}"
  echo "host=$(hostname)"
  echo "project=${PROJECT_ROOT}"
  echo "scene=${SCENE}"
  echo "model=${MODEL_DIR}"
  echo "output=${OUT_DIR}"
  echo "report_only=${REPORT_ONLY}"
  echo "write_semantic_ply=${WRITE_SEMANTIC_PLY}"
  echo "camera_indices=${CAMERA_INDICES:-<evenly-spaced>}"
  echo "view_count=${VIEW_COUNT}"
  echo "render_max_width=${RENDER_MAX_WIDTH}"
  echo "ontology=${ONTOLOGY}"
  echo "dinov3_environment=${DINOV3_ENV}"
  echo "dinov3_root=${DINOV3_ROOT}"
  echo "dinov3_commit=${DINOV3_EXPECTED_REPO_COMMIT}"
  echo "dinov3_backbone=${DINOV3_BACKBONE}"
  echo "dinov3_segmentor=${DINOV3_SEGMENTOR}"
  echo "dinov3_precision=${DINOV3_PRECISION}"
  echo "dinov3_crop_size=${DINOV3_CROP_SIZE}"
  echo "dinov3_stride=${DINOV3_STRIDE}"
  nvidia-smi
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

run_stage 01_render_real_camera_views "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov2.render_task1_views \
    --model-path "${MODEL_DIR}" \
    --output-dir "${VIEW_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --camera-indices "${CAMERA_INDICES}" \
    --count "${VIEW_COUNT}" \
    --overwrite

DINOV3_HASH_ARGS=()
if [[ -n "${DINOV3_EXPECTED_BACKBONE_SHA256}" ]]; then
  DINOV3_HASH_ARGS+=(
    --expected-backbone-sha256 "${DINOV3_EXPECTED_BACKBONE_SHA256}"
  )
fi
if [[ -n "${DINOV3_EXPECTED_SEGMENTOR_SHA256}" ]]; then
  DINOV3_HASH_ARGS+=(
    --expected-segmentor-sha256 "${DINOV3_EXPECTED_SEGMENTOR_SHA256}"
  )
fi

run_stage 02_dinov3_ade20k_mask2former "${DINOV3_ENV}" \
  python -m scripts.task1.dinov3.dinov3_segment_views \
    --input-dir "${VIEW_DIR}" \
    --dinov3-root "${DINOV3_ROOT}" \
    --backbone-checkpoint "${DINOV3_BACKBONE}" \
    --segmentor-checkpoint "${DINOV3_SEGMENTOR}" \
    --hub-entry "${DINOV3_HUB_ENTRY}" \
    --expected-repo-commit "${DINOV3_EXPECTED_REPO_COMMIT}" \
    --ontology "${ONTOLOGY}" \
    --min-pixel-confidence "${MIN_PIXEL_CONFIDENCE}" \
    --device cuda \
    --precision "${DINOV3_PRECISION}" \
    --crop-size "${DINOV3_CROP_SIZE}" \
    --stride "${DINOV3_STRIDE}" \
    "${DINOV3_HASH_ARGS[@]}" \
    --overwrite

run_stage 03_rgb_contact_sheet "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${VIEW_DIR}/rgb_renders" \
    --output "${CONTACT_DIR}/rgb_views.png" \
    --columns 4 \
    --thumb-width 360

run_stage 04_dinov3_contact_sheet "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${VIEW_DIR}/dinov3_overlays" \
    --output "${CONTACT_DIR}/dinov3_ade20k.png" \
    --columns 4 \
    --thumb-width 360

if [[ "${REPORT_ONLY}" = "1" ]]; then
  {
    echo "mode=report_only_dinov3_2d"
    echo "source=dinov3_vit7b16_ade20k_mask2former"
    echo "view_count=${VIEW_COUNT}"
    echo "semantic_vote_lifting_run=0"
    echo "semantic_labels_written=0"
    echo "semantic_ply_written=0"
  } > "${OUT_DIR}/experiment_mode.txt"
  test ! -e "${FUSION_DIR}/gaussian_labels.npy"
  test ! -e "${DELIVERABLE_DIR}/semantic_point_cloud.ply"
  find "${OUT_DIR}" -maxdepth 4 -type f | sort
  exit 0
fi

run_stage 05_lift_per_view_votes "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov2.lift_dinov2_view_votes \
    --model-path "${MODEL_DIR}" \
    --input-dir "${VIEW_DIR}" \
    --segmentation-manifest "${VIEW_DIR}/dinov3_manifest.json" \
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
SEMANTIC_PLY_ARGS=(--no-semantic-ply)
if [[ "${WRITE_SEMANTIC_PLY}" = "1" ]]; then
  SEMANTIC_PLY_ARGS=(
    --semantic-ply-path "${DELIVERABLE_DIR}/semantic_point_cloud.ply"
  )
fi

run_stage 06_exact_multiview_fusion "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov2.fuse_dinov2_multiview_votes \
    --model-path "${MODEL_DIR}" \
    --vote-dir "${VOTE_DIR}" \
    --output-dir "${FUSION_DIR}" \
    --ontology "${ONTOLOGY}" \
    --iteration 30000 \
    --scene "${SCENE}" \
    --min-views "${MIN_VIEWS}" \
    --min-agreement "${MIN_AGREEMENT}" \
    --fusion-mode "${FUSION_MODE}" \
    --min-semantic-evidence "${MIN_SEMANTIC_EVIDENCE}" \
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
    "${SEMANTIC_PLY_ARGS[@]}" \
    --overwrite

cp "${FUSION_DIR}/gaussian_labels.npy" "${DELIVERABLE_DIR}/gaussian_labels.npy"
cp "${FUSION_DIR}/label_map.json" "${DELIVERABLE_DIR}/label_map.json"

run_stage 07_semantic_overlays "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.render_auto_label_overlays \
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
    --color-mode class

run_stage 08_semantic_contact_sheet "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${QA_OVERLAY_DIR}" \
    --output "${CONTACT_DIR}/semantic_labels.png" \
    --columns 4 \
    --thumb-width 360

{
  echo "mode=dinov3_exact_multiview"
  echo "source=dinov3_vit7b16_ade20k_mask2former"
  echo "view_count=${VIEW_COUNT}"
  echo "semantic_vote_lifting_run=1"
  echo "semantic_labels_written=1"
  echo "semantic_ply_written=${WRITE_SEMANTIC_PLY}"
} > "${OUT_DIR}/experiment_mode.txt"

find "${OUT_DIR}" -maxdepth 4 -type f | sort
