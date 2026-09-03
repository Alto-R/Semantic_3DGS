#!/usr/bin/env bash
#SBATCH --job-name=task1_dense_semantic
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

# Dense-semantic (ADE20K multi-view voting) route. Replaces or complements the
# GroundingDINO+SAM front-end:
#   MODE=fill (default)  fills only base-unlabeled Gaussians with stuff labels
#                        on top of an accepted baseline (FILL_SOURCE).
#   MODE=full            labels every Gaussian from votes alone (ablation).
# Backend selection: SEG_BACKEND=mask2former (default) | dinov3.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
FLASHSPLAT_ROOT="${ROOT}/external/FlashSplat"
TASK1_ROOT="${ROOT}/outputs/eyenavgs_task1"
SCENE="${SCENE:-room}"
MODEL_DIR="${MODEL_DIR:-${ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
MODE="${MODE:-fill}"
SEG_BACKEND="${SEG_BACKEND:-mask2former}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dense_${SEG_BACKEND}_${MODE}}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
STAGE_ROOT="${OUT_DIR}/stages"
SEG_DIR="${STAGE_ROOT}/01_dense_seg"
LIFT_DIR="${STAGE_ROOT}/02_vote_lift"
FUSION_DIR="${STAGE_ROOT}/03_vote_fusion"
DELIVERABLE_DIR="${OUT_DIR}/deliverables"
VISUALIZATION_DIR="${OUT_DIR}/visualizations"
PLY_VIS_DIR="${VISUALIZATION_DIR}/ply"
SEMANTIC_OVERLAY_DIR="${VISUALIZATION_DIR}/overlays/semantic_labels"
CONTACT_DIR="${VISUALIZATION_DIR}/contact_sheets"
VALIDATION_DIR="${OUT_DIR}/validation"
RUN_LOG_DIR="${OUT_DIR}/logs"

ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.dense_backends.json}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
# Empty camera list plus VIEW_COUNT=0 uses every cameras.json camera.
SEMANTIC_CAMERA_INDICES="${SEMANTIC_CAMERA_INDICES:-}"
VIEW_COUNT="${VIEW_COUNT:-0}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.35}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-0.05}"
CLASS_BATCH_SIZE="${CLASS_BATCH_SIZE:-32}"
MIN_VIEWS="${MIN_VIEWS:-2}"
MIN_AGREEMENT="${MIN_AGREEMENT:-0.5}"
MIN_VISIBLE_RATIO="${MIN_VISIBLE_RATIO:-0.0}"
MIN_THING_GAUSSIANS="${MIN_THING_GAUSSIANS:-5000}"
MIN_STUFF_GAUSSIANS="${MIN_STUFF_GAUSSIANS:-10000}"
FILL_THINGS="${FILL_THINGS:-0}"
CONTACT_THUMB_WIDTH="${CONTACT_THUMB_WIDTH:-360}"
RESET_OUTPUT="${RESET_OUTPUT:-1}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"
REPORT_ONLY="${REPORT_ONLY:-0}"

# fill mode baseline (accepted GroundingDINO result).
FILL_SOURCE="${FILL_SOURCE:-${TASK1_ROOT}/accepted/${SCENE}}"

# mask2former backend.
MASK2FORMER_MODEL="${MASK2FORMER_MODEL:-facebook/mask2former-swin-large-ade-semantic}"

# dinov3 backend (requires a local repo clone plus downloaded checkpoints).
DINOV3_REPO="${DINOV3_REPO:-${ROOT}/external/dinov3}"
DINOV3_BACKBONE_WEIGHTS="${DINOV3_BACKBONE_WEIGHTS:-${ROOT}/data/models/dinov3/dinov3_vit7b16_pretrain_lvd1689m.pth}"
DINOV3_SEGMENTOR_WEIGHTS="${DINOV3_SEGMENTOR_WEIGHTS:-${ROOT}/data/models/dinov3/dinov3_vit7b16_ade20k_m2f_head.pth}"
DINOV3_HUB_ENTRY="${DINOV3_HUB_ENTRY:-dinov3_vit7b16_ms}"

TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TOKENIZERS_PARALLELISM

test -d "${MODEL_DIR}"
test -f "${MODEL_DIR}/cameras.json"
test -f "${ONTOLOGY}"

if [ "${MODE}" != "full" ] && [ "${MODE}" != "fill" ]; then
  echo "MODE must be full or fill, got: ${MODE}" >&2
  exit 2
fi
if [ "${REPORT_ONLY}" != "0" ] && [ "${REPORT_ONLY}" != "1" ]; then
  echo "REPORT_ONLY must be 0 or 1, got: ${REPORT_ONLY}" >&2
  exit 2
fi

# Baseline layouts differ across pipeline generations (GroundingDINO runs use
# stages/03_semantic_fusion, the DINOv2 production route uses
# stages/03_exact_fusion), so BASE_LABELS/BASE_LABEL_MAP are individually
# overridable and only default to the FILL_SOURCE convention.
BASE_LABELS="${BASE_LABELS:-}"
BASE_LABEL_MAP="${BASE_LABEL_MAP:-}"
if [ "${MODE}" = "fill" ] && [ "${REPORT_ONLY}" != "1" ]; then
  if [ -z "${BASE_LABELS}" ]; then
    for candidate in       "${FILL_SOURCE}/stages/03_semantic_fusion/gaussian_labels.npy"       "${FILL_SOURCE}/stages/03_exact_fusion/gaussian_labels.npy"       "${FILL_SOURCE}/deliverables/gaussian_labels.npy"; do
      if [ -f "${candidate}" ]; then
        BASE_LABELS="${candidate}"
        break
      fi
    done
  fi
  BASE_LABEL_MAP="${BASE_LABEL_MAP:-${FILL_SOURCE}/deliverables/label_map.json}"
  test -f "${BASE_LABELS}"
  test -f "${BASE_LABEL_MAP}"
fi

if [ "${SEG_BACKEND}" = "dinov3" ]; then
  test -d "${DINOV3_REPO}"
  test -f "${DINOV3_BACKBONE_WEIGHTS}"
  test -f "${DINOV3_SEGMENTOR_WEIGHTS}"
fi

if [ "${CONFIG_ONLY}" = "1" ]; then
  printf 'scene=%s\nmodel=%s\noutput=%s\nmode=%s\nbackend=%s\nontology=%s\nview_count=%s\nfill_source=%s\nreport_only=%s\n' \
    "${SCENE}" "${MODEL_DIR}" "${OUT_DIR}" "${MODE}" "${SEG_BACKEND}" \
    "${ONTOLOGY}" "${VIEW_COUNT}" "${FILL_SOURCE:-<none>}" "${REPORT_ONLY}"
  exit 0
fi

if [ "${RESET_OUTPUT}" = "1" ] && [ -d "${OUT_DIR}" ]; then
  case "${OUT_DIR}" in
    "${TASK1_ROOT}"/*) rm -rf -- "${OUT_DIR}" ;;
    *) echo "Refusing to reset unexpected output path: ${OUT_DIR}" >&2; exit 2 ;;
  esac
fi

mkdir -p \
  "${SEG_DIR}" \
  "${LIFT_DIR}" \
  "${FUSION_DIR}" \
  "${DELIVERABLE_DIR}" \
  "${PLY_VIS_DIR}" \
  "${SEMANTIC_OVERLAY_DIR}" \
  "${CONTACT_DIR}" \
  "${VALIDATION_DIR}" \
  "${RUN_LOG_DIR}" \
  "${TASK1_ROOT}/logs"

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

{
  echo "job=${SLURM_JOB_ID:-local}"
  echo "host=$(hostname)"
  echo "project=${PROJECT_ROOT}"
  echo "scene=${SCENE}"
  echo "model=${MODEL_DIR}"
  echo "output=${OUT_DIR}"
  echo "mode=${MODE}"
  echo "seg_backend=${SEG_BACKEND}"
  echo "ontology=${ONTOLOGY}"
  echo "render_max_width=${RENDER_MAX_WIDTH}"
  echo "semantic_camera_indices=${SEMANTIC_CAMERA_INDICES:-<all-or-evenly-spaced>}"
  echo "view_count=${VIEW_COUNT}"
  echo "min_confidence=${MIN_CONFIDENCE}"
  echo "support_threshold=${SUPPORT_THRESHOLD}"
  echo "class_batch_size=${CLASS_BATCH_SIZE}"
  echo "min_views=${MIN_VIEWS}"
  echo "min_agreement=${MIN_AGREEMENT}"
  echo "min_visible_ratio=${MIN_VISIBLE_RATIO}"
  echo "min_thing_gaussians=${MIN_THING_GAUSSIANS}"
  echo "min_stuff_gaussians=${MIN_STUFF_GAUSSIANS}"
  echo "fill_source=${FILL_SOURCE:-<none>}"
  echo "fill_things=${FILL_THINGS}"
  echo "report_only=${REPORT_ONLY}"
  echo "mask2former_model=${MASK2FORMER_MODEL}"
  echo "dinov3_repo=${DINOV3_REPO}"
  nvidia-smi
} | tee "${RUN_LOG_DIR}/00_run_configuration.log"

SEG_BACKEND_ARGS=()
if [ "${SEG_BACKEND}" = "mask2former" ]; then
  SEG_BACKEND_ARGS+=(--mask2former-model "${MASK2FORMER_MODEL}")
else
  SEG_BACKEND_ARGS+=(
    --dinov3-repo "${DINOV3_REPO}"
    --dinov3-backbone-weights "${DINOV3_BACKBONE_WEIGHTS}"
    --dinov3-segmentor-weights "${DINOV3_SEGMENTOR_WEIGHTS}"
    --dinov3-hub-entry "${DINOV3_HUB_ENTRY}"
  )
fi

run_stage 01_dense_seg \
  python -m scripts.task1.dense_seg.segment_views_semantic \
    --model-path "${MODEL_DIR}" \
    --output-dir "${SEG_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --ontology "${ONTOLOGY}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --camera-indices "${SEMANTIC_CAMERA_INDICES}" \
    --count "${VIEW_COUNT}" \
    --backend "${SEG_BACKEND}" \
    "${SEG_BACKEND_ARGS[@]}"

if [ "${REPORT_ONLY}" = "1" ]; then
  run_stage 02_dense_seg_contact_sheet \
    python -m scripts.task1.qa.make_contact_sheet \
      --image-dir "${SEG_DIR}/overlays" \
      --output "${CONTACT_DIR}/dense_seg.png" \
      --columns 5 \
      --thumb-width "${CONTACT_THUMB_WIDTH}"

  {
    echo "mode=report_only_dense_seg"
    echo "backend=${SEG_BACKEND}"
    echo "view_count=${VIEW_COUNT}"
    echo "semantic_labels_written=0"
    echo "semantic_ply_written=0"
  } > "${OUT_DIR}/experiment_mode.txt"

  find "${OUT_DIR}" -maxdepth 4 -type f | sort
  exit 0
fi

# Later stages need an explicit camera list; recover the exact cameras that
# stage 01 actually segmented (VIEW_COUNT=0 means "all cameras", which
# render_auto_label_overlays cannot express directly).
if [ -z "${SEMANTIC_CAMERA_INDICES}" ]; then
  command -v jq >/dev/null 2>&1
  SEMANTIC_CAMERA_INDICES="$(jq -r '[.frames[].camera_index] | map(tostring) | join(",")' "${SEG_DIR}/dense_seg_manifest.json")"
  VIEW_COUNT="$(jq -r '.frames | length' "${SEG_DIR}/dense_seg_manifest.json")"
  test -n "${SEMANTIC_CAMERA_INDICES}"
fi

run_stage 02_vote_lift \
  python -m scripts.task1.dense_seg.lift_semantic_votes \
    --model-path "${MODEL_DIR}" \
    --seg-dir "${SEG_DIR}" \
    --output-dir "${LIFT_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --ontology "${ONTOLOGY}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --min-confidence "${MIN_CONFIDENCE}" \
    --support-threshold "${SUPPORT_THRESHOLD}" \
    --class-batch-size "${CLASS_BATCH_SIZE}"

FUSE_ARGS=(
  --votes "${LIFT_DIR}/votes.npz"
  --model-path "${MODEL_DIR}"
  --iteration 30000
  --scene "${SCENE}"
  --mode "${MODE}"
  --min-views "${MIN_VIEWS}"
  --min-agreement "${MIN_AGREEMENT}"
  --min-visible-ratio "${MIN_VISIBLE_RATIO}"
  --min-thing-gaussians "${MIN_THING_GAUSSIANS}"
  --min-stuff-gaussians "${MIN_STUFF_GAUSSIANS}"
  --labels-path "${FUSION_DIR}/gaussian_labels.npy"
  --label-map-path "${DELIVERABLE_DIR}/label_map.json"
  --semantic-ply-path "${DELIVERABLE_DIR}/semantic_point_cloud.ply"
  --summary-path "${FUSION_DIR}/vote_summary.json"
  --overwrite
)
if [ "${MODE}" = "fill" ]; then
  FUSE_ARGS+=(--base-labels "${BASE_LABELS}" --base-label-map "${BASE_LABEL_MAP}")
  if [ "${FILL_THINGS}" = "1" ]; then
    FUSE_ARGS+=(--fill-things)
  fi
fi

run_stage 03_vote_fusion \
  python -m scripts.task1.dense_seg.fuse_semantic_votes "${FUSE_ARGS[@]}"

run_report \
  "${VALIDATION_DIR}/semantic_point_cloud_inspection.json" \
  04_semantic_ply_inspection \
  python -m scripts.task1.qa.inspect_ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" --json

run_stage 05_supersplat_debug_ply \
  python -m scripts.task1.qa.export_supersplat_label_colors \
    --input-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-ply "${PLY_VIS_DIR}/semantic_point_cloud_supersplat_debug.ply" \
    --metadata-json "${PLY_VIS_DIR}/semantic_point_cloud_supersplat_debug.json" \
    --overwrite

run_stage 06_semantic_overlays \
  python -m scripts.task1.qa.render_auto_label_overlays \
    --model-path "${MODEL_DIR}" \
    --labels-npy "${FUSION_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --output-dir "${SEMANTIC_OVERLAY_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration 30000 \
    --max-width "${RENDER_MAX_WIDTH}" \
    --camera-indices "${SEMANTIC_CAMERA_INDICES}" \
    --count "${VIEW_COUNT}"

run_stage 07_visible_coverage \
  python -m scripts.task1.qa.measure_overlay_coverage \
    --rgb-dir "${SEG_DIR}/rgb_renders" \
    --overlay-dir "${SEMANTIC_OVERLAY_DIR}" \
    --output "${VALIDATION_DIR}/visible_overlay_coverage.json" \
    --difference-threshold 2 \
    --exclude-border 1

run_stage 08_dense_seg_contact_sheet \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${SEG_DIR}/overlays" \
    --output "${CONTACT_DIR}/dense_seg.png" \
    --columns 5 \
    --thumb-width "${CONTACT_THUMB_WIDTH}"

run_stage 09_semantic_contact_sheet \
  python -m scripts.task1.qa.make_contact_sheet \
    --image-dir "${SEMANTIC_OVERLAY_DIR}" \
    --output "${CONTACT_DIR}/semantic_labels.png" \
    --columns 5 \
    --thumb-width "${CONTACT_THUMB_WIDTH}"

run_report \
  "${VALIDATION_DIR}/task1_validation.json" \
  10_task1_validation \
  python -m scripts.task1.qa.validate_task1_outputs \
    --labels-npy "${FUSION_DIR}/gaussian_labels.npy" \
    --label-map "${DELIVERABLE_DIR}/label_map.json" \
    --semantic-ply "${DELIVERABLE_DIR}/semantic_point_cloud.ply" \
    --visible-coverage "${VALIDATION_DIR}/visible_overlay_coverage.json"

find "${OUT_DIR}" -maxdepth 4 -type f | sort
