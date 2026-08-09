#!/usr/bin/env bash
# General per-scene DINOv3 end-to-end abstention-recovery materialization.
#
# One invocation runs the full chain for any scene:
#   real-camera render -> DINOv3 Mask2Former segmentation -> dense FlashSplat
#   vote lift -> equal-camera strict-majority consensus -> leave-one-camera-out
#   reliability calibration -> calibrated weighted abstention recovery ->
#   materialized labels/label map/semantic PLY/SuperSplat PLY -> validation.
#
# The script works with or without Slurm; run it directly with
# `bash scripts/slurm/slurm_task1_dinov3_end_to_end_recovery_scene.sbatch`
# on clusters that do not run Slurm.
# Outputs are review materializations and are not auto-accepted.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
EXTERNAL_ROOT="${WORKSPACE_ROOT}/external"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${EXTERNAL_ROOT}/FlashSplat}"
DINOV3_ROOT="${DINOV3_ROOT:-${EXTERNAL_ROOT}/dinov3}"
TASK1_ROOT="${WORKSPACE_ROOT}/outputs/eyenavgs_task1"

SCENE="${SCENE:?Set SCENE to the target scene}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_dinov3_abstention_recovery_review_v1}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
VIEW_DIR="${OUT_DIR}/stages/01_real_camera_views"
VOTE_DIR="${OUT_DIR}/stages/02_flashsplat_votes"
LOG_DIR="${OUT_DIR}/logs"

GAUSSIAN_ENV="${GAUSSIAN_ENV:-semantic_3dgs_renderer}"
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
DINOV3_CHECKPOINT_LOAD_MODE="${DINOV3_CHECKPOINT_LOAD_MODE:-local_mmap}"
DINOV3_MAX_CUDA_MEMORY_GIB="${DINOV3_MAX_CUDA_MEMORY_GIB:-42}"
ONTOLOGY="${ONTOLOGY:-${PROJECT_ROOT}/configs/ade20k_to_project.json}"

CAMERA_INDICES="${CAMERA_INDICES:-}"
VIEW_COUNT="${VIEW_COUNT:-0}"
ITERATION="${ITERATION:-30000}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"
RESET_OUTPUT="${RESET_OUTPUT:-0}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"

if [[ "${CONFIG_ONLY}" != "0" && "${CONFIG_ONLY}" != "1" ]]; then
  echo "CONFIG_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ "${RESET_OUTPUT}" != "0" && "${RESET_OUTPUT}" != "1" ]]; then
  echo "RESET_OUTPUT must be 0 or 1" >&2
  exit 2
fi
if ! [[ "${VIEW_COUNT}" =~ ^[0-9]+$ ]]; then
  echo "VIEW_COUNT must be a nonnegative integer" >&2
  exit 2
fi
if ! [[ "${ITERATION}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ITERATION must be a positive integer" >&2
  exit 2
fi
if ! [[ "${RENDER_MAX_WIDTH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RENDER_MAX_WIDTH must be a positive integer" >&2
  exit 2
fi
if [[ "${DINOV3_PRECISION}" != "bfloat16" && "${DINOV3_PRECISION}" != "float32" ]]; then
  echo "DINOV3_PRECISION must be bfloat16 or float32" >&2
  exit 2
fi
if ! [[ "${DINOV3_CROP_SIZE}" =~ ^[1-9][0-9]*$ ]] || (( DINOV3_CROP_SIZE % 32 != 0 )); then
  echo "DINOV3_CROP_SIZE must be a positive multiple of 32" >&2
  exit 2
fi
if ! [[ "${DINOV3_STRIDE}" =~ ^[1-9][0-9]*$ ]] || (( DINOV3_STRIDE > DINOV3_CROP_SIZE )); then
  echo "DINOV3_STRIDE must be a positive integer no greater than CROP_SIZE" >&2
  exit 2
fi
if [[ "${DINOV3_CHECKPOINT_LOAD_MODE}" != "standard" \
   && "${DINOV3_CHECKPOINT_LOAD_MODE}" != "local_mmap" ]]; then
  echo "DINOV3_CHECKPOINT_LOAD_MODE must be standard or local_mmap" >&2
  exit 2
fi
if [[ -n "${DINOV3_MAX_CUDA_MEMORY_GIB}" \
   && ! "${DINOV3_MAX_CUDA_MEMORY_GIB}" =~ ^[1-9][0-9]*([.][0-9]+)?$ ]]; then
  echo "DINOV3_MAX_CUDA_MEMORY_GIB must be empty or a positive GiB value" >&2
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

for path in \
  "${MODEL_DIR}/cameras.json" \
  "${MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply" \
  "${FLASHSPLAT_ROOT}" \
  "${DINOV3_ROOT}" \
  "${DINOV3_BACKBONE}" \
  "${DINOV3_SEGMENTOR}" \
  "${ONTOLOGY}"; do
  test -e "${path}"
done
test "$(git -C "${DINOV3_ROOT}" rev-parse HEAD)" = "${DINOV3_EXPECTED_REPO_COMMIT}"
test -z "$(git -C "${DINOV3_ROOT}" status --porcelain --untracked-files=no)"

if [[ "${CONFIG_ONLY}" = "1" ]]; then
  printf 'scene=%s\nmodel=%s\noutput=%s\nview_count=%s\ncamera_indices=%s\nrender_max_width=%s\niteration=%s\nontology=%s\ndinov3_root=%s\ndinov3_commit=%s\ndinov3_backbone=%s\ndinov3_segmentor=%s\ndinov3_precision=%s\ndinov3_crop_size=%s\ndinov3_stride=%s\ndinov3_checkpoint_load_mode=%s\ndinov3_max_cuda_memory_gib=%s\n' \
    "${SCENE}" "${MODEL_DIR}" "${OUT_DIR}" "${VIEW_COUNT}" \
    "${CAMERA_INDICES:-<all>}" "${RENDER_MAX_WIDTH}" "${ITERATION}" \
    "${ONTOLOGY}" "${DINOV3_ROOT}" "${DINOV3_EXPECTED_REPO_COMMIT}" \
    "${DINOV3_BACKBONE}" "${DINOV3_SEGMENTOR}" "${DINOV3_PRECISION}" \
    "${DINOV3_CROP_SIZE}" "${DINOV3_STRIDE}" \
    "${DINOV3_CHECKPOINT_LOAD_MODE}" "${DINOV3_MAX_CUDA_MEMORY_GIB}"
  exit 0
fi

if [[ -e "${OUT_DIR}" ]]; then
  if [[ "${RESET_OUTPUT}" = "1" ]]; then
    case "${OUT_DIR}" in
      "${TASK1_ROOT}"/*) rm -rf -- "${OUT_DIR}" ;;
      *) echo "Refusing to reset unexpected output path: ${OUT_DIR}" >&2; exit 2 ;;
    esac
  else
    echo "Output exists; choose a new OUTPUT_NAME or set RESET_OUTPUT=1:" >&2
    echo "${OUT_DIR}" >&2
    exit 2
  fi
fi

mkdir -p "${VIEW_DIR}" "${VOTE_DIR}" "${LOG_DIR}" "${TASK1_ROOT}/logs"
cd "${PROJECT_ROOT}"

run_stage() {
  local stage_name="$1"
  local environment="$2"
  shift 2
  echo "===== ${stage_name} ====="
  conda run --no-capture-output -n "${environment}" "$@" \
    2>&1 | tee "${LOG_DIR}/${stage_name}.log"
}

{
  echo "job=${SLURM_JOB_ID:-local}"
  echo "host=$(hostname)"
  echo "project=${PROJECT_ROOT}"
  echo "scene=${SCENE}"
  echo "model=${MODEL_DIR}"
  echo "output=${OUT_DIR}"
  echo "view_count=${VIEW_COUNT}"
  echo "camera_indices=${CAMERA_INDICES:-<all>}"
  echo "render_max_width=${RENDER_MAX_WIDTH}"
  echo "iteration=${ITERATION}"
  echo "ontology=${ONTOLOGY}"
  echo "dinov3_environment=${DINOV3_ENV}"
  echo "dinov3_root=${DINOV3_ROOT}"
  echo "dinov3_commit=${DINOV3_EXPECTED_REPO_COMMIT}"
  echo "dinov3_backbone=${DINOV3_BACKBONE}"
  echo "dinov3_segmentor=${DINOV3_SEGMENTOR}"
  echo "dinov3_precision=${DINOV3_PRECISION}"
  echo "dinov3_crop_size=${DINOV3_CROP_SIZE}"
  echo "dinov3_stride=${DINOV3_STRIDE}"
  echo "dinov3_checkpoint_load_mode=${DINOV3_CHECKPOINT_LOAD_MODE}"
  echo "dinov3_max_cuda_memory_gib=${DINOV3_MAX_CUDA_MEMORY_GIB}"
  nvidia-smi
} | tee "${LOG_DIR}/00_run_configuration.log"

run_stage 01_render_real_camera_views "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov2.render_task1_views \
    --model-path "${MODEL_DIR}" \
    --output-dir "${VIEW_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --iteration "${ITERATION}" \
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
DINOV3_MEMORY_LIMIT_ARGS=()
if [[ -n "${DINOV3_MAX_CUDA_MEMORY_GIB}" ]]; then
  DINOV3_MEMORY_LIMIT_ARGS=(
    --max-cuda-memory-gib "${DINOV3_MAX_CUDA_MEMORY_GIB}"
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
    --min-pixel-confidence 0.0 \
    --device cuda \
    --precision "${DINOV3_PRECISION}" \
    --crop-size "${DINOV3_CROP_SIZE}" \
    --stride "${DINOV3_STRIDE}" \
    --checkpoint-load-mode "${DINOV3_CHECKPOINT_LOAD_MODE}" \
    "${DINOV3_MEMORY_LIMIT_ARGS[@]}" \
    "${DINOV3_HASH_ARGS[@]}" \
    --overwrite

run_stage 03_lift_per_view_votes "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.lift_dense_view_votes \
    --model-path "${MODEL_DIR}" \
    --input-dir "${VIEW_DIR}" \
    --segmentation-manifest "${VIEW_DIR}/dinov3_manifest.json" \
    --output-dir "${VOTE_DIR}" \
    --flashsplat-root "${FLASHSPLAT_ROOT}" \
    --ontology "${ONTOLOGY}" \
    --iteration "${ITERATION}" \
    --max-width "${RENDER_MAX_WIDTH}"

run_stage 04_recover_and_materialize "${GAUSSIAN_ENV}" \
  python -m scripts.task1.dinov3.recover_dinov3_abstentions_pipeline \
    --scene "${SCENE}" \
    --vote-manifest "${VOTE_DIR}/vote_manifest.json" \
    --source-ply "${MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply" \
    --ontology "${ONTOLOGY}" \
    --output-dir "${OUT_DIR}"

run_stage 05_validate_semantic_ply "${GAUSSIAN_ENV}" \
  python -m scripts.task1.qa.validate_task1_outputs \
    --labels-npy "${OUT_DIR}/gaussian_labels.npy" \
    --label-map "${OUT_DIR}/label_map.json" \
    --semantic-ply "${OUT_DIR}/semantic_point_cloud.ply"

cat > "${OUT_DIR}/experiment_mode.txt" <<EOF
mode=review_dinov3_abstention_recovery_materialization
scene=${SCENE}
source=dinov3_vit7b16_ade20k_mask2former_dense_flashsplat_votes
view_count=${VIEW_COUNT}
camera_indices=${CAMERA_INDICES:-<all>}
camera_vote_policy=one_unique_max_class_per_camera_and_gaussian_else_abstain
consensus_policy=at_least_two_cameras_unique_strict_majority_else_abstain
recovery_policy=calibrated_weighted_strict_majority_within_camera_mass
manual_camera_selection_used=0
manual_gaussian_selection_used=0
scene_specific_rules=0
class_specific_rules=0
zero_camera_fill_used=0
propagation_used=0
semantic_ply=${OUT_DIR}/semantic_point_cloud.ply
supersplat_ply=${OUT_DIR}/semantic_point_cloud_supersplat_debug.ply
EOF

for path in \
  "${OUT_DIR}/gaussian_labels.npy" \
  "${OUT_DIR}/recovery_source_codes.npy" \
  "${OUT_DIR}/original_consensus_status.npy" \
  "${OUT_DIR}/label_map.json" \
  "${OUT_DIR}/summary.json" \
  "${OUT_DIR}/semantic_point_cloud.ply" \
  "${OUT_DIR}/semantic_point_cloud_supersplat_debug.ply"; do
  test -f "${path}"
done

find "${OUT_DIR}" -maxdepth 4 -type f | sort
