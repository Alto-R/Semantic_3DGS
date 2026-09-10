#!/usr/bin/env bash
#SBATCH --job-name=task1_sam3_instance
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

# Per-scene SAM3 instance route: concept segmentation -> per-concept
# FlashSplat lift -> cross-view association -> per-instance consensus ->
# hierarchy/scene graph -> QA overlays.
#
# The route consumes an existing render stage (all-camera RGB views plus the
# render manifest with camera indices), for example stage 01 of the DINOv3
# end-to-end pipeline. It never re-renders and never touches DINOv3 outputs.
#
# The script works with or without Slurm; run it directly with
# `bash scripts/slurm/slurm_task1_sam3_instance_scene.sbatch` on clusters
# that do not run Slurm. Outputs are pilot review materializations and are
# not auto-accepted. Cluster verification of the SAM3 checkpoint is pending;
# see docs/EXTERNAL_REPOS.md.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd -P)}"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
TASK1_ROOT="${TASK1_ROOT:-${WORKSPACE_ROOT}/outputs/eyenavgs_task1}"

SCENE="${SCENE:?Set SCENE to the target scene}"
RGB_DIR="${RGB_DIR:?Set RGB_DIR to the rendered all-camera RGB views}"
RENDER_MANIFEST="${RENDER_MANIFEST:?Set RENDER_MANIFEST to the render-stage manifest}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}}"
OUTPUT_NAME="${OUTPUT_NAME:-${SCENE}_sam3_instance_pilot_v1}"
OUT_DIR="${TASK1_ROOT}/${OUTPUT_NAME}"
LOG_DIR="${OUT_DIR}/logs"

VOCABULARY="${VOCABULARY:-${PROJECT_ROOT}/configs/task1_sam3_vocabulary.${SCENE}.json}"
FLASHSPLAT_ROOT="${FLASHSPLAT_ROOT:-${WORKSPACE_ROOT}/external/FlashSplat}"
ITERATION="${ITERATION:-30000}"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-960}"

SAM3_ENV="${SAM3_ENV:-sam3_semantic}"
GAUSSIAN_ENV="${GAUSSIAN_ENV:-semantic_3dgs_renderer}"
SAM3_MODEL_ID="${SAM3_MODEL_ID:-facebook/sam3}"
SAM3_MODEL_REVISION="${SAM3_MODEL_REVISION:?Set SAM3_MODEL_REVISION to the pinned checkpoint revision}"
SAM3_MIN_SCORE="${SAM3_MIN_SCORE:-0.4}"
ASSOCIATION_THRESHOLD="${ASSOCIATION_THRESHOLD:-0.3}"
MIN_WEIGHT="${MIN_WEIGHT:-0.5}"
# Set OVERWRITE=1 to rerun into an existing output root after a failed run.
OVERWRITE_FLAG="${OVERWRITE:+--overwrite}"

S1_DIR="${OUT_DIR}/stages/01_sam3_masks"
S2_DIR="${OUT_DIR}/stages/02_mask_votes"
S3_REGISTRY="${OUT_DIR}/stages/03_instance_registry.json"
S4_DIR="${OUT_DIR}/stages/04_membership"
S5_DIR="${OUT_DIR}/stages/05_scene_graph"
QA_DIR="${OUT_DIR}/visualizations"

mkdir -p "${LOG_DIR}"

run_stage() {
  local stage_name="$1"
  local environment="$2"
  shift 2
  conda run --no-capture-output -n "${environment}" "$@" \
    2>&1 | tee "${LOG_DIR}/${stage_name}.log"
}

{
  echo "scene=${SCENE}"
  echo "rgb_dir=${RGB_DIR}"
  echo "render_manifest=${RENDER_MANIFEST}"
  echo "model_dir=${MODEL_DIR}"
  echo "output=${OUT_DIR}"
  echo "vocabulary=${VOCABULARY}"
  echo "sam3_model=${SAM3_MODEL_ID}@${SAM3_MODEL_REVISION}"
  echo "min_score=${SAM3_MIN_SCORE} threshold=${ASSOCIATION_THRESHOLD} min_weight=${MIN_WEIGHT}"
} | tee "${LOG_DIR}/00_run_configuration.log"

run_stage 01_sam3_segment_views "${SAM3_ENV}" \
  python -m scripts.task1.sam3.sam3_segment_views \
  --rgb-dir "${RGB_DIR}" \
  --vocabulary "${VOCABULARY}" \
  --output-dir "${S1_DIR}" \
  --backend transformers \
  --model-id "${SAM3_MODEL_ID}" \
  --model-revision "${SAM3_MODEL_REVISION}" \
  --min-score "${SAM3_MIN_SCORE}" ${OVERWRITE_FLAG}

run_stage 02_lift_mask_votes "${GAUSSIAN_ENV}" \
  python -m scripts.task1.sam3.lift_mask_view_votes \
  --model-path "${MODEL_DIR}" \
  --masks-manifest "${S1_DIR}/sam3_masks_manifest.json" \
  --masks-dir "${S1_DIR}/masks" \
  --render-manifest "${RENDER_MANIFEST}" \
  --output-dir "${S2_DIR}" \
  --flashsplat-root "${FLASHSPLAT_ROOT}" \
  --iteration "${ITERATION}" \
  --max-width "${RENDER_MAX_WIDTH}" ${OVERWRITE_FLAG}

run_stage 03_associate_instances "${GAUSSIAN_ENV}" \
  python -m scripts.task1.sam3.associate_instances \
  --masks-manifest "${S1_DIR}/sam3_masks_manifest.json" \
  --votes-manifest "${S2_DIR}/sam3_vote_manifest.json" \
  --output "${S3_REGISTRY}" \
  --threshold "${ASSOCIATION_THRESHOLD}" ${OVERWRITE_FLAG}

run_stage 04_instance_membership "${GAUSSIAN_ENV}" \
  python -m scripts.task1.sam3.instance_membership \
  --masks-manifest "${S1_DIR}/sam3_masks_manifest.json" \
  --votes-manifest "${S2_DIR}/sam3_vote_manifest.json" \
  --registry "${S3_REGISTRY}" \
  --output-dir "${S4_DIR}" \
  --min-weight "${MIN_WEIGHT}" ${OVERWRITE_FLAG}

run_stage 05_scene_graph "${GAUSSIAN_ENV}" \
  python -m scripts.task1.sam3.instance_hierarchy \
  --membership "${S4_DIR}/membership.npz" \
  --registry "${S3_REGISTRY}" \
  --source-ply "${MODEL_DIR}/point_cloud/iteration_${ITERATION}/point_cloud.ply" \
  --vocabulary "${VOCABULARY}" \
  --output-dir "${S5_DIR}" ${OVERWRITE_FLAG}

run_stage 06_qa_overlays "${GAUSSIAN_ENV}" \
  python -m scripts.task1.sam3.render_instance_overlays \
  --masks-manifest "${S1_DIR}/sam3_masks_manifest.json" \
  --masks-dir "${S1_DIR}/masks" \
  --rgb-dir "${RGB_DIR}" \
  --output-dir "${QA_DIR}" ${OVERWRITE_FLAG}

echo "SAM3 instance route complete: ${OUT_DIR}"
