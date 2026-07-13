#!/usr/bin/env bash
#SBATCH --job-name=task1_bicycle_pilot
#SBATCH --account=gpulab02
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH --nodelist=l40gpu002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/logs/%x-%j.out
#SBATCH --error=/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/logs/%x-%j.err

set -euo pipefail

ROOT=/lab/haoq_lab/cse12312032
PROJECT_ROOT="${ROOT}/projects/pku-3dgs-vr"
GRAPHDECO_ROOT="${ROOT}/external/gaussian-splatting"
MODEL_DIR="${ROOT}/data/3dgs_models/graphdeco/bicycle"
PLY="${MODEL_DIR}/point_cloud/iteration_30000/point_cloud.ply"
OUT_DIR="${ROOT}/outputs/eyenavgs_task1/bicycle"
RENDER_MAX_WIDTH="${RENDER_MAX_WIDTH:-320}"
RENDER_COUNT="${RENDER_COUNT:-20}"
RENDER_CAMERA_INDICES="${RENDER_CAMERA_INDICES:-41,43,148,70,115,68,64,87,69,39,85,120,42,153,123,88,37,40,7,111}"

mkdir -p "${OUT_DIR}/rgb_renders" "${OUT_DIR}/overlay_renders" "${ROOT}/outputs/eyenavgs_task1/logs"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

conda activate gaussian_grouping_true

cd "${PROJECT_ROOT}"

echo "job=${SLURM_JOB_ID:-local}"
echo "host=$(hostname)"
echo "project=${PROJECT_ROOT}"
echo "model=${MODEL_DIR}"
nvidia-smi

python scripts/task1/inspect_ply.py "${PLY}" --json > "${OUT_DIR}/point_cloud_inspection.json"

if [ ! -f "${OUT_DIR}/semantic_point_cloud.ply" ]; then
  python archive/task1_legacy/scripts/add_dummy_labels.py "${PLY}" "${OUT_DIR}/semantic_point_cloud.ply" --label 0
fi

python scripts/task1/inspect_ply.py "${OUT_DIR}/semantic_point_cloud.ply" --json > "${OUT_DIR}/semantic_point_cloud_inspection.json"

if [ ! -f "${OUT_DIR}/label_map.json" ]; then
  cat > "${OUT_DIR}/label_map.json" <<'JSON'
{
  "scene": "bicycle",
  "labels": [
    {"id": 0, "name": "unlabeled", "class": "unlabeled"}
  ]
}
JSON
fi

render_args=(
  --model-path "${MODEL_DIR}"
  --output-dir "${OUT_DIR}/rgb_renders"
  --graphdeco-root "${GRAPHDECO_ROOT}"
  --iteration 30000
  --max-width "${RENDER_MAX_WIDTH}"
)

if [ -n "${RENDER_CAMERA_INDICES}" ]; then
  render_args+=(--camera-indices "${RENDER_CAMERA_INDICES}")
else
  render_args+=(--count "${RENDER_COUNT}")
fi

PYTHONPATH="${GRAPHDECO_ROOT}:${PYTHONPATH:-}" python archive/task1_legacy/scripts/render_graphdeco_cameras.py "${render_args[@]}"

find "${OUT_DIR}" -maxdepth 2 -type f | sort
