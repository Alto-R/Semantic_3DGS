#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/lab/haoq_lab/cse12312032}"
ENV_NAME="${ENV_NAME:-gaussian_grouping_true}"
RASTERIZER_ROOT="${RASTERIZER_ROOT:-${ROOT}/external/gaussian-splatting/submodules/diff-gaussian-rasterization}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

conda activate "${ENV_NAME}"

cd "${RASTERIZER_ROOT}"
MAX_JOBS="${MAX_JOBS:-4}" python setup.py build_ext --inplace

python - <<'PY'
import inspect
import pathlib
import sys

root = pathlib.Path.cwd()
sys.path.insert(0, str(root))
import diff_gaussian_rasterization as raster

print(raster.__file__)
print(inspect.signature(raster.GaussianRasterizationSettings))
PY
