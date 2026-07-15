#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-gaussian_grouping_true}"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
EXTERNAL_ROOT="${2:-${WORKSPACE_ROOT}/external}"
CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.6+PTX}"
BUILD_JOBS="${MAX_JOBS:-4}"

echo "conda env:     $ENV_NAME"
echo "external root: $EXTERNAL_ROOT"
echo "CUDA arches:   $CUDA_ARCH_LIST"
echo "build jobs:    $BUILD_JOBS"

install_editable() {
  local path="$1"
  local name="$2"
  if [[ ! -d "$path" ]]; then
    echo "missing $name at $path" >&2
    return 1
  fi
  echo
  echo "installing $name"
  TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST" \
    MAX_JOBS="$BUILD_JOBS" \
    conda run -n "$ENV_NAME" python -m pip install \
      --no-build-isolation \
      --no-deps \
      --no-cache-dir \
      --force-reinstall \
      -e "$path"
}

install_editable \
  "$EXTERNAL_ROOT/FlashSplat/submodules/flashsplat-rasterization" \
  flashsplat_rasterization

install_editable \
  "$EXTERNAL_ROOT/SegAnyGAussians/submodules/diff-gaussian-rasterization_contrastive_f" \
  diff_gaussian_rasterization_contrastive_f

install_editable \
  "$EXTERNAL_ROOT/SegAnyGAussians/submodules/diff-gaussian-rasterization-depth" \
  diff_gaussian_rasterization_depth

echo
echo "verifying imports"
conda run -n "$ENV_NAME" python -c '
import importlib.util
mods = [
    "flashsplat_rasterization",
    "diff_gaussian_rasterization_contrastive_f",
    "diff_gaussian_rasterization_depth",
]
missing = []
for mod in mods:
    ok = importlib.util.find_spec(mod) is not None
    print(mod, "OK" if ok else "MISSING")
    if not ok:
        missing.append(mod)
if missing:
    raise SystemExit(f"missing imports: {missing}")
'
