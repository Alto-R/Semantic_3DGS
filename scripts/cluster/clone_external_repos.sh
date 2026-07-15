#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
ROOT="${1:-${WORKSPACE_ROOT}/external}"
mkdir -p "$ROOT"
cd "$ROOT"

export GIT_LFS_SKIP_SMUDGE=1

clone_or_update() {
  local name="$1"
  local url="$2"

  if [[ -d "$name/.git" ]]; then
    echo "updating $name"
    git -C "$name" pull --ff-only
  else
    echo "cloning $name"
    git clone --depth 1 "$url" "$name"
  fi
}

clone_or_update EyeNavGS_Software https://github.com/symmru/EyeNavGS_Software.git
clone_or_update EyeNavGS_Rutgers_Dataset https://github.com/symmru/EyeNavGS_Rutgers_Dataset.git
clone_or_update EyeNavGS_NTHU_Dataset https://github.com/sawalee0811/EyeNavGS_NTHU_Dataset.git
clone_or_update gaussian-splatting https://github.com/graphdeco-inria/gaussian-splatting.git
clone_or_update FlashSplat https://github.com/florinshen/FlashSplat.git
clone_or_update SegAnyGAussians https://github.com/Jumpat/SegAnyGAussians.git
clone_or_update dinov2 https://github.com/facebookresearch/dinov2.git

echo
echo "initializing required submodules"
git -C gaussian-splatting submodule update --init --recursive --depth 1
git -C FlashSplat submodule update --init --recursive --depth 1

# SAGA records GitHub SSH URLs in .gitmodules; use HTTPS on clusters without a
# GitHub SSH key.
git -C SegAnyGAussians config --file .gitmodules \
  submodule.third_party/kmeans_pytorch.url \
  https://github.com/subhadarship/kmeans_pytorch.git
git -C SegAnyGAussians config --file .gitmodules \
  submodule.third_party/segment-anything.url \
  https://github.com/facebookresearch/segment-anything.git
git -C SegAnyGAussians submodule sync third_party/kmeans_pytorch third_party/segment-anything
git -C SegAnyGAussians submodule update --init --recursive --depth 1

echo
echo "Recorded versions:"
bash "$(dirname "$0")/record_external_repos.sh" "$ROOT"
