#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/lab/haoq_lab/cse12312032/external}"
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

echo
echo "Recorded versions:"
bash "$(dirname "$0")/record_external_repos.sh" "$ROOT"

