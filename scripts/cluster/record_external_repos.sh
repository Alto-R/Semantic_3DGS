#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
ROOT="${1:-${WORKSPACE_ROOT}/external}"

for git_dir in "$ROOT"/*/.git; do
  [[ -d "$git_dir" ]] || continue
  worktree="$(dirname "$git_dir")"
  name="$(basename "$worktree")"
  commit="$(git -C "$worktree" rev-parse HEAD)"
  url="$(git -C "$worktree" remote get-url origin)"
  printf "%s\t%s\t%s\n" "$name" "$commit" "$url"
done | sort
