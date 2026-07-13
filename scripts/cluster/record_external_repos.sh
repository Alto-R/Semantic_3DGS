#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/lab/haoq_lab/cse12312032/external}"

for git_dir in "$ROOT"/*/.git; do
  [[ -d "$git_dir" ]] || continue
  worktree="$(dirname "$git_dir")"
  name="$(basename "$worktree")"
  commit="$(git -C "$worktree" rev-parse HEAD)"
  url="$(git -C "$worktree" remote get-url origin)"
  printf "%s\t%s\t%s\n" "$name" "$commit" "$url"
done | sort

