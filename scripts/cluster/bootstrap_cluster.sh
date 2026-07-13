#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/lab/haoq_lab/cse12312032}"

mkdir -p \
  "$ROOT/projects/pku-3dgs-vr" \
  "$ROOT/data/EyeNavGS" \
  "$ROOT/data/3dgs_models" \
  "$ROOT/external" \
  "$ROOT/outputs/eyenavgs_task1"

cat <<EOF
Cluster directories are ready:
  $ROOT/projects/pku-3dgs-vr
  $ROOT/data/EyeNavGS
  $ROOT/data/3dgs_models
  $ROOT/external
  $ROOT/outputs/eyenavgs_task1

Next:
  1. Clone or pull the project repo in $ROOT/projects/pku-3dgs-vr.
  2. Clone third-party repos under $ROOT/external.
  3. Download EyeNavGS data and 3DGS models under $ROOT/data.
EOF

