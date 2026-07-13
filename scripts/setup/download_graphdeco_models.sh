#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
ROOT="${1:-${WORKSPACE_ROOT}/data/3dgs_models/graphdeco}"
ARCHIVE_DIR="$(dirname "$ROOT")/_downloads"
ARCHIVE="$ARCHIVE_DIR/graphdeco_pretrained_models.zip"
URL="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip"

mkdir -p "$ROOT" "$ARCHIVE_DIR"

echo "download target: $ARCHIVE"
echo "extract target:  $ROOT"
echo "url:             $URL"

if [[ ! -f "$ARCHIVE" ]]; then
  echo "downloading archive"
else
  echo "resuming/verifying existing archive"
fi

wget -c "$URL" -O "$ARCHIVE"

echo "extracting archive"
unzip -n "$ARCHIVE" -d "$ROOT"

echo "available point clouds:"
find "$ROOT" -name point_cloud.ply | sort
