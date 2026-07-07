#!/usr/bin/env python3
"""Find 3DGS point_cloud.ply files under a model root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from ply_utils import read_ply_header


def infer_scene_name(path: Path) -> str:
    parts = path.parts
    if len(parts) >= 4 and parts[-4] not in {"output", "models"}:
        return parts[-4]
    if len(parts) >= 4:
        return parts[-4]
    return path.parent.name


def summarize_point_cloud(path: Path, root: Path) -> Dict[str, Any]:
    header = read_ply_header(path)
    vertex = header.element("vertex")
    return {
        "scene": infer_scene_name(path),
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "format": header.fmt,
        "vertex_count": vertex.count if vertex is not None else None,
        "property_count": len(vertex.properties) if vertex is not None else None,
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    paths = sorted(root.rglob("point_cloud.ply"))
    summaries: List[Dict[str, Any]] = [summarize_point_cloud(path, root) for path in paths]

    if args.json:
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return

    if not summaries:
        print(f"No point_cloud.ply files found under {root}")
        return

    for item in summaries:
        size_mb = item["size_bytes"] / (1024 * 1024)
        print(
            f"{item['scene']}\t{item['vertex_count']}\t{size_mb:.1f} MB\t"
            f"{item['relative_path']}"
        )


if __name__ == "__main__":
    main()

