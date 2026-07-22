#!/usr/bin/env python3
"""Inspect a PLY header without loading the full Gaussian point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from scripts.task1.common.ply_utils import element_stride, read_ply_header


def build_summary(path: Path) -> Dict[str, Any]:
    header = read_ply_header(path)
    vertex = header.element("vertex")

    summary: Dict[str, Any] = {
        "path": str(path),
        "format": header.fmt,
        "version": header.version,
        "header_bytes": header.header_bytes,
        "elements": [
            {
                "name": element.name,
                "count": element.count,
                "properties": [
                    {
                        "name": prop.name,
                        "type": prop.data_type,
                        "is_list": prop.is_list,
                        "count_type": prop.count_type,
                    }
                    for prop in element.properties
                ],
            }
            for element in header.elements
        ],
    }

    if vertex is not None:
        summary["vertex_count"] = vertex.count
        summary["vertex_properties"] = [prop.name for prop in vertex.properties]
        try:
            summary["vertex_stride_bytes"] = element_stride(vertex)
        except ValueError as exc:
            summary["vertex_stride_bytes"] = None
            summary["vertex_stride_error"] = str(exc)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    summary = build_summary(args.ply)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print(f"path: {summary['path']}")
    print(f"format: {summary['format']} {summary['version']}")
    print(f"header_bytes: {summary['header_bytes']}")
    if "vertex_count" in summary:
        print(f"vertex_count: {summary['vertex_count']}")
        print(f"vertex_stride_bytes: {summary.get('vertex_stride_bytes')}")
        print("vertex_properties:")
        for name in summary["vertex_properties"]:
            print(f"  - {name}")
    print("elements:")
    for element in summary["elements"]:
        print(f"  - {element['name']}: {element['count']} properties={len(element['properties'])}")


if __name__ == "__main__":
    main()
