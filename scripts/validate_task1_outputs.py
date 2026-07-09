#!/usr/bin/env python3
"""Validate Task 1 semantic-label outputs before counting a scene as done."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ply_utils import read_ply_header


def load_label_map(path: Path) -> dict[str, Any]:
    label_map = json.loads(path.read_text(encoding="utf-8"))
    if "scene" not in label_map:
        raise ValueError(f"{path} is missing scene")
    if not isinstance(label_map.get("labels"), list):
        raise ValueError(f"{path} must contain a labels list")
    return label_map


def validate_label_map(label_map: dict[str, Any], allow_object_candidate: bool) -> dict[int, dict[str, Any]]:
    labels_by_id: dict[int, dict[str, Any]] = {}
    for item in label_map["labels"]:
        if "id" not in item or "name" not in item or "class" not in item:
            raise ValueError(f"Label-map item is missing id/name/class: {item}")
        label_id = int(item["id"])
        if label_id in labels_by_id:
            raise ValueError(f"Duplicate label id {label_id} in label_map")
        name = str(item["name"]).strip()
        semantic_class = str(item["class"]).strip()
        if not name:
            raise ValueError(f"Label id {label_id} has an empty name")
        if not semantic_class:
            raise ValueError(f"Label id {label_id} has an empty class")
        if label_id != 0 and semantic_class == "object_candidate" and not allow_object_candidate:
            raise ValueError(
                f"Label id {label_id} still has class object_candidate; "
                "this is an automatic candidate, not a final D1 semantic label"
            )
        labels_by_id[label_id] = item
    if 0 not in labels_by_id:
        raise ValueError("label_map must include id 0 for unlabeled")
    return labels_by_id


def validate_semantic_ply(path: Path, expected_count: int) -> dict[str, Any]:
    header = read_ply_header(path)
    vertex = header.element("vertex")
    if vertex is None:
        raise ValueError(f"{path} has no vertex element")
    if vertex.count != expected_count:
        raise ValueError(
            f"{path} vertex count {vertex.count} does not match label count {expected_count}"
        )
    if not any(prop.name == "label" for prop in vertex.properties):
        raise ValueError(f"{path} vertex element has no label property")
    return {
        "path": str(path),
        "format": header.fmt,
        "vertex_count": vertex.count,
        "has_label_property": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-npy", required=True, type=Path)
    parser.add_argument("--label-map", required=True, type=Path)
    parser.add_argument("--semantic-ply", type=Path)
    parser.add_argument("--max-unlabeled-ratio", default=1.0, type=float)
    parser.add_argument("--allow-object-candidate", action="store_true")
    args = parser.parse_args()

    labels = np.load(args.labels_npy)
    if labels.ndim != 1:
        raise ValueError(f"{args.labels_npy} must contain a one-dimensional label array")

    label_map = load_label_map(args.label_map)
    labels_by_id = validate_label_map(label_map, args.allow_object_candidate)
    histogram = {int(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))}

    missing_ids = sorted(set(histogram) - set(labels_by_id))
    if missing_ids:
        raise ValueError(f"labels_npy contains ids not present in label_map: {missing_ids}")

    unlabeled_count = histogram.get(0, 0)
    unlabeled_ratio = unlabeled_count / float(max(labels.shape[0], 1))
    if unlabeled_ratio > args.max_unlabeled_ratio:
        raise ValueError(
            f"unlabeled ratio {unlabeled_ratio:.4f} exceeds limit {args.max_unlabeled_ratio:.4f}"
        )

    ply_summary = None
    if args.semantic_ply:
        ply_summary = validate_semantic_ply(args.semantic_ply, expected_count=int(labels.shape[0]))

    summary = {
        "scene": label_map["scene"],
        "label_count": int(labels.shape[0]),
        "final_label_count": len(histogram),
        "label_histogram": {str(label): count for label, count in histogram.items()},
        "unlabeled_ratio": unlabeled_ratio,
        "semantic_ply": ply_summary,
        "status": "ok",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
