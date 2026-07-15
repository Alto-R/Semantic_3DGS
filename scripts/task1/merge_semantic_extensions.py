#!/usr/bin/env python3
"""Overlay accepted GroundingDINO semantic groups on a DINOv2 Gaussian base."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from add_labels_from_npy import write_ply_with_labels
from dinov2_ontology import load_ontology, normalize_class_name
from ply_utils import read_ply_header
from semantic_palette import PALETTE_VERSION, rgb8_for_class


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def normalized_unique(values: list[Any], label: str) -> list[str]:
    result = [normalize_class_name(value) for value in values]
    if any(not value for value in result):
        raise ValueError(f"{label} contains an empty class")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate classes: {result}")
    return result


def load_extension_config(path: Path, scene: str) -> dict[str, Any]:
    config = load_json(path)
    config_scene = str(config.get("scene", "")).strip()
    if config_scene != scene:
        raise ValueError(f"Extension config scene {config_scene!r} does not match {scene!r}")
    candidates = normalized_unique(list(config.get("candidate_classes", [])), "candidate_classes")
    defaults = normalized_unique(
        list(config.get("default_enabled_classes", [])),
        "default_enabled_classes",
    )
    unknown_defaults = [value for value in defaults if value not in candidates]
    if unknown_defaults:
        raise ValueError(f"Default extension classes are not candidates: {unknown_defaults}")
    if not candidates:
        raise ValueError("Extension config must contain candidate_classes")
    return {
        **config,
        "candidate_classes": candidates,
        "default_enabled_classes": defaults,
    }


def resolve_selected_classes(config: dict[str, Any], include_classes: str) -> list[str]:
    if include_classes.strip():
        selected = normalized_unique(
            [value for value in include_classes.split(",") if value.strip()],
            "include_classes",
        )
    else:
        selected = list(config["default_enabled_classes"])
    if not selected:
        raise ValueError(
            "No extension classes are enabled; provide --include-classes during QA or "
            "populate default_enabled_classes after review"
        )
    unknown = [value for value in selected if value not in config["candidate_classes"]]
    if unknown:
        raise ValueError(f"Selected extension classes are not candidates: {unknown}")
    return selected


def label_items_by_id(label_map: dict[str, Any], path: Path) -> dict[int, dict[str, Any]]:
    raw_items = label_map.get("labels")
    if not isinstance(raw_items, list):
        raise ValueError(f"{path} must contain a labels list")
    result: dict[int, dict[str, Any]] = {}
    for raw in raw_items:
        if not isinstance(raw, dict) or not {"id", "name", "class"}.issubset(raw):
            raise ValueError(f"Malformed label-map item in {path}: {raw}")
        item = dict(raw)
        label_id = int(item["id"])
        if label_id in result:
            raise ValueError(f"Duplicate label id {label_id} in {path}")
        item["id"] = label_id
        item["class"] = normalize_class_name(item["class"])
        result[label_id] = item
    if 0 not in result:
        raise ValueError(f"{path} is missing label id 0")
    return result


def validate_label_array(labels: np.ndarray, items: dict[int, dict[str, Any]], path: Path) -> np.ndarray:
    if labels.ndim != 1:
        raise ValueError(f"{path} must contain a one-dimensional label array")
    if labels.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{path} must contain integer labels")
    labels = labels.astype(np.int32, copy=False)
    missing = sorted(set(int(value) for value in np.unique(labels)) - set(items))
    if missing:
        raise ValueError(f"{path} contains ids absent from its label map: {missing}")
    return labels


def manifest_classes(manifest: dict[str, Any]) -> list[str]:
    raw_classes = manifest.get("classes", [])
    if not isinstance(raw_classes, list):
        raise ValueError("GroundingDINO manifest classes must be a list")
    return normalized_unique(
        [item.get("class", "") if isinstance(item, dict) else item for item in raw_classes],
        "GroundingDINO manifest classes",
    )


def histogram(labels: np.ndarray) -> dict[int, int]:
    ids, counts = np.unique(labels, return_counts=True)
    return {int(label_id): int(count) for label_id, count in zip(ids, counts)}


def transition_records(
    base_labels: np.ndarray,
    mask: np.ndarray,
    base_items: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    ids, counts = np.unique(base_labels[mask], return_counts=True)
    return [
        {
            "base_label_id": int(label_id),
            "base_name": str(base_items[int(label_id)].get("name", "")),
            "base_class": str(base_items[int(label_id)].get("class", "")),
            "gaussian_count": int(count),
        }
        for label_id, count in zip(ids, counts)
    ]


def merge_extensions(
    base_labels: np.ndarray,
    base_items: dict[int, dict[str, Any]],
    extension_labels: np.ndarray,
    extension_items: dict[int, dict[str, Any]],
    selected_classes: list[str],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    if base_labels.shape != extension_labels.shape:
        raise ValueError(
            f"Base label shape {base_labels.shape} does not match extension shape "
            f"{extension_labels.shape}"
        )
    merged = base_labels.copy()
    next_id = max(base_items) + 1
    appended_items: list[dict[str, Any]] = []
    class_reports: list[dict[str, Any]] = []
    changed_mask = np.zeros(base_labels.shape, dtype=bool)

    for class_name in selected_classes:
        source_items = sorted(
            (
                item
                for label_id, item in extension_items.items()
                if label_id > 0 and item["class"] == class_name
            ),
            key=lambda item: int(item["id"]),
        )
        group_reports: list[dict[str, Any]] = []
        for source_item in source_items:
            source_id = int(source_item["id"])
            mask = extension_labels == source_id
            count = int(mask.sum())
            if count <= 0:
                continue
            new_id = next_id
            next_id += 1
            changed_mask |= mask
            merged[mask] = new_id
            copied = dict(source_item)
            copied.update(
                {
                    "id": new_id,
                    "gaussian_count": count,
                    "source_pipeline": "groundingdino_sam_flashsplat",
                    "source_label_id": source_id,
                    "color_key": class_name,
                    "rgb": rgb8_for_class(class_name),
                }
            )
            appended_items.append(copied)
            group_reports.append(
                {
                    "source_label_id": source_id,
                    "output_label_id": new_id,
                    "name": str(source_item.get("name", "")),
                    "gaussian_count": count,
                    "newly_labeled_count": int(np.count_nonzero(base_labels[mask] == 0)),
                    "relabeled_count": int(np.count_nonzero(base_labels[mask] != 0)),
                    "base_transitions": transition_records(base_labels, mask, base_items),
                }
            )
        class_reports.append(
            {
                "class": class_name,
                "status": "merged" if group_reports else "no_final_group",
                "group_count": len(group_reports),
                "gaussian_count": sum(item["gaussian_count"] for item in group_reports),
                "groups": group_reports,
            }
        )

    if not np.array_equal(merged[~changed_mask], base_labels[~changed_mask]):
        raise AssertionError("DINOv2 labels changed outside selected extension masks")
    report = {
        "selected_classes": selected_classes,
        "merged_classes": [item["class"] for item in class_reports if item["status"] == "merged"],
        "no_final_group_classes": [
            item["class"] for item in class_reports if item["status"] == "no_final_group"
        ],
        "merged_group_count": len(appended_items),
        "changed_gaussian_count": int(changed_mask.sum()),
        "newly_labeled_count": int(np.count_nonzero(changed_mask & (base_labels == 0))),
        "relabeled_count": int(np.count_nonzero(changed_mask & (base_labels != 0))),
        "unchanged_outside_extension_masks": True,
        "classes": class_reports,
    }
    return merged, appended_items, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-labels", required=True, type=Path)
    parser.add_argument("--base-label-map", required=True, type=Path)
    parser.add_argument("--extension-labels", required=True, type=Path)
    parser.add_argument("--extension-label-map", required=True, type=Path)
    parser.add_argument("--extension-manifest", required=True, type=Path)
    parser.add_argument("--extension-config", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--include-classes", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_paths = {
        "labels": args.output_dir / "gaussian_labels.npy",
        "label_map": args.output_dir / "label_map.json",
        "summary": args.output_dir / "hybrid_merge_summary.json",
        "semantic_ply": args.output_dir / "semantic_point_cloud.ply",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Hybrid outputs exist; pass --overwrite: {existing}")

    config = load_extension_config(args.extension_config, args.scene)
    selected_classes = resolve_selected_classes(config, args.include_classes)
    ontology = load_ontology(args.ontology)
    ontology_classes = {item.project_class for item in ontology.classes}
    collisions = [value for value in selected_classes if value in ontology_classes]
    if collisions:
        raise ValueError(f"Extension classes collide with the DINOv2 ontology: {collisions}")

    base_map = load_json(args.base_label_map)
    extension_map = load_json(args.extension_label_map)
    for label, value in (("base", base_map), ("extension", extension_map)):
        if str(value.get("scene", "")) != args.scene:
            raise ValueError(f"{label} label-map scene does not match {args.scene!r}")
    base_items = label_items_by_id(base_map, args.base_label_map)
    extension_items = label_items_by_id(extension_map, args.extension_label_map)
    base_labels = validate_label_array(np.load(args.base_labels), base_items, args.base_labels)
    extension_labels = validate_label_array(
        np.load(args.extension_labels), extension_items, args.extension_labels
    )

    manifest = load_json(args.extension_manifest)
    configured_classes = manifest_classes(manifest)
    absent_from_manifest = [value for value in selected_classes if value not in configured_classes]
    if absent_from_manifest:
        raise ValueError(
            "Selected classes are absent from the GroundingDINO manifest: "
            f"{absent_from_manifest}"
        )
    header = read_ply_header(args.source_ply)
    vertex = header.element("vertex")
    if vertex is None:
        raise ValueError(f"{args.source_ply} has no vertex element")
    if vertex.count != int(base_labels.shape[0]):
        raise ValueError(
            f"Source PLY has {vertex.count} vertices but labels contain {base_labels.shape[0]}"
        )

    merged, appended_items, merge_report = merge_extensions(
        base_labels,
        base_items,
        extension_labels,
        extension_items,
        selected_classes,
    )
    merged_histogram = histogram(merged)
    final_items: list[dict[str, Any]] = []
    for label_id in sorted(base_items):
        count = merged_histogram.get(label_id, 0)
        if label_id != 0 and count <= 0:
            continue
        item = dict(base_items[label_id])
        item["gaussian_count"] = count
        item["source_pipeline"] = item.get("source_pipeline", "dinov2_multiview_voting")
        item["color_key"] = item["class"]
        item["rgb"] = rgb8_for_class(item["class"])
        final_items.append(item)
    final_items.extend(appended_items)

    output_map = {
        "scene": args.scene,
        "source": "dinov2_groundingdino_hybrid",
        "palette_version": PALETTE_VERSION,
        "base_source": str(args.base_label_map),
        "extension_source": str(args.extension_label_map),
        "selected_extension_classes": selected_classes,
        "labels": final_items,
    }
    base_histogram = histogram(base_labels)
    summary = {
        "scene": args.scene,
        "status": "ok" if merge_report["merged_group_count"] else "no_extension_evidence",
        "palette_version": PALETTE_VERSION,
        "gaussian_count": int(merged.shape[0]),
        "base_unlabeled_ratio": base_histogram.get(0, 0) / float(max(merged.shape[0], 1)),
        "hybrid_unlabeled_ratio": merged_histogram.get(0, 0) / float(max(merged.shape[0], 1)),
        "unlabeled_ratio_delta": (
            merged_histogram.get(0, 0) - base_histogram.get(0, 0)
        ) / float(max(merged.shape[0], 1)),
        "sources": {
            "base_labels": {"path": str(args.base_labels), "sha256": sha256_file(args.base_labels)},
            "base_label_map": {
                "path": str(args.base_label_map),
                "sha256": sha256_file(args.base_label_map),
            },
            "extension_labels": {
                "path": str(args.extension_labels),
                "sha256": sha256_file(args.extension_labels),
            },
            "extension_label_map": {
                "path": str(args.extension_label_map),
                "sha256": sha256_file(args.extension_label_map),
            },
            "extension_manifest": {
                "path": str(args.extension_manifest),
                "sha256": sha256_file(args.extension_manifest),
            },
            "extension_config": {
                "path": str(args.extension_config),
                "sha256": sha256_file(args.extension_config),
            },
            "ontology": {"path": str(args.ontology), "sha256": sha256_file(args.ontology)},
            "source_ply": {"path": str(args.source_ply), "vertex_count": vertex.count},
        },
        "base_label_histogram": {str(key): value for key, value in base_histogram.items()},
        "hybrid_label_histogram": {str(key): value for key, value in merged_histogram.items()},
        "merge": merge_report,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_paths["labels"], merged.astype(np.int32, copy=False))
    output_paths["label_map"].write_text(json.dumps(output_map, indent=2), encoding="utf-8")
    output_paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_ply_with_labels(args.source_ply, output_paths["semantic_ply"], merged)
    print(json.dumps({key: value for key, value in summary.items() if key != "sources"}, indent=2))


if __name__ == "__main__":
    main()
