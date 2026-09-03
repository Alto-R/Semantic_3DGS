#!/usr/bin/env python3
"""Combine direct-region labels with multiview component plurality labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.common.ply_utils import read_ply_header
from scripts.task1.dinov2.dinov2_ontology import Ontology, load_ontology
from scripts.task1.dinov3.materialize_plurality_component_labels import (
    CONTRACT as PLURALITY_CONTRACT,
    SOURCE as PLURALITY_SOURCE,
)


SOURCE = "dinov3_direct_regions_with_plurality_component_override"
CONTRACT = "direct_regions_plus_unique_camera_plurality_components_v1"
REGION_SOURCE = "dinov3_query_regions_direct_multiview_majority"
REGION_CONTRACT = "per_gaussian_one_vote_per_camera_strict_majority_v1"
CAMERA_OWNED_COMPONENT_SOURCE = (
    "dinov3_plurality_components_unique_camera_support_ownership"
)
CAMERA_OWNED_COMPONENT_CONTRACT = (
    "unique_maximum_component_camera_support_per_gaussian_v1"
)
ALLOWED_COMPONENT_CONTRACTS = {
    (PLURALITY_SOURCE, PLURALITY_CONTRACT),
    (CAMERA_OWNED_COMPONENT_SOURCE, CAMERA_OWNED_COMPONENT_CONTRACT),
}


def validate_labels(
    labels: np.ndarray,
    *,
    name: str,
    vertex_count: int,
    ontology: Ontology,
) -> np.ndarray:
    values = np.asarray(labels)
    if values.shape != (vertex_count,):
        raise ValueError(f"{name} must contain one label per source PLY vertex")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must contain integer project class IDs")
    if values.size and (
        int(values.min()) < 0 or int(values.max()) > ontology.class_count
    ):
        raise ValueError(f"{name} contains a class outside the ontology")
    return values.astype(np.int32, copy=False)


def combine_seed_labels(
    region_labels: np.ndarray,
    component_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Overlay component identity on region labels and record provenance."""

    region = np.asarray(region_labels)
    components = np.asarray(component_labels)
    if region.ndim != 1 or components.shape != region.shape:
        raise ValueError("region and component labels must be matching vectors")
    combined = np.array(region, dtype=np.int32, copy=True)
    component_mask = components > 0
    component_addition = component_mask & (region == 0)
    component_override = component_mask & (region > 0) & (region != components)
    combined[component_mask] = components[component_mask].astype(np.int32, copy=False)
    source_codes = np.zeros(region.shape, dtype=np.uint8)
    source_codes[region > 0] = np.uint8(1)
    source_codes[component_mask] = np.uint8(2)
    return combined, source_codes, component_addition, component_override


def validate_summary(
    path: Path,
    *,
    allowed_contracts: set[tuple[str, str]],
    source_ply: Path,
    scene: str,
    vertex_count: int,
) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    source_contract = (str(summary.get("source")), str(summary.get("contract")))
    if source_contract not in allowed_contracts:
        raise ValueError(f"{path} has the wrong source contract")
    if summary.get("v5_used") is not False or summary.get("dinov2_used") is not False:
        raise ValueError(f"{path} is not DINOv3-only")
    if str(summary.get("scene")) != scene:
        raise ValueError(f"{path} has the wrong scene")
    if int(summary.get("vertex_count", -1)) != vertex_count:
        raise ValueError(f"{path} has the wrong Gaussian count")
    if Path(str(summary.get("source_ply", ""))).resolve() != source_ply.resolve():
        raise ValueError(f"{path} refers to a different source PLY")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-labels", required=True, type=Path)
    parser.add_argument("--region-summary", required=True, type=Path)
    parser.add_argument("--component-labels", required=True, type=Path)
    parser.add_argument("--component-summary", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    args = parser.parse_args()

    for path in (
        args.region_labels,
        args.region_summary,
        args.component_labels,
        args.component_summary,
        args.ontology,
        args.source_ply,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    ontology = load_ontology(args.ontology)
    header = read_ply_header(args.source_ply)
    if not header.elements or header.elements[0].name != "vertex":
        raise ValueError("source PLY does not begin with a vertex element")
    vertex_count = int(header.elements[0].count)
    region_summary = validate_summary(
        args.region_summary,
        allowed_contracts={(REGION_SOURCE, REGION_CONTRACT)},
        source_ply=args.source_ply,
        scene=args.scene,
        vertex_count=vertex_count,
    )
    component_summary = validate_summary(
        args.component_summary,
        allowed_contracts=ALLOWED_COMPONENT_CONTRACTS,
        source_ply=args.source_ply,
        scene=args.scene,
        vertex_count=vertex_count,
    )
    region_labels = validate_labels(
        np.load(args.region_labels, mmap_mode="r", allow_pickle=False),
        name="region labels",
        vertex_count=vertex_count,
        ontology=ontology,
    )
    component_labels = validate_labels(
        np.load(args.component_labels, mmap_mode="r", allow_pickle=False),
        name="component labels",
        vertex_count=vertex_count,
        ontology=ontology,
    )
    combined, source_codes, addition_mask, override_mask = combine_seed_labels(
        region_labels,
        component_labels,
    )

    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "gaussian_labels.npy", combined)
    np.save(args.output_dir / "gaussian_project_class_ids.npy", combined)
    np.save(args.output_dir / "seed_source_codes.npy", source_codes)
    np.save(args.output_dir / "component_addition_mask.npy", addition_mask)
    np.save(args.output_dir / "component_override_mask.npy", override_mask)

    labels_json: list[dict[str, Any]] = [
        {"id": 0, "name": "unlabeled", "class": "unlabeled", "type": "unlabeled"}
    ]
    for project_id_value in np.unique(combined[combined > 0]):
        item = ontology.by_project_id[int(project_id_value)]
        labels_json.append(
            {
                "id": item.project_id,
                "name": item.project_class,
                "class": item.project_class,
                "project_id": item.project_id,
                "ade_id": item.ade_id,
                "type": item.kind,
            }
        )
    label_map = {
        "scene": args.scene,
        "source": SOURCE,
        "ontology": str(args.ontology),
        "region_summary": str(args.region_summary),
        "component_summary": str(args.component_summary),
        "labels": labels_json,
    }
    (args.output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2), encoding="utf-8"
    )

    assigned = combined > 0
    class_counts = {
        ontology.by_project_id[int(project_id)].project_class: int(count)
        for project_id, count in zip(*np.unique(combined[assigned], return_counts=True))
    }
    summary = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": args.scene,
        "source_ply": str(args.source_ply),
        "ontology": str(args.ontology),
        "region_labels": str(args.region_labels),
        "region_summary": str(args.region_summary),
        "component_labels": str(args.component_labels),
        "component_summary": str(args.component_summary),
        "component_seed_source": str(component_summary["source"]),
        "component_seed_contract": str(component_summary["contract"]),
        "v5_used": False,
        "dinov2_used": False,
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "offline_threshold_sweep_used": False,
        "combination_policy": "plurality_component_identity_overrides_direct_region_identity",
        "cross_component_conflicts": "already_unlabeled_by_component_materialization",
        "vertex_count": vertex_count,
        "region_seed_gaussian_count": int(np.count_nonzero(region_labels)),
        "component_seed_gaussian_count": int(np.count_nonzero(component_labels)),
        "component_added_gaussian_count": int(np.count_nonzero(addition_mask)),
        "component_overridden_region_gaussian_count": int(np.count_nonzero(override_mask)),
        "combined_seed_gaussian_count": int(np.count_nonzero(assigned)),
        "combined_seed_ratio": float(np.mean(assigned)),
        "class_combined_seed_counts": dict(sorted(class_counts.items())),
        "seed_source_codes": {
            "0": "unlabeled",
            "1": "direct_region_vote",
            "2": "plurality_physical_component",
        },
        "input_region_summary": region_summary,
        "input_component_summary": component_summary,
    }
    (args.output_dir / "combined_seed_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
