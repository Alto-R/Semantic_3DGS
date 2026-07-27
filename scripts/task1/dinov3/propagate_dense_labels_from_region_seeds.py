#!/usr/bin/env python3
"""Propagate dense DINOv3 labels through validated semantic seed components.

Direct query-region fusion or a validated DINOv3 seed-combination stage
supplies immutable semantic seeds. Dense per-pixel fusion supplies candidate
labels. A connected dense-class component may fill seed-unlabeled Gaussians
only when it contains at least one matching seed and contains no conflicting
nonzero seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.common.ply_utils import vertex_data_memmap
from scripts.task1.dinov2.dinov2_ontology import Ontology, load_ontology
from scripts.task1.dinov3.combine_plurality_component_seeds import (
    CONTRACT as COMBINED_SEED_CONTRACT,
    SOURCE as COMBINED_SEED_SOURCE,
)
from scripts.task1.grounding.cluster_semantic_flashsplat_proposals import (
    voxel_components,
)
from scripts.task1.qa.add_labels_from_npy import write_ply_with_labels


COMPONENT_NOT_DENSE_CANDIDATE = 0
COMPONENT_ACCEPTED = 1
COMPONENT_UNSEEDED = 2
COMPONENT_CONFLICTING = 3

COMPONENT_STATUS_NAMES = {
    COMPONENT_NOT_DENSE_CANDIDATE: "not_dense_candidate",
    COMPONENT_ACCEPTED: "accepted_matching_seed_no_conflict",
    COMPONENT_UNSEEDED: "unseeded",
    COMPONENT_CONFLICTING: "conflicting_nonzero_seed",
}

LABEL_SOURCE_UNLABELED = 0
LABEL_SOURCE_REGION_SEED = 1
LABEL_SOURCE_DENSE_PROPAGATION = 2

REGION_SOURCE = "dinov3_query_regions_direct_multiview_majority"
REGION_CONTRACT = "per_gaussian_one_vote_per_camera_strict_majority_v1"
DENSE_SOURCE = "dinov3_dense_pixel_flashsplat_soft_class_mass"
DENSE_CONTRACT = "complete_dense_pixels_unique_soft_argmax_v1"
ALLOWED_SEED_CONTRACTS = {
    (REGION_SOURCE, REGION_CONTRACT),
    (COMBINED_SEED_SOURCE, COMBINED_SEED_CONTRACT),
}


def adaptive_voxel_size(
    log_scales: np.ndarray,
    *,
    voxel_scale_multiplier: float,
    min_voxel_size: float,
    max_voxel_size: float,
) -> tuple[float, float]:
    """Derive one class-component voxel size from its Gaussian scales."""

    scales = np.asarray(log_scales, dtype=np.float64)
    if scales.ndim != 2 or scales.shape[1] != 3 or scales.shape[0] == 0:
        raise ValueError("log_scales must have shape (N, 3) with N > 0")
    if voxel_scale_multiplier <= 0.0:
        raise ValueError("voxel_scale_multiplier must be positive")
    if min_voxel_size <= 0.0:
        raise ValueError("min_voxel_size must be positive")
    if max_voxel_size < 0.0:
        raise ValueError("max_voxel_size must be non-negative")
    if max_voxel_size > 0.0 and max_voxel_size < min_voxel_size:
        raise ValueError("max_voxel_size must be zero or at least min_voxel_size")

    gaussian_scales = np.exp(np.clip(scales.max(axis=1), -20.0, 5.0))
    finite = gaussian_scales[np.isfinite(gaussian_scales)]
    if finite.size == 0:
        raise ValueError("Gaussian scales contain no finite values")
    median_scale = float(np.median(finite))
    voxel_size = max(min_voxel_size, median_scale * voxel_scale_multiplier)
    if max_voxel_size > 0.0:
        voxel_size = min(voxel_size, max_voxel_size)
    return median_scale, float(voxel_size)


def component_seed_guard(
    component_ids: np.ndarray,
    seed_labels: np.ndarray,
    *,
    candidate_class_id: int,
    component_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Classify components and return the seed-unlabeled fill mask.

    A component with both matching and conflicting seeds is conflicting.  This
    keeps every seed immutable while ensuring conflict always wins over fill.
    """

    components = np.asarray(component_ids)
    seeds = np.asarray(seed_labels)
    if components.ndim != 1 or seeds.shape != components.shape:
        raise ValueError("component_ids and seed_labels must be matching vectors")
    if candidate_class_id <= 0:
        raise ValueError("candidate_class_id must be positive")
    if components.size and np.any(components < 0):
        raise ValueError("component IDs must be non-negative")
    inferred_count = int(components.max()) + 1 if components.size else 0
    count = inferred_count if component_count is None else int(component_count)
    if count < inferred_count or count < 0:
        raise ValueError("component_count does not cover every component ID")

    matching = seeds == candidate_class_id
    conflicting = (seeds != 0) & ~matching
    matching_counts = np.bincount(
        components[matching].astype(np.int64, copy=False), minlength=count
    ).astype(np.int64, copy=False)
    conflicting_counts = np.bincount(
        components[conflicting].astype(np.int64, copy=False), minlength=count
    ).astype(np.int64, copy=False)

    statuses = np.full((count,), COMPONENT_UNSEEDED, dtype=np.uint8)
    statuses[(matching_counts > 0) & (conflicting_counts == 0)] = COMPONENT_ACCEPTED
    statuses[conflicting_counts > 0] = COMPONENT_CONFLICTING
    fill = (seeds == 0) & (statuses[components] == COMPONENT_ACCEPTED)
    return statuses, fill, matching_counts, conflicting_counts


def validate_project_labels(
    labels: np.ndarray,
    *,
    name: str,
    vertex_count: int,
    ontology: Ontology,
) -> np.ndarray:
    values = np.asarray(labels)
    if values.shape != (vertex_count,):
        raise ValueError(f"{name} must contain one label per source-Ply vertex")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must contain integer project class IDs")
    if values.size and (
        int(values.min()) < 0 or int(values.max()) > ontology.class_count
    ):
        raise ValueError(f"{name} contains a project class outside the ontology")
    return values.astype(np.int32, copy=False)


def load_source_summary(
    path: Path,
    *,
    allowed_contracts: set[tuple[str, str]],
    source_ply: Path,
    vertex_count: int,
    scene: str,
) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    source_contract = (str(summary.get("source")), str(summary.get("contract")))
    if source_contract not in allowed_contracts:
        raise ValueError(f"{path} has an unsupported DINOv3 source contract")
    if summary.get("v5_used") is not False or summary.get("dinov2_used") is not False:
        raise ValueError(f"{path} is not independent of v5 and DINOv2")
    if int(summary.get("vertex_count", -1)) != vertex_count:
        raise ValueError(f"{path} has the wrong Gaussian count")
    if str(summary.get("scene")) != scene:
        raise ValueError(f"{path} has the wrong scene")
    summary_ply = Path(str(summary.get("source_ply", "")))
    if summary_ply.resolve() != source_ply.resolve():
        raise ValueError(f"{path} refers to a different source PLY")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-labels", required=True, type=Path)
    parser.add_argument("--seed-summary", required=True, type=Path)
    parser.add_argument("--dense-labels", required=True, type=Path)
    parser.add_argument("--dense-summary", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--voxel-scale-multiplier", default=4.0, type=float)
    parser.add_argument("--min-voxel-size", default=0.01, type=float)
    parser.add_argument("--max-voxel-size", default=0.20, type=float)
    parser.add_argument("--no-semantic-ply", action="store_true")
    args = parser.parse_args()

    for path in (
        args.seed_labels,
        args.seed_summary,
        args.dense_labels,
        args.dense_summary,
        args.ontology,
        args.source_ply,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.scene.strip():
        raise ValueError("scene must be non-empty")

    ontology = load_ontology(args.ontology)
    ply_header, vertices = vertex_data_memmap(args.source_ply)
    vertex_count = int(ply_header.elements[0].count)
    required = {"x", "y", "z", "scale_0", "scale_1", "scale_2"}
    missing = sorted(required - set(vertices.dtype.names or ()))
    if missing:
        raise ValueError(f"source PLY is missing geometry properties: {missing}")

    seed_summary = load_source_summary(
        args.seed_summary,
        allowed_contracts=ALLOWED_SEED_CONTRACTS,
        source_ply=args.source_ply,
        vertex_count=vertex_count,
        scene=args.scene,
    )
    dense_summary = load_source_summary(
        args.dense_summary,
        allowed_contracts={(DENSE_SOURCE, DENSE_CONTRACT)},
        source_ply=args.source_ply,
        vertex_count=vertex_count,
        scene=args.scene,
    )
    seed_labels = validate_project_labels(
        np.load(args.seed_labels, mmap_mode="r", allow_pickle=False),
        name="seed labels",
        vertex_count=vertex_count,
        ontology=ontology,
    )
    dense_labels = validate_project_labels(
        np.load(args.dense_labels, mmap_mode="r", allow_pickle=False),
        name="dense labels",
        vertex_count=vertex_count,
        ontology=ontology,
    )

    final_labels = np.array(seed_labels, dtype=np.int32, copy=True)
    dense_component_ids = np.zeros((vertex_count,), dtype=np.int32)
    dense_component_status_codes = np.zeros((vertex_count,), dtype=np.uint8)
    propagated_fill_mask = np.zeros((vertex_count,), dtype=bool)
    label_source_codes = np.zeros((vertex_count,), dtype=np.uint8)
    label_source_codes[seed_labels > 0] = LABEL_SOURCE_REGION_SEED

    class_reports: list[dict[str, Any]] = []
    next_component_id = 1
    for project_id_value in np.unique(dense_labels[dense_labels > 0]):
        project_id = int(project_id_value)
        indices = np.flatnonzero(dense_labels == project_id)
        points = np.column_stack(
            [vertices[axis][indices].astype(np.float64) for axis in ("x", "y", "z")]
        )
        log_scales = np.column_stack(
            [
                vertices[axis][indices].astype(np.float64)
                for axis in ("scale_0", "scale_1", "scale_2")
            ]
        )
        median_scale, voxel_size = adaptive_voxel_size(
            log_scales,
            voxel_scale_multiplier=args.voxel_scale_multiplier,
            min_voxel_size=args.min_voxel_size,
            max_voxel_size=args.max_voxel_size,
        )
        local_components, component_sizes, geometry_stats = voxel_components(
            points, voxel_size
        )
        statuses, fill, matching_counts, conflicting_counts = component_seed_guard(
            local_components,
            seed_labels[indices],
            candidate_class_id=project_id,
            component_count=int(component_sizes.shape[0]),
        )
        global_components = local_components + next_component_id
        dense_component_ids[indices] = global_components.astype(np.int32, copy=False)
        dense_component_status_codes[indices] = statuses[local_components]
        added_indices = indices[fill]
        final_labels[added_indices] = project_id
        propagated_fill_mask[added_indices] = True
        label_source_codes[added_indices] = LABEL_SOURCE_DENSE_PROPAGATION

        ontology_class = ontology.by_project_id[project_id]
        class_reports.append(
            {
                "project_id": project_id,
                "class": ontology_class.project_class,
                "ade_id": ontology_class.ade_id,
                "type": ontology_class.kind,
                "dense_candidate_gaussian_count": int(indices.shape[0]),
                "median_gaussian_scale": median_scale,
                "voxel_size": voxel_size,
                "component_count": int(component_sizes.shape[0]),
                "accepted_component_count": int(
                    np.count_nonzero(statuses == COMPONENT_ACCEPTED)
                ),
                "unseeded_component_count": int(
                    np.count_nonzero(statuses == COMPONENT_UNSEEDED)
                ),
                "conflicting_component_count": int(
                    np.count_nonzero(statuses == COMPONENT_CONFLICTING)
                ),
                "matching_seed_gaussian_count": int(matching_counts.sum()),
                "conflicting_seed_gaussian_count": int(conflicting_counts.sum()),
                "added_gaussian_count": int(fill.sum()),
                **geometry_stats,
            }
        )
        next_component_id += int(component_sizes.shape[0])
        print(
            f"{ontology_class.project_class}: {component_sizes.shape[0]} components, "
            f"{int(np.count_nonzero(statuses == COMPONENT_ACCEPTED))} accepted, "
            f"{int(fill.sum())} Gaussians added"
        )

    seed_mask = seed_labels > 0
    if not np.array_equal(final_labels[seed_mask], seed_labels[seed_mask]):
        raise RuntimeError("immutable input seeds changed during propagation")
    label_source_codes[final_labels == 0] = LABEL_SOURCE_UNLABELED

    args.output_dir.mkdir(parents=True)
    outputs = {
        "gaussian_labels.npy": final_labels,
        "gaussian_project_class_ids.npy": final_labels,
        "seed_mask.npy": seed_mask,
        "region_seed_mask.npy": seed_mask,
        "propagated_fill_mask.npy": propagated_fill_mask,
        "label_source_codes.npy": label_source_codes,
        "dense_component_ids.npy": dense_component_ids,
        "dense_component_status_codes.npy": dense_component_status_codes,
    }
    for filename, array in outputs.items():
        np.save(args.output_dir / filename, array)

    labels_json: list[dict[str, Any]] = [
        {"id": 0, "name": "unlabeled", "class": "unlabeled", "type": "unlabeled"}
    ]
    for project_id_value in np.unique(final_labels[final_labels > 0]):
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
    combined_seed_input = seed_summary["source"] == COMBINED_SEED_SOURCE
    output_source = (
        "dinov3_plurality_component_seeded_dense_3d_propagation"
        if combined_seed_input
        else "dinov3_region_seed_guarded_dense_3d_propagation"
    )
    output_contract = (
        "plurality_component_override_seeds_matching_dense_components_v1"
        if combined_seed_input
        else "immutable_region_seeds_matching_conflict_free_dense_components_v1"
    )
    label_map = {
        "scene": args.scene,
        "source": output_source,
        "ontology": str(args.ontology),
        "seed_summary": str(args.seed_summary),
        "dense_summary": str(args.dense_summary),
        "labels": labels_json,
    }
    (args.output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2), encoding="utf-8"
    )

    semantic_ply = args.output_dir / "semantic_point_cloud.ply"
    if not args.no_semantic_ply:
        partial = semantic_ply.with_suffix(".ply.partial")
        write_ply_with_labels(args.source_ply, partial, final_labels)
        partial.replace(semantic_ply)

    class_counts = {
        ontology.by_project_id[int(project_id)].project_class: int(count)
        for project_id, count in zip(
            *np.unique(final_labels[final_labels > 0], return_counts=True)
        )
    }
    accepted_component_count = sum(
        int(report["accepted_component_count"]) for report in class_reports
    )
    unseeded_component_count = sum(
        int(report["unseeded_component_count"]) for report in class_reports
    )
    conflicting_component_count = sum(
        int(report["conflicting_component_count"]) for report in class_reports
    )
    assigned_count = int(np.count_nonzero(final_labels))
    region_seed_count = int(np.count_nonzero(seed_mask))
    component_seed_count = 0
    if combined_seed_input:
        region_seed_count = int(seed_summary["region_seed_gaussian_count"])
        component_seed_count = int(seed_summary["component_seed_gaussian_count"])
        if int(seed_summary["combined_seed_gaussian_count"]) != int(
            np.count_nonzero(seed_mask)
        ):
            raise ValueError("combined seed summary count differs from seed labels")

    summary = {
        "source": output_source,
        "contract": output_contract,
        "scene": args.scene,
        "source_ply": str(args.source_ply),
        "ontology": str(args.ontology),
        "seed_labels": str(args.seed_labels),
        "seed_summary": str(args.seed_summary),
        "dense_labels": str(args.dense_labels),
        "dense_summary": str(args.dense_summary),
        "v5_used": False,
        "dinov2_used": False,
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "offline_threshold_sweep_used": False,
        "seed_policy": (
            "immutable_direct_regions_with_plurality_component_override"
            if combined_seed_input
            else "immutable_direct_region_vote_project_classes"
        ),
        "candidate_policy": "dense_project_class_components",
        "component_acceptance_policy": (
            "at_least_one_matching_seed_and_no_conflicting_nonzero_seed"
        ),
        "fill_policy": "seed_unlabeled_gaussians_only",
        "geometry_policy": {
            "connectivity": "26_neighbor_voxel_components",
            "voxel_scale_multiplier": args.voxel_scale_multiplier,
            "min_voxel_size": args.min_voxel_size,
            "max_voxel_size": args.max_voxel_size,
            "minimum_component_size_used": False,
        },
        "component_status_codes": {
            str(code): name for code, name in COMPONENT_STATUS_NAMES.items()
        },
        "label_source_codes": {
            "0": "unlabeled",
            "1": "immutable_input_seed",
            "2": "accepted_dense_component_fill",
        },
        "vertex_count": vertex_count,
        "seed_gaussian_count": int(np.count_nonzero(seed_mask)),
        "region_seed_gaussian_count": region_seed_count,
        "plurality_component_seed_gaussian_count": component_seed_count,
        "dense_candidate_gaussian_count": int(np.count_nonzero(dense_labels)),
        "seed_dense_disagreement_count": int(
            np.count_nonzero(seed_mask & (dense_labels != seed_labels))
        ),
        "dense_component_count": next_component_id - 1,
        "accepted_component_count": accepted_component_count,
        "unseeded_component_count": unseeded_component_count,
        "conflicting_component_count": conflicting_component_count,
        "added_gaussian_count": int(np.count_nonzero(propagated_fill_mask)),
        "assigned_gaussian_count": assigned_count,
        "assigned_ratio": assigned_count / float(vertex_count),
        "unlabeled_gaussian_count": vertex_count - assigned_count,
        "class_assigned_gaussian_counts": dict(sorted(class_counts.items())),
        "class_component_reports": class_reports,
        "semantic_labels_written": True,
        "label_map_written": True,
        "semantic_ply_written": not args.no_semantic_ply,
        "input_seed_summary": seed_summary,
        "input_region_summary": seed_summary.get("input_region_summary", seed_summary),
        "input_component_summary": seed_summary.get("input_component_summary"),
        "input_dense_summary": dense_summary,
    }
    (args.output_dir / "propagation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
