#!/usr/bin/env python3
"""Audit v5-style partial-anchor completion using DINOv3 evidence only.

This stage is permanently report-only. Existing DINOv3 component instances
provide immutable anchors and dense DINOv3 votes provide same-class completion
candidates. The audit writes proposal masks and diagnostics, but never writes
semantic labels, a label map, or a semantic PLY.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.common.ply_utils import vertex_data_memmap
from scripts.task1.dinov2.dinov2_ontology import Ontology, load_ontology
from scripts.task1.dinov3.combine_plurality_component_seeds import (
    CONTRACT as COMBINED_SEED_CONTRACT,
    SOURCE as COMBINED_SEED_SOURCE,
)
from scripts.task1.dinov3.materialize_camera_owned_component_labels import (
    CONTRACT as CAMERA_COMPONENT_CONTRACT,
    SOURCE as CAMERA_COMPONENT_SOURCE,
)
from scripts.task1.dinov3.propagate_dense_labels_from_region_seeds import (
    DENSE_CONTRACT,
    DENSE_SOURCE,
    adaptive_voxel_size,
    load_source_summary,
    validate_project_labels,
)
from scripts.task1.grounding.cluster_semantic_flashsplat_proposals import (
    voxel_components,
)


SOURCE = "dinov3_partial_anchor_dense_completion_audit"
CONTRACT = "report_only_v5_style_partial_anchor_dense_completion_v1"


@dataclass(frozen=True)
class AuditThresholds:
    min_anchor_gaussians: int = 500
    min_anchor_coverage: float = 0.80
    min_anchor_precision: float = 0.10
    max_anchor_precision: float = 0.25
    min_robust_fraction: float = 0.90
    min_source_views: int = 5
    min_candidate_supporting_views: int = 5
    max_competing_thing_fraction: float = 0.05
    min_spatial_keep_fraction: float = 0.95
    max_partial_competing_thing_risk: float = 0.40
    voxel_scale_multiplier: float = 4.0
    min_voxel_size: float = 0.01
    max_voxel_size: float = 0.20

    def validate(self) -> None:
        if self.min_anchor_gaussians < 1:
            raise ValueError("min_anchor_gaussians must be positive")
        for name, value in (
            ("min_anchor_coverage", self.min_anchor_coverage),
            ("min_anchor_precision", self.min_anchor_precision),
            ("max_anchor_precision", self.max_anchor_precision),
            ("min_robust_fraction", self.min_robust_fraction),
            ("max_competing_thing_fraction", self.max_competing_thing_fraction),
            ("min_spatial_keep_fraction", self.min_spatial_keep_fraction),
            (
                "max_partial_competing_thing_risk",
                self.max_partial_competing_thing_risk,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_anchor_precision < self.min_anchor_precision:
            raise ValueError(
                "max_anchor_precision must be at least min_anchor_precision"
            )
        if self.min_source_views < 1 or self.min_candidate_supporting_views < 1:
            raise ValueError("view-support thresholds must be positive")
        adaptive_voxel_size(
            np.zeros((1, 3), dtype=np.float64),
            voxel_scale_multiplier=self.voxel_scale_multiplier,
            min_voxel_size=self.min_voxel_size,
            max_voxel_size=self.max_voxel_size,
        )


def validate_vector(
    values: np.ndarray,
    *,
    name: str,
    vertex_count: int,
    integer: bool = True,
) -> np.ndarray:
    result = np.asarray(values)
    if result.shape != (vertex_count,):
        raise ValueError(f"{name} must contain one value per source-Ply vertex")
    if integer and not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    return result


def validate_component_items(
    label_map: dict[str, Any],
    *,
    ontology: Ontology,
    scene: str,
) -> dict[int, dict[str, Any]]:
    if str(label_map.get("scene")) != scene:
        raise ValueError("component label map has the wrong scene")
    if str(label_map.get("source")) != CAMERA_COMPONENT_SOURCE:
        raise ValueError("component label map has the wrong source")
    raw_labels = label_map.get("labels")
    if not isinstance(raw_labels, list):
        raise ValueError("component label map labels must be a list")

    items: dict[int, dict[str, Any]] = {}
    for raw in raw_labels:
        if not isinstance(raw, dict):
            raise ValueError("component label records must be objects")
        label_id = int(raw["id"])
        if label_id in items:
            raise ValueError(f"duplicate component label ID {label_id}")
        if label_id == 0:
            items[0] = dict(raw)
            continue
        project_id = int(raw["project_id"])
        try:
            ontology_class = ontology.by_project_id[project_id]
        except KeyError as exc:
            raise ValueError(f"unknown component project ID {project_id}") from exc
        item_type = str(raw.get("type", raw.get("kind", "")))
        if (
            str(raw.get("class")) != ontology_class.project_class
            or int(raw.get("ade_id", -1)) != ontology_class.ade_id
            or item_type != ontology_class.kind
        ):
            raise ValueError(f"component label {label_id} disagrees with ontology")
        if int(raw.get("source_view_count", 0)) < 1:
            raise ValueError(f"component label {label_id} has no source views")
        items[label_id] = {**raw, "type": item_type}
    if 0 not in items:
        raise ValueError("component label map is missing unlabeled ID 0")
    return items


def competing_thing_metrics(
    candidate_indices: np.ndarray,
    component_instance_labels: np.ndarray,
    component_items: dict[int, dict[str, Any]],
    *,
    target_class: str,
) -> dict[str, Any]:
    """Adapt the v5 dominant competing thing-instance risk calculation."""

    empty = {
        "dominant_class": "",
        "dominant_class_fraction": 0.0,
        "dominant_instance_label_id": 0,
        "dominant_instance_overlap": 0,
        "dominant_instance_gaussian_count": 0,
        "dominant_instance_coverage": 0.0,
        "partial_overlap_risk": 0.0,
    }
    if candidate_indices.size == 0:
        return empty

    candidate_labels = component_instance_labels[candidate_indices]
    competing_by_class: dict[str, list[int]] = {}
    for label_id_value in np.unique(candidate_labels[candidate_labels > 0]):
        label_id = int(label_id_value)
        try:
            item = component_items[label_id]
        except KeyError as exc:
            raise ValueError(
                f"component instance label {label_id} is absent from label map"
            ) from exc
        class_name = str(item["class"])
        if str(item["type"]) == "thing" and class_name != target_class:
            competing_by_class.setdefault(class_name, []).append(label_id)
    if not competing_by_class:
        return empty

    class_counts = {
        class_name: int(np.count_nonzero(np.isin(candidate_labels, label_ids)))
        for class_name, label_ids in competing_by_class.items()
    }
    dominant_class = sorted(
        class_counts,
        key=lambda class_name: (-class_counts[class_name], class_name),
    )[0]
    dominant_class_fraction = class_counts[dominant_class] / float(
        candidate_indices.size
    )
    label_overlaps = {
        label_id: int(np.count_nonzero(candidate_labels == label_id))
        for label_id in competing_by_class[dominant_class]
    }
    dominant_label_id = sorted(
        label_overlaps,
        key=lambda label_id: (-label_overlaps[label_id], label_id),
    )[0]
    dominant_overlap = label_overlaps[dominant_label_id]
    dominant_size = int(
        np.count_nonzero(component_instance_labels == dominant_label_id)
    )
    dominant_coverage = dominant_overlap / float(max(dominant_size, 1))
    risk = dominant_class_fraction * (1.0 - dominant_coverage)
    return {
        "dominant_class": dominant_class,
        "dominant_class_fraction": dominant_class_fraction,
        "dominant_instance_label_id": dominant_label_id,
        "dominant_instance_overlap": dominant_overlap,
        "dominant_instance_gaussian_count": dominant_size,
        "dominant_instance_coverage": dominant_coverage,
        "partial_overlap_risk": risk,
    }


def dense_class_geometry(
    indices: np.ndarray,
    points: np.ndarray,
    log_scales: np.ndarray,
    thresholds: AuditThresholds,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    median_scale, voxel_size = adaptive_voxel_size(
        log_scales[indices],
        voxel_scale_multiplier=thresholds.voxel_scale_multiplier,
        min_voxel_size=thresholds.min_voxel_size,
        max_voxel_size=thresholds.max_voxel_size,
    )
    component_ids, component_sizes, geometry = voxel_components(
        points[indices], voxel_size
    )
    return component_ids, component_sizes, {
        "median_gaussian_scale": median_scale,
        "voxel_size": voxel_size,
        **geometry,
    }


def audit_partial_anchor_completions(
    *,
    combined_seed_labels: np.ndarray,
    component_instance_labels: np.ndarray,
    component_project_labels: np.ndarray,
    dense_labels: np.ndarray,
    supporting_views: np.ndarray,
    winner_share: np.ndarray,
    points: np.ndarray,
    log_scales: np.ndarray,
    component_items: dict[int, dict[str, Any]],
    thresholds: AuditThresholds,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Return deterministic per-anchor reports and accepted fill proposals."""

    thresholds.validate()
    seeds = np.asarray(combined_seed_labels)
    instances = np.asarray(component_instance_labels)
    component_projects = np.asarray(component_project_labels)
    dense = np.asarray(dense_labels)
    views = np.asarray(supporting_views)
    shares = np.asarray(winner_share, dtype=np.float64)
    vertex_count = int(seeds.shape[0])
    for values, name in (
        (instances, "component_instance_labels"),
        (component_projects, "component_project_labels"),
        (dense, "dense_labels"),
        (views, "supporting_views"),
        (shares, "winner_share"),
    ):
        validate_vector(values, name=name, vertex_count=vertex_count, integer=name != "winner_share")
    if points.shape != (vertex_count, 3) or log_scales.shape != (vertex_count, 3):
        raise ValueError("points and log_scales must both have shape (N, 3)")
    if np.any(views < 0):
        raise ValueError("supporting_views cannot be negative")
    if not np.all(np.isfinite(shares)) or np.any((shares < 0.0) | (shares > 1.0)):
        raise ValueError("winner_share must contain finite values in [0, 1]")

    positive_instances = set(int(value) for value in np.unique(instances) if value > 0)
    missing_items = sorted(positive_instances - set(component_items))
    if missing_items:
        raise ValueError(f"component labels are absent from label map: {missing_items}")

    class_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    accepted_fills: list[np.ndarray] = []
    for instance_label_id in sorted(component_items):
        if instance_label_id == 0:
            continue
        item = component_items[instance_label_id]
        if str(item["type"]) != "thing":
            continue
        project_id = int(item["project_id"])
        class_name = str(item["class"])
        anchor_indices = np.flatnonzero(instances == instance_label_id)
        if anchor_indices.size == 0:
            raise ValueError(f"component label {instance_label_id} has no Gaussians")
        if np.any(component_projects[anchor_indices] != project_id):
            raise ValueError(
                f"component label {instance_label_id} has inconsistent project IDs"
            )

        if project_id not in class_cache:
            dense_indices = np.flatnonzero(dense == project_id)
            if dense_indices.size:
                component_ids, component_sizes, geometry = dense_class_geometry(
                    dense_indices, points, log_scales, thresholds
                )
            else:
                component_ids = np.zeros((0,), dtype=np.int64)
                component_sizes = np.zeros((0,), dtype=np.int64)
                geometry = {
                    "median_gaussian_scale": None,
                    "voxel_size": None,
                    "voxel_count": 0,
                    "component_count": 0,
                    "largest_component_gaussians": 0,
                }
            class_cache[project_id] = (
                dense_indices,
                component_ids,
                component_sizes,
                geometry,
            )
        dense_indices, component_ids, component_sizes, geometry = class_cache[project_id]

        anchor_membership = instances[dense_indices] == instance_label_id
        anchor_counts = np.bincount(
            component_ids[anchor_membership],
            minlength=component_sizes.size,
        ).astype(np.int64, copy=False)
        component_precision = np.divide(
            anchor_counts,
            np.maximum(component_sizes, 1),
            dtype=np.float64,
        )
        initially_kept = np.flatnonzero(
            (anchor_counts > 0)
            & (component_precision >= thresholds.min_anchor_precision)
        )
        maximum_initial_precision = float(
            component_precision.max() if component_precision.size else 0.0
        )

        selected_indices = np.zeros((0,), dtype=np.int64)
        if initially_kept.size == 1:
            selected_indices = dense_indices[
                component_ids == int(initially_kept[0])
            ]
        conflicting_seed_mask = (
            (seeds[selected_indices] != 0)
            & (seeds[selected_indices] != project_id)
        )
        conflict_free_indices = selected_indices[~conflicting_seed_mask]

        retained_component_count = 0
        retained_indices = np.zeros((0,), dtype=np.int64)
        retained_geometry: dict[str, Any] = {
            "voxel_count": 0,
            "component_count": 0,
            "largest_component_gaussians": 0,
        }
        if conflict_free_indices.size:
            retained_local, _retained_sizes, retained_geometry = voxel_components(
                points[conflict_free_indices], float(geometry["voxel_size"])
            )
            retained_anchor = (
                instances[conflict_free_indices] == instance_label_id
            )
            retained_anchor_components = np.unique(retained_local[retained_anchor])
            retained_component_count = int(retained_anchor_components.size)
            if retained_anchor_components.size:
                retained_indices = conflict_free_indices[
                    np.isin(retained_local, retained_anchor_components)
                ]

        anchor_overlap = int(
            np.count_nonzero(instances[retained_indices] == instance_label_id)
        )
        anchor_coverage = anchor_overlap / float(anchor_indices.size)
        anchor_precision = anchor_overlap / float(max(retained_indices.size, 1))
        spatial_keep_fraction = retained_indices.size / float(
            max(selected_indices.size, 1)
        )
        robust = views[retained_indices] >= thresholds.min_candidate_supporting_views
        robust_fraction = float(np.mean(robust)) if retained_indices.size else 0.0
        winner_share_values = shares[retained_indices]
        competing = competing_thing_metrics(
            selected_indices,
            instances,
            component_items,
            target_class=class_name,
        )
        proposed_fill = retained_indices[seeds[retained_indices] == 0]

        reasons: list[str] = []
        if initially_kept.size != 1:
            reasons.append(f"anchor_connected_components!={1}")
        if retained_component_count != 1:
            reasons.append(f"retained_anchor_components!={1}")
        if anchor_overlap < thresholds.min_anchor_gaussians:
            reasons.append(f"anchor_overlap<{thresholds.min_anchor_gaussians}")
        if anchor_coverage < thresholds.min_anchor_coverage:
            reasons.append(f"anchor_coverage<{thresholds.min_anchor_coverage}")
        if anchor_precision < thresholds.min_anchor_precision:
            reasons.append(f"anchor_precision<{thresholds.min_anchor_precision}")
        if anchor_precision > thresholds.max_anchor_precision:
            reasons.append(f"anchor_precision>{thresholds.max_anchor_precision}")
        if int(item["source_view_count"]) < thresholds.min_source_views:
            reasons.append(f"source_views<{thresholds.min_source_views}")
        if robust_fraction < thresholds.min_robust_fraction:
            reasons.append(f"robust_fraction<{thresholds.min_robust_fraction}")
        if spatial_keep_fraction < thresholds.min_spatial_keep_fraction:
            reasons.append(
                f"spatial_keep_fraction<{thresholds.min_spatial_keep_fraction}"
            )
        if (
            float(competing["dominant_class_fraction"])
            > thresholds.max_competing_thing_fraction
        ):
            reasons.append(
                "competing_thing_fraction>"
                f"{thresholds.max_competing_thing_fraction}"
            )
        if (
            float(competing["partial_overlap_risk"])
            > thresholds.max_partial_competing_thing_risk
        ):
            reasons.append(
                "partial_competing_thing_risk>"
                f"{thresholds.max_partial_competing_thing_risk}"
            )
        if proposed_fill.size == 0:
            reasons.append("no_seed_unlabeled_fill")

        accepted = not reasons
        if accepted:
            accepted_fills.append(proposed_fill.astype(np.uint32, copy=False))
        records.append(
            {
                "component_instance_label_id": instance_label_id,
                "component_id": int(item["component_id"]),
                "project_id": project_id,
                "class": class_name,
                "type": str(item["type"]),
                "status": "accepted_proposed_completion" if accepted else "rejected",
                "reasons": reasons,
                "source_view_count": int(item["source_view_count"]),
                "anchor_gaussian_count": int(anchor_indices.size),
                "dense_same_class_gaussian_count": int(dense_indices.size),
                "dense_anchor_overlap_before_spatial_guard": int(anchor_membership.sum()),
                "initial_anchor_connected_component_count": int(initially_kept.size),
                "maximum_initial_component_anchor_precision": maximum_initial_precision,
                "selected_component_gaussian_count": int(selected_indices.size),
                "conflicting_seed_gaussian_count": int(conflicting_seed_mask.sum()),
                "retained_anchor_component_count": retained_component_count,
                "retained_gaussian_count": int(retained_indices.size),
                "anchor_overlap": anchor_overlap,
                "anchor_coverage": anchor_coverage,
                "anchor_precision": anchor_precision,
                "spatial_keep_fraction": spatial_keep_fraction,
                "robust_supporting_view_threshold": (
                    thresholds.min_candidate_supporting_views
                ),
                "robust_gaussian_count": int(np.count_nonzero(robust)),
                "robust_fraction": robust_fraction,
                "winner_share_mean": float(
                    winner_share_values.mean() if winner_share_values.size else 0.0
                ),
                "winner_share_min": float(
                    winner_share_values.min() if winner_share_values.size else 0.0
                ),
                "dominant_competing_thing_class": str(competing["dominant_class"]),
                "dominant_competing_thing_fraction": float(
                    competing["dominant_class_fraction"]
                ),
                "dominant_competing_thing_instance_label_id": int(
                    competing["dominant_instance_label_id"]
                ),
                "dominant_competing_thing_overlap": int(
                    competing["dominant_instance_overlap"]
                ),
                "dominant_competing_thing_instance_gaussian_count": int(
                    competing["dominant_instance_gaussian_count"]
                ),
                "dominant_competing_thing_instance_coverage": float(
                    competing["dominant_instance_coverage"]
                ),
                "dominant_competing_thing_partial_overlap_risk": float(
                    competing["partial_overlap_risk"]
                ),
                "proposed_fill_gaussian_count": int(proposed_fill.size),
                "dense_class_geometry": geometry,
                "retained_geometry": retained_geometry,
            }
        )
    return records, accepted_fills


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-seed-labels", required=True, type=Path)
    parser.add_argument("--combined-seed-summary", required=True, type=Path)
    parser.add_argument("--component-instance-labels", required=True, type=Path)
    parser.add_argument("--component-project-labels", required=True, type=Path)
    parser.add_argument("--component-label-map", required=True, type=Path)
    parser.add_argument("--component-summary", required=True, type=Path)
    parser.add_argument("--dense-labels", required=True, type=Path)
    parser.add_argument("--dense-supporting-views", required=True, type=Path)
    parser.add_argument("--dense-winner-share", required=True, type=Path)
    parser.add_argument("--dense-summary", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--min-anchor-gaussians", default=500, type=int)
    parser.add_argument("--min-anchor-coverage", default=0.80, type=float)
    parser.add_argument("--min-anchor-precision", default=0.10, type=float)
    parser.add_argument("--max-anchor-precision", default=0.25, type=float)
    parser.add_argument("--min-robust-fraction", default=0.90, type=float)
    parser.add_argument("--min-source-views", default=5, type=int)
    parser.add_argument("--min-candidate-supporting-views", default=5, type=int)
    parser.add_argument("--max-competing-thing-fraction", default=0.05, type=float)
    parser.add_argument("--min-spatial-keep-fraction", default=0.95, type=float)
    parser.add_argument(
        "--max-partial-competing-thing-risk", default=0.40, type=float
    )
    parser.add_argument("--voxel-scale-multiplier", default=4.0, type=float)
    parser.add_argument("--min-voxel-size", default=0.01, type=float)
    parser.add_argument("--max-voxel-size", default=0.20, type=float)
    args = parser.parse_args()

    input_paths = (
        args.combined_seed_labels,
        args.combined_seed_summary,
        args.component_instance_labels,
        args.component_project_labels,
        args.component_label_map,
        args.component_summary,
        args.dense_labels,
        args.dense_supporting_views,
        args.dense_winner_share,
        args.dense_summary,
        args.ontology,
        args.source_ply,
    )
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.scene.strip():
        raise ValueError("scene must be non-empty")

    thresholds = AuditThresholds(
        min_anchor_gaussians=args.min_anchor_gaussians,
        min_anchor_coverage=args.min_anchor_coverage,
        min_anchor_precision=args.min_anchor_precision,
        max_anchor_precision=args.max_anchor_precision,
        min_robust_fraction=args.min_robust_fraction,
        min_source_views=args.min_source_views,
        min_candidate_supporting_views=args.min_candidate_supporting_views,
        max_competing_thing_fraction=args.max_competing_thing_fraction,
        min_spatial_keep_fraction=args.min_spatial_keep_fraction,
        max_partial_competing_thing_risk=args.max_partial_competing_thing_risk,
        voxel_scale_multiplier=args.voxel_scale_multiplier,
        min_voxel_size=args.min_voxel_size,
        max_voxel_size=args.max_voxel_size,
    )
    thresholds.validate()

    ontology = load_ontology(args.ontology)
    ply_header, vertices = vertex_data_memmap(args.source_ply)
    vertex_count = int(ply_header.elements[0].count)
    required_properties = {"x", "y", "z", "scale_0", "scale_1", "scale_2"}
    missing = sorted(required_properties - set(vertices.dtype.names or ()))
    if missing:
        raise ValueError(f"source PLY is missing geometry properties: {missing}")

    combined_summary = load_source_summary(
        args.combined_seed_summary,
        allowed_contracts={(COMBINED_SEED_SOURCE, COMBINED_SEED_CONTRACT)},
        source_ply=args.source_ply,
        vertex_count=vertex_count,
        scene=args.scene,
    )
    component_summary = load_source_summary(
        args.component_summary,
        allowed_contracts={(CAMERA_COMPONENT_SOURCE, CAMERA_COMPONENT_CONTRACT)},
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
    component_label_map = json.loads(
        args.component_label_map.read_text(encoding="utf-8")
    )
    component_items = validate_component_items(
        component_label_map, ontology=ontology, scene=args.scene
    )

    combined_seeds = validate_project_labels(
        np.load(args.combined_seed_labels, mmap_mode="r", allow_pickle=False),
        name="combined seed labels",
        vertex_count=vertex_count,
        ontology=ontology,
    )
    component_instances = validate_vector(
        np.load(args.component_instance_labels, mmap_mode="r", allow_pickle=False),
        name="component instance labels",
        vertex_count=vertex_count,
    ).astype(np.int32, copy=False)
    component_projects = validate_project_labels(
        np.load(args.component_project_labels, mmap_mode="r", allow_pickle=False),
        name="component project labels",
        vertex_count=vertex_count,
        ontology=ontology,
    )
    dense_labels = validate_project_labels(
        np.load(args.dense_labels, mmap_mode="r", allow_pickle=False),
        name="dense labels",
        vertex_count=vertex_count,
        ontology=ontology,
    )
    supporting_views = validate_vector(
        np.load(args.dense_supporting_views, mmap_mode="r", allow_pickle=False),
        name="supporting views",
        vertex_count=vertex_count,
    )
    winner_share = validate_vector(
        np.load(args.dense_winner_share, mmap_mode="r", allow_pickle=False),
        name="winner share",
        vertex_count=vertex_count,
        integer=False,
    )
    points = np.column_stack(
        [vertices[axis].astype(np.float64) for axis in ("x", "y", "z")]
    )
    log_scales = np.column_stack(
        [
            vertices[axis].astype(np.float64)
            for axis in ("scale_0", "scale_1", "scale_2")
        ]
    )

    records, accepted_fills = audit_partial_anchor_completions(
        combined_seed_labels=combined_seeds,
        component_instance_labels=component_instances,
        component_project_labels=component_projects,
        dense_labels=dense_labels,
        supporting_views=supporting_views,
        winner_share=winner_share,
        points=points,
        log_scales=log_scales,
        component_items=component_items,
        thresholds=thresholds,
    )

    args.output_dir.mkdir(parents=True)
    owner_count = np.zeros((vertex_count,), dtype=np.uint16)
    offsets = [0]
    proposal_indices: list[np.ndarray] = []
    accepted_records = [
        record for record in records if record["status"] == "accepted_proposed_completion"
    ]
    for fill in accepted_fills:
        if np.any(combined_seeds[fill] != 0):
            raise RuntimeError("a proposed fill overlaps an immutable DINOv3 seed")
        if owner_count[fill].size and int(owner_count[fill].max()) == np.iinfo(np.uint16).max:
            raise OverflowError("proposed fill owner count exceeds uint16 capacity")
        owner_count[fill] += np.uint16(1)
        proposal_indices.append(fill)
        offsets.append(offsets[-1] + int(fill.size))
    concatenated = (
        np.concatenate(proposal_indices).astype(np.uint32, copy=False)
        if proposal_indices
        else np.zeros((0,), dtype=np.uint32)
    )
    proposed_fill_mask = owner_count > 0
    overlap_mask = owner_count > 1
    np.save(args.output_dir / "proposed_fill_mask.npy", proposed_fill_mask)
    np.save(args.output_dir / "proposed_fill_owner_count.npy", owner_count)
    np.save(args.output_dir / "proposed_fill_overlap_mask.npy", overlap_mask)
    np.savez_compressed(
        args.output_dir / "proposed_fill_supports.npz",
        component_instance_label_ids=np.asarray(
            [record["component_instance_label_id"] for record in accepted_records],
            dtype=np.int32,
        ),
        project_ids=np.asarray(
            [record["project_id"] for record in accepted_records], dtype=np.int32
        ),
        offsets=np.asarray(offsets, dtype=np.int64),
        indices=concatenated,
    )

    rejection_counts: dict[str, int] = {}
    for record in records:
        for reason in record["reasons"]:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    summary = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": args.scene,
        "source_ply": str(args.source_ply),
        "ontology": str(args.ontology),
        "combined_seed_labels": str(args.combined_seed_labels),
        "combined_seed_summary": str(args.combined_seed_summary),
        "component_instance_labels": str(args.component_instance_labels),
        "component_project_labels": str(args.component_project_labels),
        "component_label_map": str(args.component_label_map),
        "component_summary": str(args.component_summary),
        "dense_labels": str(args.dense_labels),
        "dense_supporting_views": str(args.dense_supporting_views),
        "dense_winner_share": str(args.dense_winner_share),
        "dense_summary": str(args.dense_summary),
        "v5_used": False,
        "v5_labels_used": False,
        "v5_method_adapted": True,
        "dinov2_used": False,
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "offline_threshold_sweep_used": False,
        "audit_scope": "dinov3_camera_owned_thing_component_instances_only",
        "anchor_policy": "immutable_dinov3_component_instance",
        "candidate_policy": "same_project_class_complete_dense_dinov3_votes",
        "conflict_policy": (
            "remove_conflicting_nonzero_seeds_then_recompute_anchor_connectivity"
        ),
        "completion_policy": (
            "exactly_one_strong_partial_anchor_component_under_global_v5_style_gates"
        ),
        "competing_instance_risk": (
            "dominant_competing_thing_fraction_times_one_minus_instance_coverage"
        ),
        "thresholds": {
            field: getattr(thresholds, field)
            for field in thresholds.__dataclass_fields__
        },
        "vertex_count": vertex_count,
        "component_instance_count": len(component_items) - 1,
        "thing_anchor_audit_count": len(records),
        "accepted_proposal_count": len(accepted_records),
        "rejected_proposal_count": len(records) - len(accepted_records),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "proposed_fill_gaussian_count": int(np.count_nonzero(proposed_fill_mask)),
        "overlapping_proposal_gaussian_count": int(np.count_nonzero(overlap_mask)),
        "immutable_seed_gaussian_count": int(np.count_nonzero(combined_seeds)),
        "immutable_seed_overlap_with_proposed_fill_count": int(
            np.count_nonzero((combined_seeds > 0) & proposed_fill_mask)
        ),
        "component_audits": records,
        "semantic_labels_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
        "proposal_masks_written": True,
        "input_combined_seed_summary": combined_summary,
        "input_component_summary": component_summary,
        "input_dense_summary": dense_summary,
    }
    (args.output_dir / "partial_anchor_completion_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
