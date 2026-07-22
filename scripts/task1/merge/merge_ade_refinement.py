#!/usr/bin/env python3
"""Guarded same-ontology refinement of DINO labels with Grounded-SAM evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.common.ply_utils import (
    read_ply_header,
    resolve_semantic_ply_output,
    vertex_data_memmap,
)
from scripts.task1.common.semantic_palette import PALETTE_VERSION, rgb8_for_class
from scripts.task1.dinov2.dinov2_ontology import load_ontology, normalize_class_name
from scripts.task1.grounding.cluster_semantic_flashsplat_proposals import voxel_components
from scripts.task1.merge.merge_semantic_extensions import (
    histogram,
    label_items_by_id,
    load_json,
    transition_records,
    validate_label_array,
)
from scripts.task1.qa.add_labels_from_npy import write_ply_with_labels


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def class_array(labels: np.ndarray, items: dict[int, dict[str, Any]]) -> np.ndarray:
    max_id = max(items)
    lookup = np.empty((max_id + 1,), dtype=object)
    lookup[:] = ""
    for label_id, item in items.items():
        lookup[label_id] = normalize_class_name(item.get("class", ""))
    return lookup[labels]


def load_refinement_config(path: Path, scene: str) -> dict[str, Any]:
    config = load_json(path)
    if str(config.get("scene", "")) != scene:
        raise ValueError("Refinement config scene does not match the requested scene")
    missing_vocabulary_enabled = bool(config.get("missing_vocabulary_enabled", False))
    if missing_vocabulary_enabled:
        raw_extensions = config.get("selected_extension_classes")
        if not isinstance(raw_extensions, list) or not raw_extensions:
            raise ValueError(
                "Unified refinement config enables missing vocabulary without extensions"
            )
    raw_classes = config.get("selected_refinement_classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise ValueError("Refinement config has no selected_refinement_classes")
    classes = [normalize_class_name(value) for value in raw_classes]
    if any(not value for value in classes) or len(classes) != len(set(classes)):
        raise ValueError("Refinement classes must be unique non-empty names")
    return {**config, "selected_refinement_classes": classes}


def load_unique_claims(
    evidence_dir: Path,
    eligible_classes: list[str],
    gaussian_count: int,
    min_positive_views: int,
    min_ratio: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = evidence_dir / "class_evidence_manifest.json"
    manifest = load_json(manifest_path)
    raw_records = manifest.get("classes")
    if not isinstance(raw_records, list):
        raise ValueError("Class-evidence manifest must contain a classes list")
    records: dict[str, dict[str, Any]] = {}
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("Every class-evidence record must be an object")
        class_name = normalize_class_name(raw.get("class", ""))
        if class_name:
            records[class_name] = raw

    claim_count = np.zeros((gaussian_count,), dtype=np.uint16)
    robust_by_class: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    missing: list[str] = []
    for class_name in eligible_classes:
        if class_name not in records:
            robust_by_class[class_name] = np.zeros((0,), dtype=np.uint32)
            counts[class_name] = 0
            missing.append(class_name)
            continue
        path = evidence_dir / str(records[class_name]["file"])
        with np.load(path) as data:
            indices = data["indices"].astype(np.uint32, copy=False)
            positive = data["positive_views"].astype(np.float32, copy=False)
            negative = data["negative_views"].astype(np.float32, copy=False)
        if indices.ndim != 1 or positive.shape != indices.shape or negative.shape != indices.shape:
            raise ValueError(f"Malformed class evidence for {class_name}")
        if indices.shape[0] and int(indices[-1]) >= gaussian_count:
            raise ValueError(f"Class evidence index exceeds Gaussian count for {class_name}")
        ratio = positive / np.maximum(positive + negative, 1.0)
        robust = np.unique(indices[(positive >= min_positive_views) & (ratio >= min_ratio)])
        robust_by_class[class_name] = robust
        claim_count[robust] += 1
        counts[class_name] = int(robust.shape[0])
    report = {
        "manifest": str(manifest_path),
        "min_positive_views": min_positive_views,
        "min_ratio": min_ratio,
        "classes_without_evidence": missing,
        "robust_claim_counts": counts,
        "unclaimed_gaussian_count": int(np.count_nonzero(claim_count == 0)),
        "unique_claim_gaussian_count": int(np.count_nonzero(claim_count == 1)),
        "ambiguous_claim_gaussian_count": int(np.count_nonzero(claim_count > 1)),
    }
    return claim_count, robust_by_class, report


def mask_from_sorted_indices(indices: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(indices, candidates)
    matched = positions < indices.shape[0]
    if matched.any():
        matched_positions = positions[matched]
        matched[matched] = indices[matched_positions] == candidates[matched]
    return matched


def spatial_anchor_components(
    guarded_indices: np.ndarray,
    anchor_membership: np.ndarray,
    vertex_data: np.ndarray | None,
    min_anchor_precision: float,
    voxel_scale_multiplier: float,
    min_voxel_size: float,
    max_voxel_size: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep only spatial components sufficiently seeded by the DINO anchor."""
    if guarded_indices.shape != anchor_membership.shape:
        raise ValueError("Guarded indices and anchor membership must have matching shapes")
    if guarded_indices.shape[0] == 0:
        return np.zeros((0,), dtype=bool), {
            "voxel_size": None,
            "component_count": 0,
            "kept_component_count": 0,
            "removed_gaussian_count": 0,
        }

    if vertex_data is None:
        component_ids = np.zeros(guarded_indices.shape, dtype=np.int64)
        component_sizes = np.asarray([guarded_indices.shape[0]], dtype=np.int64)
        voxel_size = None
        component_stats = {"voxel_count": 0, "component_count": 1}
    else:
        required = {"x", "y", "z", "scale_0", "scale_1", "scale_2"}
        missing = sorted(required - set(vertex_data.dtype.names or ()))
        if missing:
            raise ValueError(f"PLY is missing ADE spatial-guard properties: {missing}")
        points = np.column_stack(
            [vertex_data[axis][guarded_indices].astype(np.float64) for axis in ("x", "y", "z")]
        )
        log_scales = np.column_stack(
            [
                vertex_data[axis][guarded_indices].astype(np.float64)
                for axis in ("scale_0", "scale_1", "scale_2")
            ]
        )
        gaussian_scales = np.exp(np.clip(log_scales.max(axis=1), -20.0, 5.0))
        finite_scales = gaussian_scales[np.isfinite(gaussian_scales)]
        if finite_scales.shape[0] == 0:
            raise ValueError("Could not derive a finite Gaussian scale for ADE spatial guard")
        median_scale = float(np.median(finite_scales))
        voxel_size = max(min_voxel_size, median_scale * voxel_scale_multiplier)
        if max_voxel_size > 0:
            voxel_size = min(voxel_size, max_voxel_size)
        component_ids, component_sizes, component_stats = voxel_components(points, voxel_size)

    anchor_counts = np.bincount(
        component_ids,
        weights=anchor_membership.astype(np.int64),
        minlength=component_sizes.shape[0],
    ).astype(np.int64)
    component_precision = anchor_counts / np.maximum(component_sizes, 1)
    kept_components = np.flatnonzero(
        (anchor_counts > 0) & (component_precision >= min_anchor_precision)
    )
    keep = np.isin(component_ids, kept_components)
    return keep, {
        "voxel_size": voxel_size,
        "component_count": int(component_sizes.shape[0]),
        "kept_component_count": int(kept_components.shape[0]),
        "removed_gaussian_count": int(np.count_nonzero(~keep)),
        "largest_component_gaussians": int(component_sizes.max()),
        "largest_kept_component_gaussians": int(
            component_sizes[kept_components].max() if kept_components.shape[0] else 0
        ),
        "max_component_anchor_precision": float(component_precision.max()),
        "min_kept_component_anchor_precision": float(
            component_precision[kept_components].min() if kept_components.shape[0] else 0.0
        ),
        **component_stats,
    }


def robust_geometry_signature(
    indices: np.ndarray,
    vertex_data: np.ndarray | None,
    quantile: float = 0.01,
) -> np.ndarray | None:
    """Return rotation-invariant robust PCA extents for a Gaussian region."""
    if vertex_data is None or indices.shape[0] < 3:
        return None
    if not 0.0 <= quantile < 0.5:
        raise ValueError("Geometry-signature quantile must be in [0, 0.5)")
    points = np.column_stack(
        [vertex_data[axis][indices].astype(np.float64) for axis in ("x", "y", "z")]
    )
    center = np.median(points, axis=0)
    centered = points - center
    covariance = centered.T @ centered / float(max(points.shape[0] - 1, 1))
    _values, axes = np.linalg.eigh(covariance)
    projected = centered @ axes
    lower = np.quantile(projected, quantile, axis=0)
    upper = np.quantile(projected, 1.0 - quantile, axis=0)
    extents = np.sort(np.maximum(upper - lower, 1e-8))[::-1]
    return extents


def prototype_geometry_matches(
    candidate: np.ndarray | None,
    prototypes: list[np.ndarray],
    max_shape_ratio: float,
    max_size_ratio: float,
) -> bool:
    if candidate is None or not prototypes:
        return False
    if max_shape_ratio < 1.0 or max_size_ratio < 1.0:
        raise ValueError("Prototype geometry ratios must be at least one")
    candidate_shape = candidate / max(float(candidate[0]), 1e-8)
    for prototype in prototypes:
        prototype_shape = prototype / max(float(prototype[0]), 1e-8)
        shape_ratio = np.maximum(
            candidate_shape / np.maximum(prototype_shape, 1e-8),
            prototype_shape / np.maximum(candidate_shape, 1e-8),
        )
        size_ratio = max(
            float(candidate[0]) / max(float(prototype[0]), 1e-8),
            float(prototype[0]) / max(float(candidate[0]), 1e-8),
        )
        if float(shape_ratio.max()) <= max_shape_ratio and size_ratio <= max_size_ratio:
            return True
    return False


def competing_thing_metrics(
    candidate_indices: np.ndarray,
    base_labels: np.ndarray,
    base_items: dict[int, dict[str, Any]],
    base_classes: np.ndarray,
    thing_class_values: np.ndarray,
    target_class: str,
) -> dict[str, Any]:
    """Measure semantic conflict and its impact on the dominant base instance."""
    empty = {
        "dominant_class": "",
        "dominant_class_fraction": 0.0,
        "dominant_instance_label_id": 0,
        "dominant_instance_overlap": 0,
        "dominant_instance_gaussian_count": 0,
        "dominant_instance_coverage": 0.0,
        "partial_overlap_risk": 0.0,
    }
    if candidate_indices.shape[0] == 0 or thing_class_values.shape[0] == 0:
        return empty

    candidate_classes = base_classes[candidate_indices]
    competing = np.isin(candidate_classes, thing_class_values) & (
        candidate_classes != target_class
    )
    if not np.any(competing):
        return empty

    competing_candidate_indices = candidate_indices[competing]
    competing_classes = candidate_classes[competing]
    class_names, class_counts = np.unique(competing_classes, return_counts=True)
    dominant_class_position = int(np.argmax(class_counts))
    dominant_class = str(class_names[dominant_class_position])
    dominant_class_fraction = float(class_counts[dominant_class_position]) / float(
        max(candidate_indices.shape[0], 1)
    )

    competing_label_ids = base_labels[
        competing_candidate_indices[competing_classes == dominant_class]
    ]
    competing_label_ids = competing_label_ids[competing_label_ids > 0]
    if competing_label_ids.shape[0] == 0:
        return {
            **empty,
            "dominant_class": dominant_class,
            "dominant_class_fraction": dominant_class_fraction,
        }
    label_ids, label_counts = np.unique(competing_label_ids, return_counts=True)
    dominant_position = int(np.argmax(label_counts))
    dominant_label_id = int(label_ids[dominant_position])
    dominant_overlap = int(label_counts[dominant_position])
    dominant_size = int(np.count_nonzero(base_labels == dominant_label_id))
    dominant_coverage = dominant_overlap / float(max(dominant_size, 1))

    if dominant_label_id not in base_items:
        raise ValueError(f"Competing base label {dominant_label_id} is absent from label map")
    partial_overlap_risk = dominant_class_fraction * (1.0 - dominant_coverage)
    return {
        "dominant_class": dominant_class,
        "dominant_class_fraction": dominant_class_fraction,
        "dominant_instance_label_id": dominant_label_id,
        "dominant_instance_overlap": dominant_overlap,
        "dominant_instance_gaussian_count": dominant_size,
        "dominant_instance_coverage": dominant_coverage,
        "partial_overlap_risk": partial_overlap_risk,
    }


def anchor_envelope_membership(
    anchor_indices: np.ndarray,
    candidate_indices: np.ndarray,
    vertex_data: np.ndarray | None,
    quantile: float,
    margin_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Test candidates against a robust oriented envelope of a DINO instance."""
    if candidate_indices.shape[0] == 0:
        return np.zeros((0,), dtype=bool), {
            "enabled": vertex_data is not None,
            "anchor_gaussian_count": int(anchor_indices.shape[0]),
            "candidate_gaussian_count": 0,
            "kept_gaussian_count": 0,
        }
    if vertex_data is None or anchor_indices.shape[0] < 3:
        return np.ones(candidate_indices.shape, dtype=bool), {
            "enabled": False,
            "anchor_gaussian_count": int(anchor_indices.shape[0]),
            "candidate_gaussian_count": int(candidate_indices.shape[0]),
            "kept_gaussian_count": int(candidate_indices.shape[0]),
        }
    if not 0.0 <= quantile < 0.5:
        raise ValueError("Anchor-envelope quantile must be in [0, 0.5)")
    if margin_ratio < 0.0:
        raise ValueError("Anchor-envelope margin ratio must be non-negative")

    anchor_points = np.column_stack(
        [vertex_data[axis][anchor_indices].astype(np.float64) for axis in ("x", "y", "z")]
    )
    candidate_points = np.column_stack(
        [
            vertex_data[axis][candidate_indices].astype(np.float64)
            for axis in ("x", "y", "z")
        ]
    )
    center = np.median(anchor_points, axis=0)
    centered_anchor = anchor_points - center
    covariance = centered_anchor.T @ centered_anchor / float(
        max(anchor_points.shape[0] - 1, 1)
    )
    _values, axes = np.linalg.eigh(covariance)
    anchor_projected = centered_anchor @ axes
    candidate_projected = (candidate_points - center) @ axes
    lower = np.quantile(anchor_projected, quantile, axis=0)
    upper = np.quantile(anchor_projected, 1.0 - quantile, axis=0)
    spans = np.maximum(upper - lower, 1e-8)
    margin = spans * margin_ratio
    keep = np.all(
        (candidate_projected >= (lower - margin))
        & (candidate_projected <= (upper + margin)),
        axis=1,
    )
    return keep, {
        "enabled": True,
        "quantile": quantile,
        "margin_ratio": margin_ratio,
        "anchor_gaussian_count": int(anchor_indices.shape[0]),
        "candidate_gaussian_count": int(candidate_indices.shape[0]),
        "kept_gaussian_count": int(np.count_nonzero(keep)),
        "robust_extents": [float(value) for value in np.sort(spans)[::-1]],
    }


def spatial_unanchored_components(
    guarded_indices: np.ndarray,
    vertex_data: np.ndarray | None,
    min_component_gaussians: int,
    voxel_scale_multiplier: float,
    min_voxel_size: float,
    max_voxel_size: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prune small islands from a prototype-backed, otherwise unanchored group."""
    if guarded_indices.shape[0] == 0 or vertex_data is None:
        return np.ones(guarded_indices.shape, dtype=bool), {
            "voxel_size": None,
            "component_count": int(guarded_indices.shape[0] > 0),
            "kept_component_count": int(guarded_indices.shape[0] > 0),
            "removed_gaussian_count": 0,
        }
    points = np.column_stack(
        [vertex_data[axis][guarded_indices].astype(np.float64) for axis in ("x", "y", "z")]
    )
    log_scales = np.column_stack(
        [
            vertex_data[axis][guarded_indices].astype(np.float64)
            for axis in ("scale_0", "scale_1", "scale_2")
        ]
    )
    scales = np.exp(np.clip(log_scales.max(axis=1), -20.0, 5.0))
    voxel_size = max(min_voxel_size, float(np.median(scales)) * voxel_scale_multiplier)
    if max_voxel_size > 0:
        voxel_size = min(voxel_size, max_voxel_size)
    component_ids, component_sizes, component_stats = voxel_components(points, voxel_size)
    threshold = max(min_component_gaussians, int(np.ceil(component_sizes.max() * 0.10)))
    kept_components = np.flatnonzero(component_sizes >= threshold)
    keep = np.isin(component_ids, kept_components)
    return keep, {
        "voxel_size": voxel_size,
        "component_count": int(component_sizes.shape[0]),
        "kept_component_count": int(kept_components.shape[0]),
        "component_keep_threshold": threshold,
        "removed_gaussian_count": int(np.count_nonzero(~keep)),
        **component_stats,
    }


def refine_ade_labels(
    base_labels: np.ndarray,
    base_items: dict[int, dict[str, Any]],
    grounding_labels: np.ndarray,
    grounding_items: dict[int, dict[str, Any]],
    eligible_classes: list[str],
    claim_count: np.ndarray,
    robust_by_class: dict[str, np.ndarray],
    min_anchor_gaussians: int,
    min_anchor_coverage: float,
    min_group_proposals: int,
    min_group_source_views: int,
    min_anchor_precision: float = 0.10,
    vertex_data: np.ndarray | None = None,
    spatial_voxel_scale_multiplier: float = 4.0,
    spatial_min_voxel_size: float = 0.01,
    spatial_max_voxel_size: float = 0.20,
    class_kinds: dict[str, str] | None = None,
    anchor_envelope_quantile: float = 0.01,
    anchor_envelope_margin_ratio: float = 0.05,
    ambiguous_thing_min_group_anchor_ratio: float = 0.75,
    thing_halo_min_anchor_precision: float = 0.75,
    nested_thing_min_containment: float = 0.80,
    nested_thing_envelope_margin_ratio: float = 0.25,
    nested_thing_min_parent_source_ratio: float = 2.00,
    nested_thing_min_parent_overlap_coverage: float = 0.10,
    prototype_min_robust_fraction: float = 0.50,
    prototype_min_source_views: int = 3,
    prototype_strong_source_views: int = 5,
    prototype_max_dominant_competing_thing_fraction: float = 0.50,
    prototype_max_partial_competing_thing_risk: float = 0.40,
    prototype_ambiguous_min_dominant_competing_thing_fraction: float = 0.90,
    prototype_max_shape_ratio: float = 2.50,
    prototype_max_size_ratio: float = 3.00,
    partial_anchor_extension_min_anchor_coverage: float = 0.80,
    partial_anchor_extension_max_anchor_precision: float = 0.25,
    partial_anchor_extension_min_robust_fraction: float = 0.90,
    partial_anchor_extension_min_source_views: int = 5,
    partial_anchor_extension_max_competing_thing_fraction: float = 0.05,
    partial_anchor_extension_min_spatial_keep_fraction: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    if base_labels.shape != grounding_labels.shape or base_labels.shape != claim_count.shape:
        raise ValueError("Base, Grounding, and claim-count arrays must have matching shapes")
    if min_anchor_gaussians < 1:
        raise ValueError("min_anchor_gaussians must be positive")
    if not 0.0 <= min_anchor_coverage <= 1.0:
        raise ValueError("min_anchor_coverage must be between zero and one")
    if not 0.0 <= min_anchor_precision <= 1.0:
        raise ValueError("min_anchor_precision must be between zero and one")
    if spatial_voxel_scale_multiplier <= 0 or spatial_min_voxel_size <= 0:
        raise ValueError("Spatial voxel parameters must be positive")
    if min_group_proposals < 1 or min_group_source_views < 1:
        raise ValueError("Grounding group support thresholds must be positive")
    if not 0.0 <= ambiguous_thing_min_group_anchor_ratio:
        raise ValueError("Ambiguous-thing group/anchor ratio must be non-negative")
    if not 0.0 <= thing_halo_min_anchor_precision <= 1.0:
        raise ValueError("Thing-halo anchor precision must be between zero and one")
    if not 0.0 <= nested_thing_min_containment <= 1.0:
        raise ValueError("Nested-thing containment must be between zero and one")
    if nested_thing_envelope_margin_ratio < anchor_envelope_margin_ratio:
        raise ValueError(
            "Nested-thing envelope margin cannot be tighter than the anchor envelope"
        )
    if nested_thing_min_parent_source_ratio < 1.0:
        raise ValueError("Nested-thing parent source ratio must be at least one")
    if not 0.0 <= nested_thing_min_parent_overlap_coverage <= 1.0:
        raise ValueError("Nested-thing parent overlap coverage must be between zero and one")
    if not 0.0 <= prototype_min_robust_fraction <= 1.0:
        raise ValueError("Prototype robust fraction must be between zero and one")
    if prototype_min_source_views < 1:
        raise ValueError("Prototype source-view threshold must be positive")
    if prototype_strong_source_views < prototype_min_source_views:
        raise ValueError(
            "Strong prototype source-view threshold cannot be below the minimum"
        )
    if not 0.0 <= prototype_max_dominant_competing_thing_fraction <= 1.0:
        raise ValueError(
            "Prototype competing-thing fraction must be between zero and one"
        )
    if not 0.0 <= prototype_max_partial_competing_thing_risk <= 1.0:
        raise ValueError(
            "Prototype partial competing-thing risk must be between zero and one"
        )
    if not 0.0 <= prototype_ambiguous_min_dominant_competing_thing_fraction <= 1.0:
        raise ValueError(
            "Ambiguous prototype competing-thing fraction must be between zero and one"
        )
    if not 0.0 <= partial_anchor_extension_min_anchor_coverage <= 1.0:
        raise ValueError("Partial-anchor minimum coverage must be between zero and one")
    if not 0.0 <= partial_anchor_extension_max_anchor_precision <= 1.0:
        raise ValueError("Partial-anchor maximum precision must be between zero and one")
    if not 0.0 <= partial_anchor_extension_min_robust_fraction <= 1.0:
        raise ValueError("Partial-anchor robust fraction must be between zero and one")
    if partial_anchor_extension_min_source_views < 1:
        raise ValueError("Partial-anchor source-view threshold must be positive")
    if not 0.0 <= partial_anchor_extension_max_competing_thing_fraction <= 1.0:
        raise ValueError(
            "Partial-anchor competing-thing fraction must be between zero and one"
        )
    if not 0.0 <= partial_anchor_extension_min_spatial_keep_fraction <= 1.0:
        raise ValueError("Partial-anchor spatial keep fraction must be between zero and one")

    eligible = set(eligible_classes)
    base_classes = class_array(base_labels, base_items)
    base_class_names = {
        normalize_class_name(item.get("class", ""))
        for label_id, item in base_items.items()
        if label_id > 0
    }
    missing_base = sorted(eligible - base_class_names)
    if missing_base:
        raise ValueError(f"Refinement classes have no DINO anchor class: {missing_base}")
    missing_claims = sorted(eligible - set(robust_by_class))
    if missing_claims:
        raise ValueError(f"Refinement classes have no robust-claim arrays: {missing_claims}")

    merged = base_labels.copy()
    changes = np.zeros(base_labels.shape, dtype=np.int32)
    next_id = max(base_items) + 1
    appended: list[dict[str, Any]] = []
    group_reports: list[dict[str, Any]] = []
    accepted_union = np.zeros(base_labels.shape, dtype=bool)
    thing_classes = {
        name for name, kind in (class_kinds or {}).items() if kind == "thing"
    }
    thing_class_values = np.asarray(sorted(thing_classes), dtype=object)
    anchored_prototypes: dict[str, list[np.ndarray]] = {}
    accepted_thing_regions: list[dict[str, Any]] = []

    for source_id in sorted(grounding_items):
        if source_id == 0:
            continue
        source_item = grounding_items[source_id]
        class_name = normalize_class_name(source_item.get("class", ""))
        source_mask = grounding_labels == source_id
        source_count = int(source_mask.sum())
        report: dict[str, Any] = {
            "source_label_id": source_id,
            "name": str(source_item.get("name", "")),
            "class": class_name,
            "grounding_gaussian_count": source_count,
            "proposal_count": int(source_item.get("proposal_count", 0)),
            "source_view_count": int(source_item.get("source_view_count", 0)),
        }
        reasons: list[str] = []
        if class_name not in eligible:
            reasons.append("class_not_eligible")
        if report["proposal_count"] < min_group_proposals:
            reasons.append(f"proposal_count<{min_group_proposals}")
        if report["source_view_count"] < min_group_source_views:
            reasons.append(f"source_view_count<{min_group_source_views}")
        if reasons:
            report.update({"status": "rejected", "reasons": reasons})
            group_reports.append(report)
            continue

        source_indices = np.flatnonzero(source_mask)
        robust_here = mask_from_sorted_indices(robust_by_class[class_name], source_indices)
        unique_here = claim_count[source_indices] == 1
        robust_indices = source_indices[robust_here]
        guarded_indices = source_indices[robust_here & unique_here]
        pre_spatial_guarded_count = int(guarded_indices.shape[0])
        guarded_mask = np.zeros(base_labels.shape, dtype=bool)
        guarded_mask[guarded_indices] = True
        same_class = guarded_mask & (base_classes == class_name)
        anchor_ids, anchor_counts = np.unique(base_labels[same_class], return_counts=True)
        valid_anchor = anchor_ids > 0
        anchor_ids = anchor_ids[valid_anchor]
        anchor_counts = anchor_counts[valid_anchor]
        if anchor_ids.shape[0]:
            dominant_position = int(np.argmax(anchor_counts))
            dominant_anchor_id = int(anchor_ids[dominant_position])
            dominant_overlap = int(anchor_counts[dominant_position])
            dominant_size = int(np.count_nonzero(base_labels == dominant_anchor_id))
        else:
            dominant_anchor_id = 0
            dominant_overlap = 0
            dominant_size = 0
        coverage = dominant_overlap / float(max(dominant_size, 1))
        pre_spatial_precision = dominant_overlap / float(max(pre_spatial_guarded_count, 1))
        class_kind = (class_kinds or {}).get(class_name, "")
        is_thing = class_kind == "thing"
        anchor_overlap_ok = dominant_overlap >= min_anchor_gaussians
        anchor_coverage_ok = coverage >= min_anchor_coverage
        anchor_precision_fallback = (
            is_thing and anchor_overlap_ok and pre_spatial_precision >= min_anchor_precision
        )
        anchored_acceptance = anchor_overlap_ok and (
            anchor_coverage_ok or anchor_precision_fallback
        )
        robust_fraction = pre_spatial_guarded_count / float(max(source_count, 1))
        report.update(
            {
                "robust_unique_gaussian_count": pre_spatial_guarded_count,
                "robust_class_gaussian_count": int(robust_indices.shape[0]),
                "ambiguous_or_weak_removed_count": source_count - pre_spatial_guarded_count,
                "robust_unique_fraction": robust_fraction,
                "dominant_anchor_label_id": dominant_anchor_id,
                "dominant_anchor_overlap": dominant_overlap,
                "dominant_anchor_gaussian_count": dominant_size,
                "dominant_anchor_coverage": coverage,
                "pre_spatial_anchor_precision": pre_spatial_precision,
                "thing_anchor_precision_fallback": anchor_precision_fallback,
            }
        )
        nested_parent: dict[str, Any] | None = None
        nested_parent_keep = np.zeros((0,), dtype=bool)
        nested_parent_containment = 0.0
        nested_parent_overlap_coverage = 0.0
        current_anchor_indices = (
            np.flatnonzero(base_labels == dominant_anchor_id)
            if dominant_anchor_id > 0
            else np.zeros((0,), dtype=np.int64)
        )
        if is_thing and current_anchor_indices.shape[0]:
            for parent in accepted_thing_regions:
                if parent["class"] == class_name:
                    continue
                parent_keep, _parent_envelope_report = anchor_envelope_membership(
                    parent["anchor_indices"],
                    current_anchor_indices,
                    vertex_data,
                    anchor_envelope_quantile,
                    nested_thing_envelope_margin_ratio,
                )
                containment = float(np.count_nonzero(parent_keep)) / float(
                    max(current_anchor_indices.shape[0], 1)
                )
                parent_overlap = int(
                    np.count_nonzero(
                        grounding_labels[current_anchor_indices] == parent["source_id"]
                    )
                )
                parent_overlap_coverage = parent_overlap / float(
                    max(current_anchor_indices.shape[0], 1)
                )
                parent_source_ratio = float(parent["source_count"]) / float(
                    max(source_count, 1)
                )
                if (
                    containment >= nested_thing_min_containment
                    and parent_source_ratio >= nested_thing_min_parent_source_ratio
                    and parent_overlap >= min_anchor_gaussians
                    and parent_overlap_coverage
                    >= nested_thing_min_parent_overlap_coverage
                    and containment > nested_parent_containment
                ):
                    nested_parent = parent
                    nested_parent_keep = parent_keep
                    nested_parent_containment = containment
                    nested_parent_overlap_coverage = parent_overlap_coverage

        report.update(
            {
                "nested_parent_class": nested_parent["class"] if nested_parent else "",
                "nested_parent_containment": nested_parent_containment,
                "nested_parent_overlap_coverage": nested_parent_overlap_coverage,
            }
        )
        if nested_parent is not None:
            takeover_indices = current_anchor_indices[nested_parent_keep]
            takeover_mask = np.zeros(base_labels.shape, dtype=bool)
            takeover_mask[takeover_indices] = True
            takeover_mask &= base_classes != nested_parent["class"]
            changed_count = int(np.count_nonzero(takeover_mask))
            if changed_count > 0:
                output_id = next_id
                next_id += 1
                merged[takeover_mask] = output_id
                changes[takeover_mask] = output_id
                copied = dict(nested_parent["source_item"])
                copied.update(
                    {
                        "id": output_id,
                        "name": (
                            f"{nested_parent['source_item'].get('name', nested_parent['class'])}"
                            f"_nested_{source_id:02d}"
                        ),
                        "class": nested_parent["class"],
                        "gaussian_count": changed_count,
                        "source_pipeline": (
                            "groundingdino_sam_nested_thing_instance_takeover"
                        ),
                        "source_label_id": nested_parent["source_id"],
                        "suppressed_source_label_id": source_id,
                        "anchor_label_id": dominant_anchor_id,
                        "color_key": nested_parent["class"],
                        "rgb": rgb8_for_class(nested_parent["class"]),
                    }
                )
                appended.append(copied)
                accepted_union |= takeover_mask
                report.update(
                    {
                        "status": "nested_parent_takeover",
                        "output_label_id": output_id,
                        "takeover_class": nested_parent["class"],
                        "changed_gaussian_count": changed_count,
                        "newly_labeled_count": int(
                            np.count_nonzero(base_labels[takeover_mask] == 0)
                        ),
                        "relabeled_count": int(
                            np.count_nonzero(base_labels[takeover_mask] != 0)
                        ),
                        "base_transitions": transition_records(
                            base_labels, takeover_mask, base_items
                        ),
                        "reasons": [],
                    }
                )
                group_reports.append(report)
                continue
        prototype_ambiguous_geometry_recovery = False
        prototype_recovered_ambiguous_count = 0
        prototype_candidate_robust_fraction = 0.0
        prototype_candidate_dominant_competing_thing_fraction = 0.0
        prototype_signature = robust_geometry_signature(guarded_indices, vertex_data)
        if (
            is_thing
            and not anchored_acceptance
            and class_name in anchored_prototypes
            and report["source_view_count"] >= prototype_strong_source_views
        ):
            prototype_candidate_indices = robust_indices[
                ~accepted_union[robust_indices]
            ]
            prototype_candidate_signature = robust_geometry_signature(
                prototype_candidate_indices, vertex_data
            )
            prototype_candidate_fraction = prototype_candidate_indices.shape[0] / float(
                max(source_count, 1)
            )
            candidate_competing_metrics = competing_thing_metrics(
                prototype_candidate_indices,
                base_labels,
                base_items,
                base_classes,
                thing_class_values,
                class_name,
            )
            prototype_candidate_dominant_competing_thing_fraction = float(
                candidate_competing_metrics["dominant_class_fraction"]
            )
            prototype_candidate_robust_fraction = prototype_candidate_fraction
            if (
                prototype_candidate_indices.shape[0] >= min_anchor_gaussians
                and prototype_candidate_fraction >= prototype_min_robust_fraction
                and prototype_candidate_dominant_competing_thing_fraction
                >= prototype_ambiguous_min_dominant_competing_thing_fraction
                and prototype_geometry_matches(
                    prototype_candidate_signature,
                    anchored_prototypes[class_name],
                    prototype_max_shape_ratio,
                    prototype_max_size_ratio,
                )
            ):
                prototype_ambiguous_geometry_recovery = True
                prototype_recovered_ambiguous_count = int(
                    prototype_candidate_indices.shape[0] - guarded_indices.shape[0]
                )
                guarded_indices = prototype_candidate_indices
                pre_spatial_guarded_count = int(guarded_indices.shape[0])
                robust_fraction = pre_spatial_guarded_count / float(max(source_count, 1))
                prototype_signature = prototype_candidate_signature
                report.update(
                    {
                        "robust_unique_gaussian_count": pre_spatial_guarded_count,
                        "ambiguous_or_weak_removed_count": (
                            source_count - pre_spatial_guarded_count
                        ),
                    }
                )
        competing_metrics = competing_thing_metrics(
            guarded_indices,
            base_labels,
            base_items,
            base_classes,
            thing_class_values,
            class_name,
        )
        dominant_competing_thing_fraction = float(
            competing_metrics["dominant_class_fraction"]
        )
        dominant_competing_thing_partial_overlap_risk = float(
            competing_metrics["partial_overlap_risk"]
        )
        prototype_geometry_match = prototype_geometry_matches(
            prototype_signature,
            anchored_prototypes.get(class_name, []),
            prototype_max_shape_ratio,
            prototype_max_size_ratio,
        )
        prototype_strong_view_fallback = (
            class_name in anchored_prototypes
            and report["source_view_count"] >= prototype_strong_source_views
            and dominant_competing_thing_fraction
            <= prototype_max_dominant_competing_thing_fraction
            and dominant_competing_thing_partial_overlap_risk
            <= prototype_max_partial_competing_thing_risk
        )
        prototype_acceptance = (
            is_thing
            and not anchored_acceptance
            and pre_spatial_guarded_count >= min_anchor_gaussians
            and robust_fraction >= prototype_min_robust_fraction
            and report["source_view_count"]
            >= max(min_group_source_views, prototype_min_source_views)
            and (prototype_geometry_match or prototype_strong_view_fallback)
        )
        report.update(
            {
                "prototype_geometry_match": prototype_geometry_match,
                "prototype_strong_view_fallback": prototype_strong_view_fallback,
                "prototype_ambiguous_geometry_recovery": (
                    prototype_ambiguous_geometry_recovery
                ),
                "prototype_recovered_ambiguous_count": (
                    prototype_recovered_ambiguous_count
                ),
                "prototype_candidate_robust_fraction": (
                    prototype_candidate_robust_fraction
                ),
                "prototype_candidate_dominant_competing_thing_fraction": (
                    prototype_candidate_dominant_competing_thing_fraction
                ),
                "dominant_competing_thing_fraction": dominant_competing_thing_fraction,
                "dominant_competing_thing_class": str(
                    competing_metrics["dominant_class"]
                ),
                "dominant_competing_thing_label_id": int(
                    competing_metrics["dominant_instance_label_id"]
                ),
                "dominant_competing_thing_overlap": int(
                    competing_metrics["dominant_instance_overlap"]
                ),
                "dominant_competing_thing_gaussian_count": int(
                    competing_metrics["dominant_instance_gaussian_count"]
                ),
                "dominant_competing_thing_instance_coverage": float(
                    competing_metrics["dominant_instance_coverage"]
                ),
                "dominant_competing_thing_partial_overlap_risk": (
                    dominant_competing_thing_partial_overlap_risk
                ),
            }
        )
        if not anchored_acceptance and not prototype_acceptance:
            if not anchor_overlap_ok:
                reasons.append(f"anchor_overlap<{min_anchor_gaussians}")
            if not anchor_coverage_ok and not anchor_precision_fallback:
                reasons.append(f"anchor_coverage<{min_anchor_coverage}")
            if is_thing and class_name in anchored_prototypes:
                reasons.append("no_matching_anchored_instance_prototype")
                if (
                    dominant_competing_thing_partial_overlap_risk
                    > prototype_max_partial_competing_thing_risk
                ):
                    reasons.append(
                        "partial_competing_thing_risk>"
                        f"{prototype_max_partial_competing_thing_risk}"
                    )
            report.update({"status": "rejected", "reasons": reasons})
            group_reports.append(report)
            continue

        if anchored_acceptance:
            acceptance_mode = (
                "thing_anchor_precision_fallback"
                if anchor_precision_fallback and not anchor_coverage_ok
                else "anchor_coverage"
            )
            spatial_keep, spatial_report = spatial_anchor_components(
                guarded_indices,
                base_labels[guarded_indices] == dominant_anchor_id,
                vertex_data,
                min_anchor_precision,
                spatial_voxel_scale_multiplier,
                spatial_min_voxel_size,
                spatial_max_voxel_size,
            )
        else:
            acceptance_mode = "anchored_instance_prototype"
            spatial_keep, spatial_report = spatial_unanchored_components(
                guarded_indices,
                vertex_data,
                min_anchor_gaussians,
                spatial_voxel_scale_multiplier,
                spatial_min_voxel_size,
                spatial_max_voxel_size,
            )
        guarded_indices = guarded_indices[spatial_keep]
        guarded_mask[:] = False
        guarded_mask[guarded_indices] = True
        post_spatial_anchor_overlap = int(
            np.count_nonzero(base_labels[guarded_indices] == dominant_anchor_id)
        )
        post_spatial_precision = post_spatial_anchor_overlap / float(
            max(guarded_indices.shape[0], 1)
        )
        spatial_keep_fraction = guarded_indices.shape[0] / float(
            max(pre_spatial_guarded_count, 1)
        )
        partial_anchor_component_extension = (
            anchored_acceptance
            and is_thing
            and coverage >= partial_anchor_extension_min_anchor_coverage
            and post_spatial_precision <= partial_anchor_extension_max_anchor_precision
            and robust_fraction >= partial_anchor_extension_min_robust_fraction
            and report["source_view_count"] >= partial_anchor_extension_min_source_views
            and dominant_competing_thing_fraction
            <= partial_anchor_extension_max_competing_thing_fraction
            and int(spatial_report.get("kept_component_count", 0)) == 1
            and spatial_keep_fraction
            >= partial_anchor_extension_min_spatial_keep_fraction
        )
        report.update(
            {
                "spatial_anchor_guard": spatial_report,
                "spatially_guarded_gaussian_count": int(guarded_indices.shape[0]),
                "spatially_removed_gaussian_count": pre_spatial_guarded_count
                - int(guarded_indices.shape[0]),
                "post_spatial_anchor_overlap": post_spatial_anchor_overlap,
                "post_spatial_anchor_precision": post_spatial_precision,
                "spatial_keep_fraction": spatial_keep_fraction,
                "acceptance_mode": acceptance_mode,
                "partial_anchor_component_extension": (
                    partial_anchor_component_extension
                ),
            }
        )
        if guarded_indices.shape[0] == 0:
            reasons.append(f"no_spatial_component_anchor_precision>={min_anchor_precision}")
        if reasons:
            report.update({"status": "rejected", "reasons": reasons})
            group_reports.append(report)
            continue

        if anchored_acceptance and prototype_signature is not None:
            anchored_prototypes.setdefault(class_name, []).append(prototype_signature)

        envelope_global = np.zeros(base_labels.shape, dtype=bool)
        envelope_report: dict[str, Any] = {"enabled": False}
        partial_anchor_extension_added_count = 0
        if anchored_acceptance and is_thing:
            anchor_instance_indices = np.flatnonzero(base_labels == dominant_anchor_id)
            envelope_keep, envelope_report = anchor_envelope_membership(
                anchor_instance_indices,
                guarded_indices,
                vertex_data,
                anchor_envelope_quantile,
                anchor_envelope_margin_ratio,
            )
            envelope_global[guarded_indices[envelope_keep]] = True
            if partial_anchor_component_extension:
                partial_anchor_extension_added_count = int(
                    np.count_nonzero(~envelope_keep)
                )
                envelope_global[guarded_indices] = True

            group_anchor_ratio = source_count / float(max(dominant_size, 1))
            ambiguous_indices = source_indices[robust_here & ~unique_here]
            ambiguous_base_thing = (
                np.isin(base_classes[ambiguous_indices], thing_class_values)
                if thing_class_values.shape[0]
                else np.zeros(ambiguous_indices.shape, dtype=bool)
            )
            ambiguous_indices = ambiguous_indices[
                ambiguous_base_thing & (base_classes[ambiguous_indices] != class_name)
            ]
            ambiguous_added = np.zeros((0,), dtype=np.int64)
            if group_anchor_ratio >= ambiguous_thing_min_group_anchor_ratio:
                ambiguous_envelope_keep, _ambiguous_envelope_report = (
                    anchor_envelope_membership(
                        anchor_instance_indices,
                        ambiguous_indices,
                        vertex_data,
                        anchor_envelope_quantile,
                        anchor_envelope_margin_ratio,
                    )
                )
                ambiguous_added = ambiguous_indices[ambiguous_envelope_keep]
                if ambiguous_added.shape[0]:
                    guarded_mask[ambiguous_added] = True
                    envelope_global[ambiguous_added] = True
                    guarded_indices = np.flatnonzero(guarded_mask)
            weak_indices = source_indices[~robust_here]
            weak_base_thing = (
                np.isin(base_classes[weak_indices], thing_class_values)
                if thing_class_values.shape[0]
                else np.zeros(weak_indices.shape, dtype=bool)
            )
            weak_indices = weak_indices[
                weak_base_thing & (base_classes[weak_indices] != class_name)
            ]
            weak_added = np.zeros((0,), dtype=np.int64)
            if (
                group_anchor_ratio >= ambiguous_thing_min_group_anchor_ratio
                and post_spatial_precision >= thing_halo_min_anchor_precision
            ):
                weak_envelope_keep, _weak_envelope_report = anchor_envelope_membership(
                    anchor_instance_indices,
                    weak_indices,
                    vertex_data,
                    anchor_envelope_quantile,
                    anchor_envelope_margin_ratio,
                )
                weak_added = weak_indices[weak_envelope_keep]
                if weak_added.shape[0]:
                    guarded_mask[weak_added] = True
                    envelope_global[weak_added] = True
                    guarded_indices = np.flatnonzero(guarded_mask)
            report.update(
                {
                    "grounding_group_to_anchor_size_ratio": group_anchor_ratio,
                    "ambiguous_strong_thing_candidate_count": int(
                        ambiguous_indices.shape[0]
                    ),
                    "ambiguous_strong_thing_added_count": int(ambiguous_added.shape[0]),
                    "fusion_thing_halo_candidate_count": int(weak_indices.shape[0]),
                    "fusion_thing_halo_added_count": int(weak_added.shape[0]),
                }
            )
        report["anchor_envelope_guard"] = envelope_report
        report["partial_anchor_extension_added_count"] = (
            partial_anchor_extension_added_count
        )
        if anchored_acceptance and is_thing:
            accepted_thing_regions.append(
                {
                    "class": class_name,
                    "source_id": source_id,
                    "source_item": source_item,
                    "source_count": source_count,
                    "anchor_indices": np.flatnonzero(
                        base_labels == dominant_anchor_id
                    ),
                }
            )

        change_mask = guarded_mask & (base_classes != class_name)
        protected_base_thing_count = 0
        if is_thing and anchored_acceptance and thing_class_values.shape[0]:
            base_thing = np.isin(base_classes, thing_class_values)
            envelope_removed = change_mask & ~base_thing & ~envelope_global
            report["anchor_envelope_removed_change_count"] = int(
                np.count_nonzero(envelope_removed)
            )
            change_mask &= base_thing | envelope_global
        if class_kinds is not None and class_kind == "stuff":
            protected_base_thing = change_mask & np.isin(
                base_classes,
                thing_class_values,
            )
            protected_base_thing_count = int(np.count_nonzero(protected_base_thing))
            change_mask &= ~protected_base_thing
        report["protected_base_thing_gaussian_count"] = protected_base_thing_count
        changed_count = int(change_mask.sum())
        accepted_union |= guarded_mask
        if changed_count == 0:
            report.update(
                {
                    "status": "agreement_only",
                    "changed_gaussian_count": 0,
                    "reasons": [],
                }
            )
            group_reports.append(report)
            continue

        output_id = next_id
        next_id += 1
        merged[change_mask] = output_id
        changes[change_mask] = output_id
        copied = dict(source_item)
        copied.update(
            {
                "id": output_id,
                "gaussian_count": changed_count,
                "source_pipeline": "groundingdino_sam_guarded_ade_refinement",
                "source_label_id": source_id,
                "anchor_label_id": dominant_anchor_id,
                "color_key": class_name,
                "rgb": rgb8_for_class(class_name),
            }
        )
        appended.append(copied)
        report.update(
            {
                "status": "refined",
                "output_label_id": output_id,
                "changed_gaussian_count": changed_count,
                "newly_labeled_count": int(np.count_nonzero(base_labels[change_mask] == 0)),
                "relabeled_count": int(np.count_nonzero(base_labels[change_mask] != 0)),
                "base_transitions": transition_records(base_labels, change_mask, base_items),
                "reasons": [],
            }
        )
        group_reports.append(report)

    if not np.array_equal(merged[~accepted_union], base_labels[~accepted_union]):
        raise AssertionError("DINO labels changed outside accepted ADE refinement masks")
    refined_statuses = {"refined", "nested_parent_takeover"}
    refined = [item for item in group_reports if item["status"] in refined_statuses]
    summary = {
        "eligible_classes": eligible_classes,
        "grounding_group_count": len(group_reports),
        "accepted_group_count": sum(
            item["status"] in {*refined_statuses, "agreement_only"}
            for item in group_reports
        ),
        "refined_group_count": len(refined),
        "rejected_group_count": sum(item["status"] == "rejected" for item in group_reports),
        "changed_gaussian_count": int(np.count_nonzero(changes)),
        "newly_labeled_count": int(np.count_nonzero((changes != 0) & (base_labels == 0))),
        "relabeled_count": int(np.count_nonzero((changes != 0) & (base_labels != 0))),
        "unchanged_outside_accepted_masks": True,
        "groups": group_reports,
    }
    return merged, changes, appended, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-labels", required=True, type=Path)
    parser.add_argument("--base-label-map", required=True, type=Path)
    parser.add_argument("--grounding-labels", required=True, type=Path)
    parser.add_argument("--grounding-label-map", required=True, type=Path)
    parser.add_argument("--class-evidence-dir", required=True, type=Path)
    parser.add_argument("--refinement-config", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--semantic-ply-path", type=Path)
    parser.add_argument("--no-semantic-ply", action="store_true")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--min-anchor-gaussians", default=500, type=int)
    parser.add_argument("--min-anchor-coverage", default=0.10, type=float)
    parser.add_argument("--min-anchor-precision", default=0.10, type=float)
    parser.add_argument("--min-group-proposals", default=2, type=int)
    parser.add_argument("--min-group-source-views", default=2, type=int)
    parser.add_argument("--evidence-min-positive-views", default=2, type=int)
    parser.add_argument("--evidence-min-ratio", default=0.50, type=float)
    parser.add_argument("--spatial-voxel-scale-multiplier", default=4.0, type=float)
    parser.add_argument("--spatial-min-voxel-size", default=0.01, type=float)
    parser.add_argument("--spatial-max-voxel-size", default=0.20, type=float)
    parser.add_argument("--anchor-envelope-quantile", default=0.01, type=float)
    parser.add_argument("--anchor-envelope-margin-ratio", default=0.05, type=float)
    parser.add_argument(
        "--ambiguous-thing-min-group-anchor-ratio", default=0.75, type=float
    )
    parser.add_argument("--thing-halo-min-anchor-precision", default=0.75, type=float)
    parser.add_argument("--nested-thing-min-containment", default=0.80, type=float)
    parser.add_argument("--nested-thing-envelope-margin-ratio", default=0.25, type=float)
    parser.add_argument("--nested-thing-min-parent-source-ratio", default=2.00, type=float)
    parser.add_argument(
        "--nested-thing-min-parent-overlap-coverage", default=0.10, type=float
    )
    parser.add_argument("--prototype-min-robust-fraction", default=0.50, type=float)
    parser.add_argument("--prototype-min-source-views", default=3, type=int)
    parser.add_argument("--prototype-strong-source-views", default=5, type=int)
    parser.add_argument(
        "--prototype-max-dominant-competing-thing-fraction", default=0.50, type=float
    )
    parser.add_argument(
        "--prototype-max-partial-competing-thing-risk", default=0.40, type=float
    )
    parser.add_argument(
        "--prototype-ambiguous-min-dominant-competing-thing-fraction",
        default=0.90,
        type=float,
    )
    parser.add_argument("--prototype-max-shape-ratio", default=2.50, type=float)
    parser.add_argument("--prototype-max-size-ratio", default=3.00, type=float)
    parser.add_argument(
        "--partial-anchor-extension-min-anchor-coverage", default=0.80, type=float
    )
    parser.add_argument(
        "--partial-anchor-extension-max-anchor-precision", default=0.25, type=float
    )
    parser.add_argument(
        "--partial-anchor-extension-min-robust-fraction", default=0.90, type=float
    )
    parser.add_argument(
        "--partial-anchor-extension-min-source-views", default=5, type=int
    )
    parser.add_argument(
        "--partial-anchor-extension-max-competing-thing-fraction",
        default=0.05,
        type=float,
    )
    parser.add_argument(
        "--partial-anchor-extension-min-spatial-keep-fraction",
        default=0.95,
        type=float,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    semantic_ply = resolve_semantic_ply_output(
        args.output_dir,
        semantic_ply_path=args.semantic_ply_path,
        disabled=args.no_semantic_ply,
    )
    paths = {
        "labels": args.output_dir / "gaussian_labels.npy",
        "changes": args.output_dir / "refinement_changes.npy",
        "label_map": args.output_dir / "label_map.json",
        "summary": args.output_dir / "ade_refinement_summary.json",
    }
    output_files = [*paths.values(), *([semantic_ply] if semantic_ply is not None else [])]
    existing = [path for path in output_files if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"ADE refinement outputs exist; pass --overwrite: {existing}")

    base_map = load_json(args.base_label_map)
    grounding_map = load_json(args.grounding_label_map)
    for label, value in (("base", base_map), ("Grounding", grounding_map)):
        if str(value.get("scene", "")) != args.scene:
            raise ValueError(f"{label} label-map scene does not match {args.scene!r}")
    config = load_refinement_config(args.refinement_config, args.scene)
    eligible_classes = list(config["selected_refinement_classes"])
    ontology = load_ontology(args.ontology)
    ontology_classes = {item.project_class for item in ontology.classes}
    ontology_kinds = {item.project_class: item.kind for item in ontology.classes}
    outside_ontology = sorted(set(eligible_classes) - ontology_classes)
    if outside_ontology:
        raise ValueError(f"Refinement classes are outside ADE ontology: {outside_ontology}")

    base_items = label_items_by_id(base_map, args.base_label_map)
    grounding_items = label_items_by_id(grounding_map, args.grounding_label_map)
    base_labels = validate_label_array(np.load(args.base_labels), base_items, args.base_labels)
    grounding_labels = validate_label_array(
        np.load(args.grounding_labels), grounding_items, args.grounding_labels
    )
    header = read_ply_header(args.source_ply)
    vertex = header.element("vertex")
    if vertex is None or vertex.count != int(base_labels.shape[0]):
        raise ValueError("Source PLY vertex count does not match base labels")
    _, vertex_data = vertex_data_memmap(args.source_ply)

    claim_count, robust_by_class, evidence_report = load_unique_claims(
        args.class_evidence_dir,
        eligible_classes,
        int(base_labels.shape[0]),
        args.evidence_min_positive_views,
        args.evidence_min_ratio,
    )
    merged, changes, appended_items, merge_report = refine_ade_labels(
        base_labels,
        base_items,
        grounding_labels,
        grounding_items,
        eligible_classes,
        claim_count,
        robust_by_class,
        args.min_anchor_gaussians,
        args.min_anchor_coverage,
        args.min_group_proposals,
        args.min_group_source_views,
        args.min_anchor_precision,
        vertex_data,
        args.spatial_voxel_scale_multiplier,
        args.spatial_min_voxel_size,
        args.spatial_max_voxel_size,
        ontology_kinds,
        args.anchor_envelope_quantile,
        args.anchor_envelope_margin_ratio,
        args.ambiguous_thing_min_group_anchor_ratio,
        args.thing_halo_min_anchor_precision,
        args.nested_thing_min_containment,
        args.nested_thing_envelope_margin_ratio,
        args.nested_thing_min_parent_source_ratio,
        args.nested_thing_min_parent_overlap_coverage,
        args.prototype_min_robust_fraction,
        args.prototype_min_source_views,
        args.prototype_strong_source_views,
        args.prototype_max_dominant_competing_thing_fraction,
        args.prototype_max_partial_competing_thing_risk,
        args.prototype_ambiguous_min_dominant_competing_thing_fraction,
        args.prototype_max_shape_ratio,
        args.prototype_max_size_ratio,
        args.partial_anchor_extension_min_anchor_coverage,
        args.partial_anchor_extension_max_anchor_precision,
        args.partial_anchor_extension_min_robust_fraction,
        args.partial_anchor_extension_min_source_views,
        args.partial_anchor_extension_max_competing_thing_fraction,
        args.partial_anchor_extension_min_spatial_keep_fraction,
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
        "source": "dinov2_groundingdino_guarded_ade_refinement",
        "palette_version": PALETTE_VERSION,
        "missing_vocabulary_enabled": bool(config.get("missing_vocabulary_enabled", False)),
        "base_source": str(args.base_label_map),
        "grounding_source": str(args.grounding_label_map),
        "selected_refinement_classes": eligible_classes,
        "labels": final_items,
    }
    base_histogram = histogram(base_labels)
    summary = {
        "scene": args.scene,
        "status": "ok" if merge_report["refined_group_count"] else "no_refinement_changes",
        "palette_version": PALETTE_VERSION,
        "gaussian_count": int(merged.shape[0]),
        "missing_vocabulary_enabled": bool(config.get("missing_vocabulary_enabled", False)),
        "base_unlabeled_ratio": base_histogram.get(0, 0) / float(max(merged.shape[0], 1)),
        "refined_unlabeled_ratio": merged_histogram.get(0, 0) / float(max(merged.shape[0], 1)),
        "parameters": {
            "min_anchor_gaussians": args.min_anchor_gaussians,
            "min_anchor_coverage": args.min_anchor_coverage,
            "min_anchor_precision": args.min_anchor_precision,
            "min_group_proposals": args.min_group_proposals,
            "min_group_source_views": args.min_group_source_views,
            "evidence_min_positive_views": args.evidence_min_positive_views,
            "evidence_min_ratio": args.evidence_min_ratio,
            "spatial_voxel_scale_multiplier": args.spatial_voxel_scale_multiplier,
            "spatial_min_voxel_size": args.spatial_min_voxel_size,
            "spatial_max_voxel_size": args.spatial_max_voxel_size,
            "anchor_envelope_quantile": args.anchor_envelope_quantile,
            "anchor_envelope_margin_ratio": args.anchor_envelope_margin_ratio,
            "ambiguous_thing_min_group_anchor_ratio": (
                args.ambiguous_thing_min_group_anchor_ratio
            ),
            "thing_halo_min_anchor_precision": args.thing_halo_min_anchor_precision,
            "nested_thing_min_containment": args.nested_thing_min_containment,
            "nested_thing_envelope_margin_ratio": (
                args.nested_thing_envelope_margin_ratio
            ),
            "nested_thing_min_parent_source_ratio": (
                args.nested_thing_min_parent_source_ratio
            ),
            "nested_thing_min_parent_overlap_coverage": (
                args.nested_thing_min_parent_overlap_coverage
            ),
            "prototype_min_robust_fraction": args.prototype_min_robust_fraction,
            "prototype_min_source_views": args.prototype_min_source_views,
            "prototype_strong_source_views": args.prototype_strong_source_views,
            "prototype_max_dominant_competing_thing_fraction": (
                args.prototype_max_dominant_competing_thing_fraction
            ),
            "prototype_max_partial_competing_thing_risk": (
                args.prototype_max_partial_competing_thing_risk
            ),
            "prototype_ambiguous_min_dominant_competing_thing_fraction": (
                args.prototype_ambiguous_min_dominant_competing_thing_fraction
            ),
            "prototype_max_shape_ratio": args.prototype_max_shape_ratio,
            "prototype_max_size_ratio": args.prototype_max_size_ratio,
            "partial_anchor_extension_min_anchor_coverage": (
                args.partial_anchor_extension_min_anchor_coverage
            ),
            "partial_anchor_extension_max_anchor_precision": (
                args.partial_anchor_extension_max_anchor_precision
            ),
            "partial_anchor_extension_min_robust_fraction": (
                args.partial_anchor_extension_min_robust_fraction
            ),
            "partial_anchor_extension_min_source_views": (
                args.partial_anchor_extension_min_source_views
            ),
            "partial_anchor_extension_max_competing_thing_fraction": (
                args.partial_anchor_extension_max_competing_thing_fraction
            ),
            "partial_anchor_extension_min_spatial_keep_fraction": (
                args.partial_anchor_extension_min_spatial_keep_fraction
            ),
            "ambiguity_rule": "exactly_one_robust_eligible_ade_class_claim",
            "spatial_rule": (
                "keep_anchor_seeded_components; complete_one_strong_partial-anchor_"
                "component_under_multiview_and_conflict_guards"
            ),
            "prototype_competing_instance_rule": (
                "candidate_conflict_weighted_by_uncovered_fraction_of_dominant_"
                "competing_instance"
            ),
            "ontology_rule": "grounding_stuff_does_not_overwrite_base_thing_instances",
        },
        "evidence": evidence_report,
        "merge": merge_report,
        "sources": {
            "base_labels": {"path": str(args.base_labels), "sha256": sha256_file(args.base_labels)},
            "base_label_map": {
                "path": str(args.base_label_map),
                "sha256": sha256_file(args.base_label_map),
            },
            "grounding_labels": {
                "path": str(args.grounding_labels),
                "sha256": sha256_file(args.grounding_labels),
            },
            "grounding_label_map": {
                "path": str(args.grounding_label_map),
                "sha256": sha256_file(args.grounding_label_map),
            },
            "refinement_config": {
                "path": str(args.refinement_config),
                "sha256": sha256_file(args.refinement_config),
            },
            "ontology": {"path": str(args.ontology), "sha256": sha256_file(args.ontology)},
            "source_ply": {"path": str(args.source_ply), "vertex_count": vertex.count},
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if semantic_ply is not None:
        semantic_ply.parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["labels"], merged.astype(np.int32, copy=False))
    np.save(paths["changes"], changes.astype(np.int32, copy=False))
    paths["label_map"].write_text(json.dumps(output_map, indent=2), encoding="utf-8")
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if semantic_ply is not None:
        write_ply_with_labels(args.source_ply, semantic_ply, merged)
    print(json.dumps({key: value for key, value in summary.items() if key != "sources"}, indent=2))


if __name__ == "__main__":
    main()
