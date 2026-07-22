#!/usr/bin/env python3
"""Audit independent multiview evidence for unresolved singleton proposals.

This stage is deliberately report-only. It never writes semantic labels or a
semantic PLY. Candidate projections are used only to associate an independently
generated GroundingDINO detection with the original singleton. GroundingDINO
creates every verification box, SAM segments that box, and FlashSplat measures
which Gaussians actually contribute to the resulting mask in that camera.

The opt-in v10 mode adds a source-identity consistency gate, candidate-adaptive
3D radius components, and report-only lifting of associated near-threshold
quality failures. The default v9 mode remains reproducible.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from apply_singleton_point_recovery import (
    evaluate_seed_policy,
    resolve_verification_view_manifest,
)
from cluster_semantic_flashsplat_proposals import (
    load_proposals,
    load_stuff_classes,
)
from dinov2_ontology import load_ontology, normalize_class_name
from merge_semantic_extensions import label_items_by_id, load_json, validate_label_array
from ply_utils import read_ply_header, vertex_data_memmap
from recover_cross_view_sam_masks import (
    ProjectionPrompt,
    candidate_view_prompts,
    frame_mask_path,
    load_cameras,
    load_mask_stack,
    nested_competing_thing,
    proposal_grounding_score,
    proposal_sam_score,
    rgb_directory,
    singleton_proposals,
    thing_classes_from_config,
)
from semantic_palette import rgb8_for_class


@dataclass(frozen=True)
class SeparatedView:
    prompt: ProjectionPrompt
    source_angle_degrees: float
    geometry_score: float


@dataclass
class IndependentDetection:
    mask: np.ndarray
    class_name: str
    phrase: str
    grounding_score: float
    sam_score: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass
class CandidateContext:
    proposal: Any
    source_frame: dict[str, Any]
    report: dict[str, Any]
    separated_views: list[SeparatedView]


def unit_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1.0e-9:
        return None
    return vector.astype(np.float64, copy=False) / norm


def angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    left_unit = unit_vector(left)
    right_unit = unit_vector(right)
    if left_unit is None or right_unit is None:
        return 0.0
    cosine = float(np.clip(np.dot(left_unit, right_unit), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def camera_view_angle_degrees(
    left_camera: dict[str, Any],
    right_camera: dict[str, Any],
    target: np.ndarray,
) -> float:
    left_position = np.asarray(left_camera["position"], dtype=np.float64)
    right_position = np.asarray(right_camera["position"], dtype=np.float64)
    return angle_degrees(target - left_position, target - right_position)


def select_geometry_separated_views(
    prompts: list[ProjectionPrompt],
    source_camera: dict[str, Any],
    cameras: list[dict[str, Any]],
    target: np.ndarray,
    min_source_angle_degrees: float,
    min_pairwise_angle_degrees: float,
    max_views: int,
) -> list[SeparatedView]:
    """Greedily retain visible views that provide genuinely different geometry."""
    ranked: list[SeparatedView] = []
    for prompt in prompts:
        target_camera = cameras[prompt.target_camera_index]
        source_angle = camera_view_angle_degrees(
            source_camera, target_camera, target
        )
        if source_angle < min_source_angle_degrees:
            continue
        # Reward both seed visibility and parallax. This is intentionally not
        # nearest-frame ranking: sequential frames have a near-zero sine term.
        geometry_score = float(
            prompt.projected_fraction
            * math.sin(math.radians(min(source_angle, 90.0)))
        )
        ranked.append(
            SeparatedView(
                prompt=prompt,
                source_angle_degrees=source_angle,
                geometry_score=geometry_score,
            )
        )
    ranked.sort(
        key=lambda item: (
            item.geometry_score,
            item.source_angle_degrees,
            item.prompt.projected_fraction,
            -item.prompt.baseline_depth_ratio,
        ),
        reverse=True,
    )

    selected: list[SeparatedView] = []
    for candidate in ranked:
        candidate_camera = cameras[candidate.prompt.target_camera_index]
        if any(
            camera_view_angle_degrees(
                candidate_camera,
                cameras[existing.prompt.target_camera_index],
                target,
            )
            < min_pairwise_angle_degrees
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_views:
            break
    return selected


def box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / float(max(left_area + right_area - intersection, 1.0e-9))


def projected_membership(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    height, width = mask.shape
    pixels = np.rint(xy).astype(np.int64)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    return mask[pixels[:, 1], pixels[:, 0]].astype(bool, copy=False)


def match_independent_detection(
    prompt: ProjectionPrompt,
    detections: list[IndependentDetection],
) -> tuple[IndependentDetection | None, dict[str, float]]:
    """Associate independent detections after inference, never before it."""
    best: IndependentDetection | None = None
    best_metrics = {
        "projected_seed_coverage": 0.0,
        "projected_box_iou": 0.0,
        "association_score": 0.0,
    }
    projected_box = tuple(float(value) for value in prompt.bbox_xyxy)
    for detection in detections:
        if normalize_class_name(detection.class_name) != normalize_class_name(
            prompt.class_name
        ):
            continue
        membership = projected_membership(detection.mask, prompt.projected_xy)
        coverage = float(membership.mean()) if membership.size else 0.0
        overlap = box_iou(projected_box, detection.bbox_xyxy)
        # Either overlap signal may establish association. Scores only break
        # ties; they do not create a detection or a SAM prompt.
        association = max(coverage, overlap) + 0.05 * (
            detection.grounding_score + detection.sam_score
        )
        if best is None or association > best_metrics["association_score"]:
            best = detection
            best_metrics = {
                "projected_seed_coverage": coverage,
                "projected_box_iou": overlap,
                "association_score": float(association),
            }
    return best, best_metrics


def contribution_support(
    positive: np.ndarray,
    negative: np.ndarray,
    min_total_contribution: float,
    min_positive_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep Gaussians visibly dominated by the independent SAM mask."""
    if positive.shape != negative.shape or positive.ndim != 1:
        raise ValueError("positive and negative contributions must be equal 1D arrays")
    total = positive.astype(np.float64, copy=False) + negative.astype(
        np.float64, copy=False
    )
    fraction = np.divide(
        positive,
        total,
        out=np.zeros_like(total, dtype=np.float64),
        where=total > 0.0,
    )
    selected = (total >= min_total_contribution) & (
        fraction >= min_positive_fraction
    )
    indices = np.flatnonzero(selected).astype(np.uint32)
    return indices, {
        "visible_gaussian_count": int(np.count_nonzero(total >= min_total_contribution)),
        "positive_gaussian_count": int(indices.size),
        "max_total_contribution": float(total.max(initial=0.0)),
        "mean_positive_fraction": (
            float(fraction[selected].mean()) if np.any(selected) else 0.0
        ),
    }


def identity_consistency_gate(
    seed_contribution_fractions: list[float],
    min_second_fraction: float,
    min_second_to_best_ratio: float,
) -> tuple[bool, dict[str, Any]]:
    """Require two separated views to support the same source-seed identity."""
    if not 0.0 <= min_second_fraction <= 1.0:
        raise ValueError("min_second_fraction must be in [0, 1]")
    if not 0.0 <= min_second_to_best_ratio <= 1.0:
        raise ValueError("min_second_to_best_ratio must be in [0, 1]")
    ranked = sorted((float(value) for value in seed_contribution_fractions), reverse=True)
    best = ranked[0] if ranked else 0.0
    second = ranked[1] if len(ranked) >= 2 else 0.0
    relative = second / best if best > 0.0 else 0.0
    reasons: list[str] = []
    if len(ranked) < 2:
        reasons.append("identity_view_count<2")
    if second < min_second_fraction:
        reasons.append(f"second_seed_contribution_fraction<{min_second_fraction}")
    if relative < min_second_to_best_ratio:
        reasons.append(
            f"second_to_best_seed_contribution_ratio<{min_second_to_best_ratio}"
        )
    return not reasons, {
        "best_seed_contribution_fraction": best,
        "second_seed_contribution_fraction": second,
        "second_to_best_seed_contribution_ratio": relative,
        "min_second_seed_contribution_fraction": min_second_fraction,
        "min_second_to_best_seed_contribution_ratio": min_second_to_best_ratio,
        "passed": not reasons,
        "reasons": reasons,
    }


def diagnostic_lift_decision(
    grounding_score: float,
    sam_score: float,
    association_overlap: float,
    min_grounding_score: float,
    min_sam_score: float,
    min_association_overlap: float,
    max_grounding_shortfall: float,
    max_sam_shortfall: float,
) -> dict[str, Any]:
    """Select associated near-threshold quality failures for measurement only."""
    quality_accepted = (
        grounding_score >= min_grounding_score
        and sam_score >= min_sam_score
        and association_overlap >= min_association_overlap
    )
    associated = association_overlap >= min_association_overlap
    within_diagnostic_band = (
        grounding_score >= min_grounding_score - max_grounding_shortfall
        and sam_score >= min_sam_score - max_sam_shortfall
    )
    return {
        "quality_accepted": quality_accepted,
        "association_accepted": associated,
        "within_diagnostic_quality_band": within_diagnostic_band,
        "diagnostic_lift": (
            not quality_accepted and associated and within_diagnostic_band
        ),
        "grounding_score_shortfall": max(0.0, min_grounding_score - grounding_score),
        "sam_score_shortfall": max(0.0, min_sam_score - sam_score),
    }


def estimate_local_gaussian_spacing(
    points: np.ndarray,
    neighbor_k: int,
    max_samples: int,
    chunk_size: int = 64,
) -> tuple[float, dict[str, Any]]:
    """Estimate a candidate-local k-neighbor spacing without a SciPy dependency."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if neighbor_k < 1:
        raise ValueError("neighbor_k must be positive")
    if max_samples < 1 or chunk_size < 1:
        raise ValueError("spacing sample and chunk sizes must be positive")
    point_count = int(points.shape[0])
    if point_count < 2:
        raise ValueError("at least two points are required to estimate spacing")
    effective_k = min(neighbor_k, point_count - 1)
    sample_count = min(point_count, max_samples)
    sample_indices = np.unique(
        np.linspace(0, point_count - 1, num=sample_count, dtype=np.int64)
    )
    float_points = points.astype(np.float32, copy=False)
    neighbor_distances: list[np.ndarray] = []
    for start in range(0, sample_indices.size, chunk_size):
        chunk_indices = sample_indices[start : start + chunk_size]
        difference = (
            float_points[chunk_indices, np.newaxis, :]
            - float_points[np.newaxis, :, :]
        )
        squared = np.einsum("ijk,ijk->ij", difference, difference)
        squared[np.arange(chunk_indices.size), chunk_indices] = np.inf
        kth_squared = np.partition(squared, effective_k - 1, axis=1)[
            :, effective_k - 1
        ]
        neighbor_distances.append(np.sqrt(kth_squared.astype(np.float64)))
    distances = np.concatenate(neighbor_distances)
    finite_positive = distances[np.isfinite(distances) & (distances > 0.0)]
    if finite_positive.size == 0:
        raise ValueError("local Gaussian spacing is zero or non-finite")
    spacing = float(np.median(finite_positive))
    return spacing, {
        "point_count": point_count,
        "sample_count": int(sample_indices.size),
        "neighbor_k": effective_k,
        "median_neighbor_spacing": spacing,
        "p25_neighbor_spacing": float(np.percentile(finite_positive, 25.0)),
        "p75_neighbor_spacing": float(np.percentile(finite_positive, 75.0)),
    }


def radius_connected_components(
    points: np.ndarray,
    radius: float,
    pair_chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return exact Euclidean radius-graph components using a 3D spatial hash."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if pair_chunk_size < 1:
        raise ValueError("pair_chunk_size must be positive")
    point_count = int(points.shape[0])
    if point_count == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), {
            "radius": radius,
            "occupied_cell_count": 0,
            "edge_count": 0,
            "component_count": 0,
            "largest_component_gaussians": 0,
        }

    shifted = points.astype(np.float64, copy=False) - np.min(points, axis=0)
    cell_coordinates = np.floor(shifted / radius).astype(np.int64)
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, coordinate in enumerate(cell_coordinates):
        cells[tuple(int(value) for value in coordinate)].append(index)
    cell_arrays = {
        key: np.asarray(indices, dtype=np.int64) for key, indices in cells.items()
    }
    parent = np.arange(point_count, dtype=np.int64)
    rank = np.zeros(point_count, dtype=np.uint8)

    def find_root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find_root(left)
        right_root = find_root(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    radius_squared = radius * radius
    edge_count = 0
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    for cell_key in sorted(cell_arrays):
        left_indices = cell_arrays[cell_key]
        for dx, dy, dz in offsets:
            neighbor_key = (cell_key[0] + dx, cell_key[1] + dy, cell_key[2] + dz)
            if neighbor_key < cell_key:
                continue
            right_indices = cell_arrays.get(neighbor_key)
            if right_indices is None:
                continue
            for start in range(0, left_indices.size, pair_chunk_size):
                left_chunk = left_indices[start : start + pair_chunk_size]
                delta = (
                    points[left_chunk, np.newaxis, :]
                    - points[np.newaxis, right_indices, :]
                ).astype(np.float64, copy=False)
                squared = np.einsum("ijk,ijk->ij", delta, delta)
                connected = squared <= radius_squared
                if neighbor_key == cell_key:
                    connected &= right_indices[np.newaxis, :] > left_chunk[:, np.newaxis]
                rows, columns = np.nonzero(connected)
                edge_count += int(rows.size)
                for row, column in zip(rows.tolist(), columns.tolist()):
                    union(int(left_chunk[row]), int(right_indices[column]))

    roots = np.fromiter(
        (find_root(index) for index in range(point_count)),
        dtype=np.int64,
        count=point_count,
    )
    _, component_ids = np.unique(roots, return_inverse=True)
    component_sizes = np.bincount(component_ids).astype(np.int64)
    return component_ids, component_sizes, {
        "radius": radius,
        "occupied_cell_count": len(cell_arrays),
        "edge_count": edge_count,
        "component_count": int(component_sizes.size),
        "largest_component_gaussians": int(component_sizes.max(initial=0)),
    }


def retain_seed_touching_components(
    intersection: np.ndarray,
    seed_indices: np.ndarray,
    xyz: np.ndarray,
    neighbor_k: int,
    radius_multiplier: float,
    spacing_max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Keep exactly the intersection components containing source-seed points."""
    if radius_multiplier <= 0.0:
        raise ValueError("radius_multiplier must be positive")
    intersection = np.unique(intersection).astype(np.uint32, copy=False)
    if intersection.size < 2:
        component_ids = np.zeros(intersection.size, dtype=np.int64)
        seed_membership = np.isin(intersection, seed_indices, assume_unique=True)
        retained = intersection[seed_membership]
        removed = intersection[~seed_membership]
        return retained, removed, component_ids, {
            "spacing": None,
            "connectivity_radius": None,
            "component_count": int(intersection.size),
            "seed_touching_component_count": int(retained.size),
            "retained_gaussian_count": int(retained.size),
            "removed_gaussian_count": int(removed.size),
            "components": [
                {
                    "component_id": 0,
                    "gaussian_count": 1,
                    "seed_gaussian_count": int(seed_membership[0]),
                    "touches_source_seed": bool(seed_membership[0]),
                }
            ]
            if intersection.size
            else [],
        }

    spacing, spacing_metrics = estimate_local_gaussian_spacing(
        xyz[intersection], neighbor_k, spacing_max_samples
    )
    radius = spacing * radius_multiplier
    component_ids, component_sizes, graph_metrics = radius_connected_components(
        xyz[intersection], radius
    )
    seed_membership = np.isin(intersection, seed_indices, assume_unique=True)
    seed_counts = np.bincount(
        component_ids[seed_membership], minlength=component_sizes.size
    ).astype(np.int64)
    touching_ids = np.flatnonzero(seed_counts > 0)
    retained_mask = np.isin(component_ids, touching_ids)
    retained = intersection[retained_mask]
    removed = intersection[~retained_mask]
    components = [
        {
            "component_id": int(component_id),
            "gaussian_count": int(component_sizes[component_id]),
            "seed_gaussian_count": int(seed_counts[component_id]),
            "touches_source_seed": bool(seed_counts[component_id] > 0),
        }
        for component_id in range(component_sizes.size)
    ]
    return retained, removed, component_ids, {
        "spacing": spacing_metrics,
        "connectivity_radius": radius,
        **graph_metrics,
        "seed_touching_component_count": int(touching_ids.size),
        "retained_gaussian_count": int(retained.size),
        "removed_gaussian_count": int(removed.size),
        "components": components,
    }


def histogram_records(
    labels: np.ndarray,
    label_items: dict[int, dict[str, Any]],
    limit: int = 12,
) -> list[dict[str, Any]]:
    counts = Counter(int(value) for value in labels)
    records: list[dict[str, Any]] = []
    for label_id, count in counts.most_common(limit):
        item = label_items.get(label_id, {})
        records.append(
            {
                "label_id": label_id,
                "class": normalize_class_name(item.get("class", "unlabeled")),
                "name": str(item.get("name", "unlabeled")),
                "count": int(count),
                "fraction": count / float(max(labels.size, 1)),
            }
        )
    return records


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0


def save_evidence_overlay(
    rgb: np.ndarray,
    detection: IndependentDetection,
    prompt: ProjectionPrompt,
    output_path: Path,
    seed_key: str,
) -> None:
    overlay = rgb.copy()
    color = np.asarray(rgb8_for_class(detection.class_name), dtype=np.uint8)
    selected = detection.mask.astype(bool)
    overlay[selected] = (0.55 * overlay[selected] + 0.45 * color).astype(np.uint8)
    image = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    x1, y1, x2, y2 = detection.bbox_xyxy
    draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
    px1, py1, px2, py2 = prompt.bbox_xyxy
    draw.rectangle((px1, py1, px2, py2), outline=(255, 220, 0), width=2)
    draw.text(
        (max(0, int(x1)), max(0, int(y1) - 13)),
        f"independent {detection.class_name} {detection.grounding_score:.2f}",
        fill=(0, 255, 0),
        font=font,
    )
    draw.text(
        (6, 6),
        f"seed {seed_key}; yellow=projected association only",
        fill=(255, 220, 0),
        font=font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--ground-output-dir", required=True, type=Path)
    parser.add_argument("--proposal-dir", required=True, type=Path)
    parser.add_argument("--class-config", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--baseline-labels", required=True, type=Path)
    parser.add_argument("--baseline-label-map", required=True, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--source-view-manifest", required=True, type=Path)
    parser.add_argument("--groundingdino-root", required=True, type=Path)
    parser.add_argument("--groundingdino-config", required=True, type=Path)
    parser.add_argument("--groundingdino-checkpoint", required=True, type=Path)
    parser.add_argument("--segment-anything-root", required=True, type=Path)
    parser.add_argument("--sam-checkpoint", required=True, type=Path)
    parser.add_argument("--flashsplat-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--input-manifest-name", default="grounded_sam_manifest.json")
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--max-width", default=960, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam-arch", default="vit_h")
    parser.add_argument("--merge-iou", default=0.35, type=float)
    parser.add_argument("--containment-threshold", default=0.70, type=float)
    parser.add_argument("--min-proposal-gaussians", default=500, type=int)
    parser.add_argument("--min-seed-gaussians", default=1500, type=int)
    parser.add_argument("--min-source-sam-score", default=0.85, type=float)
    parser.add_argument("--min-source-grounding-score", default=0.25, type=float)
    parser.add_argument("--min-source-mask-area-ratio", default=0.005, type=float)
    parser.add_argument("--max-breadcrumb-fraction", default=0.50, type=float)
    parser.add_argument("--min-unresolved-fraction", default=0.50, type=float)
    parser.add_argument("--max-seeds", default=48, type=int)
    parser.add_argument("--min-projected-gaussians", default=250, type=int)
    parser.add_argument("--min-projected-fraction", default=0.05, type=float)
    parser.add_argument("--max-baseline-depth-ratio", default=0.90, type=float)
    parser.add_argument("--box-padding-ratio", default=0.05, type=float)
    parser.add_argument("--min-box-area-ratio", default=0.0025, type=float)
    parser.add_argument("--max-box-area-ratio", default=0.85, type=float)
    parser.add_argument("--min-source-angle-degrees", default=8.0, type=float)
    parser.add_argument("--min-pairwise-angle-degrees", default=8.0, type=float)
    parser.add_argument("--max-target-views", default=6, type=int)
    parser.add_argument("--box-threshold", default=0.30, type=float)
    parser.add_argument("--text-threshold", default=0.25, type=float)
    parser.add_argument("--min-independent-grounding-score", default=0.30, type=float)
    parser.add_argument("--min-independent-sam-score", default=0.90, type=float)
    parser.add_argument("--min-association-overlap", default=0.05, type=float)
    parser.add_argument("--min-total-contribution", default=0.05, type=float)
    parser.add_argument("--min-positive-contribution-fraction", default=0.60, type=float)
    parser.add_argument("--min-evidence-views", default=2, type=int)
    parser.add_argument("--min-intersection-gaussians", default=250, type=int)
    parser.add_argument("--min-intersection-fraction", default=0.05, type=float)
    parser.add_argument("--audit-mode", choices=("v9", "v10"), default="v9")
    parser.add_argument("--min-second-seed-contribution-fraction", default=0.20, type=float)
    parser.add_argument("--min-second-to-best-contribution-ratio", default=0.50, type=float)
    parser.add_argument("--component-neighbor-k", default=4, type=int)
    parser.add_argument("--component-radius-multiplier", default=2.0, type=float)
    parser.add_argument("--component-spacing-max-samples", default=512, type=int)
    parser.add_argument("--diagnostic-lift-quality-failures", action="store_true")
    parser.add_argument("--diagnostic-max-grounding-shortfall", default=0.05, type=float)
    parser.add_argument("--diagnostic-max-sam-shortfall", default=0.025, type=float)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_evidence_views < 2:
        raise ValueError("--min-evidence-views must be at least 2")
    if args.max_target_views < args.min_evidence_views:
        raise ValueError("--max-target-views must be at least --min-evidence-views")
    if args.audit_mode == "v10" and not args.diagnostic_lift_quality_failures:
        raise ValueError("v10 requires --diagnostic-lift-quality-failures")
    if args.audit_mode == "v10" and args.min_evidence_views != 2:
        raise ValueError("v10 identity consistency requires exactly two evidence views")
    if args.component_neighbor_k < 1 or args.component_spacing_max_samples < 1:
        raise ValueError("component spacing parameters must be positive")
    if args.component_radius_multiplier <= 0.0:
        raise ValueError("--component-radius-multiplier must be positive")
    if (
        args.diagnostic_max_grounding_shortfall < 0.0
        or args.diagnostic_max_sam_shortfall < 0.0
    ):
        raise ValueError("diagnostic quality shortfalls must be non-negative")
    summary_name = (
        "identity_component_audit_summary.json"
        if args.audit_mode == "v10"
        else "independent_multiview_evidence_summary.json"
    )
    summary_path = args.output_dir / summary_name
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {summary_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlays"
    evidence_dir = args.output_dir / "candidate_evidence"
    diagnostic_dir = args.output_dir / "diagnostic_evidence"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    if args.diagnostic_lift_quality_failures:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)

    baseline_map = load_json(args.baseline_label_map)
    if str(baseline_map.get("scene", "")) != args.scene:
        raise ValueError("Baseline label-map scene does not match --scene")
    label_items = label_items_by_id(baseline_map, args.baseline_label_map)
    baseline_labels = validate_label_array(
        np.load(args.baseline_labels), label_items, args.baseline_labels
    )
    ontology = load_ontology(args.ontology)
    ontology_kinds = {item.project_class: item.kind for item in ontology.classes}

    header = read_ply_header(args.source_ply)
    vertex = header.element("vertex")
    if vertex is None or vertex.count != int(baseline_labels.size):
        raise ValueError("Source PLY vertex count does not match baseline labels")
    _header, vertex_data = vertex_data_memmap(args.source_ply)
    xyz = np.stack(
        [np.asarray(vertex_data[axis]) for axis in ("x", "y", "z")], axis=1
    )

    grounded_manifest_path = args.ground_output_dir / args.input_manifest_name
    grounded_manifest = load_json(grounded_manifest_path)
    grounded_frames = list(grounded_manifest.get("frames", []))
    grounded_frames_by_file = {str(frame["file"]): frame for frame in grounded_frames}
    verification_manifest_path, verification_manifest = resolve_verification_view_manifest(
        grounded_manifest_path,
        grounded_manifest,
        args.source_view_manifest,
    )
    verification_frames = list(verification_manifest.get("frames", []))
    if not verification_frames:
        raise ValueError("No real-camera verification frames were found")
    rgb_dir = rgb_directory(verification_manifest_path, verification_manifest)
    cameras = load_cameras(args.model_path)

    stuff_classes = load_stuff_classes(args.class_config, "")
    thing_classes = thing_classes_from_config(args.class_config)
    proposals = load_proposals(
        args.proposal_dir / "proposal_manifest.json",
        args.proposal_dir / "proposal_supports",
        args.min_proposal_gaussians,
        0,
        True,
    )
    singleton_items = singleton_proposals(
        proposals,
        stuff_classes,
        args.merge_iou,
        args.containment_threshold,
    )

    candidate_rows: list[tuple[tuple[float, ...], CandidateContext]] = []
    rejected_reports: list[dict[str, Any]] = []
    for proposal in singleton_items:
        frame_file = str(proposal.metadata.get("frame_file", ""))
        mask_index = int(proposal.metadata.get("mask_index", -1))
        seed_key = f"{frame_file}:{mask_index}"
        report: dict[str, Any] = {
            "seed_key": seed_key,
            "proposal_id": int(proposal.proposal_id),
            "class": normalize_class_name(proposal.class_name),
            "frame_file": frame_file,
            "mask_index": mask_index,
            "gaussian_count": int(proposal.gaussian_count),
            "source_sam_score": proposal_sam_score(proposal),
            "source_grounding_score": proposal_grounding_score(proposal),
            "status": "source_rejected",
            "reasons": [],
        }
        source_frame = grounded_frames_by_file.get(frame_file)
        if source_frame is None:
            report["reasons"].append("source_frame_missing")
            rejected_reports.append(report)
            continue
        source_masks = load_mask_stack(
            frame_mask_path(grounded_manifest_path, args.ground_output_dir, source_frame)
        )
        if mask_index < 0 or mask_index >= source_masks.shape[0]:
            report["reasons"].append("source_mask_missing")
            rejected_reports.append(report)
            continue
        source_mask = source_masks[mask_index]
        mask_area_ratio = int(source_mask.sum()) / float(max(source_mask.size, 1))
        nested = nested_competing_thing(
            source_mask,
            proposal.class_name,
            source_masks,
            list(source_frame.get("masks", [])),
            thing_classes,
            1.5,
        )
        decision = evaluate_seed_policy(
            proposal.indices,
            baseline_labels,
            label_items,
            ontology_kinds,
            proposal.class_name,
            0.0,
            1.0,
            1.0,
            0.0,
        )
        unresolved_fraction = 1.0 - float(decision.metrics["breadcrumb_fraction"])
        report.update(
            {
                "source_mask_area_ratio": mask_area_ratio,
                "nested_competing_thing": nested,
                "baseline_policy": decision.metrics,
                "unresolved_fraction": unresolved_fraction,
                "source_label_histogram": histogram_records(
                    baseline_labels[proposal.indices], label_items
                ),
            }
        )
        if proposal.gaussian_count < args.min_seed_gaussians:
            report["reasons"].append(f"gaussian_count<{args.min_seed_gaussians}")
        if proposal_sam_score(proposal) < args.min_source_sam_score:
            report["reasons"].append(f"source_sam_score<{args.min_source_sam_score}")
        if proposal_grounding_score(proposal) < args.min_source_grounding_score:
            report["reasons"].append(
                f"source_grounding_score<{args.min_source_grounding_score}"
            )
        if mask_area_ratio < args.min_source_mask_area_ratio:
            report["reasons"].append(
                f"source_mask_area_ratio<{args.min_source_mask_area_ratio}"
            )
        if float(decision.metrics["breadcrumb_fraction"]) > args.max_breadcrumb_fraction:
            report["reasons"].append(
                f"breadcrumb_fraction>{args.max_breadcrumb_fraction}"
            )
        if unresolved_fraction < args.min_unresolved_fraction:
            report["reasons"].append(
                f"unresolved_fraction<{args.min_unresolved_fraction}"
            )
        if report["reasons"]:
            rejected_reports.append(report)
            continue

        source_camera = cameras[int(source_frame["camera_index"])]
        prompts = candidate_view_prompts(
            proposal,
            xyz[proposal.indices],
            source_camera,
            verification_frames,
            cameras,
            rgb_dir,
            args.min_projected_gaussians,
            args.min_projected_fraction,
            args.max_baseline_depth_ratio,
            args.box_padding_ratio,
            args.min_box_area_ratio,
            args.max_box_area_ratio,
            len(verification_frames),
        )
        centroid = np.median(xyz[proposal.indices], axis=0)
        separated = select_geometry_separated_views(
            prompts,
            source_camera,
            cameras,
            centroid,
            args.min_source_angle_degrees,
            args.min_pairwise_angle_degrees,
            args.max_target_views,
        )
        report["candidate_views"] = [
            {
                "frame_file": item.prompt.target_frame_file,
                "camera_index": item.prompt.target_camera_index,
                "projected_gaussian_count": item.prompt.projected_gaussian_count,
                "projected_fraction": item.prompt.projected_fraction,
                "baseline_depth_ratio": item.prompt.baseline_depth_ratio,
                "source_angle_degrees": item.source_angle_degrees,
                "geometry_score": item.geometry_score,
                "projected_bbox_xyxy": list(item.prompt.bbox_xyxy),
            }
            for item in separated
        ]
        if len(separated) < args.min_evidence_views:
            report["status"] = "geometry_abstained"
            report["reasons"].append(
                f"separated_view_count<{args.min_evidence_views}"
            )
            rejected_reports.append(report)
            continue
        report["status"] = "scheduled"
        opportunity = max(
            float(decision.metrics["recoverable_background_fraction"]),
            float(decision.metrics["competing_thing_fraction"]),
        )
        rank = (
            opportunity,
            proposal_sam_score(proposal),
            proposal_grounding_score(proposal),
            math.log1p(proposal.gaussian_count),
        )
        candidate_rows.append(
            (rank, CandidateContext(proposal, source_frame, report, separated))
        )

    candidate_rows.sort(key=lambda item: item[0], reverse=True)
    scheduled = [context for _rank, context in candidate_rows[: args.max_seeds]]
    for _rank, context in candidate_rows[args.max_seeds :]:
        context.report["status"] = "source_rejected"
        context.report["reasons"].append(f"max_seeds>={args.max_seeds}")
        rejected_reports.append(context.report)

    plan_path = args.output_dir / (
        "identity_component_audit_plan.json"
        if args.audit_mode == "v10"
        else "independent_multiview_evidence_plan.json"
    )
    plan_path.write_text(
        json.dumps(
            {
                "source": f"report_only_independent_multiview_evidence_{args.audit_mode}_plan",
                "scene": args.scene,
                "singleton_count": len(singleton_items),
                "scheduled_candidate_count": len(scheduled),
                "semantic_labels_written": False,
                "semantic_ply_written": False,
                "candidates": [context.report for context in scheduled],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.plan_only:
        print(
            json.dumps(
                {
                    "plan": str(plan_path),
                    "singleton_count": len(singleton_items),
                    "scheduled_candidate_count": len(scheduled),
                    "scheduled_proposal_ids": [
                        int(context.proposal.proposal_id) for context in scheduled
                    ],
                    "semantic_labels_written": False,
                    "semantic_ply_written": False,
                },
                indent=2,
            )
        )
        return

    # CUDA-heavy imports remain below source/geometry screening so the policy
    # and geometry helpers stay importable in CPU-only unit tests.
    import torch

    from flashsplat_cameras import (
        background_tensor,
        default_pipeline,
        load_flashsplat,
        load_gaussians,
        make_camera,
        point_cloud_path,
        render_flashsplat,
    )
    from generate_grounded_sam_masks import (
        class_from_phrase,
        cxcywh_to_xyxy_pixels,
        load_grounding_model,
        load_sam_predictor,
        parse_class_specs,
        run_grounding_dino,
        run_sam_for_boxes,
        text_prompt_from_specs,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the independent evidence experiment")
    specs = parse_class_specs(args.class_config, None)
    specs_by_name = {normalize_class_name(spec.name): spec for spec in specs}
    missing_specs = sorted(
        {
            normalize_class_name(context.proposal.class_name)
            for context in scheduled
            if normalize_class_name(context.proposal.class_name) not in specs_by_name
        }
    )
    if missing_specs:
        raise ValueError(f"Candidate classes are absent from the class config: {missing_specs}")

    grounding_model, transforms_module, phrase_from_posmap = load_grounding_model(
        args.groundingdino_root,
        args.groundingdino_config,
        args.groundingdino_checkpoint,
        args.device,
    )
    sam_predictor = load_sam_predictor(
        args.segment_anything_root,
        args.sam_checkpoint,
        args.sam_arch,
        args.device,
    )
    modules = load_flashsplat(args.flashsplat_root)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    pipeline = default_pipeline()
    background = background_tensor(False)

    requests_by_frame: dict[str, list[tuple[CandidateContext, SeparatedView]]] = defaultdict(list)
    for context in scheduled:
        for separated in context.separated_views:
            requests_by_frame[separated.prompt.target_frame_file].append(
                (context, separated)
            )

    accepted_supports: dict[str, list[tuple[dict[str, Any], np.ndarray]]] = defaultdict(list)
    for frame_file, requests in sorted(requests_by_frame.items()):
        rgb = np.asarray(Image.open(rgb_dir / frame_file).convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        requested_classes = sorted(
            {normalize_class_name(context.proposal.class_name) for context, _ in requests}
        )
        frame_specs = [specs_by_name[class_name] for class_name in requested_classes]
        text_prompt = text_prompt_from_specs(frame_specs, None)
        boxes, phrases, grounding_scores = run_grounding_dino(
            grounding_model,
            transforms_module,
            phrase_from_posmap,
            rgb,
            text_prompt,
            args.box_threshold,
            args.text_threshold,
            args.device,
        )
        boxes_xyxy = cxcywh_to_xyxy_pixels(boxes, width, height)
        masks, sam_scores = run_sam_for_boxes(
            sam_predictor, rgb, boxes_xyxy, args.device
        )
        detections: list[IndependentDetection] = []
        for index in range(masks.shape[0]):
            phrase = phrases[index] if index < len(phrases) else ""
            class_name = class_from_phrase(phrase, frame_specs)
            detections.append(
                IndependentDetection(
                    mask=masks[index],
                    class_name=class_name,
                    phrase=phrase,
                    grounding_score=(
                        float(grounding_scores[index])
                        if index < len(grounding_scores)
                        else 0.0
                    ),
                    sam_score=(
                        float(sam_scores[index]) if index < len(sam_scores) else 0.0
                    ),
                    bbox_xyxy=tuple(float(value) for value in boxes_xyxy[index].tolist()),
                )
            )

        contribution_cache: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
        for context, separated in requests:
            prompt = separated.prompt
            detection, association = match_independent_detection(prompt, detections)
            view_report: dict[str, Any] = {
                "frame_file": frame_file,
                "camera_index": prompt.target_camera_index,
                "source_angle_degrees": separated.source_angle_degrees,
                "baseline_depth_ratio": prompt.baseline_depth_ratio,
                "projected_fraction": prompt.projected_fraction,
                **association,
                "status": "rejected",
                "reasons": [],
            }
            context.report.setdefault("independent_view_checks", []).append(view_report)
            if detection is None:
                view_report["reasons"].append("same_class_grounding_detection_missing")
                continue
            view_report.update(
                {
                    "detected_class": normalize_class_name(detection.class_name),
                    "phrase": detection.phrase,
                    "grounding_score": detection.grounding_score,
                    "sam_score": detection.sam_score,
                    "detection_bbox_xyxy": list(detection.bbox_xyxy),
                    "mask_area": int(detection.mask.sum()),
                    "mask_area_ratio": int(detection.mask.sum()) / float(max(detection.mask.size, 1)),
                }
            )
            if detection.grounding_score < args.min_independent_grounding_score:
                view_report["reasons"].append(
                    f"grounding_score<{args.min_independent_grounding_score}"
                )
            if detection.sam_score < args.min_independent_sam_score:
                view_report["reasons"].append(f"sam_score<{args.min_independent_sam_score}")
            if max(
                association["projected_seed_coverage"],
                association["projected_box_iou"],
            ) < args.min_association_overlap:
                view_report["reasons"].append(
                    f"association_overlap<{args.min_association_overlap}"
                )
            association_overlap = max(
                association["projected_seed_coverage"],
                association["projected_box_iou"],
            )
            diagnostic_decision = diagnostic_lift_decision(
                detection.grounding_score,
                detection.sam_score,
                association_overlap,
                args.min_independent_grounding_score,
                args.min_independent_sam_score,
                args.min_association_overlap,
                args.diagnostic_max_grounding_shortfall,
                args.diagnostic_max_sam_shortfall,
            )
            diagnostic_lift = bool(
                args.diagnostic_lift_quality_failures
                and diagnostic_decision["diagnostic_lift"]
            )
            view_report.update(
                {
                    **diagnostic_decision,
                    "diagnostic_lift_enabled": bool(
                        args.diagnostic_lift_quality_failures
                    ),
                }
            )
            if view_report["reasons"] and not diagnostic_lift:
                continue

            detection_index = next(
                index for index, item in enumerate(detections) if item is detection
            )
            if detection_index not in contribution_cache:
                camera_json = cameras[prompt.target_camera_index]
                camera = make_camera(camera_json, modules, args.max_width)
                mask = resize_mask(
                    detection.mask,
                    int(camera.image_width),
                    int(camera.image_height),
                )
                gt_mask = torch.from_numpy(mask.astype(np.float32)).to(
                    device=args.device
                )
                render_pkg = render_flashsplat(
                    camera,
                    gaussians,
                    modules,
                    pipeline,
                    background,
                    gt_mask=gt_mask,
                    obj_num=2,
                )
                used_count = render_pkg["used_count"].detach().cpu().numpy()
                support, contribution_metrics = contribution_support(
                    used_count[1],
                    used_count[0],
                    args.min_total_contribution,
                    args.min_positive_contribution_fraction,
                )
                contribution_cache[detection_index] = (support, contribution_metrics)
                del used_count
                del render_pkg
                del gt_mask
                torch.cuda.empty_cache()
            support, contribution_metrics = contribution_cache[detection_index]
            seed_overlap = np.intersect1d(
                support,
                context.proposal.indices,
                assume_unique=True,
            ).astype(np.uint32, copy=False)
            view_report.update(
                {
                    **contribution_metrics,
                    "seed_contribution_gaussian_count": int(seed_overlap.size),
                    "seed_contribution_fraction": seed_overlap.size
                    / float(max(context.proposal.gaussian_count, 1)),
                    "status": (
                        "accepted"
                        if diagnostic_decision["quality_accepted"]
                        else "diagnostic_quality_failed"
                    ),
                }
            )
            if diagnostic_decision["quality_accepted"]:
                accepted_supports[context.report["seed_key"]].append(
                    (view_report, support)
                )
            else:
                diagnostic_name = (
                    f"proposal_{context.proposal.proposal_id:06d}__"
                    f"{safe_name(frame_file)}.npz"
                )
                np.savez_compressed(
                    diagnostic_dir / diagnostic_name,
                    support=support,
                    seed_overlap=seed_overlap,
                )
                view_report["diagnostic_evidence_file"] = str(
                    Path("diagnostic_evidence") / diagnostic_name
                )
            overlay_name = (
                f"{'proposal' if diagnostic_decision['quality_accepted'] else 'diagnostic'}_"
                f"{context.proposal.proposal_id:06d}__"
                f"{safe_name(frame_file)}.png"
            )
            save_evidence_overlay(
                rgb,
                detection,
                prompt,
                overlay_dir / overlay_name,
                context.report["seed_key"],
            )
            view_report["overlay_file"] = str(Path("overlays") / overlay_name)

    completed_reports: list[dict[str, Any]] = []
    for context in scheduled:
        report = context.report
        accepted = sorted(
            accepted_supports.get(report["seed_key"], []),
            key=lambda item: (
                float(item[0]["seed_contribution_fraction"]),
                max(
                    float(item[0]["projected_seed_coverage"]),
                    float(item[0]["projected_box_iou"]),
                ),
                float(item[0]["source_angle_degrees"]),
            ),
            reverse=True,
        )
        report["accepted_independent_view_count"] = len(accepted)
        if len(accepted) < args.min_evidence_views:
            report["status"] = "semantic_abstained"
            report["reasons"].append(
                f"accepted_independent_view_count<{args.min_evidence_views}"
            )
            completed_reports.append(report)
            continue
        identity_passed = True
        if args.audit_mode == "v10":
            identity_passed, identity_metrics = identity_consistency_gate(
                [
                    float(item[0]["seed_contribution_fraction"])
                    for item in accepted
                ],
                args.min_second_seed_contribution_fraction,
                args.min_second_to_best_contribution_ratio,
            )
            report["identity_consistency"] = identity_metrics
        selected = accepted[: args.min_evidence_views]
        intersection = np.unique(selected[0][1]).astype(np.uint32, copy=False)
        for _view_report, support in selected[1:]:
            intersection = np.intersect1d(
                intersection,
                np.unique(support),
                assume_unique=True,
            ).astype(np.uint32, copy=False)
        seed_intersection = np.intersect1d(
            intersection,
            context.proposal.indices,
            assume_unique=True,
        ).astype(np.uint32, copy=False)
        outside_seed = np.setdiff1d(
            intersection,
            context.proposal.indices,
            assume_unique=True,
        ).astype(np.uint32, copy=False)
        intersection_fraction = intersection.size / float(
            max(context.proposal.gaussian_count, 1)
        )
        report.update(
            {
                "selected_evidence_views": [item[0]["frame_file"] for item in selected],
                "multiview_intersection_gaussian_count": int(intersection.size),
                "multiview_intersection_vs_seed_fraction": intersection_fraction,
                "intersection_inside_seed_gaussian_count": int(seed_intersection.size),
                "intersection_outside_seed_gaussian_count": int(outside_seed.size),
                "intersection_label_histogram": histogram_records(
                    baseline_labels[intersection], label_items
                ),
                "outside_seed_label_histogram": histogram_records(
                    baseline_labels[outside_seed], label_items
                ),
            }
        )
        evidence_file = f"proposal_{context.proposal.proposal_id:06d}.npz"
        if args.audit_mode == "v10" and identity_passed:
            retained, component_removed, component_ids, component_metrics = (
                retain_seed_touching_components(
                    intersection,
                    context.proposal.indices,
                    xyz,
                    args.component_neighbor_k,
                    args.component_radius_multiplier,
                    args.component_spacing_max_samples,
                )
            )
            component_seed = np.intersect1d(
                retained,
                context.proposal.indices,
                assume_unique=True,
            ).astype(np.uint32, copy=False)
            component_outside = np.setdiff1d(
                retained,
                context.proposal.indices,
                assume_unique=True,
            ).astype(np.uint32, copy=False)
            for component in component_metrics["components"]:
                member_indices = intersection[
                    component_ids == int(component["component_id"])
                ]
                component["label_histogram"] = histogram_records(
                    baseline_labels[member_indices], label_items
                )
            retained_fraction = retained.size / float(
                max(context.proposal.gaussian_count, 1)
            )
            report.update(
                {
                    "component_audit": component_metrics,
                    "component_retained_gaussian_count": int(retained.size),
                    "component_retained_vs_seed_fraction": retained_fraction,
                    "component_retained_inside_seed_gaussian_count": int(
                        component_seed.size
                    ),
                    "component_retained_outside_seed_gaussian_count": int(
                        component_outside.size
                    ),
                    "component_removed_gaussian_count": int(component_removed.size),
                    "component_retained_label_histogram": histogram_records(
                        baseline_labels[retained], label_items
                    ),
                    "component_retained_outside_seed_label_histogram": histogram_records(
                        baseline_labels[component_outside], label_items
                    ),
                }
            )
            np.savez_compressed(
                evidence_dir / evidence_file,
                intersection=intersection,
                inside_seed=seed_intersection,
                outside_seed=outside_seed,
                component_ids=component_ids,
                retained_seed_touching_components=retained,
                removed_non_seed_components=component_removed,
            )
            evidence_count = int(retained.size)
            evidence_fraction = retained_fraction
        elif args.audit_mode == "v10":
            np.savez_compressed(
                evidence_dir / evidence_file,
                intersection=intersection,
                inside_seed=seed_intersection,
                outside_seed=outside_seed,
                component_ids=np.full(intersection.size, -1, dtype=np.int64),
                retained_seed_touching_components=np.zeros((0,), dtype=np.uint32),
                removed_non_seed_components=np.zeros((0,), dtype=np.uint32),
            )
            evidence_count = 0
            evidence_fraction = 0.0
        else:
            np.savez_compressed(
                evidence_dir / evidence_file,
                intersection=intersection,
                inside_seed=seed_intersection,
                outside_seed=outside_seed,
            )
            evidence_count = int(intersection.size)
            evidence_fraction = intersection_fraction
        report["evidence_file"] = str(Path("candidate_evidence") / evidence_file)
        if args.audit_mode == "v10" and not identity_passed:
            report["reasons"].extend(report["identity_consistency"]["reasons"])
            report["status"] = "identity_abstained"
            completed_reports.append(report)
            continue
        if evidence_count < args.min_intersection_gaussians:
            count_name = (
                "evidence_gaussian_count"
                if args.audit_mode == "v10"
                else "intersection_gaussian_count"
            )
            report["reasons"].append(
                f"{count_name}<{args.min_intersection_gaussians}"
            )
        if evidence_fraction < args.min_intersection_fraction:
            fraction_name = (
                "evidence_fraction"
                if args.audit_mode == "v10"
                else "intersection_fraction"
            )
            report["reasons"].append(
                f"{fraction_name}<{args.min_intersection_fraction}"
            )
        if args.audit_mode == "v10":
            report["status"] = (
                "component_evidence_supported"
                if not report["reasons"]
                else "component_evidence_abstained"
            )
        else:
            report["status"] = (
                "evidence_supported" if not report["reasons"] else "evidence_abstained"
            )
        completed_reports.append(report)

    all_reports = sorted(
        [*completed_reports, *rejected_reports],
        key=lambda item: int(item["proposal_id"]),
    )
    status_counts = Counter(str(item["status"]) for item in all_reports)
    diagnostic_lifted_view_count = sum(
        1
        for report in all_reports
        for view in report.get("independent_view_checks", [])
        if view.get("status") == "diagnostic_quality_failed"
    )
    summary = {
        "source": (
            "report_only_identity_consistent_adaptive_component_audit_v10"
            if args.audit_mode == "v10"
            else "report_only_independent_groundingdino_sam_flashsplat_audit"
        ),
        "audit_mode": args.audit_mode,
        "scene": args.scene,
        "invariants": {
            "semantic_labels_written": False,
            "semantic_ply_written": False,
            "groundingdino_boxes_are_independent": True,
            "seed_projection_used_only_for_post_detection_association": True,
            "verification_views_are_geometry_separated": True,
            "flashsplat_support_uses_positive_vs_negative_contribution": True,
            "identity_gate_uses_only_quality_passing_views": (
                args.audit_mode == "v10"
            ),
            "diagnostic_quality_failures_excluded_from_identity_gate": (
                args.audit_mode == "v10"
            ),
            "component_radius_derived_from_candidate_local_spacing": (
                args.audit_mode == "v10"
            ),
            "only_seed_touching_components_retained": args.audit_mode == "v10",
        },
        "inputs": {
            "baseline_labels": str(args.baseline_labels),
            "baseline_label_map": str(args.baseline_label_map),
            "source_ply": str(args.source_ply),
            "proposal_manifest": str(args.proposal_dir / "proposal_manifest.json"),
            "grounded_sam_manifest": str(grounded_manifest_path),
            "verification_view_manifest": str(verification_manifest_path),
        },
        "thresholds": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool))
        },
        "singleton_count": len(singleton_items),
        "scheduled_candidate_count": len(scheduled),
        "diagnostic_lifted_view_count": diagnostic_lifted_view_count,
        "status_counts": dict(sorted(status_counts.items())),
        "candidate_reports": all_reports,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "scheduled_candidate_count": len(scheduled),
        "status_counts": dict(sorted(status_counts.items())),
        "semantic_labels_written": False,
        "semantic_ply_written": False,
    }, indent=2))


if __name__ == "__main__":
    main()
