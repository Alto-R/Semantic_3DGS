#!/usr/bin/env python3
"""Apply a conservative two-tier singleton patch after the accepted v5 merge.

The v5 labels are immutable input. A one-view Grounded-SAM singleton may only
change Gaussians already contained in its original lifted support. Alternate
SAM verification may use every real camera view, not only the subset selected
for GroundingDINO. A verification mask acts as a point-level filter: it can
remove unsupported seed points but can never add points or alter global
proposal clustering. Tier 1 confirms a partial same-class breadcrumb. Tier 2
may recover a low- or zero-breadcrumb candidate only from an unambiguous,
geometry-compatible Tier-1 anchor of the same class and the intersection of
two alternate-view confirmations.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from add_labels_from_npy import write_ply_with_labels
from cluster_semantic_flashsplat_proposals import load_proposals, load_stuff_classes
from dinov2_ontology import load_ontology, normalize_class_name
from merge_semantic_extensions import (
    histogram,
    label_items_by_id,
    load_json,
    transition_records,
    validate_label_array,
)
from ply_utils import read_ply_header, resolve_semantic_ply_output, vertex_data_memmap
from recover_cross_view_sam_masks import (
    ProjectionPrompt,
    candidate_view_prompts,
    frame_mask_path,
    load_cameras,
    load_mask_stack,
    nested_competing_thing,
    projected_mask_membership,
    proposal_grounding_score,
    proposal_sam_score,
    rgb_directory,
    singleton_proposals,
    thing_classes_from_config,
)


@dataclass(frozen=True)
class SeedPolicyDecision:
    accepted: bool
    reasons: tuple[str, ...]
    target_label_id: int | None
    breadcrumb_local: np.ndarray
    patchable_local: np.ndarray
    metrics: dict[str, Any]


@dataclass(frozen=True)
class PointPatch:
    seed_key: str
    target_label_id: int
    indices: np.ndarray


@dataclass(frozen=True)
class GeometrySignature:
    major_extent: float
    normalized_extents: np.ndarray


@dataclass(frozen=True)
class AcceptedAnchor:
    seed_key: str
    class_name: str
    target_label_id: int
    signature: GeometrySignature


def resolve_verification_view_manifest(
    grounded_manifest_path: Path,
    grounded_manifest: dict[str, Any],
    override: Path | None,
) -> tuple[Path, dict[str, Any]]:
    """Load the full real-camera manifest used for alternate-view checks.

    Grounded-SAM normally records the source real-camera manifest from which
    its targeted subset was selected. An explicit path takes precedence so a
    replay does not depend on stale absolute paths embedded in cached output.
    Falling back to the Grounded-SAM manifest preserves compatibility with
    older caches while making that reduced verification pool explicit in the
    report.
    """
    configured: Path | None = override
    if configured is None:
        raw = str(grounded_manifest.get("source_view_manifest", "")).strip()
        if raw:
            configured = Path(raw)
            if not configured.is_absolute():
                configured = grounded_manifest_path.parent / configured
    if configured is None:
        return grounded_manifest_path, grounded_manifest
    if not configured.exists():
        raise FileNotFoundError(configured)
    return configured, load_json(configured)


def robust_geometry_signature(points: np.ndarray) -> GeometrySignature:
    """Describe 3D scale and shape without depending on world orientation."""
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError("points must have shape (N,3) with at least three points")
    centered = points.astype(np.float64, copy=False) - np.median(points, axis=0)
    _u, _singular, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes.T
    low, high = np.quantile(projected, [0.02, 0.98], axis=0)
    extents = np.sort(np.maximum(high - low, 1.0e-6))[::-1]
    major = float(extents[0])
    # The thinnest PCA extent is noisy for surfaces. A small floor prevents
    # sampling thickness from dominating an otherwise compatible shape test.
    normalized = np.maximum(extents / major, 0.05)
    return GeometrySignature(major_extent=major, normalized_extents=normalized)


def geometry_compatibility(
    candidate: GeometrySignature,
    anchor: GeometrySignature,
    max_size_ratio: float,
    max_shape_ratio: float,
) -> tuple[bool, dict[str, float]]:
    """Compare two proposal supports using global scale and PCA shape gates."""
    size_ratio = max(
        candidate.major_extent / anchor.major_extent,
        anchor.major_extent / candidate.major_extent,
    )
    ratios = np.maximum(
        candidate.normalized_extents / anchor.normalized_extents,
        anchor.normalized_extents / candidate.normalized_extents,
    )
    shape_ratio = float(np.max(ratios))
    return size_ratio <= max_size_ratio and shape_ratio <= max_shape_ratio, {
        "size_ratio": float(size_ratio),
        "shape_ratio": shape_ratio,
    }


def evaluate_anchor_candidate_policy(
    decision: SeedPolicyDecision,
    tier1_min_breadcrumb_fraction: float,
    max_breadcrumb_fraction: float,
    max_competing_thing_fraction: float,
    min_recoverable_background_fraction: float,
) -> tuple[str, ...]:
    """Gate a low-breadcrumb candidate without requiring a target label yet."""
    metrics = decision.metrics
    reasons: list[str] = []
    breadcrumb_fraction = float(metrics["breadcrumb_fraction"])
    if breadcrumb_fraction >= tier1_min_breadcrumb_fraction:
        reasons.append("tier1_breadcrumb_not_low")
    if breadcrumb_fraction > max_breadcrumb_fraction:
        reasons.append(f"breadcrumb_fraction>{max_breadcrumb_fraction}")
    if float(metrics["competing_thing_fraction"]) > max_competing_thing_fraction:
        reasons.append(
            f"competing_thing_fraction>{max_competing_thing_fraction}"
        )
    if (
        float(metrics["recoverable_background_fraction"])
        < min_recoverable_background_fraction
    ):
        reasons.append(
            "recoverable_background_fraction<"
            f"{min_recoverable_background_fraction}"
        )
    if not np.any(decision.patchable_local):
        reasons.append("no_patchable_background")
    return tuple(reasons)


def decision_with_anchor_target(
    decision: SeedPolicyDecision,
    target_label_id: int,
) -> SeedPolicyDecision:
    """Bind anchor evidence while preserving a candidate's own instance id."""
    metrics = dict(decision.metrics)
    metrics["anchor_target_label_id"] = int(target_label_id)
    metrics["candidate_target_label_id"] = decision.target_label_id
    metrics["requires_fresh_instance_label"] = decision.target_label_id is None
    resolved_target = (
        int(decision.target_label_id)
        if decision.target_label_id is not None
        else int(target_label_id)
    )
    return SeedPolicyDecision(
        accepted=True,
        reasons=(),
        target_label_id=resolved_target,
        breadcrumb_local=decision.breadcrumb_local,
        patchable_local=decision.patchable_local,
        metrics=metrics,
    )


def fresh_recovery_label_item(
    label_items: dict[int, dict[str, Any]],
    anchor_label_id: int,
    new_label_id: int,
    seed_key: str,
) -> dict[str, Any]:
    """Create a new same-class instance without copying the anchor identity."""
    if new_label_id in label_items:
        raise ValueError(f"Recovery label id {new_label_id} already exists")
    if anchor_label_id not in label_items:
        raise ValueError(f"Anchor label id {anchor_label_id} is absent")
    item = copy.deepcopy(label_items[anchor_label_id])
    class_name = normalize_class_name(item.get("class", ""))
    if not class_name or class_name == "unlabeled":
        raise ValueError("Anchor label must have a concrete semantic class")
    item.update(
        {
            "id": int(new_label_id),
            "name": f"{class_name}_singleton_{new_label_id:03d}",
            "class": class_name,
            "gaussian_count": 0,
            "source_pipeline": "post_v5_two_tier_singleton_recovery",
            "source_label_id": int(anchor_label_id),
            "recovery_seed_key": seed_key,
            "source_view_count": 3,
        }
    )
    return item


def intersect_confirmed_views(
    confirmations: list[np.ndarray],
    required_views: int,
) -> np.ndarray:
    """Keep only points confirmed by every one of the first required views."""
    if required_views < 1:
        raise ValueError("required_views must be at least one")
    if len(confirmations) < required_views:
        return np.zeros((0,), dtype=np.uint32)
    intersection = np.unique(confirmations[0]).astype(np.uint32, copy=False)
    for confirmed in confirmations[1:required_views]:
        intersection = np.intersect1d(
            intersection,
            np.unique(confirmed),
            assume_unique=True,
        ).astype(np.uint32, copy=False)
    return intersection


def label_class_by_id(items: dict[int, dict[str, Any]]) -> dict[int, str]:
    return {
        label_id: normalize_class_name(item.get("class", ""))
        for label_id, item in items.items()
    }


def label_kind_by_id(
    items: dict[int, dict[str, Any]],
    ontology_kinds: dict[str, str],
) -> dict[int, str]:
    result: dict[int, str] = {}
    for label_id, class_name in label_class_by_id(items).items():
        if label_id == 0 or class_name == "unlabeled":
            result[label_id] = "unlabeled"
        else:
            # Unknown extension classes are protected as things. This recovery
            # may never overwrite a label whose ontology kind is uncertain.
            result[label_id] = ontology_kinds.get(class_name, "thing")
    return result


def evaluate_seed_policy(
    seed_indices: np.ndarray,
    baseline_labels: np.ndarray,
    label_items: dict[int, dict[str, Any]],
    ontology_kinds: dict[str, str],
    target_class: str,
    min_breadcrumb_fraction: float,
    max_breadcrumb_fraction: float,
    max_competing_thing_fraction: float,
    min_recoverable_background_fraction: float,
) -> SeedPolicyDecision:
    """Evaluate semantic gates without changing any labels."""
    if seed_indices.ndim != 1 or seed_indices.size == 0:
        raise ValueError("seed_indices must be a non-empty one-dimensional array")
    local_labels = baseline_labels[seed_indices]
    classes = label_class_by_id(label_items)
    kinds = label_kind_by_id(label_items, ontology_kinds)
    target_class = normalize_class_name(target_class)
    target_ids = {
        label_id for label_id, class_name in classes.items() if class_name == target_class
    }
    breadcrumb_local = np.isin(local_labels, list(target_ids))
    breadcrumb_count = int(np.count_nonzero(breadcrumb_local))
    breadcrumb_fraction = breadcrumb_count / float(seed_indices.size)

    target_counts = Counter(int(value) for value in local_labels[breadcrumb_local])
    target_label_id = target_counts.most_common(1)[0][0] if target_counts else None

    competing_thing_local = np.asarray(
        [
            label_id != 0
            and label_id not in target_ids
            and kinds.get(int(label_id), "thing") == "thing"
            for label_id in local_labels
        ],
        dtype=bool,
    )
    competing_thing_count = int(np.count_nonzero(competing_thing_local))
    competing_thing_fraction = competing_thing_count / float(seed_indices.size)

    stuff_counts = Counter(
        int(value)
        for value in local_labels
        if int(value) != 0 and kinds.get(int(value), "thing") == "stuff"
    )
    dominant_stuff_label_id = stuff_counts.most_common(1)[0][0] if stuff_counts else None
    patchable_local = local_labels == 0
    if dominant_stuff_label_id is not None:
        patchable_local |= local_labels == dominant_stuff_label_id
    patchable_local &= ~breadcrumb_local
    recoverable_background_count = int(np.count_nonzero(patchable_local))
    recoverable_background_fraction = recoverable_background_count / float(seed_indices.size)

    reasons: list[str] = []
    if target_label_id is None:
        reasons.append("same_class_breadcrumb_missing")
    if breadcrumb_fraction < min_breadcrumb_fraction:
        reasons.append(f"breadcrumb_fraction<{min_breadcrumb_fraction}")
    if breadcrumb_fraction > max_breadcrumb_fraction:
        reasons.append(f"breadcrumb_fraction>{max_breadcrumb_fraction}")
    if competing_thing_fraction > max_competing_thing_fraction:
        reasons.append(
            f"competing_thing_fraction>{max_competing_thing_fraction}"
        )
    if recoverable_background_fraction < min_recoverable_background_fraction:
        reasons.append(
            "recoverable_background_fraction<"
            f"{min_recoverable_background_fraction}"
        )

    return SeedPolicyDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        target_label_id=target_label_id,
        breadcrumb_local=breadcrumb_local,
        patchable_local=patchable_local,
        metrics={
            "breadcrumb_gaussian_count": breadcrumb_count,
            "breadcrumb_fraction": breadcrumb_fraction,
            "target_label_id": target_label_id,
            "competing_thing_gaussian_count": competing_thing_count,
            "competing_thing_fraction": competing_thing_fraction,
            "dominant_background_label_id": dominant_stuff_label_id,
            "dominant_background_class": (
                classes.get(dominant_stuff_label_id, "")
                if dominant_stuff_label_id is not None
                else "unlabeled"
            ),
            "recoverable_background_gaussian_count": recoverable_background_count,
            "recoverable_background_fraction": recoverable_background_fraction,
        },
    )


def confirmed_patch_indices(
    seed_indices: np.ndarray,
    decision: SeedPolicyDecision,
    prompt: ProjectionPrompt,
    mask_membership: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Intersect the original seed, alternate mask, and patchable v5 support."""
    if mask_membership.shape != prompt.projected_seed_positions.shape:
        raise ValueError("Projected mask membership does not match projected seed points")
    local_positions = prompt.projected_seed_positions
    projected_breadcrumb = decision.breadcrumb_local[local_positions]
    breadcrumb_projected_count = int(np.count_nonzero(projected_breadcrumb))
    breadcrumb_confirmed_count = int(
        np.count_nonzero(mask_membership & projected_breadcrumb)
    )
    breadcrumb_coverage = breadcrumb_confirmed_count / float(
        max(breadcrumb_projected_count, 1)
    )
    confirmed_local_positions = local_positions[
        mask_membership & decision.patchable_local[local_positions]
    ]
    confirmed = np.unique(seed_indices[confirmed_local_positions]).astype(
        np.uint32, copy=False
    )
    return confirmed, {
        "projected_breadcrumb_gaussian_count": breadcrumb_projected_count,
        "confirmed_breadcrumb_gaussian_count": breadcrumb_confirmed_count,
        "breadcrumb_projection_coverage": breadcrumb_coverage,
        "confirmed_patch_gaussian_count": int(confirmed.size),
    }


def apply_point_patches(
    baseline_labels: np.ndarray,
    patches: list[PointPatch],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, int]]:
    """Apply unambiguous point patches and leave all other labels bit-identical."""
    merged = baseline_labels.copy()
    changes = np.zeros(baseline_labels.shape, dtype=np.int32)
    if not patches:
        return merged, changes, {
            "ambiguous_gaussian_count": 0,
            "changed_gaussian_count": 0,
            "unchanged_outside_patch": True,
        }, {}

    all_indices = np.concatenate([patch.indices.astype(np.uint32) for patch in patches])
    all_targets = np.concatenate(
        [
            np.full(patch.indices.shape, patch.target_label_id, dtype=np.int32)
            for patch in patches
        ]
    )
    order = np.argsort(all_indices, kind="stable")
    sorted_indices = all_indices[order]
    sorted_targets = all_targets[order]
    unique_indices, starts = np.unique(sorted_indices, return_index=True)
    ends = np.r_[starts[1:], sorted_indices.size]
    ambiguous = np.asarray(
        [
            index
            for index, start, end in zip(unique_indices, starts, ends)
            if np.unique(sorted_targets[start:end]).size > 1
        ],
        dtype=np.uint32,
    )

    per_seed: dict[str, int] = {}
    changed_union: list[np.ndarray] = []
    for patch in patches:
        indices = np.setdiff1d(patch.indices, ambiguous, assume_unique=False)
        if indices.size:
            merged[indices] = patch.target_label_id
            changes[indices] = patch.target_label_id
            changed_union.append(indices)
        per_seed[patch.seed_key] = int(indices.size)

    changed_indices = (
        np.unique(np.concatenate(changed_union))
        if changed_union
        else np.zeros((0,), dtype=np.uint32)
    )
    outside = np.ones(baseline_labels.shape, dtype=bool)
    outside[changed_indices] = False
    if not np.array_equal(merged[outside], baseline_labels[outside]):
        raise AssertionError("Baseline labels changed outside confirmed singleton points")
    actual_changed = merged != baseline_labels
    if not np.array_equal(actual_changed, changes != 0):
        raise AssertionError("Recovery change mask does not match changed labels")
    return merged, changes, {
        "ambiguous_gaussian_count": int(ambiguous.size),
        "changed_gaussian_count": int(np.count_nonzero(actual_changed)),
        "unchanged_outside_patch": True,
    }, per_seed


def update_label_map_counts(
    label_map: dict[str, Any],
    labels: np.ndarray,
    source: str,
    recovery_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(label_map)
    counts = histogram(labels)
    for item in result.get("labels", []):
        item["gaussian_count"] = counts.get(int(item["id"]), 0)
    existing_ids = {int(item["id"]) for item in result.get("labels", [])}
    for recovery_item in recovery_items or []:
        label_id = int(recovery_item["id"])
        count = counts.get(label_id, 0)
        if count <= 0:
            continue
        if label_id in existing_ids:
            raise ValueError(f"Duplicate recovery label id {label_id}")
        item = copy.deepcopy(recovery_item)
        item["gaussian_count"] = count
        result.setdefault("labels", []).append(item)
        existing_ids.add(label_id)
    result["labels"] = sorted(
        result.get("labels", []), key=lambda item: int(item["id"])
    )
    result["source"] = source
    result["post_v5_singleton_point_recovery"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--ground-output-dir", required=True, type=Path)
    parser.add_argument("--source-view-manifest", type=Path)
    parser.add_argument("--proposal-dir", required=True, type=Path)
    parser.add_argument("--class-config", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--baseline-labels", required=True, type=Path)
    parser.add_argument("--baseline-label-map", required=True, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--segment-anything-root", required=True, type=Path)
    parser.add_argument("--sam-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--semantic-ply-path", type=Path)
    parser.add_argument("--no-semantic-ply", action="store_true")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--input-manifest-name", default="grounded_sam_manifest.json")
    parser.add_argument("--sam-arch", default="vit_h")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-proposal-gaussians", default=500, type=int)
    parser.add_argument("--min-seed-gaussians", default=1500, type=int)
    parser.add_argument("--min-seed-sam-score", default=0.90, type=float)
    parser.add_argument("--min-seed-mask-area-ratio", default=0.02, type=float)
    parser.add_argument("--min-outer-area-ratio", default=1.50, type=float)
    parser.add_argument("--max-nested-thing-containment", default=0.80, type=float)
    parser.add_argument("--merge-iou", default=0.35, type=float)
    parser.add_argument("--containment-threshold", default=0.70, type=float)
    parser.add_argument("--min-breadcrumb-fraction", default=0.10, type=float)
    parser.add_argument("--max-breadcrumb-fraction", default=0.50, type=float)
    parser.add_argument("--max-competing-thing-fraction", default=0.05, type=float)
    parser.add_argument(
        "--min-recoverable-background-fraction", default=0.50, type=float
    )
    parser.add_argument("--min-projected-gaussians", default=250, type=int)
    parser.add_argument("--min-projected-fraction", default=0.05, type=float)
    parser.add_argument("--max-baseline-depth-ratio", default=0.75, type=float)
    parser.add_argument("--box-padding-ratio", default=0.05, type=float)
    parser.add_argument("--min-box-area-ratio", default=0.005, type=float)
    parser.add_argument("--max-box-area-ratio", default=0.80, type=float)
    parser.add_argument("--max-target-views", default=4, type=int)
    parser.add_argument("--min-verification-sam-score", default=0.90, type=float)
    parser.add_argument("--min-projection-coverage", default=0.60, type=float)
    parser.add_argument("--min-projected-breadcrumb-gaussians", default=25, type=int)
    parser.add_argument("--min-breadcrumb-projection-coverage", default=0.50, type=float)
    parser.add_argument("--min-confirmed-patch-gaussians", default=250, type=int)
    parser.add_argument("--min-verification-mask-area", default=100, type=int)
    parser.add_argument("--max-verification-mask-area-ratio", default=0.80, type=float)
    parser.add_argument("--max-seeds", default=24, type=int)
    parser.add_argument("--tier2-min-seed-sam-score", default=0.95, type=float)
    parser.add_argument(
        "--tier2-min-seed-mask-area-ratio", default=0.02, type=float
    )
    parser.add_argument(
        "--tier2-max-competing-thing-fraction", default=0.01, type=float
    )
    parser.add_argument(
        "--tier2-min-recoverable-background-fraction", default=0.90, type=float
    )
    parser.add_argument("--tier2-max-anchor-size-ratio", default=3.0, type=float)
    parser.add_argument("--tier2-max-anchor-shape-ratio", default=2.5, type=float)
    parser.add_argument("--tier2-max-target-views", default=8, type=int)
    parser.add_argument("--tier2-min-verification-views", default=2, type=int)
    parser.add_argument(
        "--tier2-min-projection-coverage", default=0.70, type=float
    )
    parser.add_argument(
        "--tier2-min-intersection-gaussians", default=250, type=int
    )
    parser.add_argument(
        "--tier2-min-intersection-fraction", default=0.50, type=float
    )
    parser.add_argument("--tier2-max-seeds", default=16, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.min_breadcrumb_fraction <= args.max_breadcrumb_fraction <= 1.0:
        raise ValueError("Breadcrumb fractions must satisfy 0 <= min <= max <= 1")
    if not 0.0 <= args.max_competing_thing_fraction <= 1.0:
        raise ValueError("max-competing-thing-fraction must be between 0 and 1")
    if args.max_target_views < 1:
        raise ValueError("max-target-views must be at least 1")
    if args.tier2_max_target_views < args.tier2_min_verification_views:
        raise ValueError(
            "tier2-max-target-views must be at least tier2-min-verification-views"
        )
    if args.tier2_min_verification_views < 2:
        raise ValueError("tier2-min-verification-views must be at least 2")
    if not 0.0 <= args.tier2_max_competing_thing_fraction <= 1.0:
        raise ValueError(
            "tier2-max-competing-thing-fraction must be between 0 and 1"
        )
    if not 0.0 <= args.tier2_min_recoverable_background_fraction <= 1.0:
        raise ValueError(
            "tier2-min-recoverable-background-fraction must be between 0 and 1"
        )
    if not 0.0 <= args.tier2_min_intersection_fraction <= 1.0:
        raise ValueError("tier2-min-intersection-fraction must be between 0 and 1")
    if args.no_semantic_ply and args.semantic_ply_path is not None:
        raise ValueError("--no-semantic-ply and --semantic-ply-path are mutually exclusive")

    semantic_ply = resolve_semantic_ply_output(
        args.output_dir,
        semantic_ply_path=args.semantic_ply_path,
        disabled=args.no_semantic_ply,
    )
    paths = {
        "labels": args.output_dir / "gaussian_labels.npy",
        "changes": args.output_dir / "singleton_recovery_changes.npy",
        "label_map": args.output_dir / "label_map.json",
        "summary": args.output_dir / "singleton_point_recovery_summary.json",
    }
    output_files = [*paths.values(), *([semantic_ply] if semantic_ply else [])]
    existing = [path for path in output_files if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Singleton recovery outputs exist; pass --overwrite: {existing}"
        )

    baseline_map = load_json(args.baseline_label_map)
    if str(baseline_map.get("scene", "")) != args.scene:
        raise ValueError("Baseline label-map scene does not match the requested scene")
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

    input_manifest_path = args.ground_output_dir / args.input_manifest_name
    proposal_manifest_path = args.proposal_dir / "proposal_manifest.json"
    support_dir = args.proposal_dir / "proposal_supports"
    source_manifest = load_json(input_manifest_path)
    grounded_frames = list(source_manifest.get("frames", []))
    grounded_frames_by_file = {
        str(frame["file"]): frame for frame in grounded_frames
    }
    verification_manifest_path, verification_manifest = (
        resolve_verification_view_manifest(
            input_manifest_path,
            source_manifest,
            args.source_view_manifest,
        )
    )
    verification_frames = list(verification_manifest.get("frames", []))
    if not verification_frames:
        raise ValueError(
            f"No verification frames found in {verification_manifest_path}"
        )
    rgb_dir = rgb_directory(verification_manifest_path, verification_manifest)
    cameras = load_cameras(args.model_path)
    stuff_classes = load_stuff_classes(args.class_config, "")
    thing_classes = thing_classes_from_config(args.class_config)
    proposals = load_proposals(
        proposal_manifest_path,
        support_dir,
        args.min_proposal_gaussians,
        0,
        True,
    )
    candidates = singleton_proposals(
        proposals,
        stuff_classes,
        args.merge_iou,
        args.containment_threshold,
    )

    reports: list[dict[str, Any]] = []
    tier1_contexts: dict[
        str,
        tuple[Any, SeedPolicyDecision, list[ProjectionPrompt], dict[str, Any]],
    ] = {}
    tier2_pool: dict[
        str,
        tuple[Any, SeedPolicyDecision, dict[str, Any], dict[str, Any]],
    ] = {}
    tier1_prompts: list[ProjectionPrompt] = []
    tier1_prompted_count = 0
    for proposal in candidates:
        frame_file = str(proposal.metadata.get("frame_file", ""))
        mask_index = int(proposal.metadata.get("mask_index", -1))
        seed_key = f"{frame_file}:{mask_index}"
        report: dict[str, Any] = {
            "seed_key": seed_key,
            "proposal_id": proposal.proposal_id,
            "class": proposal.class_name,
            "frame_file": frame_file,
            "mask_index": mask_index,
            "gaussian_count": proposal.gaussian_count,
            "sam_score": proposal_sam_score(proposal),
            "grounding_score": proposal_grounding_score(proposal),
            "status": "rejected",
            "reasons": [],
        }
        reports.append(report)
        source_frame = grounded_frames_by_file.get(frame_file)
        if source_frame is None:
            report["reasons"].append("source_frame_missing")
            continue
        source_masks = load_mask_stack(
            frame_mask_path(input_manifest_path, args.ground_output_dir, source_frame)
        )
        if mask_index < 0 or mask_index >= source_masks.shape[0]:
            report["reasons"].append("source_mask_missing")
            continue
        seed_mask = source_masks[mask_index]
        mask_area_ratio = int(seed_mask.sum()) / float(max(seed_mask.size, 1))
        nested = nested_competing_thing(
            seed_mask,
            proposal.class_name,
            source_masks,
            list(source_frame.get("masks", [])),
            thing_classes,
            args.min_outer_area_ratio,
        )
        report["mask_area_ratio"] = mask_area_ratio
        report["nested_competing_thing"] = nested
        source_reasons: list[str] = []
        if proposal.gaussian_count < args.min_seed_gaussians:
            source_reasons.append(f"gaussian_count<{args.min_seed_gaussians}")
        if proposal_sam_score(proposal) < args.min_seed_sam_score:
            source_reasons.append(f"sam_score<{args.min_seed_sam_score}")
        if mask_area_ratio < args.min_seed_mask_area_ratio:
            source_reasons.append(
                f"mask_area_ratio<{args.min_seed_mask_area_ratio}"
            )
        if float(nested["containment"]) > args.max_nested_thing_containment:
            source_reasons.append(
                "nested_competing_thing_containment>"
                f"{args.max_nested_thing_containment}"
            )

        decision = evaluate_seed_policy(
            proposal.indices,
            baseline_labels,
            label_items,
            ontology_kinds,
            proposal.class_name,
            args.min_breadcrumb_fraction,
            args.max_breadcrumb_fraction,
            args.max_competing_thing_fraction,
            args.min_recoverable_background_fraction,
        )
        report["baseline_policy"] = decision.metrics
        tier1_reasons = [*source_reasons, *decision.reasons]
        report["reasons"] = list(tier1_reasons)
        report["tier1"] = {
            "status": "eligible" if not tier1_reasons else "rejected",
            "reasons": list(tier1_reasons),
        }

        tier2_source_reasons: list[str] = []
        if proposal.gaussian_count < args.min_seed_gaussians:
            tier2_source_reasons.append(
                f"gaussian_count<{args.min_seed_gaussians}"
            )
        if proposal_sam_score(proposal) < args.tier2_min_seed_sam_score:
            tier2_source_reasons.append(
                f"sam_score<{args.tier2_min_seed_sam_score}"
            )
        if mask_area_ratio < args.tier2_min_seed_mask_area_ratio:
            tier2_source_reasons.append(
                "mask_area_ratio<"
                f"{args.tier2_min_seed_mask_area_ratio}"
            )
        if float(nested["containment"]) > args.max_nested_thing_containment:
            tier2_source_reasons.append(
                "nested_competing_thing_containment>"
                f"{args.max_nested_thing_containment}"
            )
        tier2_policy_reasons = evaluate_anchor_candidate_policy(
            decision,
            args.min_breadcrumb_fraction,
            args.max_breadcrumb_fraction,
            args.tier2_max_competing_thing_fraction,
            args.tier2_min_recoverable_background_fraction,
        )
        tier2_precheck_reasons = [
            *tier2_source_reasons,
            *tier2_policy_reasons,
        ]
        report["tier2_precheck"] = {
            "status": "eligible" if not tier2_precheck_reasons else "rejected",
            "reasons": list(tier2_precheck_reasons),
        }
        if not tier2_precheck_reasons:
            tier2_pool[seed_key] = (proposal, decision, source_frame, report)

        if tier1_reasons:
            continue
        if tier1_prompted_count >= args.max_seeds:
            report["reasons"].append(f"max_seeds>={args.max_seeds}")
            report["tier1"]["status"] = "rejected"
            report["tier1"]["reasons"] = list(report["reasons"])
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
            args.max_target_views,
        )
        report["candidate_views"] = [
            {
                "frame_file": prompt.target_frame_file,
                "camera_index": prompt.target_camera_index,
                "projected_gaussian_count": prompt.projected_gaussian_count,
                "projected_fraction": prompt.projected_fraction,
                "baseline_depth_ratio": prompt.baseline_depth_ratio,
                "bbox_xyxy": list(prompt.bbox_xyxy),
            }
            for prompt in prompts
        ]
        if not prompts:
            report["reasons"].append("no_usable_alternate_view")
            report["tier1"]["status"] = "rejected"
            report["tier1"]["reasons"] = list(report["reasons"])
            continue
        report["status"] = "prompted"
        report["tier1"]["status"] = "prompted"
        tier1_prompted_count += 1
        tier1_contexts[seed_key] = (proposal, decision, prompts, report)
        tier1_prompts.extend(prompts)

    sam_predictor: Any | None = None

    def run_verification_prompts(
        prompts: list[ProjectionPrompt],
        contexts: dict[
            str,
            tuple[Any, SeedPolicyDecision, list[ProjectionPrompt], dict[str, Any]],
        ],
        min_projection_coverage: float,
        require_breadcrumb: bool,
    ) -> dict[tuple[str, str], tuple[dict[str, Any], np.ndarray]]:
        nonlocal sam_predictor
        results: dict[
            tuple[str, str], tuple[dict[str, Any], np.ndarray]
        ] = {}
        if not prompts:
            return results
        import torch

        from generate_grounded_sam_masks import load_sam_predictor, run_sam_for_boxes

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for singleton point recovery")
        if sam_predictor is None:
            sam_predictor = load_sam_predictor(
                args.segment_anything_root,
                args.sam_checkpoint,
                args.sam_arch,
                args.device,
            )
        prompts_by_frame: dict[str, list[ProjectionPrompt]] = defaultdict(list)
        for prompt in prompts:
            prompts_by_frame[prompt.target_frame_file].append(prompt)
        for frame_file, frame_prompts in prompts_by_frame.items():
            rgb = np.asarray(
                Image.open(rgb_dir / frame_file).convert("RGB"), dtype=np.uint8
            )
            boxes = torch.tensor(
                [prompt.bbox_xyxy for prompt in frame_prompts],
                dtype=torch.float32,
                device=args.device,
            )
            masks, sam_scores = run_sam_for_boxes(
                sam_predictor, rgb, boxes, args.device
            )
            for index, prompt in enumerate(frame_prompts):
                proposal, decision, _prompts, _report = contexts[prompt.seed_key]
                mask = masks[index]
                sam_score = (
                    float(sam_scores[index]) if index < len(sam_scores) else 0.0
                )
                membership = projected_mask_membership(mask, prompt.projected_xy)
                projection_coverage = (
                    float(membership.mean()) if membership.size else 0.0
                )
                confirmed, detail = confirmed_patch_indices(
                    proposal.indices, decision, prompt, membership
                )
                area = int(mask.sum())
                area_ratio = area / float(max(mask.size, 1))
                reasons: list[str] = []
                if sam_score < args.min_verification_sam_score:
                    reasons.append(
                        f"sam_score<{args.min_verification_sam_score}"
                    )
                if area < args.min_verification_mask_area:
                    reasons.append(f"mask_area<{args.min_verification_mask_area}")
                if area_ratio > args.max_verification_mask_area_ratio:
                    reasons.append(
                        "mask_area_ratio>"
                        f"{args.max_verification_mask_area_ratio}"
                    )
                if projection_coverage < min_projection_coverage:
                    reasons.append(
                        f"projection_coverage<{min_projection_coverage}"
                    )
                if require_breadcrumb:
                    if (
                        detail["projected_breadcrumb_gaussian_count"]
                        < args.min_projected_breadcrumb_gaussians
                    ):
                        reasons.append(
                            "projected_breadcrumb_gaussian_count<"
                            f"{args.min_projected_breadcrumb_gaussians}"
                        )
                    if (
                        detail["breadcrumb_projection_coverage"]
                        < args.min_breadcrumb_projection_coverage
                    ):
                        reasons.append(
                            "breadcrumb_projection_coverage<"
                            f"{args.min_breadcrumb_projection_coverage}"
                        )
                if confirmed.size < args.min_confirmed_patch_gaussians:
                    reasons.append(
                        "confirmed_patch_gaussian_count<"
                        f"{args.min_confirmed_patch_gaussians}"
                    )
                results[(prompt.seed_key, frame_file)] = (
                    {
                        "frame_file": frame_file,
                        "camera_index": prompt.target_camera_index,
                        "sam_score": sam_score,
                        "mask_area": area,
                        "mask_area_ratio": area_ratio,
                        "projection_coverage": projection_coverage,
                        **detail,
                        "status": "accepted" if not reasons else "rejected",
                        "reasons": reasons,
                    },
                    confirmed,
                )
        return results

    tier1_results = run_verification_prompts(
        tier1_prompts,
        tier1_contexts,
        args.min_projection_coverage,
        True,
    )
    tier1_patches: list[PointPatch] = []
    for seed_key, (proposal, decision, prompts, report) in tier1_contexts.items():
        report["alternate_view_checks"] = []
        selected: tuple[dict[str, Any], np.ndarray] | None = None
        # Candidate views are geometry-ranked. Use the first one that passes;
        # never union confirmations from multiple alternate views.
        for prompt in prompts:
            result = tier1_results[(seed_key, prompt.target_frame_file)]
            report["alternate_view_checks"].append(result[0])
            if selected is None and result[0]["status"] == "accepted":
                selected = result
        if selected is None:
            report["status"] = "rejected"
            report["reasons"].append("no_alternate_view_passed")
            report["tier1"]["status"] = "rejected"
            report["tier1"]["reasons"] = list(report["reasons"])
            continue
        if decision.target_label_id is None:
            raise AssertionError("Accepted seed has no target label id")
        report["status"] = "confirmed"
        report["recovery_tier"] = 1
        report["tier1"]["status"] = "confirmed"
        report["selected_alternate_view"] = selected[0]["frame_file"]
        report["preliminary_patch_gaussian_count"] = int(selected[1].size)
        tier1_patches.append(
            PointPatch(
                seed_key=seed_key,
                target_label_id=decision.target_label_id,
                indices=selected[1],
            )
        )

    # Establish anchors from unambiguous Tier-1 patches only. This temporary
    # application does not modify the v5 array and prevents a conflicted Tier-1
    # seed from validating a weaker candidate.
    _tier1_labels, _tier1_changes, _tier1_report, tier1_seed_counts = (
        apply_point_patches(baseline_labels, tier1_patches)
    )
    anchors_by_class: dict[str, list[AcceptedAnchor]] = defaultdict(list)
    for patch in tier1_patches:
        if tier1_seed_counts.get(patch.seed_key, 0) <= 0:
            continue
        proposal, decision, _prompts, report = tier1_contexts[patch.seed_key]
        if decision.target_label_id is None:
            raise AssertionError("Tier-1 anchor has no target label id")
        class_name = normalize_class_name(proposal.class_name)
        anchors_by_class[class_name].append(
            AcceptedAnchor(
                seed_key=patch.seed_key,
                class_name=class_name,
                target_label_id=decision.target_label_id,
                signature=robust_geometry_signature(xyz[proposal.indices]),
            )
        )
        report["acts_as_tier2_anchor"] = True

    tier2_contexts: dict[
        str,
        tuple[Any, SeedPolicyDecision, list[ProjectionPrompt], dict[str, Any]],
    ] = {}
    tier2_prompts: list[ProjectionPrompt] = []
    tier2_prompted_count = 0
    for seed_key, (proposal, decision, source_frame, report) in tier2_pool.items():
        tier2_report: dict[str, Any] = {
            "status": "rejected",
            "reasons": [],
        }
        report["tier2"] = tier2_report
        class_name = normalize_class_name(proposal.class_name)
        anchors = anchors_by_class.get(class_name, [])
        if not anchors:
            tier2_report["reasons"].append("no_same_class_tier1_anchor")
            continue

        candidate_signature = robust_geometry_signature(xyz[proposal.indices])
        anchor_checks: list[dict[str, Any]] = []
        compatible: list[tuple[float, AcceptedAnchor, dict[str, float]]] = []
        for anchor in anchors:
            accepted, metrics = geometry_compatibility(
                candidate_signature,
                anchor.signature,
                args.tier2_max_anchor_size_ratio,
                args.tier2_max_anchor_shape_ratio,
            )
            anchor_checks.append(
                {
                    "anchor_seed_key": anchor.seed_key,
                    **metrics,
                    "status": "accepted" if accepted else "rejected",
                }
            )
            if accepted:
                score = max(
                    metrics["size_ratio"] / args.tier2_max_anchor_size_ratio,
                    metrics["shape_ratio"] / args.tier2_max_anchor_shape_ratio,
                )
                compatible.append((score, anchor, metrics))
        tier2_report["anchor_checks"] = anchor_checks
        if not compatible:
            tier2_report["reasons"].append("no_geometry_compatible_anchor")
            continue
        _score, anchor, match_metrics = min(compatible, key=lambda item: item[0])
        tier2_report["selected_anchor_seed_key"] = anchor.seed_key
        tier2_report["selected_anchor_geometry"] = match_metrics

        if tier2_prompted_count >= args.tier2_max_seeds:
            tier2_report["reasons"].append(
                f"tier2_max_seeds>={args.tier2_max_seeds}"
            )
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
            args.tier2_max_target_views,
        )
        tier2_report["candidate_views"] = [
            {
                "frame_file": prompt.target_frame_file,
                "camera_index": prompt.target_camera_index,
                "projected_gaussian_count": prompt.projected_gaussian_count,
                "projected_fraction": prompt.projected_fraction,
                "baseline_depth_ratio": prompt.baseline_depth_ratio,
                "bbox_xyxy": list(prompt.bbox_xyxy),
            }
            for prompt in prompts
        ]
        if len(prompts) < args.tier2_min_verification_views:
            tier2_report["reasons"].append(
                "candidate_view_count<"
                f"{args.tier2_min_verification_views}"
            )
            continue
        anchored_decision = decision_with_anchor_target(
            decision,
            anchor.target_label_id,
        )
        tier2_report["status"] = "prompted"
        report["status"] = "prompted_tier2"
        tier2_prompted_count += 1
        tier2_contexts[seed_key] = (
            proposal,
            anchored_decision,
            prompts,
            report,
        )
        tier2_prompts.extend(prompts)

    tier2_results = run_verification_prompts(
        tier2_prompts,
        tier2_contexts,
        args.tier2_min_projection_coverage,
        False,
    )
    tier2_patches: list[PointPatch] = []
    recovery_label_items: list[dict[str, Any]] = []
    next_label_id = max(label_items, default=0) + 1
    for seed_key, (proposal, decision, prompts, report) in tier2_contexts.items():
        tier2_report = report["tier2"]
        tier2_report["alternate_view_checks"] = []
        accepted_results: list[tuple[dict[str, Any], np.ndarray]] = []
        for prompt in prompts:
            result = tier2_results[(seed_key, prompt.target_frame_file)]
            tier2_report["alternate_view_checks"].append(result[0])
            if result[0]["status"] == "accepted":
                accepted_results.append(result)
        if len(accepted_results) < args.tier2_min_verification_views:
            report["status"] = "rejected"
            tier2_report["status"] = "rejected"
            tier2_report["reasons"].append(
                "accepted_verification_view_count<"
                f"{args.tier2_min_verification_views}"
            )
            continue

        selected_results = accepted_results[: args.tier2_min_verification_views]
        intersection = intersect_confirmed_views(
            [result[1] for result in selected_results],
            args.tier2_min_verification_views,
        )
        patchable_count = int(np.count_nonzero(decision.patchable_local))
        intersection_fraction = intersection.size / float(max(patchable_count, 1))
        tier2_report["selected_alternate_views"] = [
            result[0]["frame_file"] for result in selected_results
        ]
        tier2_report["intersection_gaussian_count"] = int(intersection.size)
        tier2_report["intersection_patchable_fraction"] = intersection_fraction
        if intersection.size < args.tier2_min_intersection_gaussians:
            tier2_report["reasons"].append(
                "intersection_gaussian_count<"
                f"{args.tier2_min_intersection_gaussians}"
            )
        if intersection_fraction < args.tier2_min_intersection_fraction:
            tier2_report["reasons"].append(
                "intersection_patchable_fraction<"
                f"{args.tier2_min_intersection_fraction}"
            )
        if tier2_report["reasons"]:
            report["status"] = "rejected"
            tier2_report["status"] = "rejected"
            continue
        if decision.target_label_id is None:
            raise AssertionError("Tier-2 candidate has no resolved target label id")
        target_label_id = int(decision.target_label_id)
        if bool(decision.metrics.get("requires_fresh_instance_label", False)):
            anchor_label_id = int(decision.metrics["anchor_target_label_id"])
            target_label_id = next_label_id
            next_label_id += 1
            item = fresh_recovery_label_item(
                label_items,
                anchor_label_id,
                target_label_id,
                seed_key,
            )
            label_items[target_label_id] = item
            recovery_label_items.append(item)
            tier2_report["allocated_fresh_instance_label"] = True
        else:
            tier2_report["allocated_fresh_instance_label"] = False
        tier2_report["output_label_id"] = target_label_id
        report["tier1_reasons"] = list(report["reasons"])
        report["reasons"] = []
        report["status"] = "confirmed"
        report["recovery_tier"] = 2
        report["preliminary_patch_gaussian_count"] = int(intersection.size)
        tier2_report["status"] = "confirmed"
        tier2_patches.append(
            PointPatch(
                seed_key=seed_key,
                target_label_id=target_label_id,
                indices=intersection,
            )
        )

    preliminary_patches = [*tier1_patches, *tier2_patches]

    merged, changes, patch_report, per_seed_counts = apply_point_patches(
        baseline_labels, preliminary_patches
    )
    for report in reports:
        if report["status"] == "confirmed":
            final_count = per_seed_counts.get(report["seed_key"], 0)
            report["final_patch_gaussian_count"] = final_count
            report["status"] = "applied" if final_count else "rejected"
            tier = int(report.get("recovery_tier", 0))
            if tier == 1:
                report["tier1"]["status"] = (
                    "applied" if final_count else "rejected"
                )
            elif tier == 2:
                report["tier2"]["status"] = (
                    "applied" if final_count else "rejected"
                )
            if not final_count:
                report["reasons"].append("all_points_ambiguous")
                if tier == 1:
                    report["tier1"]["reasons"].append("all_points_ambiguous")
                elif tier == 2:
                    report["tier2"]["reasons"].append("all_points_ambiguous")

    changed_mask = merged != baseline_labels
    accepted_seed_support = np.unique(
        np.concatenate([patch.indices for patch in preliminary_patches])
    ) if preliminary_patches else np.zeros((0,), dtype=np.uint32)
    changed_indices = np.flatnonzero(changed_mask).astype(np.uint32)
    if np.setdiff1d(changed_indices, accepted_seed_support).size:
        raise AssertionError("Recovery expanded beyond confirmed original seed support")

    output_map = update_label_map_counts(
        baseline_map,
        merged,
        "dinov2_groundingdino_adaptive_v5_with_two_tier_singleton_intersection",
        recovery_label_items,
    )
    summary = {
        "scene": args.scene,
        "status": "ok" if np.any(changed_mask) else "no_recovery_changes",
        "gaussian_count": int(merged.size),
        "parameters": {
            "min_seed_gaussians": args.min_seed_gaussians,
            "min_seed_sam_score": args.min_seed_sam_score,
            "min_seed_mask_area_ratio": args.min_seed_mask_area_ratio,
            "max_nested_thing_containment": args.max_nested_thing_containment,
            "min_breadcrumb_fraction": args.min_breadcrumb_fraction,
            "max_breadcrumb_fraction": args.max_breadcrumb_fraction,
            "max_competing_thing_fraction": args.max_competing_thing_fraction,
            "min_recoverable_background_fraction": (
                args.min_recoverable_background_fraction
            ),
            "max_target_views": args.max_target_views,
            "grounded_source_frame_count": len(grounded_frames),
            "verification_frame_count": len(verification_frames),
            "verification_uses_full_source_manifest": (
                verification_manifest_path != input_manifest_path
            ),
            "min_verification_sam_score": args.min_verification_sam_score,
            "min_projection_coverage": args.min_projection_coverage,
            "min_projected_breadcrumb_gaussians": (
                args.min_projected_breadcrumb_gaussians
            ),
            "min_breadcrumb_projection_coverage": (
                args.min_breadcrumb_projection_coverage
            ),
            "min_confirmed_patch_gaussians": args.min_confirmed_patch_gaussians,
            "tier2_min_seed_sam_score": args.tier2_min_seed_sam_score,
            "tier2_min_seed_mask_area_ratio": (
                args.tier2_min_seed_mask_area_ratio
            ),
            "tier2_max_competing_thing_fraction": (
                args.tier2_max_competing_thing_fraction
            ),
            "tier2_min_recoverable_background_fraction": (
                args.tier2_min_recoverable_background_fraction
            ),
            "tier2_max_anchor_size_ratio": args.tier2_max_anchor_size_ratio,
            "tier2_max_anchor_shape_ratio": args.tier2_max_anchor_shape_ratio,
            "tier2_max_target_views": args.tier2_max_target_views,
            "tier2_min_verification_views": args.tier2_min_verification_views,
            "tier2_min_projection_coverage": (
                args.tier2_min_projection_coverage
            ),
            "tier2_min_intersection_gaussians": (
                args.tier2_min_intersection_gaussians
            ),
            "tier2_min_intersection_fraction": (
                args.tier2_min_intersection_fraction
            ),
            "policy": (
                "tier1_low_partial_breadcrumb_intersected_with_one_alternate_"
                "sam_view; tier2_low_or_zero_breadcrumb_requires_unambiguous_"
                "tier1_same_class_anchor_geometry_match_and_two_alternate_sam_"
                "view_intersection; patch_only_unlabeled_or_dominant_stuff; "
                "protect_all_other_labels; no_union; no_anchor_cascade; no_"
                "global_reclustering"
            ),
        },
        "singleton_candidate_count": len(candidates),
        "prompted_seed_count": tier1_prompted_count + tier2_prompted_count,
        "tier1_prompted_seed_count": tier1_prompted_count,
        "tier2_prompted_seed_count": tier2_prompted_count,
        "applied_seed_count": sum(report["status"] == "applied" for report in reports),
        "tier1_applied_seed_count": sum(
            report.get("status") == "applied"
            and report.get("recovery_tier") == 1
            for report in reports
        ),
        "tier2_applied_seed_count": sum(
            report.get("status") == "applied"
            and report.get("recovery_tier") == 2
            for report in reports
        ),
        "baseline_transition_counts": transition_records(
            baseline_labels, changed_mask, label_items
        ),
        "patch": patch_report,
        "invariants": {
            "v5_input_modified": False,
            "unchanged_outside_confirmed_patch": True,
            "changed_points_subset_of_original_seed_support": True,
            "alternate_masks_can_add_points": False,
            "tier2_requires_tier1_same_class_anchor": True,
            "tier2_requires_multi_view_intersection": True,
            "tier2_anchors_cannot_cascade": True,
            "global_clustering_rerun": False,
        },
        "seeds": reports,
        "sources": {
            "baseline_labels": str(args.baseline_labels),
            "baseline_label_map": str(args.baseline_label_map),
            "grounded_sam_manifest": str(input_manifest_path),
            "verification_view_manifest": str(verification_manifest_path),
            "proposal_manifest": str(proposal_manifest_path),
            "source_ply": str(args.source_ply),
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
    print(json.dumps({key: value for key, value in summary.items() if key != "seeds"}, indent=2))


if __name__ == "__main__":
    main()
