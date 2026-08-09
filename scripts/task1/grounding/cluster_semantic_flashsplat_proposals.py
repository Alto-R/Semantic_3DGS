#!/usr/bin/env python3
"""Fuse semantic FlashSplat proposals into final per-Gaussian labels."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.common.ply_utils import (
    read_ply_header,
    resolve_semantic_ply_output,
    vertex_data_memmap,
)
from scripts.task1.qa.add_labels_from_npy import write_ply_with_labels


DEFAULT_STUFF_CLASSES = {"building", "ground", "road", "sidewalk", "sky", "vegetation", "terrain"}
VOXEL_NEIGHBOR_OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
]


@dataclass
class SemanticProposal:
    proposal_id: int
    class_name: str
    support_file: str
    gaussian_count: int
    score: float
    metadata: dict[str, Any]
    indices: np.ndarray


@dataclass
class ClassEvidence:
    class_name: str
    indices: np.ndarray
    positive_views: np.ndarray
    negative_views: np.ndarray


@dataclass
class SemanticGroup:
    group_id: int
    class_name: str
    indices: np.ndarray
    proposal_ids: list[int] = field(default_factory=list)
    source_frames: set[str] = field(default_factory=set)
    phrases: Counter[str] = field(default_factory=Counter)
    scores: list[float] = field(default_factory=list)
    is_stuff: bool = False
    assigned_count: int = 0
    assignment_reliability: float = 0.0
    assignment_peak_quality: float = 0.0

    @property
    def gaussian_count(self) -> int:
        return int(self.indices.shape[0])

    @property
    def proposal_count(self) -> int:
        return len(self.proposal_ids)

    @property
    def score(self) -> float:
        return max(self.scores) if self.scores else 0.0

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0


def normalize_class_name(value: Any) -> str:
    class_name = str(value or "").strip().lower().replace(" ", "_")
    return class_name or "unknown"


def point_cloud_path(model_path: Path, iteration: int) -> Path:
    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)
    return ply_path


def proposal_score(item: dict[str, Any]) -> float:
    for key in ["confidence", "grounding_score", "sam_score", "predicted_iou"]:
        if key in item:
            try:
                return float(item[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def load_indices(path: Path) -> np.ndarray:
    with np.load(path) as data:
        indices = data["indices"].astype(np.uint32)
    return np.unique(indices)


def load_stuff_classes(path: Path | None, override: str) -> set[str]:
    stuff = set(DEFAULT_STUFF_CLASSES)
    if path is not None and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        classes = data.get("classes", data) if isinstance(data, dict) else data
        for item in classes:
            if isinstance(item, dict) and str(item.get("type", "thing")).lower() == "stuff":
                stuff.add(normalize_class_name(item.get("class", item.get("name", ""))))
    if override:
        stuff = {normalize_class_name(item) for item in override.split(",") if item.strip()}
    return stuff


def load_assignment_priorities(path: Path | None, override: str) -> dict[str, int]:
    names: list[str] = []
    if override:
        names = [normalize_class_name(item) for item in override.split(",") if item.strip()]
    elif path is not None and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            names = [normalize_class_name(item) for item in data.get("assignment_priority", [])]
    return {class_name: len(names) - index for index, class_name in enumerate(names)}


def load_class_evidence(path: Path | None) -> dict[str, ClassEvidence] | None:
    if path is None:
        return None
    manifest_path = path / "class_evidence_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence: dict[str, ClassEvidence] = {}
    for item in manifest.get("classes", []):
        class_name = normalize_class_name(item.get("class"))
        with np.load(path / str(item["file"])) as data:
            indices = data["indices"].astype(np.uint32)
            positive_views = data["positive_views"].astype(np.uint16)
            negative_views = data["negative_views"].astype(np.uint16)
        evidence[class_name] = ClassEvidence(
            class_name=class_name,
            indices=indices,
            positive_views=positive_views,
            negative_views=negative_views,
        )
    return evidence


def group_order_key(group: SemanticGroup, priorities: dict[str, int]) -> tuple[Any, ...]:
    return (
        0 if group.is_stuff else 1,
        priorities.get(group.class_name, 0),
        group.score,
        group.proposal_count,
        group.gaussian_count,
    )


def load_proposals(
    manifest_path: Path,
    support_dir: Path,
    min_gaussians: int,
    max_gaussians: int,
    require_class: bool,
) -> list[SemanticProposal]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proposals: list[SemanticProposal] = []
    for item in manifest["proposals"]:
        count = int(item["gaussian_count"])
        if count < min_gaussians:
            continue
        if max_gaussians > 0 and count > max_gaussians:
            continue
        class_name = normalize_class_name(item.get("class_name", item.get("class", "")))
        if require_class and class_name in {"", "unknown", "object_candidate"}:
            continue
        proposals.append(
            SemanticProposal(
                proposal_id=int(item["proposal_id"]),
                class_name=class_name,
                support_file=str(item["support_file"]),
                gaussian_count=count,
                score=proposal_score(item),
                metadata=item,
                indices=load_indices(support_dir / str(item["support_file"])),
            )
        )
    proposals.sort(key=lambda proposal: (proposal.score, proposal.gaussian_count), reverse=True)
    return proposals


def proposal_manifest_view_count(manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_files = {
        str(item.get("frame_file", "")).strip()
        for item in manifest.get("proposals", [])
        if str(item.get("frame_file", "")).strip()
    }
    return len(frame_files)


def intersection_count(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.intersect1d(left, right, assume_unique=True).shape[0])


def merge_score(proposal_indices: np.ndarray, group_indices: np.ndarray) -> tuple[float, float, int]:
    inter = intersection_count(proposal_indices, group_indices)
    if inter == 0:
        return 0.0, 0.0, 0
    union = proposal_indices.shape[0] + group_indices.shape[0] - inter
    iou = inter / float(max(union, 1))
    containment = inter / float(max(min(proposal_indices.shape[0], group_indices.shape[0]), 1))
    return iou, containment, inter


def add_proposal_to_group(group: SemanticGroup, proposal: SemanticProposal) -> None:
    group.indices = np.union1d(group.indices, proposal.indices)
    group.proposal_ids.append(proposal.proposal_id)
    group.source_frames.add(str(proposal.metadata.get("frame_file", "")))
    phrase = str(proposal.metadata.get("phrase", proposal.class_name)).strip()
    if phrase:
        group.phrases[phrase] += 1
    group.scores.append(proposal.score)


def cluster_class_proposals(
    proposals: list[SemanticProposal],
    stuff_classes: set[str],
    merge_iou: float,
    containment: float,
) -> list[SemanticGroup]:
    groups: list[SemanticGroup] = []
    for proposal in proposals:
        best_group: SemanticGroup | None = None
        best_score = (0.0, 0.0, 0)
        for group in groups:
            if group.class_name != proposal.class_name:
                continue
            score = merge_score(proposal.indices, group.indices)
            if score[0] > best_score[0] or score[1] > best_score[1]:
                best_group = group
                best_score = score

        if best_group is not None and (best_score[0] >= merge_iou or best_score[1] >= containment):
            add_proposal_to_group(best_group, proposal)
        else:
            group = SemanticGroup(
                group_id=0,
                class_name=proposal.class_name,
                indices=proposal.indices,
                is_stuff=proposal.class_name in stuff_classes,
            )
            add_proposal_to_group(group, proposal)
            groups.append(group)
    return groups


def merge_stuff_groups(groups: list[SemanticGroup], stuff_classes: set[str]) -> list[SemanticGroup]:
    merged_by_class: dict[str, SemanticGroup] = {}
    kept: list[SemanticGroup] = []
    for group in groups:
        if group.class_name not in stuff_classes:
            kept.append(group)
            continue
        target = merged_by_class.get(group.class_name)
        if target is None:
            target = SemanticGroup(
                group_id=0,
                class_name=group.class_name,
                indices=group.indices,
                is_stuff=True,
            )
            merged_by_class[group.class_name] = target
        else:
            target.indices = np.union1d(target.indices, group.indices)
        target.proposal_ids.extend(group.proposal_ids)
        target.source_frames.update(group.source_frames)
        target.phrases.update(group.phrases)
        target.scores.extend(group.scores)
    kept.extend(merged_by_class.values())
    return kept


def filter_groups(
    groups: list[SemanticGroup],
    min_group_gaussians: int,
    min_group_proposals: int,
    max_groups: int,
    priorities: dict[str, int],
) -> list[SemanticGroup]:
    kept = [
        group
        for group in groups
        if group.gaussian_count >= min_group_gaussians
        and group.proposal_count >= min_group_proposals
    ]
    kept.sort(key=lambda group: group_order_key(group, priorities), reverse=True)
    if max_groups > 0:
        kept = kept[:max_groups]
    for label_id, group in enumerate(kept, start=1):
        group.group_id = label_id
    return kept


def final_label_name(group: SemanticGroup, class_counts: dict[str, int], class_ordinals: dict[str, int]) -> str:
    if group.is_stuff:
        return group.class_name
    class_ordinals[group.class_name] += 1
    return f"{group.class_name}_{class_ordinals[group.class_name]:02d}"


def frame_supports_for_group(
    group: SemanticGroup,
    proposals_by_id: dict[int, SemanticProposal],
) -> list[tuple[np.ndarray, float]]:
    proposals_by_frame: dict[str, list[SemanticProposal]] = defaultdict(list)
    for proposal_id in group.proposal_ids:
        proposal = proposals_by_id[proposal_id]
        frame_file = str(proposal.metadata.get("frame_file", proposal_id))
        proposals_by_frame[frame_file].append(proposal)

    frame_supports: list[tuple[np.ndarray, float]] = []
    for frame_proposals in proposals_by_frame.values():
        if len(frame_proposals) == 1:
            indices = frame_proposals[0].indices
        else:
            indices = np.unique(np.concatenate([proposal.indices for proposal in frame_proposals]))
        frame_supports.append((indices, max(proposal.score for proposal in frame_proposals)))
    return frame_supports


def signed_evidence_gate(
    candidate_indices: np.ndarray,
    evidence: ClassEvidence,
    min_positive_views: int,
    min_ratio: float,
) -> np.ndarray:
    positions = np.searchsorted(evidence.indices, candidate_indices)
    matched = positions < evidence.indices.shape[0]
    matched_positions = positions[matched]
    matched[matched] = evidence.indices[matched_positions] == candidate_indices[matched]

    positive = np.zeros((candidate_indices.shape[0],), dtype=np.float32)
    negative = np.zeros((candidate_indices.shape[0],), dtype=np.float32)
    if matched.any():
        positions = positions[matched]
        positive[matched] = evidence.positive_views[positions]
        negative[matched] = evidence.negative_views[positions]
    ratio = positive / np.maximum(positive + negative, 1.0)
    return (positive >= max(0, min_positive_views)) & (ratio >= min_ratio)


def assign_labels(
    vertex_count: int,
    groups: list[SemanticGroup],
    proposals: list[SemanticProposal],
    priorities: dict[str, int],
    reliability_views: float,
    min_quality: float,
    class_evidence: dict[str, ClassEvidence] | None,
    class_evidence_min_positive_views: int,
    class_evidence_min_ratio: float,
) -> np.ndarray:
    labels = np.zeros((vertex_count,), dtype=np.int32)
    best_quality = np.zeros((vertex_count,), dtype=np.float32)
    proposals_by_id = {proposal.proposal_id: proposal for proposal in proposals}

    for group in groups:
        frame_supports = frame_supports_for_group(group, proposals_by_id)
        if not frame_supports:
            continue
        evidence = np.zeros((vertex_count,), dtype=np.float32)
        total_weight = sum(max(weight, 1e-6) for _, weight in frame_supports)
        for indices, weight in frame_supports:
            evidence[indices] += max(weight, 1e-6)

        view_count = len(frame_supports)
        reliability = 1.0 if reliability_views <= 0 else view_count / (view_count + reliability_views)
        evidence *= reliability / max(total_weight, 1e-6)
        group.assignment_reliability = float(reliability)
        group.assignment_peak_quality = float(evidence.max())

        # Priority resolves exact numerical ties only; multi-view evidence owns the decision.
        tie_break = priorities.get(group.class_name, 0) * 1e-7
        candidate_quality = evidence + np.float32(tie_break)
        selected = (evidence >= min_quality) & (candidate_quality > best_quality)
        if class_evidence is not None and not group.is_stuff:
            signed = class_evidence.get(group.class_name)
            if signed is None:
                raise ValueError(f"Missing signed class evidence for {group.class_name}")
            candidate_indices = np.flatnonzero(selected)
            signed_keep = signed_evidence_gate(
                candidate_indices,
                signed,
                class_evidence_min_positive_views,
                class_evidence_min_ratio,
            )
            selected[:] = False
            selected[candidate_indices[signed_keep]] = True
        labels[selected] = group.group_id
        best_quality[selected] = candidate_quality[selected]
    return labels


def voxel_components(
    points: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), {
            "voxel_count": 0,
            "component_count": 0,
            "largest_component_gaussians": 0,
        }

    voxel_coordinates = np.floor(points / voxel_size).astype(np.int64)
    voxels, inverse, voxel_counts = np.unique(
        voxel_coordinates,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    parent = np.arange(voxels.shape[0], dtype=np.int64)

    def find_root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find_root(left)
        right_root = find_root(right)
        if left_root != right_root:
            parent[right_root] = left_root

    voxel_lookup = {tuple(int(value) for value in voxel): index for index, voxel in enumerate(voxels)}
    for index, voxel in enumerate(voxels):
        x, y, z = (int(value) for value in voxel)
        for dx, dy, dz in VOXEL_NEIGHBOR_OFFSETS:
            neighbor = voxel_lookup.get((x + dx, y + dy, z + dz))
            if neighbor is not None:
                union(index, neighbor)

    voxel_roots = np.fromiter(
        (find_root(index) for index in range(voxels.shape[0])),
        dtype=np.int64,
        count=voxels.shape[0],
    )
    point_roots = voxel_roots[inverse]
    _, point_components = np.unique(point_roots, return_inverse=True)
    component_sizes = np.bincount(point_components).astype(np.int64)
    largest = int(component_sizes.max())
    return point_components, component_sizes, {
        "voxel_count": int(voxels.shape[0]),
        "component_count": int(component_sizes.shape[0]),
        "largest_component_gaussians": largest,
    }


def voxel_component_membership(
    points: np.ndarray,
    voxel_size: float,
    min_component_gaussians: int,
    min_component_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    point_components, component_sizes, component_stats = voxel_components(points, voxel_size)
    if component_sizes.shape[0] == 0:
        return np.zeros((0,), dtype=bool), {
            **component_stats,
            "kept_component_count": 0,
            "component_keep_threshold": 0,
        }

    largest = int(component_sizes.max())
    keep_threshold = max(
        max(1, min_component_gaussians),
        int(np.ceil(largest * max(0.0, min_component_ratio))),
    )
    kept_components = np.flatnonzero(component_sizes >= keep_threshold)
    if kept_components.shape[0] == 0:
        kept_components = np.asarray([int(np.argmax(component_sizes))])
    keep = np.isin(point_components, kept_components)
    return keep, {
        **component_stats,
        "kept_component_count": int(kept_components.shape[0]),
        "component_keep_threshold": keep_threshold,
    }


def prune_thing_label_islands(
    labels: np.ndarray,
    groups: list[SemanticGroup],
    vertex_data: np.memmap,
    voxel_scale_multiplier: float,
    min_voxel_size: float,
    max_voxel_size: float,
    min_component_gaussians: int,
    min_component_ratio: float,
    include_stuff: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    required = {"x", "y", "z", "scale_0", "scale_1", "scale_2"}
    missing = sorted(required - set(vertex_data.dtype.names or ()))
    if missing:
        raise ValueError(f"PLY is missing spatial-pruning properties: {missing}")

    reports: list[dict[str, Any]] = []
    for group in groups:
        if group.is_stuff and not include_stuff:
            continue
        indices = np.flatnonzero(labels == group.group_id)
        if indices.shape[0] == 0:
            continue
        points = np.column_stack(
            [vertex_data[axis][indices].astype(np.float64) for axis in ("x", "y", "z")]
        )
        log_scales = np.column_stack(
            [vertex_data[axis][indices].astype(np.float64) for axis in ("scale_0", "scale_1", "scale_2")]
        )
        gaussian_scales = np.exp(np.clip(log_scales.max(axis=1), -20.0, 5.0))
        median_scale = float(np.median(gaussian_scales[np.isfinite(gaussian_scales)]))
        voxel_size = max(min_voxel_size, median_scale * voxel_scale_multiplier)
        if max_voxel_size > 0:
            voxel_size = min(voxel_size, max_voxel_size)

        keep, component_stats = voxel_component_membership(
            points,
            voxel_size,
            min_component_gaussians,
            min_component_ratio,
        )
        removed_indices = indices[~keep]
        labels[removed_indices] = 0
        group.assigned_count = int(keep.sum())
        reports.append(
            {
                "id": group.group_id,
                "class": group.class_name,
                "before_gaussians": int(indices.shape[0]),
                "after_gaussians": int(keep.sum()),
                "removed_gaussians": int((~keep).sum()),
                "removed_ratio": float((~keep).sum() / max(indices.shape[0], 1)),
                "median_gaussian_scale": median_scale,
                "voxel_size": voxel_size,
                **component_stats,
            }
        )
    return labels, reports


def merged_component_group(
    group_id: int,
    class_name: str,
    assigned_indices: np.ndarray,
    source_groups: list[SemanticGroup],
    source_counts: np.ndarray,
) -> SemanticGroup:
    support_indices = np.unique(np.concatenate([group.indices for group in source_groups]))
    merged = SemanticGroup(
        group_id=group_id,
        class_name=class_name,
        indices=support_indices,
        is_stuff=False,
        assigned_count=int(assigned_indices.shape[0]),
    )
    merged.proposal_ids = sorted(
        {proposal_id for group in source_groups for proposal_id in group.proposal_ids}
    )
    for group in source_groups:
        merged.source_frames.update(group.source_frames)
        merged.phrases.update(group.phrases)
        merged.scores.extend(group.scores)

    total = max(int(source_counts.sum()), 1)
    merged.assignment_reliability = float(
        sum(group.assignment_reliability * int(count) for group, count in zip(source_groups, source_counts))
        / total
    )
    merged.assignment_peak_quality = max(
        (group.assignment_peak_quality for group in source_groups),
        default=0.0,
    )
    return merged


def consolidate_thing_instances(
    labels: np.ndarray,
    groups: list[SemanticGroup],
    vertex_data: np.memmap,
    voxel_scale_multiplier: float,
    min_voxel_size: float,
    max_voxel_size: float,
    min_component_gaussians: int,
    min_component_ratio: float,
) -> tuple[np.ndarray, list[SemanticGroup], list[dict[str, Any]]]:
    """Merge connected same-class labels without splitting accepted instances."""

    required = {"x", "y", "z", "scale_0", "scale_1", "scale_2"}
    missing = sorted(required - set(vertex_data.dtype.names or ()))
    if missing:
        raise ValueError(f"PLY is missing instance-consolidation properties: {missing}")

    group_by_id = {group.group_id: group for group in groups}
    next_group_id = max(group_by_id, default=0) + 1
    rebuilt: list[SemanticGroup] = []
    reports: list[dict[str, Any]] = []
    processed_classes: set[str] = set()

    for group in groups:
        if group.is_stuff:
            rebuilt.append(group)
            continue
        if group.class_name in processed_classes:
            continue
        processed_classes.add(group.class_name)

        class_groups = [
            candidate
            for candidate in groups
            if not candidate.is_stuff and candidate.class_name == group.class_name
        ]
        class_group_ids = np.asarray([candidate.group_id for candidate in class_groups], dtype=np.int32)
        class_indices = np.flatnonzero(np.isin(labels, class_group_ids))
        if class_indices.shape[0] == 0:
            continue

        original_labels = labels[class_indices].copy()
        points = np.column_stack(
            [vertex_data[axis][class_indices].astype(np.float64) for axis in ("x", "y", "z")]
        )
        log_scales = np.column_stack(
            [
                vertex_data[axis][class_indices].astype(np.float64)
                for axis in ("scale_0", "scale_1", "scale_2")
            ]
        )
        gaussian_scales = np.exp(np.clip(log_scales.max(axis=1), -20.0, 5.0))
        finite_scales = gaussian_scales[np.isfinite(gaussian_scales)]
        median_scale = float(np.median(finite_scales)) if finite_scales.shape[0] else 0.0
        voxel_size = max(min_voxel_size, median_scale * voxel_scale_multiplier)
        if max_voxel_size > 0:
            voxel_size = min(voxel_size, max_voxel_size)

        components, component_sizes, component_stats = voxel_components(points, voxel_size)
        largest = int(component_sizes.max()) if component_sizes.shape[0] else 0
        keep_threshold = max(
            max(1, min_component_gaussians),
            int(np.ceil(largest * max(0.0, min_component_ratio))),
        )
        linking_components = np.flatnonzero(component_sizes >= keep_threshold)

        parent = np.arange(len(class_groups), dtype=np.int64)

        def find_root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        def union(left: int, right: int) -> None:
            left_root = find_root(left)
            right_root = find_root(right)
            if left_root != right_root:
                parent[right_root] = left_root

        group_position = {
            candidate.group_id: position for position, candidate in enumerate(class_groups)
        }
        linked_component_count = 0
        for component_id in linking_components:
            source_ids = np.unique(original_labels[components == component_id])
            if source_ids.shape[0] < 2:
                continue
            linked_component_count += 1
            first = group_position[int(source_ids[0])]
            for source_id in source_ids[1:]:
                union(first, group_position[int(source_id)])

        source_ids_by_root: dict[int, list[int]] = defaultdict(list)
        for position, candidate in enumerate(class_groups):
            source_ids_by_root[find_root(position)].append(candidate.group_id)
        merged_source_sets = sorted(
            source_ids_by_root.values(),
            key=lambda source_ids: min(source_ids),
        )

        labels[class_indices] = 0
        instance_reports: list[dict[str, Any]] = []
        for source_id_list in merged_source_sets:
            source_ids = np.asarray(source_id_list, dtype=np.int32)
            assigned_mask = np.isin(original_labels, source_ids)
            assigned_indices = class_indices[assigned_mask]
            counted_ids, source_counts = np.unique(
                original_labels[assigned_mask],
                return_counts=True,
            )
            source_groups = [group_by_id[int(source_id)] for source_id in counted_ids]
            rebuilt_group = merged_component_group(
                next_group_id,
                group.class_name,
                assigned_indices,
                source_groups,
                source_counts,
            )
            labels[assigned_indices] = next_group_id
            rebuilt.append(rebuilt_group)

            instance_points = points[assigned_mask]
            instance_reports.append(
                {
                    "temporary_id": next_group_id,
                    "gaussian_count": int(assigned_indices.shape[0]),
                    "source_label_ids": [int(value) for value in counted_ids],
                    "source_label_gaussian_counts": [int(value) for value in source_counts],
                    "centroid": [float(value) for value in instance_points.mean(axis=0)],
                    "bounds_min": [float(value) for value in instance_points.min(axis=0)],
                    "bounds_max": [float(value) for value in instance_points.max(axis=0)],
                }
            )
            next_group_id += 1

        reports.append(
            {
                "class": group.class_name,
                "before_group_count": len(class_groups),
                "after_instance_count": len(merged_source_sets),
                "before_gaussians": int(class_indices.shape[0]),
                "after_gaussians": int(class_indices.shape[0]),
                "removed_gaussians": 0,
                "median_gaussian_scale": median_scale,
                "voxel_size": voxel_size,
                "component_keep_threshold": keep_threshold,
                "linking_component_count": linked_component_count,
                **component_stats,
                "instances": instance_reports,
            }
        )

    return labels, rebuilt, reports


def prune_and_compact_groups(labels: np.ndarray, groups: list[SemanticGroup]) -> tuple[np.ndarray, list[SemanticGroup]]:
    histogram = {int(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))}
    active_groups = [group for group in groups if histogram.get(group.group_id, 0) > 0]
    remap = {group.group_id: new_id for new_id, group in enumerate(active_groups, start=1)}
    if len(active_groups) == len(groups) and all(remap[group.group_id] == group.group_id for group in active_groups):
        for group in active_groups:
            group.assigned_count = histogram.get(group.group_id, 0)
        return labels, active_groups

    compacted = np.zeros(labels.shape, dtype=np.int32)
    for group in active_groups:
        old_id = group.group_id
        new_id = remap[old_id]
        compacted[labels == old_id] = new_id
        group.group_id = new_id
        group.assigned_count = histogram.get(old_id, 0)
    return compacted, active_groups


def assigned_prune_threshold(
    group: SemanticGroup,
    min_assigned_gaussians: int,
    min_assigned_thing_gaussians: int,
    min_assigned_stuff_gaussians: int,
    total_source_view_count: int = 0,
    adaptive_stuff_min_view_ratio: float = 0.5,
    adaptive_stuff_threshold_ratio: float = 0.75,
    adaptive_thing_threshold_floor_ratio: float = 0.5,
) -> int:
    threshold = max(0, min_assigned_gaussians)
    if group.is_stuff:
        threshold = max(threshold, max(0, min_assigned_stuff_gaussians))
        view_ratio = len(group.source_frames) / float(max(total_source_view_count, 1))
        if total_source_view_count > 0 and view_ratio >= adaptive_stuff_min_view_ratio:
            adaptive_threshold = math.ceil(threshold * adaptive_stuff_threshold_ratio)
            threshold = max(max(0, min_assigned_gaussians), adaptive_threshold)
    else:
        threshold = max(threshold, max(0, min_assigned_thing_gaussians))
        if total_source_view_count > 0:
            view_ratio = min(
                1.0,
                len(group.source_frames) / float(total_source_view_count),
            )
            threshold_ratio = max(
                adaptive_thing_threshold_floor_ratio,
                1.0 - view_ratio,
            )
            adaptive_threshold = math.ceil(threshold * threshold_ratio)
            threshold = max(max(0, min_assigned_gaussians), adaptive_threshold)
    return threshold


def prune_assigned_groups(
    labels: np.ndarray,
    groups: list[SemanticGroup],
    min_assigned_gaussians: int,
    min_assigned_thing_gaussians: int,
    min_assigned_stuff_gaussians: int,
    min_label_score: float,
    total_source_view_count: int = 0,
    adaptive_stuff_min_view_ratio: float = 0.5,
    adaptive_stuff_threshold_ratio: float = 0.75,
    adaptive_thing_threshold_floor_ratio: float = 0.5,
) -> tuple[np.ndarray, list[SemanticGroup], list[dict[str, Any]]]:
    kept: list[SemanticGroup] = []
    pruned: list[dict[str, Any]] = []
    prune_ids: list[int] = []
    for group in groups:
        threshold = assigned_prune_threshold(
            group,
            min_assigned_gaussians,
            min_assigned_thing_gaussians,
            min_assigned_stuff_gaussians,
            total_source_view_count,
            adaptive_stuff_min_view_ratio,
            adaptive_stuff_threshold_ratio,
            adaptive_thing_threshold_floor_ratio,
        )
        reasons: list[str] = []
        if threshold > 0 and group.assigned_count < threshold:
            reasons.append(f"assigned_gaussians<{threshold}")
        if min_label_score > 0.0 and group.score < min_label_score:
            reasons.append(f"score<{min_label_score}")

        if not reasons:
            kept.append(group)
            continue

        prune_ids.append(group.group_id)
        pruned.append(
            {
                "id": group.group_id,
                "class": group.class_name,
                "is_stuff": group.is_stuff,
                "assigned_gaussian_count": group.assigned_count,
                "support_gaussian_count": group.gaussian_count,
                "proposal_count": group.proposal_count,
                "source_view_count": len(group.source_frames),
                "score": group.score,
                "mean_score": group.mean_score,
                "assigned_threshold": threshold,
                "reasons": reasons,
                "proposal_ids": group.proposal_ids,
                "phrases": dict(group.phrases),
            }
        )

    if prune_ids:
        for label_id in prune_ids:
            labels[labels == label_id] = 0
    return labels, kept, pruned


def build_label_map(scene: str, groups: list[SemanticGroup]) -> tuple[dict[str, Any], dict[int, str]]:
    class_counts = Counter(group.class_name for group in groups)
    class_ordinals: dict[str, int] = defaultdict(int)
    names_by_id: dict[int, str] = {}
    label_map: dict[str, Any] = {
        "scene": scene,
        "labels": [{"id": 0, "name": "unlabeled", "class": "unlabeled"}],
    }
    for group in sorted(groups, key=lambda item: item.group_id):
        name = final_label_name(group, class_counts, class_ordinals)
        names_by_id[group.group_id] = name
        label_map["labels"].append(
            {
                "id": group.group_id,
                "name": name,
                "class": group.class_name,
                "source": "groundingdino_sam_flashsplat",
                "proposal_count": group.proposal_count,
                "gaussian_count": group.assigned_count or group.gaussian_count,
                "support_gaussian_count": group.gaussian_count,
                "source_view_count": len(group.source_frames),
                "score": group.score,
                "mean_score": group.mean_score,
                "phrases": [phrase for phrase, _ in group.phrases.most_common(5)],
            }
        )
    return label_map, names_by_id


def group_summary(group: SemanticGroup, name: str) -> dict[str, Any]:
    return {
        "id": group.group_id,
        "name": name,
        "class": group.class_name,
        "is_stuff": group.is_stuff,
        "gaussian_count": group.assigned_count or group.gaussian_count,
        "support_gaussian_count": group.gaussian_count,
        "proposal_count": group.proposal_count,
        "source_view_count": len(group.source_frames),
        "score": group.score,
        "mean_score": group.mean_score,
        "assignment_reliability": group.assignment_reliability,
        "assignment_peak_quality": group.assignment_peak_quality,
        "proposal_ids": group.proposal_ids,
        "phrases": dict(group.phrases),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--proposal-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--class-config", type=Path)
    parser.add_argument("--stuff-classes", default="")
    parser.add_argument("--class-priority", default="")
    parser.add_argument("--assignment-reliability-views", default=2.0, type=float)
    parser.add_argument("--assignment-min-quality", default=0.03, type=float)
    parser.add_argument("--class-evidence-dir", type=Path)
    parser.add_argument("--class-evidence-min-positive-views", default=2, type=int)
    parser.add_argument("--class-evidence-min-ratio", default=0.50, type=float)
    parser.add_argument("--spatial-prune-thing-islands", action="store_true")
    parser.add_argument(
        "--spatial-prune-all-classes",
        action="store_true",
        help="Apply adaptive connected-component pruning to stuff and thing groups",
    )
    parser.add_argument("--spatial-voxel-scale-multiplier", default=4.0, type=float)
    parser.add_argument("--spatial-min-voxel-size", default=0.01, type=float)
    parser.add_argument("--spatial-max-voxel-size", default=0.20, type=float)
    parser.add_argument("--spatial-min-component-gaussians", default=500, type=int)
    parser.add_argument("--spatial-min-component-ratio", default=0.01, type=float)
    parser.add_argument("--consolidate-thing-instances", action="store_true")
    parser.add_argument("--instance-voxel-scale-multiplier", default=4.0, type=float)
    parser.add_argument("--instance-min-voxel-size", default=0.01, type=float)
    parser.add_argument("--instance-max-voxel-size", default=0.20, type=float)
    parser.add_argument("--instance-min-component-gaussians", default=500, type=int)
    parser.add_argument("--instance-min-component-ratio", default=0.01, type=float)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--min-proposal-gaussians", default=500, type=int)
    parser.add_argument("--max-proposal-gaussians", default=0, type=int)
    parser.add_argument("--merge-iou", default=0.35, type=float)
    parser.add_argument("--containment-threshold", default=0.70, type=float)
    parser.add_argument("--min-group-gaussians", default=1000, type=int)
    parser.add_argument("--min-group-proposals", default=1, type=int)
    parser.add_argument("--max-groups", default=128, type=int)
    parser.add_argument("--min-assigned-gaussians", default=0, type=int)
    parser.add_argument("--min-assigned-thing-gaussians", default=0, type=int)
    parser.add_argument("--min-assigned-stuff-gaussians", default=0, type=int)
    parser.add_argument("--adaptive-stuff-min-view-ratio", default=0.5, type=float)
    parser.add_argument("--adaptive-stuff-threshold-ratio", default=0.75, type=float)
    parser.add_argument(
        "--adaptive-thing-threshold-floor-ratio",
        default=0.5,
        type=float,
    )
    parser.add_argument("--min-label-score", default=0.0, type=float)
    parser.add_argument("--scene", default="")
    parser.add_argument("--semantic-ply-name")
    parser.add_argument("--labels-path", type=Path)
    parser.add_argument("--label-map-path", type=Path)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--semantic-ply-path", type=Path)
    parser.add_argument("--no-semantic-ply", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write only the fusion/group summary; do not write labels, label map, or PLY",
    )
    parser.add_argument("--require-class", action="store_true", default=True)
    parser.add_argument("--allow-unknown-class", dest="require_class", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.adaptive_stuff_min_view_ratio <= 1.0:
        raise ValueError("adaptive-stuff-min-view-ratio must be between 0 and 1")
    if not 0.0 < args.adaptive_stuff_threshold_ratio <= 1.0:
        raise ValueError("adaptive-stuff-threshold-ratio must be greater than 0 and at most 1")
    if not 0.0 < args.adaptive_thing_threshold_floor_ratio <= 1.0:
        raise ValueError(
            "adaptive-thing-threshold-floor-ratio must be greater than 0 and at most 1"
        )

    manifest_path = args.proposal_dir / "proposal_manifest.json"
    support_dir = args.proposal_dir / "proposal_supports"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not support_dir.exists():
        raise FileNotFoundError(support_dir)

    ply_path = point_cloud_path(args.model_path, args.iteration)
    vertex = read_ply_header(ply_path).element("vertex")
    if vertex is None:
        raise ValueError(f"{ply_path} has no vertex element")

    stuff_classes = load_stuff_classes(args.class_config, args.stuff_classes)
    assignment_priorities = load_assignment_priorities(args.class_config, args.class_priority)
    total_source_view_count = proposal_manifest_view_count(manifest_path)
    class_evidence = load_class_evidence(args.class_evidence_dir)
    proposals = load_proposals(
        manifest_path,
        support_dir,
        args.min_proposal_gaussians,
        args.max_proposal_gaussians,
        args.require_class,
    )
    groups = cluster_class_proposals(proposals, stuff_classes, args.merge_iou, args.containment_threshold)
    groups = merge_stuff_groups(groups, stuff_classes)
    groups = filter_groups(
        groups,
        args.min_group_gaussians,
        args.min_group_proposals,
        args.max_groups,
        assignment_priorities,
    )
    labels = assign_labels(
        vertex.count,
        groups,
        proposals,
        assignment_priorities,
        args.assignment_reliability_views,
        args.assignment_min_quality,
        class_evidence,
        args.class_evidence_min_positive_views,
        args.class_evidence_min_ratio,
    )
    labels, groups = prune_and_compact_groups(labels, groups)

    # Merge connected same-class fragments before applying the per-instance
    # size cutoff. Otherwise several valid fragments of one object can each be
    # discarded even when their connected union is large enough to retain.
    vertex_data: np.memmap | None = None
    instance_consolidation: list[dict[str, Any]] = []
    if args.consolidate_thing_instances:
        _, vertex_data = vertex_data_memmap(ply_path)
        labels, groups, instance_consolidation = consolidate_thing_instances(
            labels,
            groups,
            vertex_data,
            args.instance_voxel_scale_multiplier,
            args.instance_min_voxel_size,
            args.instance_max_voxel_size,
            args.instance_min_component_gaussians,
            args.instance_min_component_ratio,
        )
        labels, groups = prune_and_compact_groups(labels, groups)

    labels, groups, pruned_groups = prune_assigned_groups(
        labels,
        groups,
        args.min_assigned_gaussians,
        args.min_assigned_thing_gaussians,
        args.min_assigned_stuff_gaussians,
        args.min_label_score,
        total_source_view_count,
        args.adaptive_stuff_min_view_ratio,
        args.adaptive_stuff_threshold_ratio,
        args.adaptive_thing_threshold_floor_ratio,
    )
    labels, groups = prune_and_compact_groups(labels, groups)
    if pruned_groups:
        labels = assign_labels(
            vertex.count,
            groups,
            proposals,
            assignment_priorities,
            args.assignment_reliability_views,
            args.assignment_min_quality,
            class_evidence,
            args.class_evidence_min_positive_views,
            args.class_evidence_min_ratio,
        )
        labels, groups = prune_and_compact_groups(labels, groups)

    spatial_pruning: list[dict[str, Any]] = []
    spatial_tiny_pruned: list[dict[str, Any]] = []
    if args.spatial_prune_thing_islands:
        _, vertex_data = vertex_data_memmap(ply_path)
        labels, spatial_pruning = prune_thing_label_islands(
            labels,
            groups,
            vertex_data,
            args.spatial_voxel_scale_multiplier,
            args.spatial_min_voxel_size,
            args.spatial_max_voxel_size,
            args.spatial_min_component_gaussians,
            args.spatial_min_component_ratio,
            args.spatial_prune_all_classes,
        )
        labels, groups = prune_and_compact_groups(labels, groups)
        labels, groups, spatial_tiny_pruned = prune_assigned_groups(
            labels,
            groups,
            args.min_assigned_gaussians,
            args.min_assigned_thing_gaussians,
            args.min_assigned_stuff_gaussians,
            args.min_label_score,
            total_source_view_count,
            args.adaptive_stuff_min_view_ratio,
            args.adaptive_stuff_threshold_ratio,
            args.adaptive_thing_threshold_floor_ratio,
        )
        labels, groups = prune_and_compact_groups(labels, groups)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.labels_path or args.output_dir / "gaussian_labels.npy"
    label_map_path = args.label_map_path or args.output_dir / "label_map.json"
    summary_path = args.summary_path or args.output_dir / "semantic_group_summary.json"
    semantic_ply = resolve_semantic_ply_output(
        args.output_dir,
        semantic_ply_path=args.semantic_ply_path,
        semantic_ply_name=args.semantic_ply_name,
        disabled=args.no_semantic_ply or args.report_only,
    )
    output_files = [summary_path]
    if not args.report_only:
        output_files.extend(
            [
                labels_path,
                label_map_path,
                *([semantic_ply] if semantic_ply is not None else []),
            ]
        )
    for path in output_files:
        path.parent.mkdir(parents=True, exist_ok=True)
    for path in output_files:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    scene_name = args.scene or args.model_path.name
    label_map, names_by_id = build_label_map(scene_name, groups)
    if not args.report_only:
        np.save(labels_path, labels)
        label_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    histogram = {str(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))}
    summary = {
        "source": "groundingdino_sam_flashsplat",
        "stage": "semantic_fusion_and_pruning",
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "proposal_dir": str(args.proposal_dir),
        "outputs": {
            "report_only": bool(args.report_only),
            "semantic_labels_written": not args.report_only,
            "label_map_written": not args.report_only,
            "semantic_ply_written": semantic_ply is not None,
        },
        "parameters": {
            "min_proposal_gaussians": args.min_proposal_gaussians,
            "max_proposal_gaussians": args.max_proposal_gaussians,
            "merge_iou": args.merge_iou,
            "containment_threshold": args.containment_threshold,
            "min_group_gaussians": args.min_group_gaussians,
            "min_group_proposals": args.min_group_proposals,
            "max_groups": args.max_groups,
            "min_assigned_gaussians": args.min_assigned_gaussians,
            "min_assigned_thing_gaussians": args.min_assigned_thing_gaussians,
            "min_assigned_stuff_gaussians": args.min_assigned_stuff_gaussians,
            "total_source_view_count": total_source_view_count,
            "adaptive_stuff_min_view_ratio": args.adaptive_stuff_min_view_ratio,
            "adaptive_stuff_threshold_ratio": args.adaptive_stuff_threshold_ratio,
            "adaptive_thing_threshold_floor_ratio": (
                args.adaptive_thing_threshold_floor_ratio
            ),
            "min_label_score": args.min_label_score,
            "assignment_priority": assignment_priorities,
            "assignment_mode": "confidence_weighted_multiview",
            "assignment_reliability_views": args.assignment_reliability_views,
            "assignment_min_quality": args.assignment_min_quality,
            "class_evidence_dir": str(args.class_evidence_dir) if args.class_evidence_dir else "",
            "class_evidence_min_positive_views": args.class_evidence_min_positive_views,
            "class_evidence_min_ratio": args.class_evidence_min_ratio,
            "spatial_prune_thing_islands": args.spatial_prune_thing_islands,
            "spatial_prune_all_classes": args.spatial_prune_all_classes,
            "spatial_voxel_scale_multiplier": args.spatial_voxel_scale_multiplier,
            "spatial_min_voxel_size": args.spatial_min_voxel_size,
            "spatial_max_voxel_size": args.spatial_max_voxel_size,
            "spatial_min_component_gaussians": args.spatial_min_component_gaussians,
            "spatial_min_component_ratio": args.spatial_min_component_ratio,
            "consolidate_thing_instances": args.consolidate_thing_instances,
            "instance_voxel_scale_multiplier": args.instance_voxel_scale_multiplier,
            "instance_min_voxel_size": args.instance_min_voxel_size,
            "instance_max_voxel_size": args.instance_max_voxel_size,
            "instance_min_component_gaussians": args.instance_min_component_gaussians,
            "instance_min_component_ratio": args.instance_min_component_ratio,
        },
        "proposal_count": len(proposals),
        "group_count": len(groups),
        "pruning": {
            "pruned_group_count": len(pruned_groups) + len(spatial_tiny_pruned),
            "pruned_groups": pruned_groups + spatial_tiny_pruned,
            "reassigned_after_pruning": bool(pruned_groups),
        },
        "spatial_pruning": {
            "enabled": args.spatial_prune_thing_islands,
            "label_count": len(spatial_pruning),
            "removed_gaussian_count": sum(item["removed_gaussians"] for item in spatial_pruning),
            "labels": spatial_pruning,
        },
        "instance_consolidation": {
            "enabled": args.consolidate_thing_instances,
            "class_count": len(instance_consolidation),
            "before_group_count": sum(item["before_group_count"] for item in instance_consolidation),
            "after_instance_count": sum(item["after_instance_count"] for item in instance_consolidation),
            "removed_gaussian_count": sum(item["removed_gaussians"] for item in instance_consolidation),
            "classes": instance_consolidation,
        },
        "stuff_classes": sorted(stuff_classes),
        "label_histogram": histogram,
        "groups": [group_summary(group, names_by_id[group.group_id]) for group in groups],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.report_only:
        print("report-only fusion; semantic labels, label map, and PLY were not written")
    elif semantic_ply is not None:
        write_ply_with_labels(ply_path, semantic_ply, labels)
        print(f"wrote {semantic_ply}")
    else:
        print("semantic PLY disabled; retained labels and label map only")
    print(json.dumps(histogram, sort_keys=True))


if __name__ == "__main__":
    main()
