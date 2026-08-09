#!/usr/bin/env python3
"""Report-only component grouping for camera-observed black Gaussians.

The audit groups Gaussians before assigning a semantic class.  Its sparse
mutual-kNN graph is class agnostic and combines 3D proximity, appearance,
Gaussian shape, co-visibility, and semantic-distribution similarity.  Camera
evidence then names each component with one normalized vote per camera.
Spatial anchors only corroborate that camera-supported name.  Zero-camera
Gaussians remain black, and the audit writes no accepted labels or PLY.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from scripts.task1.common.ply_utils import vertex_data_memmap
from scripts.task1.dinov2.dinov2_ontology import Ontology, load_ontology
from scripts.task1.dinov3.recover_detected_abstentions import (
    CONTRACT as RECOVERY_CONTRACT,
    SOURCE as RECOVERY_SOURCE,
    camera_reliability_rows,
    load_camera_evidence,
    validate_vote_manifest,
    wilson_lower_bound,
)
from scripts.task1.dinov3.dinov2_second_source import (
    aggregate_component_votes_agreement_gated,
    align_dinov2_evidence,
    load_dinov2_evidence,
)
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    CONTRACT as HARD_AUDIT_CONTRACT,
    SOURCE as HARD_AUDIT_SOURCE,
    STATUS_ACCEPTED,
    consensus_statistics,
    quantile_summary,
    sha256_file,
)


SOURCE = "dinov3_observed_black_component_graph_audit"
CONTRACT = "cache_only_class_agnostic_component_graph_report_v1"

DECISION_ZERO_CAMERA = 0
DECISION_COMPONENT_TOO_WEAK = 1
DECISION_COMPONENT_CONFLICT = 2
DECISION_COMPONENT_BOUNDARY_AMBIGUOUS = 3
DECISION_COMPONENT_SCORE_TOO_LOW = 4
DECISION_ELIGIBLE_COMPONENT = 5
DECISION_NAMES = {
    DECISION_ZERO_CAMERA: "remain_black_zero_combined_semantic_cameras",
    DECISION_COMPONENT_TOO_WEAK: "remain_black_component_has_too_little_camera_evidence",
    DECISION_COMPONENT_CONFLICT: "remain_black_component_raw_and_weighted_winners_disagree",
    DECISION_COMPONENT_BOUNDARY_AMBIGUOUS: "remain_black_component_boundary_is_ambiguous",
    DECISION_COMPONENT_SCORE_TOO_LOW: "remain_black_component_score_below_calibrated_threshold",
    DECISION_ELIGIBLE_COMPONENT: "eligible_report_only_component_graph_candidate",
}

POLICY: dict[str, Any] = {
    "neighbor_count": 16,
    "anchor_support_neighbor_count": 8,
    "calibration_anchor_sample_limit": 20_000,
    "minimum_component_camera_count": 2,
    "minimum_calibration_component_size": 2,
    "minimum_calibration_component_purity": 0.90,
    "minimum_component_score_floor": 0.62,
    "minimum_component_precision_lower_bound": 0.90,
    "maximum_calibration_cross_class_edge_rate": 0.05,
    "minimum_calibration_edges": 64,
    "maximum_boundary_pressure": 0.98,
    "soft_class_reliability_floor": 0.50,
    "soft_class_reliability_ceiling": 1.00,
    "edge_weights": {
        "spatial": 0.20,
        "appearance": 0.15,
        "scale": 0.10,
        "orientation": 0.10,
        "co_visibility": 0.15,
        "semantic_distribution": 0.30,
    },
}


def score_features(scores: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("scores must have classes-plus-zero x items")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("scores must be finite and non-negative")
    semantic = values[1:]
    maximum = semantic.max(axis=0)
    winner = semantic.argmax(axis=0).astype(np.uint16) + np.uint16(1)
    tied = (semantic == maximum[None, :]).sum(axis=0) > 1
    second = (
        np.partition(semantic, -2, axis=0)[-2]
        if semantic.shape[0] > 1
        else np.zeros_like(maximum)
    )
    total = semantic.sum(axis=0, dtype=np.float32)
    share = np.divide(maximum, total, out=np.zeros_like(maximum), where=total > 0.0)
    margin = np.divide(
        maximum - second,
        total,
        out=np.zeros_like(maximum),
        where=total > 0.0,
    )
    probabilities = np.divide(
        semantic,
        total[None, :],
        out=np.zeros_like(semantic),
        where=total[None, :] > 0.0,
    )
    logs = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    logs[positive] = np.log(probabilities[positive])
    entropy = -(probabilities * logs).sum(axis=0, dtype=np.float32)
    normalized_entropy = entropy / np.float32(math.log(max(semantic.shape[0], 2)))
    accepted = (maximum > 0.0) & ~tied & (maximum * 2.0 > total)
    return {
        "winner": winner,
        "accepted": accepted,
        "tied": tied,
        "total": total,
        "winner_share": share,
        "winner_margin": margin,
        "normalized_entropy": normalized_entropy,
        "probabilities": probabilities,
    }


def semantic_cache(
    evidence: Sequence[Mapping[str, Any]],
    reliabilities: Mapping[int, float],
    gaussian_indices: np.ndarray,
    *,
    class_count: int,
) -> dict[str, np.ndarray]:
    target = np.asarray(gaussian_indices, dtype=np.int64)
    weighted = np.zeros((class_count + 1, target.size), dtype=np.float32)
    raw = np.zeros((class_count + 1, target.size), dtype=np.uint16)
    camera_count = np.zeros((target.size,), dtype=np.uint16)
    winners = np.zeros((len(evidence), target.size), dtype=np.uint16)
    masses = np.zeros((len(evidence), target.size), dtype=np.float32)
    columns = np.arange(target.size, dtype=np.int64)
    for ordinal, item in enumerate(evidence):
        local_winner = np.asarray(item["winners"], dtype=np.uint16)[target]
        local_mass = np.asarray(item["mass"], dtype=np.float32)[target]
        winners[ordinal] = local_winner
        masses[ordinal] = local_mass
        supported = local_winner > 0
        if not np.any(supported):
            continue
        rows = local_winner[supported].astype(np.int64)
        selected = columns[supported]
        raw[rows, selected] += np.uint16(1)
        contribution = (
            np.float32(reliabilities[int(item["camera_index"])])
            * local_mass[supported]
        )
        np.add.at(weighted, (rows, selected), contribution)
        camera_count[supported] += np.uint16(1)
    features = score_features(weighted)
    return {
        "raw": raw,
        "weighted": weighted,
        "camera_count": camera_count,
        "winners": winners,
        "masses": masses,
        "visibility": winners > 0,
        "probabilities": features["probabilities"],
    }


def query_neighbors(
    tree: cKDTree,
    points: np.ndarray,
    *,
    neighbor_count: int,
    workers: int,
    self_indices: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    extra = 1 if self_indices is not None else 0
    distances, indices = tree.query(
        np.asarray(points, dtype=np.float64),
        k=neighbor_count + extra,
        workers=workers,
    )
    if neighbor_count + extra == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if self_indices is None:
        return distances[:, :neighbor_count], indices[:, :neighbor_count]
    own = np.asarray(self_indices, dtype=np.int64)
    if own.shape != (distances.shape[0],):
        raise ValueError("self indices must align with query points")
    output_distances = np.full((distances.shape[0], neighbor_count), np.inf)
    output_indices = np.full((indices.shape[0], neighbor_count), tree.n, dtype=np.int64)
    for row in range(distances.shape[0]):
        keep = indices[row] != own[row]
        selected_distances = distances[row, keep][:neighbor_count]
        selected_indices = indices[row, keep][:neighbor_count]
        output_distances[row, : selected_distances.size] = selected_distances
        output_indices[row, : selected_indices.size] = selected_indices
    return output_distances, output_indices


def _matrix(vertices: Any, indices: np.ndarray, names: Sequence[str]) -> Optional[np.ndarray]:
    available = set(vertices.dtype.names or ())
    if not all(name in available for name in names):
        return None
    return np.column_stack(
        [np.asarray(vertices[name][indices], dtype=np.float32) for name in names]
    )


def feature_subset(vertices: Any, indices: np.ndarray) -> dict[str, Optional[np.ndarray]]:
    selected = np.asarray(indices, dtype=np.int64)
    available = set(vertices.dtype.names or ())
    if not {"x", "y", "z"}.issubset(available):
        raise ValueError("source PLY lacks x/y/z coordinates")
    points = np.column_stack(
        [np.asarray(vertices[name][selected], dtype=np.float64) for name in ("x", "y", "z")]
    )
    appearance = _matrix(vertices, selected, ("f_dc_0", "f_dc_1", "f_dc_2"))
    log_scale = _matrix(vertices, selected, ("scale_0", "scale_1", "scale_2"))
    rotation = _matrix(vertices, selected, ("rot_0", "rot_1", "rot_2", "rot_3"))
    normal: Optional[np.ndarray] = None
    if log_scale is not None and rotation is not None:
        quaternion = np.nan_to_num(rotation).astype(np.float64)
        norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
        quaternion = np.divide(
            quaternion,
            norm,
            out=np.zeros_like(quaternion),
            where=norm > 1e-8,
        )
        axes = np.eye(3, dtype=np.float64)[np.argmin(log_scale, axis=1)]
        vector = quaternion[:, 1:]
        cross = 2.0 * np.cross(vector, axes)
        normal = axes + quaternion[:, :1] * cross + np.cross(vector, cross)
        normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = np.divide(
            normal,
            normal_norm,
            out=np.zeros_like(normal),
            where=normal_norm > 1e-8,
        )
    return {
        "points": points,
        "appearance": None if appearance is None else np.nan_to_num(appearance),
        "log_scale": None if log_scale is None else np.nan_to_num(log_scale),
        "normal": normal,
    }


def derive_feature_scales(
    features: Mapping[str, Optional[np.ndarray]],
    src: np.ndarray,
    dst: np.ndarray,
) -> dict[str, float]:
    scales: dict[str, float] = {}
    for key, output_name in (("appearance", "appearance"), ("log_scale", "scale")):
        values = features.get(key)
        if values is None:
            continue
        distances = np.linalg.norm(values[src] - values[dst], axis=1)
        positive = distances[np.isfinite(distances) & (distances > 0.0)]
        scales[output_name] = (
            float(np.quantile(positive, 0.75)) if positive.size else 1.0
        )
    return scales


def pair_affinity(
    features: Mapping[str, Optional[np.ndarray]],
    semantic_probabilities: np.ndarray,
    visibility: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    distances: np.ndarray,
    *,
    distance_scale: float,
    feature_scales: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    left = np.asarray(src, dtype=np.int64)
    right = np.asarray(dst, dtype=np.int64)
    pair_distance = np.asarray(distances, dtype=np.float64)
    scores: dict[str, np.ndarray] = {
        "spatial": np.exp(-pair_distance / max(distance_scale, 1e-8)).astype(np.float32)
    }
    appearance = features.get("appearance")
    if appearance is not None:
        value = np.linalg.norm(appearance[left] - appearance[right], axis=1)
        scores["appearance"] = np.exp(
            -value / max(float(feature_scales.get("appearance", 1.0)), 1e-6)
        ).astype(np.float32)
    log_scale = features.get("log_scale")
    if log_scale is not None:
        value = np.linalg.norm(log_scale[left] - log_scale[right], axis=1)
        scores["scale"] = np.exp(
            -value / max(float(feature_scales.get("scale", 1.0)), 1e-6)
        ).astype(np.float32)
    normal = features.get("normal")
    if normal is not None:
        scores["orientation"] = np.clip(
            np.abs(np.sum(normal[left] * normal[right], axis=1)), 0.0, 1.0
        ).astype(np.float32)
    if visibility.size:
        left_visible = visibility[:, left]
        right_visible = visibility[:, right]
        intersection = np.count_nonzero(left_visible & right_visible, axis=0)
        union = np.count_nonzero(left_visible | right_visible, axis=0)
        scores["co_visibility"] = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=union > 0,
        )
    if semantic_probabilities.size:
        left_semantic = semantic_probabilities[:, left]
        right_semantic = semantic_probabilities[:, right]
        numerator = np.sum(left_semantic * right_semantic, axis=0)
        denominator = np.sqrt(
            np.sum(left_semantic * left_semantic, axis=0)
            * np.sum(right_semantic * right_semantic, axis=0)
        )
        scores["semantic_distribution"] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-8,
        ).astype(np.float32)
    numerator = np.zeros((left.size,), dtype=np.float32)
    denominator = 0.0
    for name, value in scores.items():
        weight = float(POLICY["edge_weights"][name])
        numerator += np.float32(weight) * value
        denominator += weight
    return numerator / np.float32(max(denominator, 1e-8)), scores


def calibrate_edge_threshold(
    affinity: np.ndarray,
    same_class: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    values = np.asarray(affinity, dtype=np.float32)
    same = np.asarray(same_class, dtype=bool)
    if values.size == 0:
        return 0.80, {"pair_count": 0, "threshold": 0.80, "fallback": True}
    candidates = np.unique(np.quantile(values, np.linspace(0.01, 0.995, 100)))
    best: Optional[tuple[float, float, float]] = None
    for threshold in candidates:
        accepted = values >= threshold
        accepted_count = int(np.count_nonzero(accepted))
        if accepted_count < int(POLICY["minimum_calibration_edges"]):
            continue
        cross_rate = float(np.count_nonzero(accepted & ~same) / accepted_count)
        if cross_rate > float(POLICY["maximum_calibration_cross_class_edge_rate"]):
            continue
        same_recall = float(
            np.count_nonzero(accepted & same) / max(np.count_nonzero(same), 1)
        )
        candidate = (same_recall, -cross_rate, -float(threshold))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        threshold = float(np.quantile(values, 0.90))
        fallback = True
    else:
        threshold = -best[2]
        fallback = False
    accepted = values >= threshold
    accepted_count = int(np.count_nonzero(accepted))
    return threshold, {
        "pair_count": int(values.size),
        "same_class_pair_count": int(np.count_nonzero(same)),
        "accepted_pair_count": accepted_count,
        "accepted_same_class_pair_count": int(np.count_nonzero(accepted & same)),
        "accepted_cross_class_pair_count": int(np.count_nonzero(accepted & ~same)),
        "accepted_cross_class_rate": (
            float(np.count_nonzero(accepted & ~same) / accepted_count)
            if accepted_count
            else 0.0
        ),
        "threshold": threshold,
        "fallback": fallback,
        "class_specific_thresholds_used": False,
    }


def mutual_knn_pairs(
    points: np.ndarray,
    *,
    neighbor_count: int,
    maximum_distance: float,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.shape[0] < 2:
        empty_i = np.empty((0,), dtype=np.int64)
        empty_f = np.empty((0,), dtype=np.float64)
        return empty_i, empty_i.copy(), empty_f
    effective_neighbors = min(neighbor_count, coordinates.shape[0] - 1)
    tree = cKDTree(coordinates)
    distances, neighbors = query_neighbors(
        tree,
        coordinates,
        neighbor_count=effective_neighbors,
        workers=workers,
        self_indices=np.arange(coordinates.shape[0], dtype=np.int64),
    )
    rows = np.repeat(np.arange(coordinates.shape[0], dtype=np.int64), effective_neighbors)
    columns = neighbors.reshape(-1)
    values = distances.reshape(-1)
    valid = (
        np.isfinite(values)
        & (values <= maximum_distance)
        & (columns >= 0)
        & (columns < coordinates.shape[0])
        & (rows != columns)
    )
    rows, columns, values = rows[valid], columns[valid], values[valid]
    low = np.minimum(rows, columns)
    high = np.maximum(rows, columns)
    codes = low * np.int64(coordinates.shape[0]) + high
    order = np.argsort(codes, kind="mergesort")
    codes = codes[order]
    values = values[order]
    unique, starts, counts = np.unique(codes, return_index=True, return_counts=True)
    mutual = counts >= 2
    unique = unique[mutual]
    starts = starts[mutual]
    counts = counts[mutual]
    maximums = np.maximum.reduceat(values, starts)
    # reduceat can include the next group for the final repeated start layout;
    # explicitly recompute the uncommon groups with more than two entries.
    repeated = counts > 2
    for position in np.flatnonzero(repeated):
        start = int(starts[position])
        maximums[position] = np.max(values[start : start + int(counts[position])])
    return (
        (unique // np.int64(coordinates.shape[0])).astype(np.int64),
        (unique % np.int64(coordinates.shape[0])).astype(np.int64),
        maximums.astype(np.float64),
    )


def build_components(
    features: Mapping[str, Optional[np.ndarray]],
    semantic_probabilities: np.ndarray,
    visibility: np.ndarray,
    *,
    class_constraint: Optional[np.ndarray] = None,
    edge_threshold: float,
    distance_scale: float,
    feature_scales: Mapping[str, float],
    neighbor_count: int,
    workers: int,
) -> dict[str, Any]:
    points = np.asarray(features["points"], dtype=np.float64)
    src, dst, distances = mutual_knn_pairs(
        points,
        neighbor_count=neighbor_count,
        maximum_distance=distance_scale * 3.0,
        workers=workers,
    )
    affinity, edge_features = pair_affinity(
        features,
        semantic_probabilities,
        visibility,
        src,
        dst,
        distances,
        distance_scale=distance_scale,
        feature_scales=feature_scales,
    )
    accepted = affinity >= np.float32(edge_threshold)
    if class_constraint is not None:
        constraint = np.asarray(class_constraint, dtype=np.int64)
        if constraint.shape != (points.shape[0],):
            raise ValueError("class_constraint must be aligned with the points")
        accepted &= constraint[src] == constraint[dst]
    parent = np.arange(points.shape[0], dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for left, right in zip(src[accepted], dst[accepted]):
        left_root = find(int(left))
        right_root = find(int(right))
        if left_root != right_root:
            parent[right_root] = left_root
    roots = np.asarray([find(index) for index in range(points.shape[0])], dtype=np.int64)
    _, component_ids = np.unique(roots, return_inverse=True)
    component_count = int(component_ids.max() + 1) if component_ids.size else 0
    component_sizes = np.bincount(component_ids, minlength=component_count).astype(np.uint32)
    internal_sum = np.zeros((component_count,), dtype=np.float32)
    internal_count = np.zeros((component_count,), dtype=np.uint32)
    boundary_max = np.zeros((component_count,), dtype=np.float32)
    if src.size:
        same_component = component_ids[src] == component_ids[dst]
        internal = same_component & accepted
        if np.any(internal):
            np.add.at(internal_sum, component_ids[src[internal]], affinity[internal])
            np.add.at(internal_count, component_ids[src[internal]], np.uint32(1))
        crossing = ~same_component
        if np.any(crossing):
            np.maximum.at(boundary_max, component_ids[src[crossing]], affinity[crossing])
            np.maximum.at(boundary_max, component_ids[dst[crossing]], affinity[crossing])
    internal_mean = np.divide(
        internal_sum,
        internal_count,
        out=np.zeros_like(internal_sum),
        where=internal_count > 0,
    )
    boundary_pressure = np.divide(
        boundary_max,
        np.float32(max(edge_threshold, 1e-6)),
        out=np.zeros_like(boundary_max),
        where=boundary_max > 0.0,
    )
    return {
        "component_ids": component_ids.astype(np.int32),
        "component_sizes": component_sizes,
        "component_internal_affinity": internal_mean,
        "component_boundary_pressure": np.clip(boundary_pressure, 0.0, 1.0),
        "edge_count": int(src.size),
        "accepted_edge_count": int(np.count_nonzero(accepted)),
        "edge_affinity": affinity,
        "edge_features": edge_features,
    }


def aggregate_component_votes(
    component_ids: np.ndarray,
    cache: Mapping[str, np.ndarray],
    evidence: Sequence[Mapping[str, Any]],
    reliabilities: Mapping[int, float],
    *,
    class_count: int,
    excluded_camera_by_component: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    components = np.asarray(component_ids, dtype=np.int64)
    component_count = int(components.max() + 1) if components.size else 0
    raw = np.zeros((class_count + 1, component_count), dtype=np.uint16)
    weighted = np.zeros((class_count + 1, component_count), dtype=np.float32)
    camera_count = np.zeros((component_count,), dtype=np.uint16)
    winners = np.asarray(cache["winners"], dtype=np.uint16)
    masses = np.asarray(cache["masses"], dtype=np.float32)
    for ordinal, item in enumerate(evidence):
        local_winner = winners[ordinal]
        local_mass = masses[ordinal]
        supported = local_winner > 0
        if excluded_camera_by_component is not None:
            supported &= excluded_camera_by_component[components] != ordinal
        if not np.any(supported):
            continue
        keys = (
            components[supported] * np.int64(class_count + 1)
            + local_winner[supported].astype(np.int64)
        )
        totals = np.bincount(
            keys,
            weights=local_mass[supported],
            minlength=component_count * (class_count + 1),
        ).reshape(component_count, class_count + 1)
        semantic = totals[:, 1:]
        maximum = semantic.max(axis=1)
        winner = semantic.argmax(axis=1).astype(np.int64) + 1
        tied = (semantic == maximum[:, None]).sum(axis=1) > 1
        total = semantic.sum(axis=1)
        unique = (maximum > 0.0) & ~tied & (maximum * 2.0 > total)
        selected_components = np.flatnonzero(unique)
        if selected_components.size == 0:
            continue
        raw[winner[selected_components], selected_components] += np.uint16(1)
        normalized_mass = np.divide(
            maximum[selected_components],
            total[selected_components],
            out=np.zeros_like(maximum[selected_components]),
            where=total[selected_components] > 0.0,
        )
        weighted[winner[selected_components], selected_components] += (
            np.float32(reliabilities[int(item["camera_index"])]) * normalized_mass
        )
        camera_count[selected_components] += np.uint16(1)
    return {"raw": raw, "weighted": weighted, "camera_count": camera_count}


def class_reliability_rows(
    confusion: np.ndarray,
    ontology: Ontology,
) -> list[dict[str, Any]]:
    counts = np.asarray(confusion, dtype=np.uint64)
    expected = (ontology.class_count + 1, ontology.class_count + 1)
    if counts.shape != expected:
        raise ValueError(f"held-out confusion matrix must have shape {expected}")
    source_totals = counts.sum(axis=1, dtype=np.uint64)
    predicted_totals = counts.sum(axis=0, dtype=np.uint64)
    rows: list[dict[str, Any]] = []
    for item in ontology.classes:
        project_id = item.project_id
        correct = int(counts[project_id, project_id])
        recall_trials = int(source_totals[project_id])
        precision_trials = int(predicted_totals[project_id])
        rows.append(
            {
                "project_id": project_id,
                "class": item.project_class,
                "type": item.kind,
                "heldout_recall_trial_count": recall_trials,
                "heldout_recall_correct_count": correct,
                "heldout_recall": correct / recall_trials if recall_trials else 0.0,
                "heldout_recall_lower_bound": wilson_lower_bound(correct, recall_trials),
                "heldout_precision_trial_count": precision_trials,
                "heldout_precision_correct_count": correct,
                "heldout_precision": correct / precision_trials if precision_trials else 0.0,
                "heldout_precision_lower_bound": wilson_lower_bound(correct, precision_trials),
            }
        )
    return rows


def soft_class_reliability(
    rows: Sequence[Mapping[str, Any]],
    class_count: int,
) -> np.ndarray:
    floor = float(POLICY["soft_class_reliability_floor"])
    ceiling = float(POLICY["soft_class_reliability_ceiling"])
    result = np.ones((class_count + 1,), dtype=np.float32)
    for row in rows:
        project_id = int(row["project_id"])
        recall = float(row["heldout_recall_lower_bound"])
        precision = float(row["heldout_precision_lower_bound"])
        if recall == 0.0 and precision == 0.0:
            value = (floor + ceiling) * 0.5
        else:
            value = math.sqrt(max(recall, 0.0) * max(precision, 0.0))
        result[project_id] = np.float32(np.clip(value, floor, ceiling))
    return result


def component_anchor_support(
    points: np.ndarray,
    component_ids: np.ndarray,
    anchor_tree: cKDTree,
    anchor_labels: np.ndarray,
    *,
    neighbor_count: int,
    maximum_distance: float,
    workers: int,
    class_count: int,
    self_anchor_positions: Optional[np.ndarray] = None,
) -> np.ndarray:
    extra = 1 if self_anchor_positions is not None else 0
    distances, neighbors = query_neighbors(
        anchor_tree,
        points,
        neighbor_count=neighbor_count + extra,
        workers=workers,
    )
    if self_anchor_positions is not None:
        own = np.asarray(self_anchor_positions, dtype=np.int64)
        filtered_distances = np.full((points.shape[0], neighbor_count), np.inf)
        filtered_neighbors = np.full(
            (points.shape[0], neighbor_count), anchor_tree.n, dtype=np.int64
        )
        for row in range(points.shape[0]):
            keep = neighbors[row] != own[row]
            selected_distances = distances[row, keep][:neighbor_count]
            selected_neighbors = neighbors[row, keep][:neighbor_count]
            filtered_distances[row, : selected_distances.size] = selected_distances
            filtered_neighbors[row, : selected_neighbors.size] = selected_neighbors
        distances, neighbors = filtered_distances, filtered_neighbors
    valid = (
        np.isfinite(distances)
        & (distances <= maximum_distance)
        & (neighbors >= 0)
        & (neighbors < anchor_labels.size)
    )
    safe = np.where(valid, neighbors, 0)
    labels = anchor_labels[safe]
    components = np.asarray(component_ids, dtype=np.int64)
    component_count = int(components.max() + 1) if components.size else 0
    support = np.zeros((component_count, class_count + 1), dtype=np.float32)
    valid_count = valid.sum(axis=1).astype(np.float32)
    for column in range(neighbor_count):
        selected = valid[:, column] & (labels[:, column] > 0)
        if not np.any(selected):
            continue
        weight = np.divide(
            np.ones(np.count_nonzero(selected), dtype=np.float32),
            valid_count[selected],
            out=np.zeros(np.count_nonzero(selected), dtype=np.float32),
            where=valid_count[selected] > 0.0,
        )
        keys = (
            components[selected] * np.int64(class_count + 1)
            + labels[selected, column].astype(np.int64)
        )
        np.add.at(support.reshape(-1), keys, weight)
    sizes = np.bincount(components, minlength=component_count).astype(np.float32)
    return np.divide(
        support,
        sizes[:, None],
        out=np.zeros_like(support),
        where=sizes[:, None] > 0.0,
    )


def component_scores(
    candidate: np.ndarray,
    semantic_features: Mapping[str, np.ndarray],
    component_ids: np.ndarray,
    component_internal_affinity: np.ndarray,
    component_boundary_pressure: np.ndarray,
    node_semantic_probabilities: np.ndarray,
    anchor_support: np.ndarray,
    class_reliability: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    candidates = np.asarray(candidate, dtype=np.uint16)
    components = np.asarray(component_ids, dtype=np.int64)
    component_count = candidates.size
    candidate_per_node = candidates[components]
    node_fit = np.zeros((components.size,), dtype=np.float32)
    valid_nodes = candidate_per_node > 0
    node_columns = np.flatnonzero(valid_nodes)
    if node_columns.size:
        node_fit[node_columns] = node_semantic_probabilities[
            candidate_per_node[node_columns].astype(np.int64) - 1,
            node_columns,
        ]
    fit_sum = np.bincount(components, weights=node_fit, minlength=component_count)
    sizes = np.bincount(components, minlength=component_count)
    semantic_fit = np.divide(
        fit_sum,
        np.maximum(sizes, 1),
        out=np.zeros_like(fit_sum),
        where=sizes > 0,
    ).astype(np.float32)
    valid_components = candidates > 0
    support = np.zeros((component_count,), dtype=np.float32)
    selected = np.flatnonzero(valid_components)
    if selected.size:
        support[selected] = anchor_support[
            selected,
            candidates[selected].astype(np.int64),
        ]
    reliability = class_reliability[candidates.astype(np.int64)]
    score = (
        np.float32(0.30) * semantic_features["winner_share"]
        + np.float32(0.15) * semantic_features["winner_margin"]
        + np.float32(0.15) * semantic_fit
        + np.float32(0.15) * component_internal_affinity
        + np.float32(0.10) * support
        + np.float32(0.10) * reliability
        - np.float32(0.05) * component_boundary_pressure
    )
    return score.astype(np.float32), {
        "semantic_fit": semantic_fit,
        "anchor_support": support,
        "class_reliability": reliability,
    }


def calibrate_component_score(
    scores: np.ndarray,
    correct: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    values = np.asarray(scores, dtype=np.float32)
    correctness = np.asarray(correct, dtype=bool)
    floor = float(POLICY["minimum_component_score_floor"])
    if values.size == 0:
        return floor, {"trial_count": 0, "threshold": floor, "fallback": True}
    best: Optional[tuple[float, float, float]] = None
    for threshold in np.unique(np.quantile(values, np.linspace(0.05, 0.99, 80))):
        selected = values >= threshold
        trial_count = int(np.count_nonzero(selected))
        if trial_count < 8:
            continue
        success_count = int(np.count_nonzero(selected & correctness))
        lower = wilson_lower_bound(success_count, trial_count)
        if lower < float(POLICY["minimum_component_precision_lower_bound"]):
            continue
        recall = success_count / max(int(np.count_nonzero(correctness)), 1)
        candidate = (recall, -float(threshold), success_count / trial_count)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        threshold = floor
        fallback = True
    else:
        threshold = max(floor, -best[1])
        fallback = False
    selected = values >= threshold
    return threshold, {
        "trial_count": int(values.size),
        "correct_trial_count": int(np.count_nonzero(correctness)),
        "selected_trial_count": int(np.count_nonzero(selected)),
        "selected_correct_count": int(np.count_nonzero(selected & correctness)),
        "threshold": threshold,
        "fallback": fallback,
    }


def validate_recovery_inputs(
    report: Mapping[str, Any],
    candidate_labels: np.ndarray,
    source_codes: np.ndarray,
    *,
    gaussian_count: int,
    scene: str,
) -> None:
    if report.get("source") != RECOVERY_SOURCE or report.get("contract") != RECOVERY_CONTRACT:
        raise ValueError("recovery report has the wrong contract")
    if report.get("scene") != scene:
        raise ValueError("recovery report belongs to a different scene")
    if int(report.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("recovery report has a different Gaussian count")
    if candidate_labels.shape != (gaussian_count,) or source_codes.shape != (gaussian_count,):
        raise ValueError("recovery arrays have a different Gaussian count")


def decision_counts(decisions: np.ndarray) -> dict[str, int]:
    values = np.asarray(decisions, dtype=np.uint8)
    return {
        DECISION_NAMES[code]: int(np.count_nonzero(values == code))
        for code in DECISION_NAMES
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--baseline-vote-manifest", required=True, type=Path)
    parser.add_argument("--additional-vote-manifest", required=True, type=Path)
    parser.add_argument("--hard-audit-report", required=True, type=Path)
    parser.add_argument("--hard-diagnostics", required=True, type=Path)
    parser.add_argument("--hard-confusion", required=True, type=Path)
    parser.add_argument("--recovery-report", required=True, type=Path)
    parser.add_argument("--recovery-candidate-labels", required=True, type=Path)
    parser.add_argument("--recovery-source-codes", required=True, type=Path)
    parser.add_argument("--dinov2-vote-manifest", default=None, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=65_536)
    parser.add_argument("--query-workers", type=int, default=-1)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    hashed_inputs = (
        args.baseline_vote_manifest,
        args.additional_vote_manifest,
        args.hard_audit_report,
        args.hard_diagnostics,
        args.hard_confusion,
        args.recovery_report,
        args.recovery_candidate_labels,
        args.recovery_source_codes,
        args.ontology,
    )
    if args.dinov2_vote_manifest is not None:
        hashed_inputs = (*hashed_inputs, args.dinov2_vote_manifest)
    hashes_before = {str(path): sha256_file(path) for path in hashed_inputs}
    source_ply_state = (args.source_ply.stat().st_size, args.source_ply.stat().st_mtime_ns)
    baseline_manifest = json.loads(args.baseline_vote_manifest.read_text(encoding="utf-8"))
    additional_manifest = json.loads(args.additional_vote_manifest.read_text(encoding="utf-8"))
    hard_audit = json.loads(args.hard_audit_report.read_text(encoding="utf-8"))
    recovery_report = json.loads(args.recovery_report.read_text(encoding="utf-8"))
    validate_vote_manifest(baseline_manifest, name="baseline")
    validate_vote_manifest(additional_manifest, name="additional")
    baseline_indices = [int(frame["camera_index"]) for frame in baseline_manifest["frames"]]
    additional_indices = [int(frame["camera_index"]) for frame in additional_manifest["frames"]]
    if len(set(baseline_indices)) != len(baseline_indices):
        raise ValueError("baseline vote manifest repeats a camera")
    if len(set(additional_indices)) != len(additional_indices):
        raise ValueError("additional vote manifest repeats a camera")
    if set(baseline_indices) & set(additional_indices):
        raise ValueError("additional votes repeat a baseline camera")
    gaussian_count = int(baseline_manifest.get("gaussian_count", -1))
    if gaussian_count <= 0 or int(additional_manifest.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("vote manifests have incompatible Gaussian counts")
    if hard_audit.get("source") != HARD_AUDIT_SOURCE or hard_audit.get("contract") != HARD_AUDIT_CONTRACT:
        raise ValueError("hard audit report has the wrong contract")
    if [int(value) for value in hard_audit.get("camera_indices", [])] != baseline_indices:
        raise ValueError("hard audit report has different baseline cameras")
    if int(hard_audit.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("hard audit report has a different Gaussian count")
    if [int(value) for value in recovery_report.get("baseline_camera_indices", [])] != baseline_indices:
        raise ValueError("recovery report has different baseline cameras")
    if [int(value) for value in recovery_report.get("additional_camera_indices", [])] != additional_indices:
        raise ValueError("recovery report has different additional cameras")
    for manifest in (baseline_manifest, additional_manifest):
        if Path(str(manifest.get("ply_path", ""))).resolve() != args.source_ply.resolve():
            raise ValueError("vote manifest belongs to a different source PLY")

    ontology = load_ontology(args.ontology)
    class_count = ontology.class_count
    candidate_labels = np.load(args.recovery_candidate_labels, mmap_mode="r")
    source_codes = np.load(args.recovery_source_codes, mmap_mode="r")
    validate_recovery_inputs(
        recovery_report,
        candidate_labels,
        source_codes,
        gaussian_count=gaussian_count,
        scene=args.scene,
    )
    if candidate_labels.size and int(np.max(candidate_labels)) > class_count:
        raise ValueError("recovery candidate contains a project class outside the ontology")
    with np.load(args.hard_diagnostics, allow_pickle=False) as diagnostics:
        expected_total = np.asarray(diagnostics["semantic_camera_count"], dtype=np.uint16)
        expected_maximum = np.asarray(diagnostics["winner_camera_count"], dtype=np.uint8)
        expected_status = np.asarray(diagnostics["consensus_status"], dtype=np.uint8)
    if any(
        array.shape != (gaussian_count,)
        for array in (expected_total, expected_maximum, expected_status)
    ):
        raise ValueError("hard diagnostics have a different Gaussian count")
    with np.load(args.hard_confusion, allow_pickle=False) as archive:
        confusion = np.asarray(archive["counts"], dtype=np.uint64)
    class_rows = class_reliability_rows(confusion, ontology)
    class_reliability = soft_class_reliability(class_rows, class_count)

    baseline_evidence = load_camera_evidence(
        args.baseline_vote_manifest,
        baseline_manifest,
        gaussian_count=gaussian_count,
        class_count=class_count,
    )
    additional_evidence = load_camera_evidence(
        args.additional_vote_manifest,
        additional_manifest,
        gaussian_count=gaussian_count,
        class_count=class_count,
    )
    all_evidence = [*baseline_evidence, *additional_evidence]
    dinov2_evidence: List[dict[str, Any]] = []
    dinov2_agreement_gate_used = False
    if args.dinov2_vote_manifest is not None:
        if not args.dinov2_vote_manifest.is_file():
            raise FileNotFoundError(args.dinov2_vote_manifest)
        dinov2_manifest = json.loads(
            args.dinov2_vote_manifest.read_text(encoding="utf-8")
        )
        raw_dinov2 = load_dinov2_evidence(
            args.dinov2_vote_manifest,
            dinov2_manifest,
            gaussian_count=gaussian_count,
            class_count=class_count,
        )
        dinov2_evidence = align_dinov2_evidence(all_evidence, raw_dinov2)
        dinov2_agreement_gate_used = True
    hard_counts = np.zeros((class_count + 1, gaussian_count), dtype=np.uint8)
    for item in baseline_evidence:
        supported = np.flatnonzero(item["winners"]).astype(np.int64)
        hard_counts[item["winners"][supported], supported] += np.uint8(1)
    baseline_statistics = consensus_statistics(hard_counts, chunk_size=args.chunk_size)
    for key, expected in (
        ("total", expected_total),
        ("maximum", expected_maximum),
        ("status", expected_status),
    ):
        if not np.array_equal(baseline_statistics[key], expected):
            raise RuntimeError(f"baseline hard evidence does not reproduce {key}")
    locked_labels = np.where(
        expected_status == STATUS_ACCEPTED,
        baseline_statistics["winner"],
        0,
    ).astype(np.uint16)
    if np.any(
        np.asarray(candidate_labels)[locked_labels > 0]
        != locked_labels[locked_labels > 0]
    ):
        raise RuntimeError("current recovery candidate changed an immutable hard anchor")
    for item in additional_evidence:
        supported = np.flatnonzero(item["winners"]).astype(np.int64)
        hard_counts[item["winners"][supported], supported] += np.uint8(1)
    combined_statistics = consensus_statistics(hard_counts, chunk_size=args.chunk_size)
    del hard_counts
    reliability_rows = camera_reliability_rows(
        baseline_evidence,
        additional_evidence,
        baseline_statistics,
        locked_labels,
    )
    reliabilities = {
        int(row["camera_index"]): float(row["reliability_weight"])
        for row in reliability_rows
    }

    header, vertices = vertex_data_memmap(args.source_ply)
    if not header.elements or int(header.elements[0].count) != gaussian_count:
        raise ValueError("source PLY has a different Gaussian count")
    anchor_indices = np.flatnonzero(locked_labels).astype(np.int64)
    if anchor_indices.size < int(POLICY["neighbor_count"]) + 1:
        raise ValueError("not enough immutable anchors for graph calibration")
    anchor_features = feature_subset(vertices, anchor_indices)
    anchor_points = np.asarray(anchor_features["points"], dtype=np.float64)
    anchor_tree = cKDTree(anchor_points)
    sample_count = min(
        anchor_indices.size,
        int(POLICY["calibration_anchor_sample_limit"]),
    )
    sample_positions = np.linspace(
        0,
        anchor_indices.size - 1,
        sample_count,
        dtype=np.int64,
    )
    sample_global = anchor_indices[sample_positions]
    sample_points = anchor_points[sample_positions]
    sample_distances, sample_neighbors = query_neighbors(
        anchor_tree,
        sample_points,
        neighbor_count=int(POLICY["neighbor_count"]),
        workers=args.query_workers,
        self_indices=sample_positions,
    )
    finite_distances = sample_distances[
        np.isfinite(sample_distances) & (sample_distances > 0.0)
    ]
    if finite_distances.size == 0:
        raise RuntimeError("could not derive graph distance scale")
    distance_scale = float(np.quantile(finite_distances, 0.90))

    valid_neighbors = (
        (sample_neighbors >= 0)
        & (sample_neighbors < anchor_indices.size)
        & np.isfinite(sample_distances)
    )
    calibration_src_global = np.repeat(sample_global, sample_neighbors.shape[1])[
        valid_neighbors.reshape(-1)
    ]
    calibration_dst_global = anchor_indices[
        sample_neighbors[valid_neighbors]
    ]
    calibration_distances = sample_distances[valid_neighbors]
    calibration_global = np.unique(
        np.concatenate([calibration_src_global, calibration_dst_global])
    )
    calibration_features = feature_subset(vertices, calibration_global)
    calibration_cache = semantic_cache(
        all_evidence,
        reliabilities,
        calibration_global,
        class_count=class_count,
    )
    calibration_src = np.searchsorted(calibration_global, calibration_src_global)
    calibration_dst = np.searchsorted(calibration_global, calibration_dst_global)
    feature_scales = derive_feature_scales(
        calibration_features,
        calibration_src,
        calibration_dst,
    )
    edge_affinity, _ = pair_affinity(
        calibration_features,
        calibration_cache["probabilities"],
        calibration_cache["visibility"],
        calibration_src,
        calibration_dst,
        calibration_distances,
        distance_scale=distance_scale,
        feature_scales=feature_scales,
    )
    same_class = (
        locked_labels[calibration_src_global]
        == locked_labels[calibration_dst_global]
    ) & (locked_labels[calibration_src_global] > 0)
    edge_threshold, edge_calibration = calibrate_edge_threshold(
        edge_affinity,
        same_class,
    )
    del calibration_features, calibration_cache, edge_affinity

    sample_features = {
        name: None if values is None else values[sample_positions]
        for name, values in anchor_features.items()
    }
    sample_cache = semantic_cache(
        all_evidence,
        reliabilities,
        sample_global,
        class_count=class_count,
    )
    sample_graph = build_components(
        sample_features,
        sample_cache["probabilities"],
        sample_cache["visibility"],
        edge_threshold=edge_threshold,
        distance_scale=distance_scale,
        feature_scales=feature_scales,
        neighbor_count=int(POLICY["neighbor_count"]),
        workers=args.query_workers,
    )
    sample_component_ids = sample_graph["component_ids"]
    sample_component_count = int(sample_graph["component_sizes"].size)
    holdout_camera = np.full((sample_component_count,), -1, dtype=np.int32)
    for camera_ordinal in range(len(baseline_evidence)):
        supported_components = np.unique(
            sample_component_ids[sample_cache["winners"][camera_ordinal] > 0]
        )
        if supported_components.size:
            missing = holdout_camera[supported_components] < 0
            holdout_camera[supported_components[missing]] = camera_ordinal
    if dinov2_evidence:
        sample_dinov2_cache = semantic_cache(
            dinov2_evidence,
            reliabilities,
            sample_global,
            class_count=class_count,
        )
        sample_votes = aggregate_component_votes_agreement_gated(
            sample_component_ids,
            sample_cache,
            sample_dinov2_cache,
            all_evidence,
            dinov2_evidence,
            reliabilities,
            class_count=class_count,
            excluded_camera_by_component=holdout_camera,
        )
    else:
        sample_votes = aggregate_component_votes(
            sample_component_ids,
            sample_cache,
            all_evidence,
            reliabilities,
            class_count=class_count,
            excluded_camera_by_component=holdout_camera,
        )
    sample_weighted = score_features(sample_votes["weighted"])
    sample_raw = score_features(sample_votes["raw"])
    sample_candidate = np.where(
        sample_weighted["accepted"]
        & ~sample_raw["tied"]
        & (sample_raw["winner"] == sample_weighted["winner"]),
        sample_weighted["winner"],
        0,
    ).astype(np.uint16)
    sample_truth = np.zeros((sample_component_count,), dtype=np.uint16)
    sample_purity = np.zeros((sample_component_count,), dtype=np.float32)
    sample_labels = locked_labels[sample_global]
    for component_id in range(sample_component_count):
        labels = sample_labels[sample_component_ids == component_id]
        values, counts = np.unique(labels[labels > 0], return_counts=True)
        if values.size:
            position = int(np.argmax(counts))
            sample_truth[component_id] = np.uint16(values[position])
            sample_purity[component_id] = np.float32(counts[position] / labels.size)
    sample_anchor_support = component_anchor_support(
        sample_points,
        sample_component_ids,
        anchor_tree,
        locked_labels[anchor_indices],
        neighbor_count=int(POLICY["anchor_support_neighbor_count"]),
        maximum_distance=distance_scale * 1.5,
        workers=args.query_workers,
        class_count=class_count,
        self_anchor_positions=sample_positions,
    )
    sample_scores, _ = component_scores(
        sample_candidate,
        sample_weighted,
        sample_component_ids,
        sample_graph["component_internal_affinity"],
        sample_graph["component_boundary_pressure"],
        sample_cache["probabilities"],
        sample_anchor_support,
        class_reliability,
    )
    calibration_components = (
        (sample_graph["component_sizes"] >= int(POLICY["minimum_calibration_component_size"]))
        & (sample_purity >= float(POLICY["minimum_calibration_component_purity"]))
        & (sample_candidate > 0)
        & (sample_votes["camera_count"] >= int(POLICY["minimum_component_camera_count"]))
        & (holdout_camera >= 0)
    )
    component_score_threshold, component_calibration = calibrate_component_score(
        sample_scores[calibration_components],
        sample_candidate[calibration_components] == sample_truth[calibration_components],
    )
    component_calibration.update(
        {
            "sampled_anchor_count": int(sample_count),
            "sampled_component_count": sample_component_count,
            "pure_heldout_component_count": int(np.count_nonzero(calibration_components)),
            "one_observing_baseline_camera_removed_per_component": True,
        }
    )
    del sample_cache, sample_features, sample_anchor_support

    black_indices = np.flatnonzero(np.asarray(candidate_labels) == 0).astype(np.int64)
    black_count = int(black_indices.size)
    combined_count = combined_statistics["total"][black_indices]
    observed_mask = combined_count > 0
    observed_indices = black_indices[observed_mask]
    observed_count = int(observed_indices.size)
    if observed_count:
        observed_features = feature_subset(vertices, observed_indices)
        observed_cache = semantic_cache(
            all_evidence,
            reliabilities,
            observed_indices,
            class_count=class_count,
        )
        if not np.array_equal(observed_cache["camera_count"], combined_count[observed_mask]):
            raise RuntimeError("combined semantic camera counts do not reproduce cached evidence")
        observed_graph = build_components(
            observed_features,
            observed_cache["probabilities"],
            observed_cache["visibility"],
            edge_threshold=edge_threshold,
            distance_scale=distance_scale,
            feature_scales=feature_scales,
            neighbor_count=int(POLICY["neighbor_count"]),
            workers=args.query_workers,
        )
        observed_component_ids = observed_graph["component_ids"]
        if dinov2_evidence:
            observed_dinov2_cache = semantic_cache(
                dinov2_evidence,
                reliabilities,
                observed_indices,
                class_count=class_count,
            )
            observed_votes = aggregate_component_votes_agreement_gated(
                observed_component_ids,
                observed_cache,
                observed_dinov2_cache,
                all_evidence,
                dinov2_evidence,
                reliabilities,
                class_count=class_count,
            )
        else:
            observed_votes = aggregate_component_votes(
                observed_component_ids,
                observed_cache,
                all_evidence,
                reliabilities,
                class_count=class_count,
            )
        observed_weighted = score_features(observed_votes["weighted"])
        observed_raw = score_features(observed_votes["raw"])
        component_candidate = np.where(
            observed_weighted["accepted"]
            & ~observed_raw["tied"]
            & (observed_raw["winner"] == observed_weighted["winner"]),
            observed_weighted["winner"],
            0,
        ).astype(np.uint16)
        observed_anchor_support = component_anchor_support(
            np.asarray(observed_features["points"], dtype=np.float64),
            observed_component_ids,
            anchor_tree,
            locked_labels[anchor_indices],
            neighbor_count=int(POLICY["anchor_support_neighbor_count"]),
            maximum_distance=distance_scale * 1.5,
            workers=args.query_workers,
            class_count=class_count,
        )
        component_score_values, score_parts = component_scores(
            component_candidate,
            observed_weighted,
            observed_component_ids,
            observed_graph["component_internal_affinity"],
            observed_graph["component_boundary_pressure"],
            observed_cache["probabilities"],
            observed_anchor_support,
            class_reliability,
        )
        component_decisions = np.full(
            component_candidate.shape,
            DECISION_COMPONENT_SCORE_TOO_LOW,
            dtype=np.uint8,
        )
        too_weak = observed_votes["camera_count"] < int(POLICY["minimum_component_camera_count"])
        conflict = ~too_weak & (component_candidate == 0)
        boundary = (
            ~too_weak
            & (component_candidate > 0)
            & (
                observed_graph["component_boundary_pressure"]
                > float(POLICY["maximum_boundary_pressure"])
            )
        )
        component_decisions[too_weak] = DECISION_COMPONENT_TOO_WEAK
        component_decisions[conflict] = DECISION_COMPONENT_CONFLICT
        component_decisions[boundary] = DECISION_COMPONENT_BOUNDARY_AMBIGUOUS
        eligible_components = (
            ~too_weak
            & (component_candidate > 0)
            & ~boundary
            & (component_score_values >= component_score_threshold)
        )
        component_decisions[eligible_components] = DECISION_ELIGIBLE_COMPONENT
    else:
        observed_graph = {
            "component_ids": np.empty((0,), dtype=np.int32),
            "component_sizes": np.empty((0,), dtype=np.uint32),
            "component_internal_affinity": np.empty((0,), dtype=np.float32),
            "component_boundary_pressure": np.empty((0,), dtype=np.float32),
            "edge_count": 0,
            "accepted_edge_count": 0,
        }
        observed_component_ids = observed_graph["component_ids"]
        component_candidate = np.empty((0,), dtype=np.uint16)
        component_score_values = np.empty((0,), dtype=np.float32)
        component_decisions = np.empty((0,), dtype=np.uint8)
        observed_votes = {"camera_count": np.empty((0,), dtype=np.uint16)}
        observed_weighted = {
            "winner_share": np.empty((0,), dtype=np.float32),
            "winner_margin": np.empty((0,), dtype=np.float32),
            "normalized_entropy": np.empty((0,), dtype=np.float32),
        }
        score_parts = {
            "anchor_support": np.empty((0,), dtype=np.float32),
            "class_reliability": np.empty((0,), dtype=np.float32),
        }

    decisions = np.full((black_count,), DECISION_ZERO_CAMERA, dtype=np.uint8)
    decisions[observed_mask] = component_decisions[observed_component_ids]
    component_id = np.full((black_count,), -1, dtype=np.int32)
    component_id[observed_mask] = observed_component_ids

    def expand_component(
        values: np.ndarray,
        *,
        dtype: Any,
        default: float = 0.0,
    ) -> np.ndarray:
        output = np.full((black_count,), default, dtype=dtype)
        if observed_count:
            output[observed_mask] = values[observed_component_ids]
        return output

    candidate_nodes = expand_component(component_candidate, dtype=np.uint16)
    component_size_nodes = expand_component(
        observed_graph["component_sizes"], dtype=np.uint32
    )
    camera_count_nodes = expand_component(observed_votes["camera_count"], dtype=np.uint16)
    share_nodes = expand_component(observed_weighted["winner_share"], dtype=np.float32)
    margin_nodes = expand_component(observed_weighted["winner_margin"], dtype=np.float32)
    entropy_nodes = expand_component(
        observed_weighted["normalized_entropy"], dtype=np.float32
    )
    score_nodes = expand_component(component_score_values, dtype=np.float32)
    internal_nodes = expand_component(
        observed_graph["component_internal_affinity"], dtype=np.float32
    )
    boundary_nodes = expand_component(
        observed_graph["component_boundary_pressure"], dtype=np.float32
    )
    support_nodes = expand_component(score_parts["anchor_support"], dtype=np.float32)
    reliability_nodes = expand_component(
        score_parts["class_reliability"], dtype=np.float32
    )

    hashes_after = {str(path): sha256_file(path) for path in hashed_inputs}
    if hashes_after != hashes_before:
        raise RuntimeError("audit inputs changed during execution")
    if (args.source_ply.stat().st_size, args.source_ply.stat().st_mtime_ns) != source_ply_state:
        raise RuntimeError("source PLY changed during execution")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    diagnostics_path = args.output_dir / "observed_black_component_graph_diagnostics.npz"
    np.savez_compressed(
        diagnostics_path,
        gaussian_index=black_indices.astype(np.uint32),
        combined_semantic_camera_count=combined_count.astype(np.uint16),
        observed_by_semantic_camera=observed_mask,
        component_id=component_id,
        component_size=component_size_nodes,
        component_candidate_project_id=candidate_nodes,
        component_camera_count=camera_count_nodes,
        component_winner_share=share_nodes.astype(np.float16),
        component_winner_margin=margin_nodes.astype(np.float16),
        component_normalized_entropy=entropy_nodes.astype(np.float16),
        component_score=score_nodes.astype(np.float16),
        component_internal_affinity=internal_nodes.astype(np.float16),
        component_boundary_pressure=boundary_nodes.astype(np.float16),
        component_anchor_support=support_nodes.astype(np.float16),
        component_soft_class_reliability=reliability_nodes.astype(np.float16),
        decision_code=decisions,
    )
    eligible = decisions == DECISION_ELIGIBLE_COMPONENT
    eligible_by_class = []
    for project_id in np.unique(candidate_nodes[eligible]):
        if int(project_id) <= 0:
            continue
        selected = eligible & (candidate_nodes == project_id)
        item = ontology.by_project_id[int(project_id)]
        eligible_by_class.append(
            {
                "project_id": int(project_id),
                "class": item.project_class,
                "type": item.kind,
                "eligible_gaussian_count": int(np.count_nonzero(selected)),
                "eligible_component_count": int(
                    np.unique(component_id[selected]).size
                ),
            }
        )
    eligible_by_class.sort(
        key=lambda row: (-row["eligible_gaussian_count"], row["project_id"])
    )
    available_features = ["3d_distance", "co_visibility", "semantic_distribution"]
    if anchor_features["appearance"] is not None:
        available_features.append("spherical_harmonic_dc_appearance")
    if anchor_features["log_scale"] is not None:
        available_features.append("gaussian_scale")
    if anchor_features["normal"] is not None:
        available_features.append("gaussian_orientation")
    report = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": args.scene,
        "report_only": True,
        "policy": POLICY,
        "baseline_vote_manifest": str(args.baseline_vote_manifest),
        "additional_vote_manifest": str(args.additional_vote_manifest),
        "hard_audit_report": str(args.hard_audit_report),
        "hard_diagnostics": str(args.hard_diagnostics),
        "hard_confusion": str(args.hard_confusion),
        "recovery_report": str(args.recovery_report),
        "source_ply": str(args.source_ply),
        "ontology": str(args.ontology),
        "dinov2_vote_manifest": (
            str(args.dinov2_vote_manifest)
            if args.dinov2_vote_manifest is not None
            else None
        ),
        "dinov2_agreement_gate_used": dinov2_agreement_gate_used,
        "gaussian_count": gaussian_count,
        "baseline_camera_count": len(baseline_evidence),
        "additional_camera_count": len(additional_evidence),
        "combined_camera_count": len(all_evidence),
        "immutable_hard_anchor_count": int(anchor_indices.size),
        "current_black_gaussian_count": black_count,
        "current_black_zero_combined_camera_count": int(
            np.count_nonzero(combined_count == 0)
        ),
        "current_black_single_combined_camera_count": int(
            np.count_nonzero(combined_count == 1)
        ),
        "current_black_multicamera_count": int(
            np.count_nonzero(combined_count >= 2)
        ),
        "camera_observed_black_gaussian_count": observed_count,
        "graph_distance_scale": distance_scale,
        "graph_feature_scales": feature_scales,
        "graph_edge_threshold": edge_threshold,
        "graph_calibration": edge_calibration,
        "component_score_threshold": component_score_threshold,
        "component_calibration": component_calibration,
        "component_count": int(observed_graph["component_sizes"].size),
        "component_edge_count": int(observed_graph["edge_count"]),
        "component_accepted_edge_count": int(observed_graph["accepted_edge_count"]),
        "combined_semantic_camera_count": quantile_summary(combined_count),
        "decision_counts": decision_counts(decisions),
        "eligible_report_only_gaussian_count": int(np.count_nonzero(eligible)),
        "eligible_ratio_of_current_black": (
            float(np.mean(eligible)) if eligible.size else 0.0
        ),
        "eligible_by_class": eligible_by_class,
        "camera_reliability": {
            "method": "95_percent_wilson_lower_bound",
            "component_fusion": "one_normalized_vote_per_camera_per_component",
            "per_camera": reliability_rows,
        },
        "heldout_class_reliability": class_rows,
        "component_graph_features": available_features,
        "component_graph_is_class_agnostic": True,
        "component_graph_uses_mutual_knn": True,
        "component_labels_selected_after_grouping": True,
        "candidate_labels_limited_to_component_camera_evidence": True,
        "class_reliability_is_soft_weighting": True,
        "global_class_veto_used": False,
        "anchor_support_is_component_normalized": True,
        "broad_surface_population_normalized": True,
        "zero_camera_gaussians_forced_black": True,
        "single_camera_gaussians_forced_black": False,
        "spatial_evidence_can_choose_semantic_class": False,
        "dinov3_inference_rerun": False,
        "flashsplat_lifting_rerun": False,
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "manual_class_selection_used": False,
        "scene_specific_rules": False,
        "accepted_gaussian_labels_written": False,
        "gaussian_project_class_array_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
        "diagnostic_archive": str(diagnostics_path),
        "diagnostic_archive_contains_only_current_black_gaussians": True,
        "input_sha256": hashes_before,
    }
    report_path = args.output_dir / "observed_black_component_graph_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "experiment_mode.txt").write_text(
        "cache_only=1\n"
        "report_only=1\n"
        "class_agnostic_mutual_knn_component_graph=1\n"
        "component_labels_selected_after_grouping=1\n"
        "one_normalized_camera_vote_per_component=1\n"
        "zero_camera_gaussians_forced_black=1\n"
        "single_camera_gaussians_forced_black=0\n"
        "global_class_veto_used=0\n"
        "broad_surface_population_normalized=1\n"
        "accepted_gaussian_labels_written=0\n"
        "semantic_ply_written=0\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source": SOURCE,
                "scene": args.scene,
                "current_black_gaussian_count": black_count,
                "camera_observed_black_gaussian_count": observed_count,
                "component_count": report["component_count"],
                "decision_counts": report["decision_counts"],
                "eligible_report_only_gaussian_count": report[
                    "eligible_report_only_gaussian_count"
                ],
            },
            indent=2,
        )
    )
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
