#!/usr/bin/env python3
"""Audit conservative recovery of camera-observed black DINOv3 Gaussians.

The audit reuses completed hard-vote caches.  It never runs DINOv3 or
FlashSplat, and it never writes accepted semantic labels, a label map, or a
PLY.  Multi-view semantic evidence chooses every candidate identity.  Nearby
same-class anchors may only corroborate that identity; they cannot create or
change it.  Broad ontology ``stuff`` classes receive stricter semantic gates
and cannot use spatial evidence to relax them.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from scripts.task1.common.ply_utils import vertex_data_memmap
from scripts.task1.dinov2.dinov2_ontology import Ontology, load_ontology
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    CONTRACT as HARD_AUDIT_CONTRACT,
    SOURCE as HARD_AUDIT_SOURCE,
    STATUS_ACCEPTED,
    VOTE_CONTRACT,
    VOTE_SOURCE,
    collapse_camera_distribution,
    consensus_statistics,
    leave_one_out_consensus,
    quantile_summary,
    sha256_file,
)
from scripts.task1.grounding.cluster_semantic_flashsplat_proposals import (
    voxel_components,
)


SOURCE = "dinov3_observed_black_calibrated_spatial_audit"
CONTRACT = "cache_only_semantic_first_calibrated_spatial_corroboration_v1"
RECOVERY_SOURCE = "dinov3_detected_abstention_recovery_audit"
RECOVERY_CONTRACT = "immutable_hard_anchor_incremental_strict_then_calibrated_v1"

DECISION_ZERO_CAMERA = 0
DECISION_SINGLE_CAMERA = 1
DECISION_NO_WEIGHTED_MAJORITY = 2
DECISION_RAW_WEIGHTED_CONFLICT = 3
DECISION_UNCALIBRATED = 4
DECISION_CLASS_UNRELIABLE = 5
DECISION_SEMANTIC_TOO_WEAK = 6
DECISION_SPATIAL_CONFLICT = 7
DECISION_COMPONENT_TOO_SMALL = 8
DECISION_ELIGIBLE_STRONG_SEMANTIC = 9
DECISION_ELIGIBLE_SPATIAL_CORROBORATION = 10
DECISION_NAMES = {
    DECISION_ZERO_CAMERA: "remain_black_zero_combined_semantic_cameras",
    DECISION_SINGLE_CAMERA: "remain_black_single_combined_semantic_camera",
    DECISION_NO_WEIGHTED_MAJORITY: "remain_black_no_unique_weighted_strict_majority",
    DECISION_RAW_WEIGHTED_CONFLICT: "remain_black_raw_and_weighted_winners_disagree",
    DECISION_UNCALIBRATED: "remain_black_uncalibrated_evidence_bin",
    DECISION_CLASS_UNRELIABLE: "remain_black_heldout_class_reliability_gate",
    DECISION_SEMANTIC_TOO_WEAK: "remain_black_semantic_share_margin_or_entropy_gate",
    DECISION_SPATIAL_CONFLICT: "remain_black_nearby_anchors_contradict_candidate",
    DECISION_COMPONENT_TOO_SMALL: "remain_black_spatial_component_support_gate",
    DECISION_ELIGIBLE_STRONG_SEMANTIC: "eligible_report_only_strong_semantic_evidence",
    DECISION_ELIGIBLE_SPATIAL_CORROBORATION: (
        "eligible_report_only_semantic_candidate_with_spatial_corroboration"
    ),
}

EVIDENCE_BIN_NAMES = ("two", "three_to_four", "five_to_nine", "ten_or_more")
SHARE_BIN_NAMES = ("half_to_0p65", "0p65_to_0p80", "0p80_to_0p90", "0p90_to_one")
MARGIN_BIN_NAMES = ("zero_to_0p15", "0p15_to_0p30", "0p30_to_0p50", "0p50_to_one")

POLICY = {
    "minimum_semantic_camera_count": 2,
    "calibration_anchor_sample_limit": 100_000,
    "minimum_class_calibration_trials": 64,
    "minimum_class_evidence_calibration_trials": 32,
    "minimum_score_region_calibration_trials": 128,
    "thing_calibration_lower_bound_floor": 0.90,
    "thing_strong_calibration_lower_bound_floor": 0.95,
    "stuff_strong_calibration_lower_bound_floor": 0.98,
    "thing_pixel_class_lower_bound_floor": 0.55,
    "stuff_pixel_class_lower_bound_floor": 0.70,
    "catastrophic_source_recall_ceiling": 0.20,
    "maximum_catastrophic_incoming_confusion_ratio": 0.05,
    "thing_minimum_semantic_share": 0.65,
    "thing_minimum_semantic_margin": 0.15,
    "thing_maximum_normalized_entropy": 0.80,
    "thing_strong_semantic_share": 0.80,
    "thing_strong_semantic_margin": 0.40,
    "thing_strong_maximum_normalized_entropy": 0.65,
    "stuff_strong_semantic_share": 0.85,
    "stuff_strong_semantic_margin": 0.50,
    "stuff_strong_maximum_normalized_entropy": 0.45,
    "neighbor_count": 16,
    "minimum_same_class_neighbors": 4,
    "thing_minimum_same_class_fraction": 0.75,
    "stuff_minimum_same_class_fraction": 0.90,
    "maximum_same_to_competing_distance_ratio": 1.0,
    "spatial_conflict_same_class_fraction_ceiling": 0.25,
    "minimum_spatial_component_gaussians": 4,
    "component_voxel_radius_ratio": 0.5,
}


def wilson_lower_bound(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> float:
    """Return a conservative 95 percent binomial lower confidence bound."""

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    if trials == 0:
        return 0.0
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z * z / (4.0 * trials * trials)
    )
    return max(0.0, (centre - radius) / denominator)


def evidence_bin(values: np.ndarray) -> np.ndarray:
    counts = np.asarray(values)
    result = np.full(counts.shape, -1, dtype=np.int8)
    result[counts == 2] = 0
    result[(counts >= 3) & (counts <= 4)] = 1
    result[(counts >= 5) & (counts <= 9)] = 2
    result[counts >= 10] = 3
    return result


def score_bin(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    result = np.searchsorted(np.asarray(edges, dtype=np.float32), array, side="right") - 1
    return np.clip(result, 0, len(edges) - 2).astype(np.int8)


def score_features(scores: np.ndarray) -> dict[str, np.ndarray]:
    """Summarize class scores and require a unique weighted strict majority."""

    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("scores must have classes-plus-zero x items")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("scores must be finite and non-negative")
    semantic = values[1:]
    maximum = semantic.max(axis=0)
    winner = semantic.argmax(axis=0).astype(np.uint16) + np.uint16(1)
    tied = (semantic == maximum[None, :]).sum(axis=0) > 1
    if semantic.shape[0] == 1:
        second = np.zeros_like(maximum)
    else:
        second = np.partition(semantic, -2, axis=0)[-2]
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
    log_probabilities = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    log_probabilities[positive] = np.log(probabilities[positive])
    entropy = -(probabilities * log_probabilities).sum(axis=0, dtype=np.float32)
    normalizer = math.log(max(semantic.shape[0], 2))
    normalized_entropy = entropy / np.float32(normalizer)
    accepted = (maximum > 0.0) & ~tied & (maximum * 2.0 > total)
    prediction = np.where(accepted, winner, 0).astype(np.uint16)
    return {
        "prediction": prediction,
        "winner": winner,
        "accepted": accepted,
        "tied": tied,
        "total": total,
        "maximum": maximum,
        "second": second,
        "winner_share": share,
        "winner_margin": margin,
        "normalized_entropy": normalized_entropy,
    }


def collapse_camera_with_mass(
    indices: np.ndarray,
    class_ids: np.ndarray,
    weights: np.ndarray,
    *,
    gaussian_count: int,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    winners, summary = collapse_camera_distribution(
        indices,
        class_ids,
        weights,
        gaussian_count=gaussian_count,
        class_count=class_count,
    )
    idx = np.asarray(indices, dtype=np.int64)
    classes = np.asarray(class_ids, dtype=np.uint16)
    values = np.asarray(weights, dtype=np.float32)
    mass = np.zeros((gaussian_count,), dtype=np.float32)
    matched = winners[idx] == classes
    if np.any(matched):
        mass[idx[matched]] = values[matched]
    mass[winners == 0] = 0.0
    return winners, mass, summary


def validate_vote_manifest(manifest: Mapping[str, Any], *, name: str) -> None:
    if manifest.get("source") != VOTE_SOURCE or manifest.get("contract") != VOTE_CONTRACT:
        raise ValueError(f"{name} vote manifest has the wrong contract")
    for field, expected in (
        ("query_region_filtering_used", False),
        ("confidence_threshold_used", False),
        ("one_normalized_vote_per_camera", True),
    ):
        if manifest.get(field) is not expected:
            raise ValueError(f"{name} vote manifest violates {field}={expected}")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{name} vote manifest has no frames")


def load_camera_evidence(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    gaussian_count: int,
    class_count: int,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for frame in manifest["frames"]:
        vote_path = manifest_path.parent / str(frame["vote_file"])
        if not vote_path.is_file():
            raise FileNotFoundError(vote_path)
        with np.load(vote_path, allow_pickle=False) as data:
            winners, mass, summary = collapse_camera_with_mass(
                data["indices"],
                data["class_ids"],
                data["weights"],
                gaussian_count=gaussian_count,
                class_count=class_count,
            )
        evidence.append(
            {
                "camera_index": int(frame["camera_index"]),
                "camera_id": int(frame["camera_id"]),
                "file": str(frame["file"]),
                "winners": winners,
                "mass": mass,
                "collapse_summary": summary,
            }
        )
    return evidence


def camera_reliability_rows(
    baseline_evidence: Sequence[Mapping[str, Any]],
    additional_evidence: Sequence[Mapping[str, Any]],
    baseline_statistics: Mapping[str, np.ndarray],
    locked_labels: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evidence in baseline_evidence:
        winners = np.asarray(evidence["winners"], dtype=np.uint16)
        target, _remaining = leave_one_out_consensus(dict(baseline_statistics), winners)
        eligible = (target > 0) & (winners > 0)
        trials = int(np.count_nonzero(eligible))
        successes = int(np.count_nonzero(eligible & (target == winners)))
        rows.append(
            {
                "camera_index": int(evidence["camera_index"]),
                "camera_id": int(evidence["camera_id"]),
                "source": "baseline_leave_one_camera_out",
                "calibration_trial_count": trials,
                "calibration_agreement_count": successes,
                "observed_agreement": successes / trials if trials else 0.0,
                "reliability_weight": wilson_lower_bound(successes, trials),
            }
        )
    locked = np.asarray(locked_labels) > 0
    for evidence in additional_evidence:
        winners = np.asarray(evidence["winners"], dtype=np.uint16)
        eligible = locked & (winners > 0)
        trials = int(np.count_nonzero(eligible))
        successes = int(np.count_nonzero(eligible & (winners == locked_labels)))
        rows.append(
            {
                "camera_index": int(evidence["camera_index"]),
                "camera_id": int(evidence["camera_id"]),
                "source": "additional_camera_vs_immutable_anchor",
                "calibration_trial_count": trials,
                "calibration_agreement_count": successes,
                "observed_agreement": successes / trials if trials else 0.0,
                "reliability_weight": wilson_lower_bound(successes, trials),
            }
        )
    return rows


def accumulate_scores(
    evidence: Sequence[Mapping[str, Any]],
    reliabilities: Mapping[int, float],
    gaussian_indices: np.ndarray,
    *,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(gaussian_indices, dtype=np.int64)
    raw = np.zeros((class_count + 1, target.size), dtype=np.uint8)
    weighted = np.zeros((class_count + 1, target.size), dtype=np.float32)
    camera_count = np.zeros((target.size,), dtype=np.uint16)
    columns = np.arange(target.size, dtype=np.int64)
    for item in evidence:
        winners = np.asarray(item["winners"], dtype=np.uint16)[target]
        mass = np.asarray(item["mass"], dtype=np.float32)[target]
        supported = winners > 0
        if not np.any(supported):
            continue
        rows = winners[supported].astype(np.int64)
        selected_columns = columns[supported]
        raw[rows, selected_columns] += np.uint8(1)
        contribution = (
            np.float32(reliabilities[int(item["camera_index"])]) * mass[supported]
        )
        np.add.at(weighted, (rows, selected_columns), contribution)
        camera_count[supported] += np.uint16(1)
    return raw, weighted, camera_count


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
    out_distances = np.full((distances.shape[0], neighbor_count), np.inf, dtype=np.float64)
    out_indices = np.full((indices.shape[0], neighbor_count), tree.n, dtype=np.int64)
    for row in range(distances.shape[0]):
        keep = indices[row] != own[row]
        selected_distances = distances[row, keep][:neighbor_count]
        selected_indices = indices[row, keep][:neighbor_count]
        out_distances[row, : selected_distances.size] = selected_distances
        out_indices[row, : selected_indices.size] = selected_indices
    return out_distances, out_indices


def neighbor_metrics(
    distances: np.ndarray,
    neighbor_indices: np.ndarray,
    anchor_labels: np.ndarray,
    target_labels: np.ndarray,
    *,
    maximum_distance: float,
) -> dict[str, np.ndarray]:
    dist = np.asarray(distances, dtype=np.float64)
    neighbor_ids = np.asarray(neighbor_indices, dtype=np.int64)
    labels = np.asarray(anchor_labels, dtype=np.uint16)
    targets = np.asarray(target_labels, dtype=np.uint16)
    if dist.ndim != 2 or neighbor_ids.shape != dist.shape:
        raise ValueError("neighbor distances and indices must be aligned")
    if targets.shape != (dist.shape[0],):
        raise ValueError("target labels must align with neighbor rows")
    if not np.isfinite(maximum_distance) or maximum_distance <= 0.0:
        raise ValueError("maximum neighbor distance must be finite and positive")
    valid = (
        np.isfinite(dist)
        & (dist <= maximum_distance)
        & (neighbor_ids >= 0)
        & (neighbor_ids < labels.size)
    )
    safe = np.where(valid, neighbor_ids, 0)
    neighbor_labels = labels[safe]
    same = valid & (neighbor_labels == targets[:, None]) & (targets[:, None] > 0)
    competing = valid & (neighbor_labels != targets[:, None])
    valid_count = valid.sum(axis=1, dtype=np.uint8)
    same_count = same.sum(axis=1, dtype=np.uint8)
    competing_count = competing.sum(axis=1, dtype=np.uint8)
    same_fraction = np.divide(
        same_count,
        np.maximum(valid_count, 1),
        dtype=np.float32,
    )
    nearest_same = np.min(np.where(same, dist, np.inf), axis=1)
    nearest_competing = np.min(np.where(competing, dist, np.inf), axis=1)
    return {
        "valid_neighbor_count": valid_count,
        "same_class_neighbor_count": same_count,
        "competing_class_neighbor_count": competing_count,
        "same_class_fraction": same_fraction,
        "nearest_same_class_distance": nearest_same,
        "nearest_competing_class_distance": nearest_competing,
    }


def class_reliability_rows(
    confusion: np.ndarray,
    ontology: Ontology,
) -> list[dict[str, Any]]:
    counts = np.asarray(confusion, dtype=np.uint64)
    expected = (ontology.class_count + 1, ontology.class_count + 1)
    if counts.shape != expected:
        raise ValueError(f"held-out confusion matrix must have shape {expected}")
    rows: list[dict[str, Any]] = []
    source_totals = counts.sum(axis=1, dtype=np.uint64)
    predicted_totals = counts.sum(axis=0, dtype=np.uint64)
    observed_recall = np.divide(
        np.diag(counts),
        source_totals,
        out=np.zeros((counts.shape[0],), dtype=np.float64),
        where=source_totals > 0,
    )
    for item in ontology.classes:
        project_id = item.project_id
        correct = int(counts[project_id, project_id])
        recall_trials = int(source_totals[project_id])
        precision_trials = int(predicted_totals[project_id])
        catastrophic_source_id = 0
        catastrophic_incoming_ratio = 0.0
        catastrophic_incoming_count = 0
        for source_id in range(1, ontology.class_count + 1):
            if source_id == project_id or source_totals[source_id] == 0:
                continue
            if observed_recall[source_id] >= POLICY["catastrophic_source_recall_ceiling"]:
                continue
            ratio = float(counts[source_id, project_id] / source_totals[source_id])
            if ratio > catastrophic_incoming_ratio:
                catastrophic_source_id = source_id
                catastrophic_incoming_ratio = ratio
                catastrophic_incoming_count = int(counts[source_id, project_id])
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
                "largest_catastrophic_incoming_source_project_id": catastrophic_source_id,
                "largest_catastrophic_incoming_source_class": (
                    ontology.by_project_id[catastrophic_source_id].project_class
                    if catastrophic_source_id
                    else None
                ),
                "largest_catastrophic_incoming_pixel_count": catastrophic_incoming_count,
                "largest_catastrophic_incoming_ratio": catastrophic_incoming_ratio,
            }
        )
    return rows


def _calibration_row(successes: int, trials: int) -> dict[str, Any]:
    return {
        "trial_count": int(trials),
        "correct_count": int(successes),
        "observed_precision": successes / trials if trials else 0.0,
        "wilson_lower_bound": wilson_lower_bound(successes, trials),
    }


def build_calibration_tables(
    predictions: np.ndarray,
    truth: np.ndarray,
    camera_count: np.ndarray,
    winner_share: np.ndarray,
    winner_margin: np.ndarray,
    interior: np.ndarray,
    ontology: Ontology,
) -> dict[str, Any]:
    predicted = np.asarray(predictions, dtype=np.uint16)
    expected = np.asarray(truth, dtype=np.uint16)
    counts = np.asarray(camera_count, dtype=np.uint16)
    shares = np.asarray(winner_share, dtype=np.float32)
    margins = np.asarray(winner_margin, dtype=np.float32)
    interiors = np.asarray(interior, dtype=bool)
    shape = predicted.shape
    if any(array.shape != shape for array in (expected, counts, shares, margins, interiors)):
        raise ValueError("calibration arrays must align")
    count_bins = evidence_bin(counts)
    share_bins = score_bin(shares, (0.50, 0.65, 0.80, 0.90, 1.000001))
    margin_bins = score_bin(margins, (0.0, 0.15, 0.30, 0.50, 1.000001))
    valid = (predicted > 0) & (count_bins >= 0)
    correct = predicted == expected

    class_global: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    class_evidence: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    score_region: dict[tuple[int, int, int], list[int]] = defaultdict(lambda: [0, 0])
    for index in np.flatnonzero(valid):
        project_id = int(predicted[index])
        succeeded = int(bool(correct[index]))
        class_global[project_id][0] += succeeded
        class_global[project_id][1] += 1
        class_evidence[(project_id, int(count_bins[index]))][0] += succeeded
        class_evidence[(project_id, int(count_bins[index]))][1] += 1
        region = int(bool(interiors[index]))
        score_region[(int(share_bins[index]), int(margin_bins[index]), region)][0] += succeeded
        score_region[(int(share_bins[index]), int(margin_bins[index]), region)][1] += 1

    class_rows = []
    for item in ontology.classes:
        successes, trials = class_global.get(item.project_id, (0, 0))
        class_rows.append(
            {
                "project_id": item.project_id,
                "class": item.project_class,
                "type": item.kind,
                **_calibration_row(int(successes), int(trials)),
            }
        )
    class_evidence_rows = []
    for (project_id, count_bin), (successes, trials) in sorted(class_evidence.items()):
        item = ontology.by_project_id[project_id]
        class_evidence_rows.append(
            {
                "project_id": project_id,
                "class": item.project_class,
                "type": item.kind,
                "evidence_bin": EVIDENCE_BIN_NAMES[count_bin],
                **_calibration_row(successes, trials),
            }
        )
    score_region_rows = []
    for (share_index, margin_index, region), (successes, trials) in sorted(
        score_region.items()
    ):
        score_region_rows.append(
            {
                "winner_share_bin": SHARE_BIN_NAMES[share_index],
                "winner_margin_bin": MARGIN_BIN_NAMES[margin_index],
                "spatial_region": "interior" if region else "boundary_or_unanchored",
                **_calibration_row(successes, trials),
            }
        )
    return {
        "eligible_leave_one_camera_out_anchor_count": int(np.count_nonzero(valid)),
        "correct_leave_one_camera_out_anchor_count": int(np.count_nonzero(valid & correct)),
        "class_global": class_rows,
        "class_evidence": class_evidence_rows,
        "score_region": score_region_rows,
    }


def calibration_lookup(tables: Mapping[str, Any]) -> dict[str, dict[Any, Mapping[str, Any]]]:
    return {
        "class_global": {
            int(row["project_id"]): row for row in tables["class_global"]
        },
        "class_evidence": {
            (int(row["project_id"]), EVIDENCE_BIN_NAMES.index(str(row["evidence_bin"]))): row
            for row in tables["class_evidence"]
        },
        "score_region": {
            (
                SHARE_BIN_NAMES.index(str(row["winner_share_bin"])),
                MARGIN_BIN_NAMES.index(str(row["winner_margin_bin"])),
                int(row["spatial_region"] == "interior"),
            ): row
            for row in tables["score_region"]
        },
    }


def calibrated_lower_bounds(
    project_ids: np.ndarray,
    camera_count: np.ndarray,
    winner_share: np.ndarray,
    winner_margin: np.ndarray,
    interior: np.ndarray,
    tables: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    projects = np.asarray(project_ids, dtype=np.uint16)
    counts = np.asarray(camera_count, dtype=np.uint16)
    shares = np.asarray(winner_share, dtype=np.float32)
    margins = np.asarray(winner_margin, dtype=np.float32)
    interiors = np.asarray(interior, dtype=bool)
    lookup = calibration_lookup(tables)
    count_bins = evidence_bin(counts)
    share_bins = score_bin(shares, (0.50, 0.65, 0.80, 0.90, 1.000001))
    margin_bins = score_bin(margins, (0.0, 0.15, 0.30, 0.50, 1.000001))
    lower = np.zeros(projects.shape, dtype=np.float32)
    calibrated = np.zeros(projects.shape, dtype=bool)
    for index in range(projects.size):
        project_id = int(projects[index])
        count_bin = int(count_bins[index])
        if project_id <= 0 or count_bin < 0:
            continue
        class_row = lookup["class_global"].get(project_id)
        evidence_row = lookup["class_evidence"].get((project_id, count_bin))
        score_row = lookup["score_region"].get(
            (int(share_bins[index]), int(margin_bins[index]), int(interiors[index]))
        )
        if class_row is None or evidence_row is None or score_row is None:
            continue
        if int(class_row["trial_count"]) < POLICY["minimum_class_calibration_trials"]:
            continue
        if int(evidence_row["trial_count"]) < POLICY[
            "minimum_class_evidence_calibration_trials"
        ]:
            continue
        if int(score_row["trial_count"]) < POLICY[
            "minimum_score_region_calibration_trials"
        ]:
            continue
        lower[index] = np.float32(
            min(
                float(class_row["wilson_lower_bound"]),
                float(evidence_row["wilson_lower_bound"]),
                float(score_row["wilson_lower_bound"]),
            )
        )
        calibrated[index] = True
    return lower, calibrated


def component_support(
    points: np.ndarray,
    project_ids: np.ndarray,
    spatial_anchor_support: np.ndarray,
    *,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(points, dtype=np.float64)
    projects = np.asarray(project_ids, dtype=np.uint16)
    anchor_support = np.asarray(spatial_anchor_support, dtype=bool)
    if coordinates.shape != (projects.size, 3) or anchor_support.shape != projects.shape:
        raise ValueError("component inputs must align")
    sizes = np.zeros(projects.shape, dtype=np.uint32)
    anchored_counts = np.zeros(projects.shape, dtype=np.uint32)
    for project_id in np.unique(projects):
        if int(project_id) == 0:
            continue
        selected = np.flatnonzero(projects == project_id)
        component_ids, component_sizes, _stats = voxel_components(
            coordinates[selected], voxel_size
        )
        anchored = np.bincount(
            component_ids,
            weights=anchor_support[selected].astype(np.uint32),
            minlength=component_sizes.size,
        ).astype(np.uint32)
        sizes[selected] = component_sizes[component_ids].astype(np.uint32)
        anchored_counts[selected] = anchored[component_ids]
    return sizes, anchored_counts


def decide_candidates(
    project_ids: np.ndarray,
    camera_count: np.ndarray,
    raw_winners: np.ndarray,
    weighted_accepted: np.ndarray,
    winner_share: np.ndarray,
    winner_margin: np.ndarray,
    normalized_entropy: np.ndarray,
    calibrated: np.ndarray,
    calibration_lower_bound: np.ndarray,
    class_rows: Sequence[Mapping[str, Any]],
    ontology: Ontology,
    spatial: Mapping[str, np.ndarray],
    component_size: np.ndarray,
    component_anchor_count: np.ndarray,
) -> np.ndarray:
    projects = np.asarray(project_ids, dtype=np.uint16)
    counts = np.asarray(camera_count, dtype=np.uint16)
    raw = np.asarray(raw_winners, dtype=np.uint16)
    accepted = np.asarray(weighted_accepted, dtype=bool)
    share = np.asarray(winner_share, dtype=np.float32)
    margin = np.asarray(winner_margin, dtype=np.float32)
    entropy = np.asarray(normalized_entropy, dtype=np.float32)
    calibrated_mask = np.asarray(calibrated, dtype=bool)
    calibration_lower = np.asarray(calibration_lower_bound, dtype=np.float32)
    decisions = np.full(projects.shape, DECISION_NO_WEIGHTED_MAJORITY, dtype=np.uint8)
    decisions[counts == 0] = DECISION_ZERO_CAMERA
    decisions[counts == 1] = DECISION_SINGLE_CAMERA
    multicamera = counts >= POLICY["minimum_semantic_camera_count"]
    weighted = multicamera & accepted & (projects > 0)
    conflict = weighted & (raw != projects)
    decisions[conflict] = DECISION_RAW_WEIGHTED_CONFLICT
    stable = weighted & ~conflict
    decisions[stable & ~calibrated_mask] = DECISION_UNCALIBRATED
    stable &= calibrated_mask

    class_by_id = {int(row["project_id"]): row for row in class_rows}
    valid_neighbors = np.asarray(spatial["valid_neighbor_count"], dtype=np.uint8)
    same_neighbors = np.asarray(spatial["same_class_neighbor_count"], dtype=np.uint8)
    same_fraction = np.asarray(spatial["same_class_fraction"], dtype=np.float32)
    nearest_same = np.asarray(spatial["nearest_same_class_distance"], dtype=np.float64)
    nearest_competing = np.asarray(
        spatial["nearest_competing_class_distance"], dtype=np.float64
    )

    for index in np.flatnonzero(stable):
        project_id = int(projects[index])
        item = ontology.by_project_id[project_id]
        class_row = class_by_id[project_id]
        pixel_floor = (
            POLICY["stuff_pixel_class_lower_bound_floor"]
            if item.kind == "stuff"
            else POLICY["thing_pixel_class_lower_bound_floor"]
        )
        class_reliable = (
            min(
                float(class_row["heldout_recall_lower_bound"]),
                float(class_row["heldout_precision_lower_bound"]),
            )
            >= pixel_floor
            and float(class_row["largest_catastrophic_incoming_ratio"])
            <= POLICY["maximum_catastrophic_incoming_confusion_ratio"]
        )
        if not class_reliable:
            decisions[index] = DECISION_CLASS_UNRELIABLE
            continue

        same_distance_ok = nearest_same[index] <= (
            nearest_competing[index]
            * POLICY["maximum_same_to_competing_distance_ratio"]
        )
        spatial_conflict = (
            valid_neighbors[index] >= POLICY["minimum_same_class_neighbors"]
            and same_fraction[index]
            < POLICY["spatial_conflict_same_class_fraction_ceiling"]
        ) or (
            np.isfinite(nearest_competing[index])
            and np.isfinite(nearest_same[index])
            and not same_distance_ok
        )
        if spatial_conflict:
            decisions[index] = DECISION_SPATIAL_CONFLICT
            continue

        if item.kind == "stuff":
            strong = (
                calibration_lower[index]
                >= POLICY["stuff_strong_calibration_lower_bound_floor"]
                and share[index] >= POLICY["stuff_strong_semantic_share"]
                and margin[index] >= POLICY["stuff_strong_semantic_margin"]
                and entropy[index] <= POLICY["stuff_strong_maximum_normalized_entropy"]
            )
            spatial_interior = (
                same_neighbors[index] >= POLICY["minimum_same_class_neighbors"]
                and same_fraction[index] >= POLICY["stuff_minimum_same_class_fraction"]
                and same_distance_ok
            )
            if strong and spatial_interior:
                decisions[index] = DECISION_ELIGIBLE_STRONG_SEMANTIC
            else:
                decisions[index] = DECISION_SEMANTIC_TOO_WEAK
            continue

        strong = (
            calibration_lower[index]
            >= POLICY["thing_strong_calibration_lower_bound_floor"]
            and share[index] >= POLICY["thing_strong_semantic_share"]
            and margin[index] >= POLICY["thing_strong_semantic_margin"]
            and entropy[index] <= POLICY["thing_strong_maximum_normalized_entropy"]
        )
        if strong:
            decisions[index] = DECISION_ELIGIBLE_STRONG_SEMANTIC
            continue
        semantic_floor = (
            calibration_lower[index] >= POLICY["thing_calibration_lower_bound_floor"]
            and share[index] >= POLICY["thing_minimum_semantic_share"]
            and margin[index] >= POLICY["thing_minimum_semantic_margin"]
            and entropy[index] <= POLICY["thing_maximum_normalized_entropy"]
        )
        spatial_support = (
            same_neighbors[index] >= POLICY["minimum_same_class_neighbors"]
            and same_fraction[index] >= POLICY["thing_minimum_same_class_fraction"]
            and same_distance_ok
            and component_anchor_count[index] > 0
        )
        if not semantic_floor or not spatial_support:
            decisions[index] = DECISION_SEMANTIC_TOO_WEAK
        elif component_size[index] < POLICY["minimum_spatial_component_gaussians"]:
            decisions[index] = DECISION_COMPONENT_TOO_SMALL
        else:
            decisions[index] = DECISION_ELIGIBLE_SPATIAL_CORROBORATION
    return decisions


def decision_counts(decisions: np.ndarray) -> dict[str, int]:
    values = np.asarray(decisions, dtype=np.uint8)
    return {
        DECISION_NAMES[code]: int(np.count_nonzero(values == code))
        for code in sorted(DECISION_NAMES)
    }


def per_class_eligibility(
    projects: np.ndarray,
    decisions: np.ndarray,
    ontology: Ontology,
) -> list[dict[str, Any]]:
    eligible_codes = np.asarray(
        [DECISION_ELIGIBLE_STRONG_SEMANTIC, DECISION_ELIGIBLE_SPATIAL_CORROBORATION],
        dtype=np.uint8,
    )
    eligible = np.isin(decisions, eligible_codes)
    rows = []
    for project_id in np.unique(projects[eligible]):
        if int(project_id) == 0:
            continue
        item = ontology.by_project_id[int(project_id)]
        selected = eligible & (projects == project_id)
        rows.append(
            {
                "project_id": int(project_id),
                "class": item.project_class,
                "type": item.kind,
                "eligible_gaussian_count": int(np.count_nonzero(selected)),
                "strong_semantic_count": int(
                    np.count_nonzero(selected & (decisions == DECISION_ELIGIBLE_STRONG_SEMANTIC))
                ),
                "spatial_corroboration_count": int(
                    np.count_nonzero(
                        selected & (decisions == DECISION_ELIGIBLE_SPATIAL_CORROBORATION)
                    )
                ),
            }
        )
    return sorted(rows, key=lambda row: (-row["eligible_gaussian_count"], row["project_id"]))


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
    if report.get("scene") != scene or report.get("report_only") is not True:
        raise ValueError("recovery report has the wrong scene or mode")
    if int(report.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("recovery report has a different Gaussian count")
    if candidate_labels.shape != (gaussian_count,) or source_codes.shape != (gaussian_count,):
        raise ValueError("recovery arrays have the wrong shape")
    if not np.issubdtype(candidate_labels.dtype, np.integer) or not np.issubdtype(
        source_codes.dtype, np.integer
    ):
        raise ValueError("recovery arrays must be integer arrays")
    if np.any(candidate_labels < 0) or np.any(source_codes < 0):
        raise ValueError("recovery arrays must be non-negative")
    if np.any((candidate_labels == 0) != (source_codes == 0)):
        raise ValueError("recovery candidate labels and source codes disagree on black Gaussians")


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
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", default=50_000, type=int)
    parser.add_argument("--query-workers", default=4, type=int)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    required_paths = (
        args.baseline_vote_manifest,
        args.additional_vote_manifest,
        args.hard_audit_report,
        args.hard_diagnostics,
        args.hard_confusion,
        args.recovery_report,
        args.recovery_candidate_labels,
        args.recovery_source_codes,
        args.source_ply,
        args.ontology,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.chunk_size < 1 or args.query_workers < 1:
        raise ValueError("chunk size and query workers must be positive")

    hash_paths = tuple(path for path in required_paths if path != args.source_ply)
    hashes_before = {str(path): sha256_file(path) for path in hash_paths}
    source_ply_state = (args.source_ply.stat().st_size, args.source_ply.stat().st_mtime_ns)
    baseline_manifest = json.loads(args.baseline_vote_manifest.read_text(encoding="utf-8"))
    additional_manifest = json.loads(args.additional_vote_manifest.read_text(encoding="utf-8"))
    hard_audit = json.loads(args.hard_audit_report.read_text(encoding="utf-8"))
    recovery_report = json.loads(args.recovery_report.read_text(encoding="utf-8"))
    validate_vote_manifest(baseline_manifest, name="baseline")
    validate_vote_manifest(additional_manifest, name="additional")
    if hard_audit.get("source") != HARD_AUDIT_SOURCE or hard_audit.get(
        "contract"
    ) != HARD_AUDIT_CONTRACT:
        raise ValueError("hard audit report has the wrong contract")
    if Path(str(hard_audit.get("vote_manifest", ""))).resolve() != args.baseline_vote_manifest.resolve():
        raise ValueError("hard audit report belongs to a different baseline vote manifest")

    gaussian_count = int(baseline_manifest.get("gaussian_count", -1))
    if gaussian_count < 1 or int(additional_manifest.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("vote manifests have different Gaussian counts")
    baseline_indices = [int(frame["camera_index"]) for frame in baseline_manifest["frames"]]
    additional_indices = [int(frame["camera_index"]) for frame in additional_manifest["frames"]]
    if len(set(baseline_indices)) != len(baseline_indices) or len(set(additional_indices)) != len(
        additional_indices
    ):
        raise ValueError("vote manifests repeat a camera")
    if set(baseline_indices) & set(additional_indices):
        raise ValueError("additional votes repeat a baseline camera")
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
    if source_codes.size and int(np.max(source_codes)) > 3:
        raise ValueError("recovery source codes contain an unknown value")
    with np.load(args.hard_diagnostics, allow_pickle=False) as diagnostics:
        hard_total = np.asarray(diagnostics["semantic_camera_count"], dtype=np.uint16)
        hard_maximum = np.asarray(diagnostics["winner_camera_count"], dtype=np.uint8)
        hard_status = np.asarray(diagnostics["consensus_status"], dtype=np.uint8)
    if any(array.shape != (gaussian_count,) for array in (hard_total, hard_maximum, hard_status)):
        raise ValueError("hard diagnostics have a different Gaussian count")
    with np.load(args.hard_confusion, allow_pickle=False) as archive:
        confusion = np.asarray(archive["counts"], dtype=np.uint64)
    class_rows = class_reliability_rows(confusion, ontology)

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
    raw_counts = np.zeros((class_count + 1, gaussian_count), dtype=np.uint8)
    for evidence in baseline_evidence:
        supported = np.flatnonzero(evidence["winners"]).astype(np.int64)
        raw_counts[evidence["winners"][supported], supported] += np.uint8(1)
    baseline_statistics = consensus_statistics(raw_counts, chunk_size=args.chunk_size)
    for key, expected in (
        ("total", hard_total),
        ("maximum", hard_maximum),
        ("status", hard_status),
    ):
        if not np.array_equal(baseline_statistics[key], expected):
            raise RuntimeError(f"baseline hard evidence does not reproduce {key}")
    locked_labels = np.where(
        hard_status == STATUS_ACCEPTED,
        baseline_statistics["winner"],
        0,
    ).astype(np.uint16)
    if np.any(candidate_labels[locked_labels > 0] != locked_labels[locked_labels > 0]):
        raise RuntimeError("current recovery candidate changed an immutable hard anchor")
    for evidence in additional_evidence:
        supported = np.flatnonzero(evidence["winners"]).astype(np.int64)
        raw_counts[evidence["winners"][supported], supported] += np.uint8(1)
    combined_statistics = consensus_statistics(raw_counts, chunk_size=args.chunk_size)
    del raw_counts

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
    required_coordinates = {"x", "y", "z"}
    if not required_coordinates.issubset(vertices.dtype.names or ()):
        raise ValueError("source PLY lacks x/y/z coordinates")
    anchor_indices = np.flatnonzero(locked_labels).astype(np.int64)
    if anchor_indices.size < POLICY["neighbor_count"] + 1:
        raise ValueError("not enough immutable anchors for spatial calibration")
    anchor_points = np.column_stack(
        [vertices[axis][anchor_indices].astype(np.float64) for axis in ("x", "y", "z")]
    )
    anchor_labels = locked_labels[anchor_indices]
    anchor_tree = cKDTree(anchor_points)
    sample_limit = int(POLICY["calibration_anchor_sample_limit"])
    if anchor_indices.size > sample_limit:
        sample_positions = np.linspace(0, anchor_indices.size - 1, sample_limit, dtype=np.int64)
    else:
        sample_positions = np.arange(anchor_indices.size, dtype=np.int64)
    sample_indices = anchor_indices[sample_positions]
    sample_points = anchor_points[sample_positions]
    anchor_distances, anchor_neighbors = query_neighbors(
        anchor_tree,
        sample_points,
        neighbor_count=int(POLICY["neighbor_count"]),
        workers=args.query_workers,
        self_indices=sample_positions,
    )
    last_neighbor = anchor_distances[:, -1]
    finite_last = last_neighbor[np.isfinite(last_neighbor) & (last_neighbor > 0.0)]
    if finite_last.size == 0:
        raise RuntimeError("could not derive an automatic spatial radius from anchors")
    maximum_neighbor_distance = float(np.quantile(finite_last, 0.95))
    anchor_spatial = neighbor_metrics(
        anchor_distances,
        anchor_neighbors,
        anchor_labels,
        locked_labels[sample_indices],
        maximum_distance=maximum_neighbor_distance,
    )
    anchor_interior = (
        anchor_spatial["same_class_neighbor_count"] >= POLICY["minimum_same_class_neighbors"]
    ) & (
        anchor_spatial["same_class_fraction"] >= POLICY["thing_minimum_same_class_fraction"]
    )

    sample_raw, sample_weighted, sample_camera_count = accumulate_scores(
        all_evidence,
        reliabilities,
        sample_indices,
        class_count=class_count,
    )
    heldout_ordinals = np.full((sample_indices.size,), -1, dtype=np.int16)
    heldout_winners = np.zeros((sample_indices.size,), dtype=np.uint16)
    heldout_masses = np.zeros((sample_indices.size,), dtype=np.float32)
    for ordinal, evidence in enumerate(baseline_evidence):
        available = (heldout_ordinals < 0) & (evidence["winners"][sample_indices] > 0)
        heldout_ordinals[available] = np.int16(ordinal)
        heldout_winners[available] = evidence["winners"][sample_indices[available]]
        heldout_masses[available] = evidence["mass"][sample_indices[available]]
    if np.any(heldout_ordinals < 0):
        raise RuntimeError("an immutable anchor has no baseline camera to hold out")
    columns = np.arange(sample_indices.size, dtype=np.int64)
    sample_raw[heldout_winners.astype(np.int64), columns] -= np.uint8(1)
    heldout_reliability = np.asarray(
        [reliabilities[baseline_indices[int(value)]] for value in heldout_ordinals],
        dtype=np.float32,
    )
    sample_weighted[heldout_winners.astype(np.int64), columns] -= (
        heldout_reliability * heldout_masses
    )
    sample_weighted = np.maximum(sample_weighted, 0.0)
    sample_camera_count -= np.uint16(1)
    sample_raw_features = score_features(sample_raw)
    sample_weighted_features = score_features(sample_weighted)
    calibration_prediction = sample_weighted_features["prediction"].copy()
    sample_raw_unique_winner = np.where(
        sample_raw_features["tied"],
        0,
        sample_raw_features["winner"],
    ).astype(np.uint16)
    calibration_prediction[
        (sample_camera_count < POLICY["minimum_semantic_camera_count"])
        | (sample_raw_unique_winner != sample_weighted_features["winner"])
    ] = np.uint16(0)
    calibration_tables = build_calibration_tables(
        calibration_prediction,
        locked_labels[sample_indices],
        sample_camera_count,
        sample_weighted_features["winner_share"],
        sample_weighted_features["winner_margin"],
        anchor_interior,
        ontology,
    )
    del sample_raw, sample_weighted

    black_indices = np.flatnonzero(np.asarray(candidate_labels) == 0).astype(np.int64)
    black_count = int(black_indices.size)
    combined_count = combined_statistics["total"][black_indices]
    raw_winner = np.zeros((black_count,), dtype=np.uint16)
    weighted_winner = np.zeros((black_count,), dtype=np.uint16)
    weighted_accepted = np.zeros((black_count,), dtype=bool)
    winner_share = np.zeros((black_count,), dtype=np.float32)
    winner_margin = np.zeros((black_count,), dtype=np.float32)
    normalized_entropy = np.zeros((black_count,), dtype=np.float32)
    for start in range(0, black_count, args.chunk_size):
        end = min(start + args.chunk_size, black_count)
        indices = black_indices[start:end]
        local_raw, local_weighted, local_count = accumulate_scores(
            all_evidence,
            reliabilities,
            indices,
            class_count=class_count,
        )
        if not np.array_equal(local_count, combined_count[start:end]):
            raise RuntimeError("combined semantic camera counts do not reproduce cached evidence")
        raw_features = score_features(local_raw)
        weighted_features = score_features(local_weighted)
        raw_winner[start:end] = np.where(
            raw_features["tied"],
            0,
            raw_features["winner"],
        ).astype(np.uint16)
        weighted_winner[start:end] = weighted_features["prediction"]
        weighted_accepted[start:end] = weighted_features["accepted"]
        winner_share[start:end] = weighted_features["winner_share"]
        winner_margin[start:end] = weighted_features["winner_margin"]
        normalized_entropy[start:end] = weighted_features["normalized_entropy"]

    black_points = np.column_stack(
        [vertices[axis][black_indices].astype(np.float64) for axis in ("x", "y", "z")]
    )
    black_distances, black_neighbors = query_neighbors(
        anchor_tree,
        black_points,
        neighbor_count=int(POLICY["neighbor_count"]),
        workers=args.query_workers,
    )
    black_spatial = neighbor_metrics(
        black_distances,
        black_neighbors,
        anchor_labels,
        weighted_winner,
        maximum_distance=maximum_neighbor_distance,
    )
    black_interior = (
        black_spatial["same_class_neighbor_count"] >= POLICY["minimum_same_class_neighbors"]
    ) & (
        black_spatial["same_class_fraction"] >= POLICY["thing_minimum_same_class_fraction"]
    )
    calibration_lower, calibrated = calibrated_lower_bounds(
        weighted_winner,
        combined_count,
        winner_share,
        winner_margin,
        black_interior,
        calibration_tables,
    )
    spatial_anchor_support = (
        black_spatial["same_class_neighbor_count"] >= POLICY["minimum_same_class_neighbors"]
    ) & (
        black_spatial["same_class_fraction"] >= POLICY["thing_minimum_same_class_fraction"]
    )
    component_voxel_size = max(
        maximum_neighbor_distance * POLICY["component_voxel_radius_ratio"],
        np.finfo(np.float64).eps,
    )
    component_projects = weighted_winner.copy()
    stuff_ids = np.asarray(
        [item.project_id for item in ontology.classes if item.kind == "stuff"],
        dtype=np.uint16,
    )
    component_projects[np.isin(component_projects, stuff_ids)] = np.uint16(0)
    component_sizes, component_anchor_counts = component_support(
        black_points,
        component_projects,
        spatial_anchor_support,
        voxel_size=component_voxel_size,
    )
    decisions = decide_candidates(
        weighted_winner,
        combined_count,
        raw_winner,
        weighted_accepted,
        winner_share,
        winner_margin,
        normalized_entropy,
        calibrated,
        calibration_lower,
        class_rows,
        ontology,
        black_spatial,
        component_sizes,
        component_anchor_counts,
    )

    hashes_after = {str(path): sha256_file(path) for path in hash_paths}
    if hashes_after != hashes_before:
        raise RuntimeError("audit inputs changed during execution")
    if (args.source_ply.stat().st_size, args.source_ply.stat().st_mtime_ns) != source_ply_state:
        raise RuntimeError("source PLY changed during execution")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    diagnostics_path = args.output_dir / "observed_black_diagnostics.npz"
    np.savez_compressed(
        diagnostics_path,
        gaussian_index=black_indices.astype(np.uint32),
        combined_semantic_camera_count=combined_count.astype(np.uint16),
        raw_winner_project_id=raw_winner,
        weighted_winner_project_id=weighted_winner,
        weighted_winner_share=winner_share.astype(np.float16),
        weighted_winner_margin=winner_margin.astype(np.float16),
        weighted_normalized_entropy=normalized_entropy.astype(np.float16),
        calibration_lower_bound=calibration_lower.astype(np.float16),
        calibrated=calibrated,
        valid_anchor_neighbor_count=black_spatial["valid_neighbor_count"],
        same_class_anchor_neighbor_count=black_spatial["same_class_neighbor_count"],
        competing_anchor_neighbor_count=black_spatial["competing_class_neighbor_count"],
        same_class_anchor_fraction=black_spatial["same_class_fraction"].astype(np.float16),
        nearest_same_class_anchor_distance=black_spatial[
            "nearest_same_class_distance"
        ].astype(np.float32),
        nearest_competing_anchor_distance=black_spatial[
            "nearest_competing_class_distance"
        ].astype(np.float32),
        semantic_candidate_component_size=component_sizes,
        semantic_candidate_component_anchor_count=component_anchor_counts,
        decision_code=decisions,
    )
    eligible = np.isin(
        decisions,
        np.asarray(
            [DECISION_ELIGIBLE_STRONG_SEMANTIC, DECISION_ELIGIBLE_SPATIAL_CORROBORATION],
            dtype=np.uint8,
        ),
    )
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
        "gaussian_count": gaussian_count,
        "baseline_camera_count": len(baseline_evidence),
        "additional_camera_count": len(additional_evidence),
        "combined_camera_count": len(all_evidence),
        "immutable_hard_anchor_count": int(anchor_indices.size),
        "current_black_gaussian_count": black_count,
        "current_black_zero_combined_camera_count": int(np.count_nonzero(combined_count == 0)),
        "current_black_single_combined_camera_count": int(np.count_nonzero(combined_count == 1)),
        "current_black_multicamera_count": int(np.count_nonzero(combined_count >= 2)),
        "automatic_spatial_radius": maximum_neighbor_distance,
        "automatic_component_voxel_size": component_voxel_size,
        "combined_semantic_camera_count": quantile_summary(combined_count),
        "decision_counts": decision_counts(decisions),
        "eligible_report_only_gaussian_count": int(np.count_nonzero(eligible)),
        "eligible_ratio_of_current_black": float(np.mean(eligible)) if eligible.size else 0.0,
        "eligible_by_class": per_class_eligibility(weighted_winner, decisions, ontology),
        "camera_reliability": {
            "method": "95_percent_wilson_lower_bound",
            "fusion_contribution": "camera_reliability_times_within_camera_winning_mass",
            "per_camera": reliability_rows,
        },
        "anchor_calibration": {
            "method": (
                "one_deterministic_observing_baseline_camera_removed_per_immutable_anchor; "
                "candidate_compared_with_the_immutable_anchor_identity"
            ),
            "sample_count": int(sample_indices.size),
            "spatial_interior_count": int(np.count_nonzero(anchor_interior)),
            **calibration_tables,
        },
        "heldout_class_reliability": class_rows,
        "decision_codes": {str(code): name for code, name in DECISION_NAMES.items()},
        "zero_camera_gaussians_forced_black": True,
        "single_camera_gaussians_forced_black": True,
        "spatial_evidence_can_choose_semantic_class": False,
        "spatial_evidence_can_relax_stuff_semantic_thresholds": False,
        "semantic_candidate_must_precede_spatial_corroboration": True,
        "diagnostic_archive": str(diagnostics_path),
        "diagnostic_archive_contains_only_current_black_gaussians": True,
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
        "input_sha256": hashes_before,
    }
    report_path = args.output_dir / "observed_black_calibrated_spatial_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "experiment_mode.txt").write_text(
        "cache_only=1\n"
        "report_only=1\n"
        "semantic_class_selected_by_multiview_evidence_only=1\n"
        "spatial_evidence_is_corroborative_only=1\n"
        "broad_surface_spatial_relaxation=0\n"
        "accepted_gaussian_labels_written=0\n"
        "label_map_written=0\n"
        "semantic_ply_written=0\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "source": SOURCE,
                "scene": args.scene,
                "current_black_gaussian_count": black_count,
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
