#!/usr/bin/env python3
"""Leakage-controlled held-out validation of component-graph candidates.

The component-graph audit creates report-only candidates for camera-observed
black Gaussians from all baseline and additional cameras.  This validator
recomputes the candidate for every original baseline camera after removing
that camera's votes from the combined evidence, then compares the candidate
and the leave-one-out hard-vote baseline against the cached DINOv3 map.  It
writes reports, overlays, and contact sheets only; no labels or PLY are
accepted.

Per fold, every evidence-derived component-graph array is rebuilt from the
remaining cameras: the semantic cache, the mutual-kNN graph edges, component
votes, component scores, and decisions.  Camera reliability weights are
recomputed from the remaining baseline and additional cameras.  The audited
edge and component-score thresholds remain fixed common policy calibrated
from immutable anchors, and the audited held-out class-reliability matrix is
reused unchanged.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from scripts.task1.common.flashsplat_cameras import (
    background_tensor,
    default_pipeline,
    load_cameras,
    load_flashsplat,
    load_gaussians,
    make_camera,
    point_cloud_path,
)
from scripts.task1.common.ply_utils import vertex_data_memmap
from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov3.abstention_round_trip_validation import (
    METRIC_KEYS,
    _metric_values,
    _metrics,
    update_metrics,
)
from scripts.task1.dinov3.observed_black_component_graph_audit import (
    CONTRACT as COMPONENT_CONTRACT,
    DECISION_COMPONENT_BOUNDARY_AMBIGUOUS,
    DECISION_COMPONENT_CONFLICT,
    DECISION_COMPONENT_SCORE_TOO_LOW,
    DECISION_COMPONENT_TOO_WEAK,
    DECISION_ELIGIBLE_COMPONENT,
    DECISION_NAMES,
    DECISION_ZERO_CAMERA,
    POLICY,
    SOURCE as COMPONENT_SOURCE,
    aggregate_component_votes,
    build_components,
    class_reliability_rows,
    component_anchor_support,
    component_scores,
    feature_subset,
    score_features,
    semantic_cache,
    soft_class_reliability,
)
from scripts.task1.dinov3.dinov2_second_source import (
    align_dinov2_evidence,
    aggregate_component_votes_agreement_gated,
    exclude_dinov2_camera,
    load_dinov2_evidence,
)
from scripts.task1.dinov3.recover_detected_abstentions import (
    CONTRACT as RECOVERY_CONTRACT,
    SOURCE as RECOVERY_SOURCE,
    camera_reliability_rows,
    load_camera_evidence,
    validate_vote_manifest,
)
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    CACHE_CONTRACT,
    CONTRACT as AUDIT_CONTRACT,
    SOURCE as AUDIT_SOURCE,
    STATUS_ACCEPTED,
    STATUS_NAMES,
    boundary_mask,
    consensus_statistics,
    leave_one_out_consensus,
    quantile_summary,
    render_binary_project_ids,
    save_visuals,
    sha256_file,
    validate_provenance,
)


SOURCE = "dinov3_observed_black_component_graph_round_trip_validation"
CONTRACT = "report_only_leave_one_camera_out_component_graph_comparison_v1"

FLOAT16_TOLERANCE = 2e-2


def _per_class_values() -> Dict[int, Dict[str, int]]:
    return {}


def update_per_class(
    values: Dict[int, Dict[str, int]],
    predicted: np.ndarray,
    valid: np.ndarray,
    source: np.ndarray,
    baseline_predicted: np.ndarray,
) -> None:
    for project_id in np.unique(source):
        key = int(project_id)
        if key <= 0:
            continue
        row = values.setdefault(
            key,
            {
                "source_pixels": 0,
                "baseline_agreed": 0,
                "candidate_agreed": 0,
                "newly_recovered_pixels": 0,
            },
        )
        selected = valid & (source == project_id)
        row["source_pixels"] += int(np.count_nonzero(selected))
        row["baseline_agreed"] += int(
            np.count_nonzero(selected & (baseline_predicted == project_id))
        )
        row["candidate_agreed"] += int(
            np.count_nonzero(selected & (predicted == project_id))
        )
        row["newly_recovered_pixels"] += int(
            np.count_nonzero(
                selected
                & (predicted == project_id)
                & (baseline_predicted != project_id)
            )
        )


def plurality_runner_up_candidate(
    raw_votes: np.ndarray,
    *,
    runner_up_cap: float,
    minimum_camera_count: int,
) -> dict[str, np.ndarray]:
    """Relaxed report-only rule: accept the raw plurality winner per component.

    A component's largest raw camera-vote class is accepted when the winner is
    unique, at least ``minimum_camera_count`` cameras voted, and the
    second-largest class holds at most ``runner_up_cap`` of the component's
    raw votes.  This intentionally drops the strict-majority, score, boundary,
    and agreement gates and is only used for report-only experiments.
    """

    semantic = np.asarray(raw_votes, dtype=np.float32)[1:]
    if semantic.ndim != 2:
        raise ValueError("raw votes must be classes x components")
    total = semantic.sum(axis=0, dtype=np.float32)
    maximum = semantic.max(axis=0)
    winner = semantic.argmax(axis=0).astype(np.uint16) + np.uint16(1)
    tied = (semantic == maximum[None, :]).sum(axis=0) > 1
    second = (
        np.partition(semantic, -2, axis=0)[-2]
        if semantic.shape[0] > 1
        else np.zeros_like(maximum)
    )
    runner_share = np.divide(
        second,
        total,
        out=np.zeros_like(maximum, dtype=np.float32),
        where=total > 0.0,
    )
    accepted = (
        ~tied
        & (maximum > 0.0)
        & (total >= minimum_camera_count)
        & (runner_share <= float(runner_up_cap))
    )
    share = np.divide(
        maximum,
        total,
        out=np.zeros_like(maximum, dtype=np.float32),
        where=total > 0.0,
    )
    margin = np.divide(
        maximum - second,
        total,
        out=np.zeros_like(maximum, dtype=np.float32),
        where=total > 0.0,
    )
    return {
        "winner": np.where(accepted, winner, 0).astype(np.uint16),
        "accepted": accepted,
        "tied": tied,
        "winner_share": share,
        "winner_margin": margin,
        "runner_share": runner_share,
    }


def per_gaussian_plurality_constraint(cache: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return each Gaussian's unique raw plurality winner for graph constraints.

    Exact ties abstain (class 0).  This is used only by the class-aware
    report-only grouping experiment so edges connect only Gaussians whose
    per-Gaussian raw camera plurality agrees.
    """

    raw = np.asarray(cache["raw"], dtype=np.float32)[1:]
    if raw.ndim != 2:
        raise ValueError("cache raw votes must be classes x Gaussians")
    total = raw.sum(axis=0, dtype=np.float32)
    maximum = raw.max(axis=0)
    winner = raw.argmax(axis=0).astype(np.uint16) + np.uint16(1)
    tied = (raw == maximum[None, :]).sum(axis=0) > 1
    return np.where((total > 0.0) & ~tied, winner, 0).astype(np.uint16)


def observed_component_candidate(
    vertices: Any,
    black_indices: np.ndarray,
    combined_total: np.ndarray,
    evidence: List[Dict[str, Any]],
    reliabilities: Mapping[int, float],
    class_reliability: np.ndarray,
    anchor_tree: Any,
    anchor_labels: np.ndarray,
    *,
    dinov2_evidence: Optional[List[Dict[str, Any]]] = None,
    runner_up_cap: Optional[float] = None,
    class_aware: bool = False,
    edge_threshold: float,
    distance_scale: float,
    feature_scales: Mapping[str, float],
    component_score_threshold: float,
    class_count: int,
    workers: int,
) -> Dict[str, np.ndarray]:
    """Rebuild one component-graph candidate from the given evidence.

    Returns per-black-index arrays aligned with ``black_indices``:
    combined count, observed mask, component id, candidate project id,
    component camera count, component score, decision code, and eligibility.
    Zero-camera black Gaussians always receive the zero-camera decision.
    """

    black_count = int(black_indices.size)
    combined_black = np.asarray(combined_total, dtype=np.uint16)[black_indices]
    observed_mask = combined_black > 0
    observed_indices = np.asarray(black_indices, dtype=np.int64)[observed_mask]
    observed_count = int(observed_indices.size)
    if observed_count:
        raw_winner = np.zeros((black_count,), dtype=np.uint16)
        weighted_winner = np.zeros((black_count,), dtype=np.uint16)
        raw_accepted = np.zeros((black_count,), dtype=bool)
        weighted_accepted = np.zeros((black_count,), dtype=bool)
        raw_tied = np.zeros((black_count,), dtype=bool)
        weighted_tied = np.zeros((black_count,), dtype=bool)
        observed_features = feature_subset(vertices, observed_indices)
        observed_cache = semantic_cache(
            evidence,
            reliabilities,
            observed_indices,
            class_count=class_count,
        )
        if not np.array_equal(
            observed_cache["camera_count"], combined_black[observed_mask]
        ):
            raise RuntimeError("combined semantic camera counts do not reproduce evidence")
        observed_graph = build_components(
            observed_features,
            observed_cache["probabilities"],
            observed_cache["visibility"],
            class_constraint=(
                per_gaussian_plurality_constraint(observed_cache)
                if (runner_up_cap is not None and class_aware)
                else None
            ),
            edge_threshold=edge_threshold,
            distance_scale=distance_scale,
            feature_scales=feature_scales,
            neighbor_count=int(POLICY["neighbor_count"]),
            workers=workers,
        )
        observed_component_ids = observed_graph["component_ids"]
        if dinov2_evidence and runner_up_cap is None:
            dinov2_cache = semantic_cache(
                dinov2_evidence,
                reliabilities,
                observed_indices,
                class_count=class_count,
            )
            observed_votes = aggregate_component_votes_agreement_gated(
                observed_component_ids,
                observed_cache,
                dinov2_cache,
                evidence,
                dinov2_evidence,
                reliabilities,
                class_count=class_count,
            )
        else:
            observed_votes = aggregate_component_votes(
                observed_component_ids,
                observed_cache,
                evidence,
                reliabilities,
                class_count=class_count,
            )
        observed_weighted = score_features(observed_votes["weighted"])
        observed_raw = score_features(observed_votes["raw"])
        if runner_up_cap is not None:
            relaxed = plurality_runner_up_candidate(
                observed_votes["raw"],
                runner_up_cap=runner_up_cap,
                minimum_camera_count=int(POLICY["minimum_component_camera_count"]),
            )
            component_candidate = np.where(
                relaxed["accepted"], relaxed["winner"], 0
            ).astype(np.uint16)
            observed_raw = {
                "winner": relaxed["winner"],
                "tied": relaxed["tied"],
                "accepted": relaxed["accepted"],
            }
            observed_weighted = {
                "winner": relaxed["winner"],
                "tied": relaxed["tied"],
                "accepted": relaxed["accepted"],
                "winner_share": relaxed["winner_share"],
                "winner_margin": relaxed["winner_margin"],
                "normalized_entropy": np.zeros_like(relaxed["winner_share"]),
            }
            component_score_values = np.zeros_like(
                component_candidate, dtype=np.float32
            )
            score_parts = {
                "anchor_support": np.zeros_like(
                    component_candidate, dtype=np.float32
                ),
                "class_reliability": np.zeros_like(
                    component_candidate, dtype=np.float32
                ),
            }
            component_decisions = np.full(
                component_candidate.shape,
                DECISION_COMPONENT_CONFLICT,
                dtype=np.uint8,
            )
            too_weak = (
                observed_votes["camera_count"]
                < int(POLICY["minimum_component_camera_count"])
            )
            component_decisions[too_weak] = DECISION_COMPONENT_TOO_WEAK
            eligible_components = relaxed["accepted"] & ~too_weak
            component_decisions[
                eligible_components
            ] = DECISION_ELIGIBLE_COMPONENT
        else:
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
                anchor_labels,
                neighbor_count=int(POLICY["anchor_support_neighbor_count"]),
                maximum_distance=distance_scale * 1.5,
                workers=workers,
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
            too_weak = (
                observed_votes["camera_count"]
                < int(POLICY["minimum_component_camera_count"])
            )
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
            component_decisions[
                eligible_components
            ] = DECISION_ELIGIBLE_COMPONENT
    else:
        observed_component_ids = np.empty((0,), dtype=np.int32)
        component_candidate = np.empty((0,), dtype=np.uint16)
        component_score_values = np.empty((0,), dtype=np.float32)
        component_decisions = np.empty((0,), dtype=np.uint8)
        observed_votes = {"camera_count": np.empty((0,), dtype=np.uint16)}
        raw_winner = np.zeros((black_count,), dtype=np.uint16)
        weighted_winner = np.zeros((black_count,), dtype=np.uint16)
        raw_accepted = np.zeros((black_count,), dtype=bool)
        weighted_accepted = np.zeros((black_count,), dtype=bool)
        raw_tied = np.zeros((black_count,), dtype=bool)
        weighted_tied = np.zeros((black_count,), dtype=bool)
        observed_weighted = {
            "winner_share": np.empty((0,), dtype=np.float32),
            "winner_margin": np.empty((0,), dtype=np.float32),
            "normalized_entropy": np.empty((0,), dtype=np.float32),
        }
        observed_graph = {
            "component_sizes": np.empty((0,), dtype=np.uint32),
            "component_internal_affinity": np.empty((0,), dtype=np.float32),
            "component_boundary_pressure": np.empty((0,), dtype=np.float32),
        }
        score_parts = {
            "anchor_support": np.empty((0,), dtype=np.float32),
            "class_reliability": np.empty((0,), dtype=np.float32),
        }

    decisions = np.full((black_count,), DECISION_ZERO_CAMERA, dtype=np.uint8)
    decisions[observed_mask] = component_decisions[observed_component_ids]
    component_id = np.full((black_count,), -1, dtype=np.int32)
    component_id[observed_mask] = observed_component_ids
    candidate = np.zeros((black_count,), dtype=np.uint16)
    camera_count = np.zeros((black_count,), dtype=np.uint16)
    score = np.zeros((black_count,), dtype=np.float32)
    component_size = np.zeros((black_count,), dtype=np.uint32)
    winner_share = np.zeros((black_count,), dtype=np.float32)
    winner_margin = np.zeros((black_count,), dtype=np.float32)
    normalized_entropy = np.zeros((black_count,), dtype=np.float32)
    internal_affinity = np.zeros((black_count,), dtype=np.float32)
    boundary_pressure = np.zeros((black_count,), dtype=np.float32)
    anchor_support = np.zeros((black_count,), dtype=np.float32)
    class_reliability_nodes = np.zeros((black_count,), dtype=np.float32)
    if observed_count:
        candidate[observed_mask] = component_candidate[observed_component_ids]
        raw_winner[observed_mask] = observed_raw["winner"][observed_component_ids]
        weighted_winner[observed_mask] = observed_weighted["winner"][
            observed_component_ids
        ]
        raw_accepted[observed_mask] = observed_raw["accepted"][observed_component_ids]
        weighted_accepted[observed_mask] = observed_weighted["accepted"][
            observed_component_ids
        ]
        raw_tied[observed_mask] = observed_raw["tied"][observed_component_ids]
        weighted_tied[observed_mask] = observed_weighted["tied"][
            observed_component_ids
        ]
        camera_count[observed_mask] = observed_votes["camera_count"][
            observed_component_ids
        ]
        score[observed_mask] = component_score_values[observed_component_ids]
        component_size[observed_mask] = observed_graph["component_sizes"][
            observed_component_ids
        ]
        winner_share[observed_mask] = observed_weighted["winner_share"][
            observed_component_ids
        ]
        winner_margin[observed_mask] = observed_weighted["winner_margin"][
            observed_component_ids
        ]
        normalized_entropy[observed_mask] = observed_weighted["normalized_entropy"][
            observed_component_ids
        ]
        internal_affinity[observed_mask] = observed_graph["component_internal_affinity"][
            observed_component_ids
        ]
        boundary_pressure[observed_mask] = observed_graph["component_boundary_pressure"][
            observed_component_ids
        ]
        anchor_support[observed_mask] = score_parts["anchor_support"][
            observed_component_ids
        ]
        class_reliability_nodes[observed_mask] = score_parts["class_reliability"][
            observed_component_ids
        ]
    eligible = decisions == DECISION_ELIGIBLE_COMPONENT
    return {
        "combined_camera_count": combined_black,
        "observed_mask": observed_mask,
        "component_id": component_id,
        "candidate_project_id": candidate,
        "raw_winner": raw_winner,
        "weighted_winner": weighted_winner,
        "raw_accepted": raw_accepted,
        "weighted_accepted": weighted_accepted,
        "raw_tied": raw_tied,
        "weighted_tied": weighted_tied,
        "component_camera_count": camera_count,
        "component_score": score,
        "component_size": component_size,
        "component_winner_share": winner_share,
        "component_winner_margin": winner_margin,
        "component_normalized_entropy": normalized_entropy,
        "component_internal_affinity": internal_affinity,
        "component_boundary_pressure": boundary_pressure,
        "component_anchor_support": anchor_support,
        "component_soft_class_reliability": class_reliability_nodes,
        "decision_code": decisions,
        "eligible": eligible,
    }


def _compare_float16(actual: np.ndarray, saved: np.ndarray, name: str) -> None:
    if actual.shape != saved.shape:
        raise RuntimeError(name + " array shape differs")
    if not np.allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(saved, dtype=np.float64),
        rtol=FLOAT16_TOLERANCE,
        atol=FLOAT16_TOLERANCE,
    ):
        raise RuntimeError(name + " does not reproduce the saved audit archive")


def _validate_component_audit_contract(
    report: Mapping[str, Any],
    *,
    scene: str,
    gaussian_count: int,
) -> None:
    if report.get("source") != COMPONENT_SOURCE or report.get("contract") != COMPONENT_CONTRACT:
        raise ValueError("component-graph audit report has the wrong contract")
    if report.get("scene") != scene or not report.get("report_only"):
        raise ValueError("component-graph audit report has the wrong scene or mode")
    if int(report.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("component-graph audit report has a different Gaussian count")
    for field in (
        "manual_camera_selection_used",
        "manual_gaussian_selection_used",
        "manual_class_selection_used",
        "scene_specific_rules",
        "accepted_gaussian_labels_written",
        "gaussian_project_class_array_written",
        "label_map_written",
        "semantic_ply_written",
    ):
        if report.get(field) not in (0, False):
            raise ValueError("component-graph audit report violates " + field)


def _validate_recovery_contract(
    report: Mapping[str, Any],
    *,
    scene: str,
    gaussian_count: int,
) -> None:
    if report.get("source") != RECOVERY_SOURCE or report.get("contract") != RECOVERY_CONTRACT:
        raise ValueError("recovery report has the wrong contract")
    if report.get("scene") != scene or not report.get("report_only"):
        raise ValueError("recovery report has the wrong scene or mode")
    if int(report.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("recovery report has a different Gaussian count")
    for field in (
        "immutable_anchor_labels_changed",
        "manual_camera_selection_used",
        "manual_gaussian_selection_used",
        "accepted_gaussian_labels_written",
        "label_map_written",
        "semantic_ply_written",
    ):
        if report.get(field) not in (0, False):
            raise ValueError("recovery report violates " + field)


def _validate_ontology_provenance(
    component_report: Mapping[str, Any],
    ontology_path: Path,
) -> None:
    recorded_ontology = str(component_report.get("ontology", ""))
    recorded_hashes = component_report.get("input_sha256", {})
    expected = recorded_hashes.get(recorded_ontology)
    if not expected:
        raise ValueError("component audit did not record an ontology hash")
    if sha256_file(ontology_path) != expected:
        raise ValueError("component audit belongs to a different ontology")


def _validate_camera_counts(
    component_report: Mapping[str, Any],
    baseline_evidence: Sequence[Any],
    additional_evidence: Sequence[Any],
) -> None:
    if int(component_report.get("baseline_camera_count", -1)) != len(baseline_evidence):
        raise ValueError("component audit has a different baseline camera count")
    if int(component_report.get("additional_camera_count", -1)) != len(additional_evidence):
        raise ValueError("component audit has a different additional camera count")


def _validate_reproduction(
    values: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, np.ndarray],
    report: Mapping[str, Any],
) -> None:
    exact_map = {
        "component_id": "component_id",
        "candidate_project_id": "component_candidate_project_id",
        "decision_code": "decision_code",
        "combined_camera_count": "combined_semantic_camera_count",
        "component_camera_count": "component_camera_count",
        "observed_mask": "observed_by_semantic_camera",
    }
    for name, diagnostic_key in exact_map.items():
        saved = np.asarray(diagnostics[diagnostic_key])
        actual = np.asarray(values[name])
        if actual.dtype != saved.dtype:
            actual = actual.astype(saved.dtype)
        if not np.array_equal(actual, saved):
            raise RuntimeError(name + " does not reproduce the saved audit archive")
    for name in (
        "component_score",
        "component_winner_share",
        "component_winner_margin",
        "component_normalized_entropy",
        "component_internal_affinity",
        "component_boundary_pressure",
        "component_anchor_support",
        "component_soft_class_reliability",
    ):
        _compare_float16(values[name], diagnostics[name], name)
    saved_size = np.asarray(diagnostics["component_size"], dtype=np.uint32)
    if not np.array_equal(np.asarray(values["component_size"]), saved_size):
        raise RuntimeError("component_size does not reproduce the saved audit archive")
    eligible = np.asarray(values["eligible"])
    if int(np.count_nonzero(eligible)) != int(report["eligible_report_only_gaussian_count"]):
        raise RuntimeError("eligible count does not reproduce the audit report")
    component_ids = np.asarray(values["component_id"])
    component_count = int(
        np.unique(component_ids[component_ids >= 0]).size
    )
    if component_count != int(report["component_count"]):
        raise RuntimeError("component count does not reproduce the audit report")
    decision_counts = {
        DECISION_NAMES[code]: int(np.count_nonzero(np.asarray(values["decision_code"]) == code))
        for code in DECISION_NAMES
    }
    if decision_counts != report.get("decision_counts"):
        raise RuntimeError("decision counts do not reproduce the audit report")


def _validate_baseline_reproduction(
    diagnostics: Mapping[str, np.ndarray],
    statistics: Dict[str, np.ndarray],
    audit: Mapping[str, Any],
) -> np.ndarray:
    for key, diagnostic_key in (
        ("total", "semantic_camera_count"),
        ("maximum", "winner_camera_count"),
        ("status", "consensus_status"),
    ):
        if not np.array_equal(statistics[key], diagnostics[diagnostic_key]):
            raise RuntimeError("baseline hard consensus does not reproduce " + key)
    status = statistics["status"]
    expected_counts = audit.get("gaussian_agreement", {}).get("status_counts")
    if expected_counts:
        actual_counts = {
            STATUS_NAMES[code]: int(np.count_nonzero(status == code))
            for code in sorted(STATUS_NAMES)
        }
        if actual_counts != expected_counts:
            raise RuntimeError("hard diagnostics do not reproduce audit status counts")
    return status


def _validate_baseline_metrics(
    actual: Mapping[str, float], expected: Mapping[str, Any]
) -> None:
    for key in (
        "projected_ratio",
        "agreement_of_projected",
        "boundary_agreement_of_projected",
        "interior_agreement_of_projected",
    ):
        if abs(float(actual[key]) - float(expected[key])) > 2e-5:
            raise RuntimeError("baseline round-trip metric does not reproduce " + key)


def prepare_component_validation(args: argparse.Namespace) -> Dict[str, Any]:
    """Load, validate, and reproduce every shared input for the round-trip stage.

    Returns the loaded manifests, evidence, consensus context, graph context,
    and the reproduced full-evidence component-graph candidate. The caller owns
    the returned temporary directory and counts memmap.
    """

    required = (
        args.source_view_dir / "view_manifest.json",
        args.source_view_dir / "dinov3_manifest.json",
        args.selected_cache_report,
        args.baseline_vote_manifest,
        args.additional_vote_manifest,
        args.hard_audit_report,
        args.hard_diagnostics,
        args.hard_confusion,
        args.recovery_report,
        args.candidate_labels,
        args.recovery_source_codes,
        args.component_audit_report,
        args.component_diagnostics,
        args.source_ply,
        args.ontology,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.chunk_size < 1:
        raise ValueError("chunk size must be positive")

    cache_report = json.loads(args.selected_cache_report.read_text(encoding="utf-8"))
    view_manifest = json.loads(
        (args.source_view_dir / "view_manifest.json").read_text(encoding="utf-8")
    )
    dino_manifest = json.loads(
        (args.source_view_dir / "dinov3_manifest.json").read_text(encoding="utf-8")
    )
    baseline_manifest = json.loads(
        args.baseline_vote_manifest.read_text(encoding="utf-8")
    )
    additional_manifest = json.loads(
        args.additional_vote_manifest.read_text(encoding="utf-8")
    )
    hard_audit = json.loads(args.hard_audit_report.read_text(encoding="utf-8"))
    recovery_report = json.loads(args.recovery_report.read_text(encoding="utf-8"))
    component_report = json.loads(
        args.component_audit_report.read_text(encoding="utf-8")
    )
    if cache_report.get("contract") != CACHE_CONTRACT:
        raise ValueError("source cache has the wrong contract")
    baseline_indices = validate_provenance(
        cache_report, view_manifest, dino_manifest, baseline_manifest
    )
    validate_vote_manifest(baseline_manifest, name="baseline")
    validate_vote_manifest(additional_manifest, name="additional")
    gaussian_count = int(baseline_manifest.get("gaussian_count", -1))
    if gaussian_count < 1 or int(additional_manifest.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("vote manifests have different Gaussian counts")
    additional_indices = [
        int(frame["camera_index"]) for frame in additional_manifest["frames"]
    ]
    if set(baseline_indices) & set(additional_indices):
        raise ValueError("additional vote manifest repeats a baseline camera")
    if Path(str(baseline_manifest.get("ply_path", ""))).resolve() != args.source_ply.resolve():
        raise ValueError("baseline vote manifest belongs to a different source PLY")
    if Path(str(additional_manifest.get("ply_path", ""))).resolve() != args.source_ply.resolve():
        raise ValueError("additional vote manifest belongs to a different source PLY")
    if hard_audit.get("source") != AUDIT_SOURCE or hard_audit.get("contract") != AUDIT_CONTRACT:
        raise ValueError("hard audit has the wrong contract")
    if Path(str(hard_audit.get("vote_manifest", ""))).resolve() != args.baseline_vote_manifest.resolve():
        raise ValueError("hard audit belongs to a different baseline vote manifest")
    if [int(value) for value in hard_audit.get("camera_indices", [])] != baseline_indices:
        raise ValueError("hard audit camera list differs from the selected cache")
    _validate_component_audit_contract(
        component_report, scene=args.scene, gaussian_count=gaussian_count
    )
    _validate_recovery_contract(
        recovery_report, scene=args.scene, gaussian_count=gaussian_count
    )
    if [int(value) for value in recovery_report.get("baseline_camera_indices", [])] != baseline_indices:
        raise ValueError("recovery report has different baseline cameras")
    if [int(value) for value in recovery_report.get("additional_camera_indices", [])] != additional_indices:
        raise ValueError("recovery report differs from the additional vote manifest")
    if Path(str(component_report.get("baseline_vote_manifest", ""))).resolve() != args.baseline_vote_manifest.resolve():
        raise ValueError("component audit belongs to a different baseline manifest")
    if Path(str(component_report.get("additional_vote_manifest", ""))).resolve() != args.additional_vote_manifest.resolve():
        raise ValueError("component audit belongs to a different additional manifest")
    if Path(str(component_report.get("hard_audit_report", ""))).resolve() != args.hard_audit_report.resolve():
        raise ValueError("component audit belongs to a different hard audit")
    if Path(str(component_report.get("hard_confusion", ""))).resolve() != args.hard_confusion.resolve():
        raise ValueError("component audit belongs to a different hard confusion")
    if Path(str(component_report.get("recovery_report", ""))).resolve() != args.recovery_report.resolve():
        raise ValueError("component audit belongs to a different recovery report")
    if Path(str(component_report.get("source_ply", ""))).resolve() != args.source_ply.resolve():
        raise ValueError("component audit belongs to a different source PLY")
    _validate_ontology_provenance(component_report, args.ontology)
    if Path(str(component_report.get("diagnostic_archive", ""))).resolve() != args.component_diagnostics.resolve():
        raise ValueError("component audit belongs to a different diagnostic archive")

    ontology = load_ontology(args.ontology)
    class_count = ontology.class_count
    candidate_labels = np.load(args.candidate_labels, allow_pickle=False)
    source_codes = np.load(args.recovery_source_codes, allow_pickle=False)
    if candidate_labels.shape != (gaussian_count,) or source_codes.shape != (gaussian_count,):
        raise ValueError("recovery arrays have the wrong shape")
    if np.any(candidate_labels > class_count):
        raise ValueError("recovery candidate contains a class outside the ontology")
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
    _validate_camera_counts(
        component_report, baseline_evidence, additional_evidence
    )
    all_evidence = [*baseline_evidence, *additional_evidence]
    dinov2_evidence: List[Dict[str, Any]] = []
    dinov2_vote_manifest = getattr(args, "dinov2_vote_manifest", None)
    runner_up_cap = getattr(args, "runner_up_cap", None)
    class_aware = bool(getattr(args, "class_aware", False))
    if dinov2_vote_manifest is not None:
        if not Path(dinov2_vote_manifest).is_file():
            raise FileNotFoundError(dinov2_vote_manifest)
        dinov2_manifest = json.loads(
            Path(dinov2_vote_manifest).read_text(encoding="utf-8")
        )
        raw_dinov2 = load_dinov2_evidence(
            Path(dinov2_vote_manifest),
            dinov2_manifest,
            gaussian_count=gaussian_count,
            class_count=class_count,
        )
        dinov2_evidence = align_dinov2_evidence(all_evidence, raw_dinov2)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(
        prefix="component_round_trip_", dir=args.output_dir.parent
    )
    counts = np.memmap(
        Path(temporary.name) / "baseline_counts.uint8",
        mode="w+",
        dtype=np.uint8,
        shape=(class_count + 1, gaussian_count),
    )
    counts[:] = 0
    for item in baseline_evidence:
        supported = np.flatnonzero(item["winners"]).astype(np.int64)
        if supported.size:
            counts[item["winners"][supported].astype(np.int64), supported] += np.uint8(1)
    baseline_statistics = consensus_statistics(counts, chunk_size=args.chunk_size)
    with np.load(args.hard_diagnostics, allow_pickle=False) as diagnostics:
        status = _validate_baseline_reproduction(diagnostics, baseline_statistics, hard_audit)
    locked = baseline_statistics["winner"].copy()
    locked_labels = np.where(status == STATUS_ACCEPTED, locked, 0).astype(np.uint16)
    if np.any(
        np.asarray(candidate_labels)[locked_labels > 0]
        != locked_labels[locked_labels > 0]
    ):
        raise RuntimeError("recovery candidate changed an immutable hard anchor")
    for item in additional_evidence:
        supported = np.flatnonzero(item["winners"]).astype(np.int64)
        if supported.size:
            counts[item["winners"][supported].astype(np.int64), supported] += np.uint8(1)
    for item in additional_evidence:
        winners = item["winners"]
        supported = np.flatnonzero(winners).astype(np.int64)
        if supported.size:
            counts[winners[supported].astype(np.int64), supported] -= np.uint8(1)

    reliability_rows = camera_reliability_rows(
        baseline_evidence,
        additional_evidence,
        baseline_statistics,
        locked_labels,
    )
    saved_camera_rows = component_report.get("camera_reliability", {}).get(
        "per_camera"
    )
    if isinstance(saved_camera_rows, list):
        recomputed = {
            int(row["camera_index"]): float(row["reliability_weight"])
            for row in reliability_rows
        }
        saved = {
            int(row["camera_index"]): float(row["reliability_weight"])
            for row in saved_camera_rows
        }
        if set(recomputed) != set(saved) or any(
            abs(recomputed[key] - saved[key]) > 2e-5 for key in recomputed
        ):
            raise RuntimeError("camera reliabilities do not reproduce the audit report")
    reliabilities = {
        int(row["camera_index"]): float(row["reliability_weight"])
        for row in reliability_rows
    }
    with np.load(args.hard_confusion, allow_pickle=False) as archive:
        confusion = np.asarray(archive["counts"], dtype=np.uint64)
    class_rows = class_reliability_rows(confusion, ontology)
    class_reliability = soft_class_reliability(class_rows, class_count)
    saved_class_rows = component_report.get("heldout_class_reliability")
    if isinstance(saved_class_rows, list):
        recomputed = {
            int(row["project_id"]): (
                float(row["heldout_recall_lower_bound"]),
                float(row["heldout_precision_lower_bound"]),
            )
            for row in class_rows
        }
        saved = {
            int(row["project_id"]): (
                float(row["heldout_recall_lower_bound"]),
                float(row["heldout_precision_lower_bound"]),
            )
            for row in saved_class_rows
        }
        if set(recomputed) != set(saved) or any(
            abs(recomputed[key][0] - saved[key][0]) > 2e-5
            or abs(recomputed[key][1] - saved[key][1]) > 2e-5
            for key in recomputed
        ):
            raise RuntimeError("class reliabilities do not reproduce the audit report")

    header, vertices = vertex_data_memmap(args.source_ply)
    if not header.elements or int(header.elements[0].count) != gaussian_count:
        raise ValueError("source PLY has a different Gaussian count")
    black_indices = np.flatnonzero(np.asarray(candidate_labels) == 0).astype(np.int64)
    anchor_indices = np.flatnonzero(locked_labels).astype(np.int64)
    if anchor_indices.size < int(POLICY["neighbor_count"]) + 1:
        raise ValueError("not enough immutable anchors for graph calibration")
    anchor_features = feature_subset(vertices, anchor_indices)
    anchor_points = np.asarray(anchor_features["points"], dtype=np.float64)
    from scipy.spatial import cKDTree

    anchor_tree = cKDTree(anchor_points)
    with np.load(args.component_diagnostics, allow_pickle=False) as diagnostics:
        saved_gaussian_index = np.asarray(diagnostics["gaussian_index"], dtype=np.uint32)
        saved_observed = np.asarray(diagnostics["observed_by_semantic_camera"], dtype=bool)
        saved_combined = np.asarray(
            diagnostics["combined_semantic_camera_count"], dtype=np.uint16
        )
    if not np.array_equal(black_indices, saved_gaussian_index):
        raise RuntimeError("component audit black indices do not reproduce")
    full_combined = combined_camera_counts(
        baseline_evidence, additional_evidence, gaussian_count=gaussian_count
    )
    full_values = observed_component_candidate(
        vertices,
        black_indices,
        full_combined,
        all_evidence,
        reliabilities,
        class_reliability,
        anchor_tree,
        locked_labels[anchor_indices],
        dinov2_evidence=dinov2_evidence,
        runner_up_cap=runner_up_cap,
        class_aware=class_aware,
        edge_threshold=float(component_report["graph_edge_threshold"]),
        distance_scale=float(component_report["graph_distance_scale"]),
        feature_scales=component_report["graph_feature_scales"],
        component_score_threshold=float(component_report["component_score_threshold"]),
        class_count=class_count,
        workers=args.query_workers,
    )
    if not np.array_equal(full_values["observed_mask"], saved_observed):
        raise RuntimeError("component audit observed mask does not reproduce")
    if not np.array_equal(full_values["combined_camera_count"], saved_combined):
        raise RuntimeError("component audit combined camera counts do not reproduce")
    if runner_up_cap is None:
        with np.load(args.component_diagnostics, allow_pickle=False) as diagnostics:
            _validate_reproduction(full_values, diagnostics, component_report)
    return {
        "cache_report": cache_report,
        "view_manifest": view_manifest,
        "dino_manifest": dino_manifest,
        "baseline_manifest": baseline_manifest,
        "additional_manifest": additional_manifest,
        "hard_audit": hard_audit,
        "recovery_report": recovery_report,
        "component_report": component_report,
        "baseline_indices": baseline_indices,
        "additional_indices": additional_indices,
        "gaussian_count": gaussian_count,
        "class_count": class_count,
        "ontology": ontology,
        "candidate_labels": candidate_labels,
        "source_codes": source_codes,
        "baseline_evidence": baseline_evidence,
        "additional_evidence": additional_evidence,
        "all_evidence": all_evidence,
        "dinov2_evidence": dinov2_evidence,
        "dinov2_vote_manifest": dinov2_vote_manifest,
        "runner_up_cap": runner_up_cap,
        "class_aware": class_aware,
        "counts": counts,
        "temporary": temporary,
        "baseline_statistics": baseline_statistics,
        "status": status,
        "locked_labels": locked_labels,
        "reliabilities": reliabilities,
        "class_reliability": class_reliability,
        "vertices": vertices,
        "black_indices": black_indices,
        "anchor_indices": anchor_indices,
        "anchor_tree": anchor_tree,
        "full_values": full_values,
    }


def main() -> None:
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source-view-dir", required=True, type=Path)
    parser.add_argument("--selected-cache-report", required=True, type=Path)
    parser.add_argument("--baseline-vote-manifest", required=True, type=Path)
    parser.add_argument("--additional-vote-manifest", required=True, type=Path)
    parser.add_argument("--hard-audit-report", required=True, type=Path)
    parser.add_argument("--hard-diagnostics", required=True, type=Path)
    parser.add_argument("--hard-confusion", required=True, type=Path)
    parser.add_argument("--recovery-report", required=True, type=Path)
    parser.add_argument("--candidate-labels", required=True, type=Path)
    parser.add_argument("--recovery-source-codes", required=True, type=Path)
    parser.add_argument("--component-audit-report", required=True, type=Path)
    parser.add_argument("--component-diagnostics", required=True, type=Path)
    parser.add_argument("--dinov2-vote-manifest", default=None, type=Path)
    parser.add_argument("--runner-up-cap", default=None, type=float)
    parser.add_argument("--class-aware", action="store_true")
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--flashsplat-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--chunk-size", default=100_000, type=int)
    parser.add_argument("--query-workers", default=-1, type=int)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    context = prepare_component_validation(args)
    cache_report = context["cache_report"]
    view_manifest = context["view_manifest"]
    dino_manifest = context["dino_manifest"]
    baseline_manifest = context["baseline_manifest"]
    additional_manifest = context["additional_manifest"]
    hard_audit = context["hard_audit"]
    recovery_report = context["recovery_report"]
    component_report = context["component_report"]
    baseline_indices = context["baseline_indices"]
    additional_indices = context["additional_indices"]
    gaussian_count = context["gaussian_count"]
    class_count = context["class_count"]
    ontology = context["ontology"]
    candidate_labels = context["candidate_labels"]
    source_codes = context["source_codes"]
    baseline_evidence = context["baseline_evidence"]
    additional_evidence = context["additional_evidence"]
    dinov2_evidence = context["dinov2_evidence"]
    all_evidence = context["all_evidence"]
    counts = context["counts"]
    temporary = context["temporary"]
    baseline_statistics = context["baseline_statistics"]
    status = context["status"]
    locked_labels = context["locked_labels"]
    reliabilities = context["reliabilities"]
    class_reliability = context["class_reliability"]
    vertices = context["vertices"]
    black_indices = context["black_indices"]
    anchor_indices = context["anchor_indices"]
    anchor_tree = context["anchor_tree"]
    full_values = context["full_values"]

    modules = load_flashsplat(args.flashsplat_root)
    cameras = load_cameras(args.model_path)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    if ply_path.resolve() != args.source_ply.resolve():
        raise ValueError("model PLY differs from the component audit source PLY")
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    if int(gaussians.get_xyz.shape[0]) != gaussian_count:
        raise ValueError("model Gaussian count differs from vote manifest")
    pipeline = default_pipeline()
    background = background_tensor(False)
    rgb_dir = args.source_view_dir / "rgb_renders"
    lookup = ontology.ade_to_project
    baseline_values = _metrics()
    candidate_values = _metrics()
    per_camera: List[Dict[str, Any]] = []
    per_class = _per_class_values()
    group_counts = {
        name: {"baseline": 0, "candidate": 0}
        for name in DECISION_NAMES.values()
    }
    newly_resolved = np.zeros((black_indices.size,), dtype=bool)
    newly_resolved_by_class: Dict[int, int] = {}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidate_overlay_dir = args.output_dir / "candidate_heldout_overlays"
    candidate_disagreement_dir = args.output_dir / "candidate_heldout_disagreement"
    candidate_overlay_dir.mkdir()
    candidate_disagreement_dir.mkdir()

    for frame in baseline_manifest["frames"]:
        camera_index = int(frame["camera_index"])
        fold = build_fold_candidate(
            counts=counts,
            baseline_statistics=baseline_statistics,
            status=status,
            baseline_evidence=baseline_evidence,
            additional_evidence=additional_evidence,
            dinov2_evidence=dinov2_evidence,
            runner_up_cap=args.runner_up_cap,
            class_aware=bool(args.class_aware),
            vertices=vertices,
            black_indices=black_indices,
            anchor_indices=anchor_indices,
            anchor_tree=anchor_tree,
            class_reliability=class_reliability,
            component_report=component_report,
            camera_index=camera_index,
            gaussian_count=gaussian_count,
            class_count=class_count,
            chunk_size=args.chunk_size,
            query_workers=args.query_workers,
        )
        baseline_labels = fold["baseline_labels"]
        candidate = fold["candidate"]
        fold_values = fold["fold_values"]
        fold_newly = fold["fold_newly"]
        remaining_views = fold["remaining_views"]
        newly_resolved |= fold_newly
        fold_classes = fold_values["candidate_project_id"][fold_newly]
        for project_id in np.unique(fold_classes):
            key = int(project_id)
            if key <= 0:
                continue
            newly_resolved_by_class[key] = newly_resolved_by_class.get(key, 0) + int(
                np.count_nonzero(fold_classes == project_id)
            )

        camera = make_camera(
            cameras[camera_index], modules, int(cache_report["render"]["max_width"])
        )
        baseline_predicted, baseline_valid, _ = render_binary_project_ids(
            baseline_labels, camera, gaussians, modules, pipeline, background,
            class_count=class_count,
        )
        candidate_predicted, candidate_valid, candidate_margin = render_binary_project_ids(
            candidate, camera, gaussians, modules, pipeline, background,
            class_count=class_count,
        )
        dino_frame = next(
            item for item in dino_manifest["frames"]
            if int(item["camera_index"]) == camera_index
        )
        with np.load(
            args.source_view_dir / str(dino_frame["segment_file"]), allow_pickle=False
        ) as segment:
            source_project = lookup[np.asarray(segment["class_id"], dtype=np.uint8)]
        if source_project.shape != baseline_predicted.shape:
            raise ValueError("held-out DINO map and 3D projection shapes differ")
        boundary = boundary_mask(source_project)
        update_metrics(baseline_values, baseline_predicted, baseline_valid, source_project, boundary)
        update_metrics(candidate_values, candidate_predicted, candidate_valid, source_project, boundary)
        update_per_class(
            per_class,
            candidate_predicted,
            candidate_valid,
            source_project,
            baseline_predicted,
        )
        base_rgb = np.asarray(
            Image.open(rgb_dir / str(frame["file"])).convert("RGB"), dtype=np.uint8
        )
        save_visuals(
            base_rgb,
            candidate_predicted,
            candidate_valid,
            source_project,
            ontology,
            candidate_overlay_dir / str(frame["file"]),
            candidate_disagreement_dir / str(frame["file"]),
        )
        for code in DECISION_NAMES:
            name = DECISION_NAMES[code]
            eligible_code = np.asarray(fold_values["decision_code"]) == code
            group_counts[name]["baseline"] += int(
                np.count_nonzero(eligible_code & (baseline_labels[black_indices] > 0))
            )
            group_counts[name]["candidate"] += int(
                np.count_nonzero(eligible_code & (candidate[black_indices] > 0))
            )
        per_camera.append(
            {
                "camera_index": camera_index,
                "camera_id": int(frame["camera_id"]),
                "file": str(frame["file"]),
                "baseline_projected_ratio": int(np.count_nonzero(baseline_valid))
                / int(source_project.size),
                "candidate_projected_ratio": int(np.count_nonzero(candidate_valid))
                / int(source_project.size),
                "baseline_agreement_of_projected": (
                    int(np.count_nonzero(baseline_valid & (baseline_predicted == source_project)))
                    / int(np.count_nonzero(baseline_valid))
                    if np.any(baseline_valid)
                    else 0.0
                ),
                "candidate_agreement_of_projected": (
                    int(np.count_nonzero(candidate_valid & (candidate_predicted == source_project)))
                    / int(np.count_nonzero(candidate_valid))
                    if np.any(candidate_valid)
                    else 0.0
                ),
                "candidate_binary_margin": quantile_summary(
                    candidate_margin[candidate_valid]
                ),
                "newly_resolved_gaussian_count": int(np.count_nonzero(fold_newly)),
                "remaining_camera_evidence": quantile_summary(
                    remaining_views[baseline_labels > 0]
                ),
            }
        )
        print(
            "held out camera %s: baseline=%.4f candidate=%.4f"
            % (
                camera_index,
                per_camera[-1]["baseline_agreement_of_projected"],
                per_camera[-1]["candidate_agreement_of_projected"],
            )
        )

    baseline_metrics = _metric_values(baseline_values)
    candidate_metrics = _metric_values(candidate_values)
    _validate_baseline_metrics(
        baseline_metrics,
        hard_audit["heldout_pixel_metrics"],
    )
    delta = {
        key: float(candidate_metrics[key]) - float(baseline_metrics[key])
        for key in METRIC_KEYS
    }
    per_class_recovery = []
    for project_id in sorted(per_class):
        row = per_class[project_id]
        item = ontology.by_project_id[project_id]
        source_pixels = row["source_pixels"]
        per_class_recovery.append(
            {
                "project_id": project_id,
                "class": item.project_class,
                "type": item.kind,
                "source_pixels": source_pixels,
                "baseline_agreed": row["baseline_agreed"],
                "candidate_agreed": row["candidate_agreed"],
                "newly_recovered_pixels": row["newly_recovered_pixels"],
                "baseline_agreement_of_source": (
                    row["baseline_agreed"] / source_pixels if source_pixels else 0.0
                ),
                "candidate_agreement_of_source": (
                    row["candidate_agreed"] / source_pixels if source_pixels else 0.0
                ),
                "newly_resolved_gaussians": newly_resolved_by_class.get(project_id, 0),
            }
        )
    report = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": args.scene,
        "report_only": True,
        "gaussian_count": gaussian_count,
        "camera_count": len(baseline_indices),
        "camera_indices": baseline_indices,
        "heldout_policy": "exclude_each_original_baseline_camera_from_baseline_and_component_graph_evidence",
        "heldout_calibration_policy": "recompute_all_camera_reliabilities_and_rebuild_the_component_graph_after_excluding_the_heldout_camera",
        "component_audit_report": str(args.component_audit_report),
        "component_diagnostics": str(args.component_diagnostics),
        "dinov2_vote_manifest": (
            str(context["dinov2_vote_manifest"])
            if context["dinov2_vote_manifest"] is not None
            else None
        ),
        "dinov2_agreement_gate_used": bool(dinov2_evidence),
        "full_evidence_candidate_reproduced": True,
        "component_candidate_sha256": sha256_file(args.component_diagnostics),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "delta": delta,
        "per_class_recovery": per_class_recovery,
        "group_label_counts": group_counts,
        "per_camera": per_camera,
        "candidate_recovered_count": int(np.count_nonzero(newly_resolved)),
        "candidate_recovered_ratio": (
            float(np.mean(newly_resolved)) if newly_resolved.size else 0.0
        ),
        "full_evidence_eligible_gaussian_count": int(
            component_report["eligible_report_only_gaussian_count"]
        ),
        "coverage_non_regression": candidate_metrics["projected_ratio"]
        >= baseline_metrics["projected_ratio"] - 1e-6,
        "overall_non_regression": candidate_metrics["agreement_of_projected"]
        >= baseline_metrics["agreement_of_projected"] - 1e-6,
        "interior_non_regression": candidate_metrics["interior_agreement_of_projected"]
        >= baseline_metrics["interior_agreement_of_projected"] - 1e-6,
        "boundary_non_regression": candidate_metrics["boundary_agreement_of_projected"]
        >= baseline_metrics["boundary_agreement_of_projected"] - 1e-6,
        "immutable_anchor_labels_changed": 0,
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "manual_class_selection_used": False,
        "scene_specific_rules": False,
        "accepted_gaussian_labels_written": False,
        "gaussian_project_class_array_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
    }
    (args.output_dir / "component_graph_round_trip_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output_dir / "experiment_mode.txt").write_text(
        "mode=report_only_leave_one_camera_out_component_graph_comparison\n"
        "accepted_gaussian_labels_written=0\n"
        "semantic_ply_written=0\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "scene", "camera_count", "baseline_metrics", "candidate_metrics", "delta",
        "candidate_recovered_count", "coverage_non_regression",
        "overall_non_regression", "interior_non_regression",
        "boundary_non_regression",
    )}, indent=2))
    del counts
    temporary.cleanup()
def combined_camera_counts(
    baseline_evidence: List[Dict[str, Any]],
    additional_evidence: List[Dict[str, Any]],
    *,
    gaussian_count: int,
) -> np.ndarray:
    """Return the combined per-Gaussian semantic camera count from evidence."""

    total = np.zeros((gaussian_count,), dtype=np.uint16)
    for item in [*baseline_evidence, *additional_evidence]:
        winners = np.asarray(item["winners"], dtype=np.uint16)
        total += (winners > 0).astype(np.uint16)
    return total


def build_fold_candidate(
    *,
    counts: np.ndarray,
    baseline_statistics: Dict[str, np.ndarray],
    status: np.ndarray,
    baseline_evidence: List[Dict[str, Any]],
    additional_evidence: List[Dict[str, Any]],
    dinov2_evidence: List[Dict[str, Any]],
    runner_up_cap: Optional[float] = None,
    class_aware: bool = False,
    vertices: Any,
    black_indices: np.ndarray,
    anchor_indices: np.ndarray,
    anchor_tree: Any,
    class_reliability: np.ndarray,
    component_report: Mapping[str, Any],
    camera_index: int,
    gaussian_count: int,
    class_count: int,
    chunk_size: int,
    query_workers: int,
) -> Dict[str, Any]:
    """Rebuild one leave-one-camera-out component-graph candidate.

    Removes the held-out camera from the baseline counts, recomputes the
    baseline consensus and camera reliabilities, rebuilds the component-graph
    candidate from the remaining evidence, and restores the counts matrix.
    """

    evidence = next(
        item for item in baseline_evidence if item["camera_index"] == camera_index
    )
    baseline_labels, remaining_views = leave_one_out_consensus(
        baseline_statistics, evidence["winners"]
    )
    supported = np.flatnonzero(evidence["winners"]).astype(np.int64)
    counts[evidence["winners"][supported].astype(np.int64), supported] -= np.uint8(1)
    try:
        fold_statistics = consensus_statistics(counts, chunk_size=chunk_size)
        fold_labels = np.where(
            fold_statistics["status"] == STATUS_ACCEPTED,
            fold_statistics["winner"],
            0,
        ).astype(np.uint16)
        if not np.array_equal(fold_labels, baseline_labels):
            raise RuntimeError("fold baseline consensus does not reproduce leave-one-out labels")
        fold_locked = np.where(
            (status == STATUS_ACCEPTED) & (fold_labels > 0),
            fold_labels,
            0,
        ).astype(np.uint16)
        remaining_baseline = [
            item
            for item in baseline_evidence
            if item["camera_index"] != camera_index
        ]
        fold_rows = camera_reliability_rows(
            remaining_baseline,
            additional_evidence,
            fold_statistics,
            fold_locked,
        )
        fold_reliabilities = {
            int(row["camera_index"]): float(row["reliability_weight"])
            for row in fold_rows
        }
        fold_evidence = [*remaining_baseline, *additional_evidence]
        fold_dinov2_evidence = exclude_dinov2_camera(
            dinov2_evidence, evidence["camera_id"]
        )
        fold_combined = combined_camera_counts(
            remaining_baseline,
            additional_evidence,
            gaussian_count=gaussian_count,
        )
        fold_values = observed_component_candidate(
            vertices,
            black_indices,
            fold_combined,
            fold_evidence,
            fold_reliabilities,
            class_reliability,
            anchor_tree,
            fold_locked[anchor_indices],
            dinov2_evidence=fold_dinov2_evidence,
            runner_up_cap=runner_up_cap,
            class_aware=class_aware,
            edge_threshold=float(component_report["graph_edge_threshold"]),
            distance_scale=float(component_report["graph_distance_scale"]),
            feature_scales=component_report["graph_feature_scales"],
            component_score_threshold=float(
                component_report["component_score_threshold"]
            ),
            class_count=class_count,
            workers=query_workers,
        )
    finally:
        counts[evidence["winners"][supported].astype(np.int64), supported] += np.uint8(1)
    candidate = baseline_labels.copy()
    fold_eligible = np.asarray(fold_values["eligible"])
    candidate[black_indices[fold_eligible]] = fold_values["candidate_project_id"][
        fold_eligible
    ]
    fold_newly = fold_eligible & (fold_values["candidate_project_id"] > 0)
    return {
        "evidence": evidence,
        "baseline_labels": baseline_labels,
        "remaining_views": remaining_views,
        "candidate": candidate,
        "fold_values": fold_values,
        "fold_newly": fold_newly,
    }


if __name__ == "__main__":
    main()
