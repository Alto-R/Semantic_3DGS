#!/usr/bin/env python3
"""Audit conservative DINOv3 proposal extension from immutable 3D anchors.

This stage reuses two completed report-only DINOv3 caches.  Strict components
from the anchor report remain immutable.  Proposals from cameras absent from
the anchor cache may attach only through a direct, unique, mutual-best match
to an anchor's original aggregate support.  Attached proposals never enlarge
the envelope used to match later proposals and can never merge two anchors.

The audit also separates exact unanimity, one-view-outlier consensus, and
genuinely mixed identity.  It writes diagnostics and sparse proposed fills,
but never writes semantic labels, a label map, or a PLY.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.dinov2.dinov2_ontology import Ontology, load_ontology
from scripts.task1.dinov3.associate_3d_query_regions import (
    QueryProposal,
    aggregate_component_support,
    candidate_edge,
    load_query_proposals,
    robust_semantic_distribution,
    semantic_stability,
)


SOURCE = "dinov3_anchored_component_association_audit"
CONTRACT = "report_only_dinov3_anchored_component_association_v1"
SOURCE_COMPONENT_CONTRACT = "report_only_class_agnostic_3d_association_v1"


@dataclass(frozen=True)
class AuditThresholds:
    min_shared_gaussians: int = 250
    min_iou: float = 0.05
    min_containment: float = 0.25
    min_feature_similarity: float = 0.0
    min_candidate_anchor_containment: float = 0.50
    min_unique_anchor_score_ratio: float = 1.10
    min_extension_views: int = 2
    min_extension_gaussians: int = 500

    def validate(self) -> None:
        if self.min_shared_gaussians < 1:
            raise ValueError("min_shared_gaussians must be positive")
        for name, value in (
            ("min_iou", self.min_iou),
            ("min_containment", self.min_containment),
            ("min_candidate_anchor_containment", self.min_candidate_anchor_containment),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if not -1.0 <= self.min_feature_similarity <= 1.0:
            raise ValueError("min_feature_similarity must be between -1 and one")
        if self.min_unique_anchor_score_ratio < 1.0:
            raise ValueError("min_unique_anchor_score_ratio must be at least one")
        if self.min_extension_views < 1:
            raise ValueError("min_extension_views must be positive")
        if self.min_extension_gaussians < 1:
            raise ValueError("min_extension_gaussians must be positive")


@dataclass(frozen=True)
class AnchorComponent:
    component_id: int
    ade_id: int
    project_id: int
    class_name: str
    kind: str
    record: dict[str, Any]
    proposals: tuple[QueryProposal, ...]
    envelope: QueryProposal


def _normalized_mean(values: list[np.ndarray]) -> np.ndarray:
    mean = np.mean(np.stack(values, axis=0).astype(np.float64), axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= np.finfo(np.float64).tiny:
        raise ValueError("anchor query-feature mean has zero norm")
    return (mean / norm).astype(np.float32)


def build_anchor_component(
    record: dict[str, Any],
    proposals: list[QueryProposal],
) -> AnchorComponent:
    """Build an immutable aggregate envelope from one accepted component."""

    if not bool(record.get("accepted", False)):
        raise ValueError("only accepted components can become immutable anchors")
    support_indices, support_scores = aggregate_component_support(proposals)
    if support_indices.size == 0:
        raise ValueError("anchor component has empty Gaussian support")
    probabilities = robust_semantic_distribution(
        np.stack([item.class_probabilities for item in proposals], axis=0)
    )
    envelope = QueryProposal(
        proposal_id=-int(record["component_id"]),
        frame_file=f"immutable_anchor_{int(record['component_id']):06d}",
        camera_index=-1,
        region_id=-1,
        indices=support_indices,
        counts=support_scores,
        class_probabilities=probabilities,
        no_object_probability=float(
            np.mean([item.no_object_probability for item in proposals])
        ),
        query_embedding=_normalized_mean(
            [item.query_embedding for item in proposals]
        ),
        quality=float(np.mean([item.quality for item in proposals])),
        metadata={"immutable_anchor": True},
    )
    return AnchorComponent(
        component_id=int(record["component_id"]),
        ade_id=int(record["ade_id"]),
        project_id=int(record["project_id"]),
        class_name=str(record["class"]),
        kind=str(record["kind"]),
        record=record,
        proposals=tuple(proposals),
        envelope=envelope,
    )


def load_accepted_anchors(
    component_report: dict[str, Any],
    proposals: list[QueryProposal],
) -> list[AnchorComponent]:
    if component_report.get("contract") != SOURCE_COMPONENT_CONTRACT:
        raise ValueError("anchor component report has the wrong contract")
    proposal_by_id = {item.proposal_id: item for item in proposals}
    anchors: list[AnchorComponent] = []
    for record in component_report.get("components", []):
        if not bool(record.get("accepted", False)):
            continue
        component_proposals: list[QueryProposal] = []
        for proposal_id in record.get("proposal_ids", []):
            proposal_id = int(proposal_id)
            if proposal_id not in proposal_by_id:
                raise ValueError(
                    f"anchor component references missing proposal {proposal_id}"
                )
            component_proposals.append(proposal_by_id[proposal_id])
        anchors.append(build_anchor_component(record, component_proposals))
    anchors.sort(key=lambda item: item.component_id)
    return anchors


def direct_anchor_edge(
    anchor: AnchorComponent,
    candidate: QueryProposal,
    thresholds: AuditThresholds,
) -> dict[str, Any] | None:
    """Return a direct anchor match without allowing transitive enlargement."""

    edge = candidate_edge(
        anchor.envelope,
        candidate,
        min_shared_gaussians=thresholds.min_shared_gaussians,
        min_iou=thresholds.min_iou,
        min_containment=thresholds.min_containment,
        min_feature_similarity=thresholds.min_feature_similarity,
    )
    if edge is None:
        return None
    _common, anchor_position, candidate_position = np.intersect1d(
        anchor.envelope.indices,
        candidate.indices,
        assume_unique=True,
        return_indices=True,
    )
    shared_weight = float(
        np.minimum(
            anchor.envelope.counts[anchor_position],
            candidate.counts[candidate_position],
        ).sum()
    )
    candidate_weight = float(candidate.counts.sum())
    anchor_weight = float(anchor.envelope.counts.sum())
    candidate_anchor_containment = shared_weight / max(
        candidate_weight, np.finfo(np.float32).eps
    )
    if candidate_anchor_containment < thresholds.min_candidate_anchor_containment:
        return None
    return {
        **edge,
        "anchor_component_id": anchor.component_id,
        "candidate_proposal_id": candidate.proposal_id,
        "candidate_camera_index": candidate.camera_index,
        "candidate_anchor_containment": candidate_anchor_containment,
        "anchor_weight_coverage": shared_weight
        / max(anchor_weight, np.finfo(np.float32).eps),
    }


def match_candidates_to_anchors(
    anchors: list[AnchorComponent],
    candidates: list[QueryProposal],
    thresholds: AuditThresholds,
) -> tuple[dict[int, list[tuple[QueryProposal, dict[str, Any]]]], dict[str, int]]:
    """Match proposals per camera using direct, unique, mutual-best edges."""

    thresholds.validate()
    by_camera: dict[int, list[QueryProposal]] = {}
    for proposal in candidates:
        by_camera.setdefault(proposal.camera_index, []).append(proposal)
    attached: dict[int, list[tuple[QueryProposal, dict[str, Any]]]] = {
        item.component_id: [] for item in anchors
    }
    diagnostics = {
        "candidate_camera_count": len(by_camera),
        "candidate_proposal_count": len(candidates),
        "direct_compatible_edge_count": 0,
        "ambiguous_anchor_proposal_count": 0,
        "non_mutual_best_proposal_count": 0,
        "attached_proposal_count": 0,
    }
    anchor_by_id = {item.component_id: item for item in anchors}
    for camera_index in sorted(by_camera):
        proposals = sorted(
            by_camera[camera_index], key=lambda item: item.proposal_id
        )
        edges: list[dict[str, Any]] = []
        for anchor in anchors:
            for proposal in proposals:
                edge = direct_anchor_edge(anchor, proposal, thresholds)
                if edge is not None:
                    edges.append(edge)
        diagnostics["direct_compatible_edge_count"] += len(edges)
        if not edges:
            continue

        by_proposal: dict[int, list[dict[str, Any]]] = {}
        by_anchor: dict[int, list[dict[str, Any]]] = {}
        for edge in edges:
            by_proposal.setdefault(int(edge["candidate_proposal_id"]), []).append(edge)
            by_anchor.setdefault(int(edge["anchor_component_id"]), []).append(edge)
        for values in by_proposal.values():
            values.sort(
                key=lambda item: (
                    -float(item["score"]),
                    int(item["anchor_component_id"]),
                )
            )
        for values in by_anchor.values():
            values.sort(
                key=lambda item: (
                    -float(item["score"]),
                    int(item["candidate_proposal_id"]),
                )
            )

        proposal_by_id = {item.proposal_id: item for item in proposals}
        for proposal_id in sorted(by_proposal):
            ranked = by_proposal[proposal_id]
            best = ranked[0]
            if len(ranked) > 1:
                best_score = float(best["score"])
                second_score = float(ranked[1]["score"])
                ratio = best_score / max(second_score, np.finfo(np.float32).eps)
                if ratio < thresholds.min_unique_anchor_score_ratio:
                    diagnostics["ambiguous_anchor_proposal_count"] += 1
                    continue
            anchor_id = int(best["anchor_component_id"])
            if int(by_anchor[anchor_id][0]["candidate_proposal_id"]) != proposal_id:
                diagnostics["non_mutual_best_proposal_count"] += 1
                continue
            attached[anchor_id].append((proposal_by_id[proposal_id], best))
            diagnostics["attached_proposal_count"] += 1

    for anchor_id in attached:
        attached[anchor_id].sort(
            key=lambda item: (item[0].camera_index, item[0].proposal_id)
        )
        if anchor_id not in anchor_by_id:
            raise AssertionError("attachment references an unknown anchor")
    return attached, diagnostics


def classify_numeric_consensus(
    stability: dict[str, Any],
    expected_winner: int,
) -> str:
    """Separate exact unanimity from a single stable outlying view."""

    winner = int(stability["winner"])
    view_count = int(stability["view_count"])
    winner_count = int(stability["winner_view_count"])
    leave_one_out = [int(item) for item in stability["leave_one_out_winners"]]
    if winner == expected_winner and bool(stability["stable"]):
        return "strict_unanimous"
    if (
        winner == expected_winner
        and view_count >= 3
        and winner_count == view_count - 1
        and leave_one_out
        and all(item == expected_winner for item in leave_one_out)
    ):
        return "one_view_outlier_leave_one_out_stable"
    return "mixed_or_unstable"


def classify_component_record(record: dict[str, Any]) -> str:
    """Classify a serialized source component without changing acceptance."""

    status = str(record.get("status", ""))
    if status == "accepted_stable_multiview_identity":
        return "strict_unanimous"
    if status == "abstained_insufficient_multiview_geometry":
        return "insufficient_multiview_geometry"
    stability = record.get("semantic_stability", {})
    view_count = int(stability.get("view_count", 0))
    winner_count = int(stability.get("winner_view_count", 0))
    class_name = str(record.get("class", ""))
    leave_one_out = [str(item) for item in stability.get("leave_one_out_winners", [])]
    if (
        view_count >= 3
        and winner_count == view_count - 1
        and leave_one_out
        and all(item == class_name for item in leave_one_out)
    ):
        return "one_view_outlier_leave_one_out_stable"
    return "mixed_or_unstable"


def audit_source_component_consensus(
    component_report: dict[str, Any],
) -> dict[str, Any]:
    if component_report.get("contract") != SOURCE_COMPONENT_CONTRACT:
        raise ValueError("candidate component report has the wrong contract")
    counts: dict[str, int] = {}
    one_outlier: list[dict[str, Any]] = []
    for record in component_report.get("components", []):
        tier = classify_component_record(record)
        counts[tier] = counts.get(tier, 0) + 1
        if tier == "one_view_outlier_leave_one_out_stable":
            stability = record["semantic_stability"]
            one_outlier.append(
                {
                    "component_id": int(record["component_id"]),
                    "class": str(record["class"]),
                    "source_view_count": int(record["source_view_count"]),
                    "winner_view_count": int(stability["winner_view_count"]),
                    "support_gaussian_count": int(record["support_gaussian_count"]),
                }
            )
    one_outlier.sort(
        key=lambda item: (-int(item["support_gaussian_count"]), int(item["component_id"]))
    )
    return {"tier_counts": counts, "one_view_outlier_components": one_outlier}


def _extension_indices(
    anchor: AnchorComponent,
    matching_proposals: list[QueryProposal],
    min_extension_views: int,
) -> np.ndarray:
    outside_supports: list[np.ndarray] = []
    for proposal in matching_proposals:
        outside = np.setdiff1d(
            proposal.indices,
            anchor.envelope.indices,
            assume_unique=True,
        )
        if outside.size:
            outside_supports.append(outside)
    if not outside_supports:
        return np.empty((0,), dtype=np.uint32)
    unique, counts = np.unique(np.concatenate(outside_supports), return_counts=True)
    return unique[counts >= min_extension_views].astype(np.uint32, copy=False)


def evaluate_anchor_extensions(
    anchors: list[AnchorComponent],
    attached: dict[int, list[tuple[QueryProposal, dict[str, Any]]]],
    thresholds: AuditThresholds,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    """Evaluate immutable-anchor identity and robust outside support."""

    records: list[dict[str, Any]] = []
    fills: dict[int, np.ndarray] = {}
    for anchor in anchors:
        matches = attached.get(anchor.component_id, [])
        candidate_proposals = [item[0] for item in matches]
        combined = [*anchor.proposals, *candidate_proposals]
        stability = semantic_stability(
            np.stack([item.class_probabilities for item in combined], axis=0)
        )
        consensus_tier = classify_numeric_consensus(stability, anchor.ade_id)
        matching = [
            item
            for item in candidate_proposals
            if int(np.argmax(item.class_probabilities)) == anchor.ade_id
        ]
        mismatching = [
            item
            for item in candidate_proposals
            if int(np.argmax(item.class_probabilities)) != anchor.ade_id
        ]
        fill = _extension_indices(
            anchor,
            matching,
            thresholds.min_extension_views,
        )
        consensus_ok = consensus_tier in {
            "strict_unanimous",
            "one_view_outlier_leave_one_out_stable",
        }
        accepted = consensus_ok and fill.size >= thresholds.min_extension_gaussians
        if not candidate_proposals:
            status = "no_direct_candidate_attachment"
        elif not consensus_ok:
            status = "rejected_mixed_or_unstable_identity"
        elif fill.size < thresholds.min_extension_gaussians:
            status = "rejected_insufficient_multiview_extension"
        else:
            status = "accepted_report_only_extension_proposal"
            fills[anchor.component_id] = fill

        edge_records = []
        for proposal, edge in matches:
            edge_records.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "camera_index": proposal.camera_index,
                    "score": float(edge["score"]),
                    "intersection_gaussians": int(edge["intersection_gaussians"]),
                    "weighted_iou": float(edge["weighted_iou"]),
                    "weighted_containment": float(edge["weighted_containment"]),
                    "candidate_anchor_containment": float(
                        edge["candidate_anchor_containment"]
                    ),
                    "anchor_weight_coverage": float(edge["anchor_weight_coverage"]),
                    "feature_similarity": float(edge["feature_similarity"]),
                    "candidate_winner_ade_id": int(
                        np.argmax(proposal.class_probabilities)
                    ),
                }
            )
        overlay_proposals = matching if accepted else candidate_proposals
        records.append(
            {
                "component_id": anchor.component_id,
                "anchor_component_id": anchor.component_id,
                "status": status,
                "accepted": bool(accepted),
                "class": anchor.class_name,
                "project_id": anchor.project_id,
                "ade_id": anchor.ade_id,
                "kind": anchor.kind,
                "proposal_ids": [item.proposal_id for item in overlay_proposals],
                "attached_proposal_ids": [
                    item.proposal_id for item in candidate_proposals
                ],
                "matching_identity_proposal_ids": [item.proposal_id for item in matching],
                "mismatching_identity_proposal_ids": [
                    item.proposal_id for item in mismatching
                ],
                "source_frames": sorted({item.frame_file for item in combined}),
                "source_view_count": len({item.camera_index for item in combined}),
                "anchor_source_view_count": len(
                    {item.camera_index for item in anchor.proposals}
                ),
                "attached_candidate_view_count": len(
                    {item.camera_index for item in candidate_proposals}
                ),
                "anchor_support_gaussian_count": int(anchor.envelope.indices.size),
                "support_gaussian_count": int(fill.size),
                "proposed_fill_gaussian_count": int(fill.size) if accepted else 0,
                "consensus_tier": consensus_tier,
                "semantic_stability": {
                    "winner": int(stability["winner"]),
                    "winner_view_count": int(stability["winner_view_count"]),
                    "view_count": int(stability["view_count"]),
                    "unanimous_view_winner": bool(
                        stability["unanimous_view_winner"]
                    ),
                    "leave_one_out_winners": [
                        int(item) for item in stability["leave_one_out_winners"]
                    ],
                    "stable": bool(stability["stable"]),
                },
                "direct_anchor_edges": edge_records,
            }
        )
    return records, fills


def resolve_fill_overlaps(
    records: list[dict[str, Any]],
    fills: dict[int, np.ndarray],
    vertex_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    owner_count = np.zeros((vertex_count,), dtype=np.uint16)
    for indices in fills.values():
        owner_count[indices] += 1
    overlap = owner_count > 1
    proposed = owner_count == 1
    exclusive: dict[int, np.ndarray] = {}
    record_by_id = {int(item["component_id"]): item for item in records}
    for component_id, indices in fills.items():
        kept = indices[owner_count[indices] == 1]
        exclusive[component_id] = kept
        record = record_by_id[component_id]
        record["overlapping_proposed_fill_gaussian_count"] = int(
            indices.size - kept.size
        )
        record["exclusive_proposed_fill_gaussian_count"] = int(kept.size)
        record["assigned_gaussian_count"] = int(kept.size)
    for record in records:
        record.setdefault("overlapping_proposed_fill_gaussian_count", 0)
        record.setdefault("exclusive_proposed_fill_gaussian_count", 0)
        record.setdefault("assigned_gaussian_count", 0)
    return proposed, owner_count, overlap, exclusive


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-component-report", required=True, type=Path)
    parser.add_argument("--anchor-proposal-manifest", required=True, type=Path)
    parser.add_argument("--candidate-component-report", required=True, type=Path)
    parser.add_argument("--candidate-proposal-manifest", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--min-shared-gaussians", default=250, type=int)
    parser.add_argument("--min-iou", default=0.05, type=float)
    parser.add_argument("--min-containment", default=0.25, type=float)
    parser.add_argument("--min-feature-similarity", default=0.0, type=float)
    parser.add_argument(
        "--min-candidate-anchor-containment", default=0.50, type=float
    )
    parser.add_argument("--min-unique-anchor-score-ratio", default=1.10, type=float)
    parser.add_argument("--min-extension-views", default=2, type=int)
    parser.add_argument("--min-extension-gaussians", default=500, type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    thresholds = AuditThresholds(
        min_shared_gaussians=args.min_shared_gaussians,
        min_iou=args.min_iou,
        min_containment=args.min_containment,
        min_feature_similarity=args.min_feature_similarity,
        min_candidate_anchor_containment=args.min_candidate_anchor_containment,
        min_unique_anchor_score_ratio=args.min_unique_anchor_score_ratio,
        min_extension_views=args.min_extension_views,
        min_extension_gaussians=args.min_extension_gaussians,
    )
    thresholds.validate()
    ontology: Ontology = load_ontology(args.ontology)
    anchor_report = _load_json(args.anchor_component_report)
    candidate_report = _load_json(args.candidate_component_report)
    anchor_proposals, anchor_manifest = load_query_proposals(
        args.anchor_proposal_manifest
    )
    candidate_proposals, candidate_manifest = load_query_proposals(
        args.candidate_proposal_manifest
    )
    anchor_vertex_count = int(anchor_manifest["vertex_count"])
    candidate_vertex_count = int(candidate_manifest["vertex_count"])
    if anchor_vertex_count != candidate_vertex_count:
        raise ValueError("anchor and candidate caches use different Gaussian counts")

    anchors = load_accepted_anchors(anchor_report, anchor_proposals)
    for anchor in anchors:
        if anchor.ade_id < 0 or anchor.ade_id >= len(ontology.classes):
            raise ValueError("anchor ADE class is outside the ontology")
        if ontology.classes[anchor.ade_id].project_id != anchor.project_id:
            raise ValueError("anchor ontology identity is inconsistent")
    anchor_cameras = {item.camera_index for item in anchor_proposals}
    novel_candidates = [
        item for item in candidate_proposals if item.camera_index not in anchor_cameras
    ]
    attached, match_diagnostics = match_candidates_to_anchors(
        anchors,
        novel_candidates,
        thresholds,
    )
    records, fills = evaluate_anchor_extensions(anchors, attached, thresholds)
    proposed, owner_count, overlap, exclusive = resolve_fill_overlaps(
        records,
        fills,
        anchor_vertex_count,
    )
    consensus_audit = audit_source_component_consensus(candidate_report)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "proposed_fill_mask.npy", proposed)
    np.save(args.output_dir / "proposed_fill_owner_count.npy", owner_count)
    np.save(args.output_dir / "proposed_fill_overlap_mask.npy", overlap)
    np.savez_compressed(
        args.output_dir / "proposed_fill_supports.npz",
        **{
            f"anchor_{component_id:06d}_indices": indices
            for component_id, indices in sorted(exclusive.items())
        },
    )

    accepted = [item for item in records if bool(item["accepted"])]
    report = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": args.scene,
        "anchor_component_report": str(args.anchor_component_report),
        "anchor_proposal_manifest": str(args.anchor_proposal_manifest),
        "candidate_component_report": str(args.candidate_component_report),
        "candidate_proposal_manifest": str(args.candidate_proposal_manifest),
        "ontology": str(args.ontology),
        "report_only": True,
        "semantic_labels_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
        "v5_used": False,
        "v5_labels_used": False,
        "v5_method_adapted": True,
        "dinov2_used": False,
        "association_policy": (
            "immutable_anchor_direct_unique_mutual_best_without_transitive_growth"
        ),
        "identity_policy": (
            "strict_unanimous_or_exactly_one_outlier_with_all_leave_one_out_"
            "winners_unchanged"
        ),
        "extension_policy": (
            "matching_identity_outside_anchor_support_from_multiple_new_cameras"
        ),
        "parameters": vars(thresholds),
        "vertex_count": anchor_vertex_count,
        "anchor_camera_count": len(anchor_cameras),
        "candidate_camera_count": len(
            {item.camera_index for item in candidate_proposals}
        ),
        "novel_candidate_camera_count": len(
            {item.camera_index for item in novel_candidates}
        ),
        "anchor_component_count": len(anchors),
        "accepted_extension_component_count": len(accepted),
        "proposed_fill_gaussian_count": int(np.count_nonzero(proposed)),
        "proposed_fill_overlap_gaussian_count": int(np.count_nonzero(overlap)),
        "match_diagnostics": match_diagnostics,
        "candidate_source_consensus_audit": consensus_audit,
        "components": records,
    }
    (args.output_dir / "anchored_component_association_audit.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
