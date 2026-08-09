#!/usr/bin/env python3
"""Associate DINOv3 query regions in 3D and fuse soft semantics.

The association graph is class-agnostic.  Semantic identity is assigned only
after physical components have been formed from shared Gaussian support.
This stage is permanently report-only: it computes candidate assignment
statistics in memory and writes JSON, never Gaussian labels or a semantic PLY.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.dinov2.dinov2_ontology import Ontology, load_ontology


@dataclass(frozen=True)
class QueryProposal:
    proposal_id: int
    frame_file: str
    camera_index: int
    region_id: int
    indices: np.ndarray
    counts: np.ndarray
    class_probabilities: np.ndarray
    no_object_probability: float
    query_embedding: np.ndarray
    quality: float
    metadata: dict[str, Any]


def support_overlap(left: QueryProposal, right: QueryProposal) -> dict[str, float | int]:
    common, left_position, right_position = np.intersect1d(
        left.indices,
        right.indices,
        assume_unique=True,
        return_indices=True,
    )
    intersection = int(common.shape[0])
    union = int(left.indices.shape[0] + right.indices.shape[0] - intersection)
    binary_iou = intersection / float(max(union, 1))
    binary_containment = intersection / float(
        max(min(left.indices.shape[0], right.indices.shape[0]), 1)
    )
    left_weight = float(left.counts.sum())
    right_weight = float(right.counts.sum())
    shared_weight = float(
        np.minimum(
            left.counts[left_position],
            right.counts[right_position],
        ).sum()
    )
    weighted_union = left_weight + right_weight - shared_weight
    weighted_iou = shared_weight / max(weighted_union, np.finfo(np.float32).eps)
    weighted_containment = shared_weight / max(
        min(left_weight, right_weight),
        np.finfo(np.float32).eps,
    )
    return {
        "intersection_gaussians": intersection,
        "binary_iou": binary_iou,
        "binary_containment": binary_containment,
        "weighted_iou": weighted_iou,
        "weighted_containment": weighted_containment,
    }


def feature_similarity(left: QueryProposal, right: QueryProposal) -> float:
    return float(np.clip(np.dot(left.query_embedding, right.query_embedding), -1.0, 1.0))


def candidate_edge(
    left: QueryProposal,
    right: QueryProposal,
    *,
    min_shared_gaussians: int,
    min_iou: float,
    min_containment: float,
    min_feature_similarity: float,
) -> dict[str, Any] | None:
    if left.frame_file == right.frame_file:
        return None
    overlap = support_overlap(left, right)
    if int(overlap["intersection_gaussians"]) < min_shared_gaussians:
        return None
    if (
        float(overlap["weighted_iou"]) < min_iou
        and float(overlap["weighted_containment"]) < min_containment
    ):
        return None
    similarity = feature_similarity(left, right)
    if similarity < min_feature_similarity:
        return None
    geometry = max(
        float(overlap["weighted_iou"]),
        0.5 * float(overlap["weighted_containment"]),
    )
    score = geometry * (0.75 + 0.25 * max(similarity, 0.0))
    return {
        "left_proposal_id": left.proposal_id,
        "right_proposal_id": right.proposal_id,
        "score": score,
        "feature_similarity": similarity,
        **overlap,
    }


def mutual_best_edges(
    proposals: list[QueryProposal],
    *,
    min_shared_gaussians: int,
    min_iou: float,
    min_containment: float,
    min_feature_similarity: float,
) -> list[dict[str, Any]]:
    by_frame: dict[str, list[QueryProposal]] = {}
    for proposal in proposals:
        by_frame.setdefault(proposal.frame_file, []).append(proposal)
    frames = sorted(by_frame)
    kept: list[dict[str, Any]] = []
    for left_index, left_frame in enumerate(frames):
        for right_frame in frames[left_index + 1 :]:
            edges: list[dict[str, Any]] = []
            for left in by_frame[left_frame]:
                for right in by_frame[right_frame]:
                    edge = candidate_edge(
                        left,
                        right,
                        min_shared_gaussians=min_shared_gaussians,
                        min_iou=min_iou,
                        min_containment=min_containment,
                        min_feature_similarity=min_feature_similarity,
                    )
                    if edge is not None:
                        edges.append(edge)
            if not edges:
                continue
            best_right: dict[int, dict[str, Any]] = {}
            best_left: dict[int, dict[str, Any]] = {}
            for edge in edges:
                left_id = int(edge["left_proposal_id"])
                right_id = int(edge["right_proposal_id"])
                if left_id not in best_right or float(edge["score"]) > float(
                    best_right[left_id]["score"]
                ):
                    best_right[left_id] = edge
                if right_id not in best_left or float(edge["score"]) > float(
                    best_left[right_id]["score"]
                ):
                    best_left[right_id] = edge
            for left_id, edge in best_right.items():
                right_id = int(edge["right_proposal_id"])
                if int(best_left[right_id]["left_proposal_id"]) == left_id:
                    kept.append(edge)
    kept.sort(key=lambda item: float(item["score"]), reverse=True)
    return kept


def associate_components(
    proposals: list[QueryProposal],
    edges: list[dict[str, Any]],
) -> list[list[QueryProposal]]:
    position = {proposal.proposal_id: index for index, proposal in enumerate(proposals)}
    parent = np.arange(len(proposals), dtype=np.int64)
    frames = [{proposal.frame_file} for proposal in proposals]

    def root(index: int) -> int:
        while int(parent[index]) != index:
            parent[index] = parent[int(parent[index])]
            index = int(parent[index])
        return index

    for edge in edges:
        left = root(position[int(edge["left_proposal_id"])])
        right = root(position[int(edge["right_proposal_id"])])
        if left == right or frames[left].intersection(frames[right]):
            continue
        if right < left:
            left, right = right, left
        parent[right] = left
        frames[left].update(frames[right])

    grouped: dict[int, list[QueryProposal]] = {}
    for index, proposal in enumerate(proposals):
        grouped.setdefault(root(index), []).append(proposal)
    components = list(grouped.values())
    for component in components:
        component.sort(key=lambda item: (item.frame_file, item.proposal_id))
    components.sort(key=lambda items: min(item.proposal_id for item in items))
    return components


def robust_semantic_distribution(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("component probabilities must have shape VxC")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("component probabilities are invalid")
    row_sums = values.sum(axis=1, keepdims=True)
    values = np.divide(values, np.maximum(row_sums, np.finfo(np.float64).tiny))
    log_values = np.log(np.maximum(values, np.finfo(np.float64).tiny))
    robust_log = np.median(log_values, axis=0)
    robust_log -= robust_log.max()
    fused = np.exp(robust_log)
    return (fused / fused.sum()).astype(np.float32)


def semantic_stability(probabilities: np.ndarray) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float32)
    fused = robust_semantic_distribution(values)
    winner = int(np.argmax(fused))
    view_winners = np.argmax(values, axis=1).astype(np.int64)
    winner_views = int(np.count_nonzero(view_winners == winner))
    unanimous = winner_views == values.shape[0]
    if values.shape[0] == 1:
        leave_one_out = []
        stable = False
    elif values.shape[0] == 2:
        leave_one_out = [int(view_winners[1]), int(view_winners[0])]
        stable = bool(view_winners[0] == view_winners[1] == winner)
    else:
        leave_one_out = [
            int(np.argmax(robust_semantic_distribution(np.delete(values, index, axis=0))))
            for index in range(values.shape[0])
        ]
        stable = unanimous and all(item == winner for item in leave_one_out)
    return {
        "distribution": fused,
        "winner": winner,
        "view_winners": view_winners,
        "winner_view_count": winner_views,
        "view_count": int(values.shape[0]),
        "unanimous_view_winner": bool(unanimous),
        "leave_one_out_winners": leave_one_out,
        "stable": bool(stable),
    }


def aggregate_component_support(
    proposals: list[QueryProposal],
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.concatenate([proposal.indices for proposal in proposals])
    counts = np.concatenate([proposal.counts for proposal in proposals])
    unique, inverse = np.unique(indices, return_inverse=True)
    accumulated = np.zeros((unique.shape[0],), dtype=np.float32)
    np.add.at(accumulated, inverse, counts.astype(np.float32, copy=False))
    accumulated /= np.float32(max(len({item.frame_file for item in proposals}), 1))
    return unique.astype(np.uint32, copy=False), accumulated


def load_query_proposals(
    proposal_manifest_path: Path,
) -> tuple[list[QueryProposal], dict[str, Any]]:
    manifest = json.loads(proposal_manifest_path.read_text(encoding="utf-8"))
    source_manifest_path = Path(str(manifest["source_manifest"]))
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("contract") != "class_agnostic_query_masks_with_soft_evidence_v1":
        raise ValueError("proposal source is not the DINOv3 class-agnostic contract")
    evidence_cache: dict[Path, dict[str, np.ndarray]] = {}
    support_dir = proposal_manifest_path.parent / "proposal_supports"
    proposals: list[QueryProposal] = []
    for metadata in manifest.get("proposals", []):
        if bool(metadata.get("semantic_identity_assigned", True)):
            raise ValueError("query proposal was semantically hardened before 3D association")
        evidence_path = source_manifest_path.parent / str(
            metadata["query_evidence_file"]
        )
        if evidence_path not in evidence_cache:
            with np.load(evidence_path, allow_pickle=False) as data:
                evidence_cache[evidence_path] = {
                    key: np.asarray(data[key])
                    for key in (
                        "query_indices",
                        "class_probabilities",
                        "no_object_probabilities",
                        "query_embeddings",
                    )
                }
        evidence = evidence_cache[evidence_path]
        row = int(metadata["query_evidence_row"])
        with np.load(
            support_dir / str(metadata["support_file"]),
            allow_pickle=False,
        ) as support:
            indices = np.asarray(support["indices"], dtype=np.uint32)
            counts = np.asarray(support["counts"], dtype=np.float32)
        if indices.ndim != 1 or counts.shape != indices.shape:
            raise ValueError("proposal support arrays must be matching vectors")
        if indices.size and np.any(indices[1:] <= indices[:-1]):
            raise ValueError("proposal support indices must be strictly increasing")
        probabilities = np.asarray(
            evidence["class_probabilities"][row], dtype=np.float32
        )
        embedding = np.asarray(evidence["query_embeddings"][row], dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(probabilities).all() or not np.isfinite(embedding).all():
            raise ValueError("proposal query evidence contains non-finite values")
        if norm <= np.finfo(np.float32).tiny:
            raise ValueError("proposal query embedding has zero norm")
        proposals.append(
            QueryProposal(
                proposal_id=int(metadata["proposal_id"]),
                frame_file=str(metadata["frame_file"]),
                camera_index=int(metadata["camera_index"]),
                region_id=int(metadata["region_id"]),
                indices=indices,
                counts=counts,
                class_probabilities=probabilities / max(
                    float(probabilities.sum()), np.finfo(np.float32).tiny
                ),
                no_object_probability=float(
                    evidence["no_object_probabilities"][row]
                ),
                query_embedding=embedding / norm,
                quality=float(metadata.get("confidence", 0.0)),
                metadata=metadata,
            )
        )
    return proposals, manifest


def component_record(
    component_id: int,
    proposals: list[QueryProposal],
    ontology: Ontology,
    *,
    min_source_views: int,
    min_component_gaussians: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    probabilities = np.stack(
        [proposal.class_probabilities for proposal in proposals], axis=0
    )
    stability = semantic_stability(probabilities)
    support_indices, support_scores = aggregate_component_support(proposals)
    winner = int(stability["winner"])
    ontology_item = ontology.classes[winner]
    source_views = len({proposal.frame_file for proposal in proposals})
    geometry_ready = (
        source_views >= min_source_views
        and support_indices.shape[0] >= min_component_gaussians
    )
    accepted = geometry_ready and bool(stability["stable"])
    if not geometry_ready:
        status = "abstained_insufficient_multiview_geometry"
    elif not bool(stability["stable"]):
        status = "abstained_unstable_multiview_identity"
    else:
        status = "accepted_stable_multiview_identity"
    distribution = np.asarray(stability.pop("distribution"), dtype=np.float32)
    order = np.argsort(distribution)[::-1][:8]
    record = {
        "component_id": component_id,
        "status": status,
        "accepted": accepted,
        "class": ontology_item.project_class,
        "project_id": ontology_item.project_id,
        "ade_id": ontology_item.ade_id,
        "kind": ontology_item.kind,
        "probability": float(distribution[winner]),
        "top_classes": [
            {
                "ade_id": int(index),
                "class": ontology.classes[int(index)].project_class,
                "probability": float(distribution[index]),
            }
            for index in order
        ],
        "proposal_ids": [item.proposal_id for item in proposals],
        "source_frames": sorted({item.frame_file for item in proposals}),
        "source_view_count": source_views,
        "support_gaussian_count": int(support_indices.shape[0]),
        "mean_no_object_probability": float(
            np.mean([item.no_object_probability for item in proposals])
        ),
        "mean_region_quality": float(np.mean([item.quality for item in proposals])),
        "semantic_stability": {
            **stability,
            "view_winners": [
                ontology.classes[int(index)].project_class
                for index in stability["view_winners"]
            ],
            "leave_one_out_winners": [
                ontology.classes[int(index)].project_class
                for index in stability["leave_one_out_winners"]
            ],
        },
    }
    return record, support_indices, support_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-manifest", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-shared-gaussians", default=250, type=int)
    parser.add_argument("--min-iou", default=0.05, type=float)
    parser.add_argument("--min-containment", default=0.25, type=float)
    parser.add_argument("--min-feature-similarity", default=0.0, type=float)
    parser.add_argument("--min-source-views", default=2, type=int)
    parser.add_argument("--min-component-gaussians", default=500, type=int)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if not args.report_only:
        raise ValueError("This DINOv3 3D-first stage is permanently report-only")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.min_shared_gaussians < 1 or args.min_source_views < 2:
        raise ValueError("multiview association minima are invalid")
    for name in ("min_iou", "min_containment"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    if not -1.0 <= args.min_feature_similarity <= 1.0:
        raise ValueError("min_feature_similarity must be between -1 and one")

    proposals, proposal_manifest = load_query_proposals(args.proposal_manifest)
    ontology = load_ontology(args.ontology)
    if proposals and any(
        proposal.class_probabilities.shape != (ontology.class_count,)
        for proposal in proposals
    ):
        raise ValueError("query class probabilities do not match the ontology")
    edges = mutual_best_edges(
        proposals,
        min_shared_gaussians=args.min_shared_gaussians,
        min_iou=args.min_iou,
        min_containment=args.min_containment,
        min_feature_similarity=args.min_feature_similarity,
    )
    components = associate_components(proposals, edges)
    vertex_count = int(proposal_manifest["vertex_count"])
    best_score = np.zeros((vertex_count,), dtype=np.float32)
    best_component = np.zeros((vertex_count,), dtype=np.int32)
    records: list[dict[str, Any]] = []
    supports: dict[int, np.ndarray] = {}
    for component_id, component in enumerate(components, start=1):
        record, indices, scores = component_record(
            component_id,
            component,
            ontology,
            min_source_views=args.min_source_views,
            min_component_gaussians=args.min_component_gaussians,
        )
        records.append(record)
        supports[component_id] = indices
        if bool(record["accepted"]):
            selected = scores > best_score[indices]
            best_score[indices[selected]] = scores[selected]
            best_component[indices[selected]] = component_id

    for record in records:
        component_id = int(record["component_id"])
        record["assigned_gaussian_count"] = int(
            np.count_nonzero(best_component == component_id)
        ) if bool(record["accepted"]) else 0
    accepted = [record for record in records if bool(record["accepted"])]
    conflicts: list[dict[str, Any]] = []
    for left_index, left in enumerate(accepted):
        for right in accepted[left_index + 1 :]:
            overlap = np.intersect1d(
                supports[int(left["component_id"])],
                supports[int(right["component_id"])],
                assume_unique=True,
            )
            if overlap.size:
                conflicts.append(
                    {
                        "left_component_id": int(left["component_id"]),
                        "right_component_id": int(right["component_id"]),
                        "intersection_gaussians": int(overlap.size),
                    }
                )

    output = {
        "source": "dinov3_mask2former_queries_3d_first",
        "contract": "report_only_class_agnostic_3d_association_v1",
        "proposal_manifest": str(args.proposal_manifest),
        "ontology": str(args.ontology),
        "v5_used": False,
        "dinov2_used": False,
        "semantic_identity_hardened_before_3d": False,
        "parameters": {
            "min_shared_gaussians": args.min_shared_gaussians,
            "min_iou": args.min_iou,
            "min_containment": args.min_containment,
            "min_feature_similarity": args.min_feature_similarity,
            "min_source_views": args.min_source_views,
            "min_component_gaussians": args.min_component_gaussians,
            "semantic_acceptance": (
                "strict_multiview_leave_one_out_stability_without_class_share_gate"
            ),
        },
        "proposal_count": len(proposals),
        "association_edge_count": len(edges),
        "component_count": len(records),
        "accepted_component_count": len(accepted),
        "candidate_assigned_gaussian_count": int(np.count_nonzero(best_component)),
        "association_edges": edges,
        "components": records,
        "candidate_component_conflicts": conflicts,
        "outputs": {
            "report_only": True,
            "semantic_labels_written": False,
            "label_map_written": False,
            "semantic_ply_written": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
