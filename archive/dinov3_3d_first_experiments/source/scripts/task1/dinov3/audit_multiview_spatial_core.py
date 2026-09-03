#!/usr/bin/env python3
"""Audit stable multiview spatial cores from a cached DINOv3 3D report.

This report-only stage keeps source components whose identity is either
strictly unanimous or has exactly one dissenting camera while every
leave-one-out fusion retains the same winner.  A dissenting proposal is
removed before geometry is evaluated.  Remaining support is counted once per
camera, filtered to multiview Gaussians, and split into adaptive 26-neighbor
voxel components.  Cross-class overlap is assigned only to a uniquely larger
supporting-camera count; equal counts abstain.

The stage never writes semantic labels, a label map, or a PLY.
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
from scripts.task1.dinov3.associate_3d_query_regions import (
    QueryProposal,
    load_query_proposals,
    semantic_stability,
)
from scripts.task1.dinov3.audit_anchored_component_association import (
    classify_numeric_consensus,
)
from scripts.task1.dinov3.materialize_plurality_component_labels import (
    validate_report,
)
from scripts.task1.dinov3.propagate_dense_labels_from_region_seeds import (
    adaptive_voxel_size,
)
from scripts.task1.grounding.cluster_semantic_flashsplat_proposals import (
    voxel_components,
)


SOURCE = "dinov3_multiview_spatial_core_audit"
CONTRACT = "report_only_dinov3_multiview_spatial_core_v1"


@dataclass(frozen=True)
class AuditThresholds:
    min_gaussian_camera_support: int = 2
    min_spatial_component_gaussians: int = 500
    min_spatial_component_cameras: int = 2
    voxel_scale_multiplier: float = 4.0
    min_voxel_size: float = 0.01
    max_voxel_size: float = 0.20

    def validate(self) -> None:
        if self.min_gaussian_camera_support < 2:
            raise ValueError("min_gaussian_camera_support must be at least two")
        if self.min_spatial_component_gaussians < 1:
            raise ValueError("min_spatial_component_gaussians must be positive")
        if self.min_spatial_component_cameras < 2:
            raise ValueError("min_spatial_component_cameras must be at least two")
        if self.voxel_scale_multiplier <= 0.0:
            raise ValueError("voxel_scale_multiplier must be positive")
        if self.min_voxel_size <= 0.0:
            raise ValueError("min_voxel_size must be positive")
        if self.max_voxel_size < 0.0:
            raise ValueError("max_voxel_size must be non-negative")
        if 0.0 < self.max_voxel_size < self.min_voxel_size:
            raise ValueError("max_voxel_size must be zero or at least min_voxel_size")


@dataclass(frozen=True)
class SpatialCoreSupport:
    component_id: int
    source_component_id: int
    project_id: int
    indices: np.ndarray
    camera_counts: np.ndarray
    record: dict[str, Any]


def aggregate_camera_presence(
    proposals: list[QueryProposal],
) -> tuple[np.ndarray, np.ndarray]:
    """Count each agreeing camera at most once at every Gaussian."""

    if not proposals:
        return (
            np.empty((0,), dtype=np.uint32),
            np.empty((0,), dtype=np.uint16),
        )
    camera_indices = [item.camera_index for item in proposals]
    frame_files = [item.frame_file for item in proposals]
    if len(set(camera_indices)) != len(camera_indices):
        raise ValueError("component contains multiple proposals from one camera")
    if len(set(frame_files)) != len(frame_files):
        raise ValueError("component contains multiple proposals from one frame")
    concatenated = np.concatenate(
        [np.asarray(item.indices, dtype=np.uint32) for item in proposals]
    )
    unique, counts = np.unique(concatenated, return_counts=True)
    if counts.size and int(counts.max()) > np.iinfo(np.uint16).max:
        raise ValueError("supporting camera count exceeds uint16 capacity")
    return (
        unique.astype(np.uint32, copy=False),
        counts.astype(np.uint16, copy=False),
    )


def classify_component_identity(
    record: dict[str, Any],
    proposals: list[QueryProposal],
    ontology: Ontology,
) -> tuple[str, list[QueryProposal], list[QueryProposal], dict[str, Any]]:
    """Return the consensus tier and agreeing/dissenting camera proposals."""

    source_view_count = int(record["source_view_count"])
    if len(proposals) != source_view_count:
        raise ValueError("component proposal count differs from source_view_count")
    if not proposals:
        return "mixed_or_unstable", [], [], {}
    probabilities = np.stack(
        [item.class_probabilities for item in proposals], axis=0
    )
    stability = semantic_stability(probabilities)
    winner = int(stability["winner"])
    if winner < 0 or winner >= len(ontology.classes):
        raise ValueError("component winner is outside the ontology")
    if int(record["ade_id"]) != winner:
        raise ValueError("component report identity differs from cached evidence")
    ontology_item = ontology.classes[winner]
    if str(record["class"]) != ontology_item.project_class:
        raise ValueError("component class differs from the ontology winner")
    if int(record["project_id"]) != ontology_item.project_id:
        raise ValueError("component project ID differs from the ontology winner")

    tier = classify_numeric_consensus(stability, winner)
    view_winners = np.asarray(stability["view_winners"], dtype=np.int64)
    agreeing = [
        item for item, view_winner in zip(proposals, view_winners) if view_winner == winner
    ]
    dissenting = [
        item for item, view_winner in zip(proposals, view_winners) if view_winner != winner
    ]
    if tier == "strict_unanimous" and dissenting:
        raise AssertionError("strict component contains a dissenting proposal")
    if tier == "one_view_outlier_leave_one_out_stable" and len(dissenting) != 1:
        raise AssertionError("one-outlier component does not have exactly one dissent")
    if tier == "mixed_or_unstable":
        agreeing = []
    serialized = {
        "winner_ade_id": winner,
        "winner_view_count": int(stability["winner_view_count"]),
        "view_count": int(stability["view_count"]),
        "unanimous_view_winner": bool(stability["unanimous_view_winner"]),
        "leave_one_out_winners": [
            int(item) for item in stability["leave_one_out_winners"]
        ],
        "stable": bool(stability["stable"]),
    }
    return tier, agreeing, dissenting, serialized


def supporting_cameras_by_spatial_component(
    retained_indices: np.ndarray,
    point_components: np.ndarray,
    component_count: int,
    proposals: list[QueryProposal],
) -> list[set[int]]:
    """Collect independent camera IDs intersecting every spatial component."""

    indices = np.asarray(retained_indices, dtype=np.uint32)
    labels = np.asarray(point_components, dtype=np.int64)
    if labels.shape != indices.shape:
        raise ValueError("point component IDs must align with retained indices")
    cameras = [set() for _ in range(component_count)]
    for proposal in proposals:
        positions = np.searchsorted(indices, proposal.indices)
        inside = positions < indices.size
        positions = positions[inside]
        proposal_indices = proposal.indices[inside]
        exact = indices[positions] == proposal_indices
        if not np.any(exact):
            continue
        for component_id in np.unique(labels[positions[exact]]):
            cameras[int(component_id)].add(int(proposal.camera_index))
    return cameras


def split_multiview_spatial_support(
    source_component_id: int,
    retained_indices: np.ndarray,
    camera_counts: np.ndarray,
    agreeing_proposals: list[QueryProposal],
    points: np.ndarray,
    log_scales: np.ndarray,
    thresholds: AuditThresholds,
) -> tuple[list[dict[str, Any]], float, float, dict[str, Any]]:
    """Split filtered support and return per-spatial-component diagnostics."""

    indices = np.asarray(retained_indices, dtype=np.uint32)
    counts = np.asarray(camera_counts, dtype=np.uint16)
    if counts.shape != indices.shape:
        raise ValueError("camera counts must align with retained indices")
    if points.shape != (indices.size, 3) or log_scales.shape != (indices.size, 3):
        raise ValueError("geometry must align with retained indices")
    if indices.size == 0:
        return [], 0.0, 0.0, {
            "voxel_count": 0,
            "component_count": 0,
            "largest_component_gaussians": 0,
        }

    median_scale, voxel_size = adaptive_voxel_size(
        log_scales,
        voxel_scale_multiplier=thresholds.voxel_scale_multiplier,
        min_voxel_size=thresholds.min_voxel_size,
        max_voxel_size=thresholds.max_voxel_size,
    )
    point_components, component_sizes, geometry = voxel_components(
        np.asarray(points, dtype=np.float64), voxel_size
    )
    component_cameras = supporting_cameras_by_spatial_component(
        indices,
        point_components,
        int(component_sizes.size),
        agreeing_proposals,
    )
    results: list[dict[str, Any]] = []
    for local_component_id, size_value in enumerate(component_sizes):
        selected = point_components == local_component_id
        size = int(size_value)
        independent_cameras = sorted(component_cameras[local_component_id])
        large_enough = size >= thresholds.min_spatial_component_gaussians
        enough_cameras = (
            len(independent_cameras) >= thresholds.min_spatial_component_cameras
        )
        accepted = large_enough and enough_cameras
        if not large_enough:
            status = "rejected_insufficient_spatial_component_gaussians"
        elif not enough_cameras:
            status = "rejected_insufficient_independent_cameras"
        else:
            status = "accepted_report_only_multiview_spatial_core"
        local_counts = counts[selected]
        unique_counts, frequencies = np.unique(local_counts, return_counts=True)
        results.append(
            {
                "source_component_id": source_component_id,
                "local_spatial_component_id": local_component_id,
                "status": status,
                "accepted": bool(accepted),
                "support_gaussian_count": size,
                "independent_camera_count": len(independent_cameras),
                "independent_camera_indices": independent_cameras,
                "minimum_gaussian_camera_support": int(local_counts.min()),
                "maximum_gaussian_camera_support": int(local_counts.max()),
                "camera_support_count_histogram": {
                    str(int(value)): int(frequency)
                    for value, frequency in zip(unique_counts, frequencies)
                },
                "_indices": indices[selected],
                "_camera_counts": local_counts,
            }
        )
    return results, median_scale, voxel_size, geometry


def resolve_spatial_core_ownership(
    cores: list[SpatialCoreSupport],
    vertex_count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, tuple[np.ndarray, np.ndarray]],
]:
    """Resolve overlaps by unique maximum camera count across classes."""

    if vertex_count < 1:
        raise ValueError("vertex_count must be positive")
    owner_count = np.zeros((vertex_count,), dtype=np.uint16)
    maximum_camera_count = np.zeros((vertex_count,), dtype=np.uint16)
    winning_component = np.zeros((vertex_count,), dtype=np.int32)
    winning_class = np.zeros((vertex_count,), dtype=np.int32)
    first_class = np.zeros((vertex_count,), dtype=np.int32)
    cross_class_overlap = np.zeros((vertex_count,), dtype=bool)
    cross_class_tie = np.zeros((vertex_count,), dtype=bool)

    for core in cores:
        indices = np.asarray(core.indices, dtype=np.int64)
        counts = np.asarray(core.camera_counts)
        if indices.ndim != 1 or counts.shape != indices.shape:
            raise ValueError("core indices and camera counts must align")
        if indices.size and (indices[0] < 0 or indices[-1] >= vertex_count):
            raise ValueError("core index is outside the source PLY")
        if not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("core camera counts must be integers")
        if owner_count[indices].size and np.any(
            owner_count[indices] == np.iinfo(np.uint16).max
        ):
            raise ValueError("spatial core owner count exceeds uint16 capacity")
        owner_count[indices] += np.uint16(1)

        existing_first = first_class[indices]
        empty_first = existing_first == 0
        different_first = (~empty_first) & (existing_first != core.project_id)
        cross_class_overlap[indices[different_first]] = True
        first_class[indices[empty_first]] = core.project_id

        current_counts = maximum_camera_count[indices]
        current_classes = winning_class[indices]
        greater = counts > current_counts
        equal = counts == current_counts
        greater_indices = indices[greater]
        maximum_camera_count[greater_indices] = counts[greater]
        winning_component[greater_indices] = core.component_id
        winning_class[greater_indices] = core.project_id
        cross_class_tie[greater_indices] = False

        equal_cross_class = (
            equal & (current_counts > 0) & (current_classes != core.project_id)
        )
        cross_class_tie[indices[equal_cross_class]] = True

    proposed = (winning_component > 0) & ~cross_class_tie
    exclusive: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for core in cores:
        selected = proposed[core.indices] & (
            winning_component[core.indices] == core.component_id
        )
        exclusive[core.component_id] = (
            core.indices[selected],
            core.camera_counts[selected],
        )
        core.record["overlapping_candidate_gaussian_count"] = int(
            np.count_nonzero(owner_count[core.indices] > 1)
        )
        core.record["assigned_gaussian_count"] = int(np.count_nonzero(selected))
        core.record["abstained_or_other_owner_gaussian_count"] = int(
            core.indices.size - np.count_nonzero(selected)
        )
    return (
        proposed,
        owner_count,
        owner_count > 1,
        cross_class_overlap,
        cross_class_tie,
        exclusive,
    )


def _component_proposals(
    record: dict[str, Any],
    proposal_by_id: dict[int, QueryProposal],
) -> list[QueryProposal]:
    proposals: list[QueryProposal] = []
    for proposal_id_value in record.get("proposal_ids", []):
        proposal_id = int(proposal_id_value)
        try:
            proposals.append(proposal_by_id[proposal_id])
        except KeyError as exc:
            raise ValueError(
                f"component references missing proposal {proposal_id}"
            ) from exc
    return proposals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-report", required=True, type=Path)
    parser.add_argument("--proposal-manifest", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--min-gaussian-camera-support", default=2, type=int)
    parser.add_argument("--min-spatial-component-gaussians", default=500, type=int)
    parser.add_argument("--min-spatial-component-cameras", default=2, type=int)
    parser.add_argument("--voxel-scale-multiplier", default=4.0, type=float)
    parser.add_argument("--min-voxel-size", default=0.01, type=float)
    parser.add_argument("--max-voxel-size", default=0.20, type=float)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (args.component_report, args.proposal_manifest, args.ontology):
        if not path.is_file():
            raise FileNotFoundError(path)
    thresholds = AuditThresholds(
        min_gaussian_camera_support=args.min_gaussian_camera_support,
        min_spatial_component_gaussians=args.min_spatial_component_gaussians,
        min_spatial_component_cameras=args.min_spatial_component_cameras,
        voxel_scale_multiplier=args.voxel_scale_multiplier,
        min_voxel_size=args.min_voxel_size,
        max_voxel_size=args.max_voxel_size,
    )
    thresholds.validate()

    component_report = json.loads(args.component_report.read_text(encoding="utf-8"))
    proposals, proposal_manifest = load_query_proposals(args.proposal_manifest)
    validate_report(proposal_manifest, component_report)
    ontology = load_ontology(args.ontology)
    if proposals and any(
        item.class_probabilities.shape != (ontology.class_count,)
        for item in proposals
    ):
        raise ValueError("query class probabilities do not match the ontology")
    proposal_by_id = {item.proposal_id: item for item in proposals}
    if len(proposal_by_id) != len(proposals):
        raise ValueError("proposal IDs are not unique")

    source_ply = Path(str(proposal_manifest["ply_path"]))
    if not source_ply.is_file():
        raise FileNotFoundError(source_ply)
    header, vertices = vertex_data_memmap(source_ply)
    vertex_count = int(proposal_manifest["vertex_count"])
    if not header.elements or int(header.elements[0].count) != vertex_count:
        raise ValueError("source PLY vertex count differs from proposal manifest")
    required_fields = {"x", "y", "z", "scale_0", "scale_1", "scale_2"}
    if not required_fields.issubset(vertices.dtype.names or ()):
        raise ValueError("source PLY lacks coordinates or Gaussian scales")

    source_records: list[dict[str, Any]] = []
    visualization_components: list[dict[str, Any]] = []
    core_records: list[dict[str, Any]] = []
    accepted_cores: list[SpatialCoreSupport] = []
    consensus_counts: dict[str, int] = {}
    next_core_id = 1
    for record in component_report.get("components", []):
        source_component_id = int(record["component_id"])
        component_proposals = _component_proposals(record, proposal_by_id)
        tier, agreeing, dissenting, stability = classify_component_identity(
            record, component_proposals, ontology
        )
        consensus_counts[tier] = consensus_counts.get(tier, 0) + 1
        source_record: dict[str, Any] = {
            "source_component_id": source_component_id,
            "class": str(record["class"]),
            "project_id": int(record["project_id"]),
            "ade_id": int(record["ade_id"]),
            "kind": str(record["kind"]),
            "source_status": str(record["status"]),
            "consensus_tier": tier,
            "source_view_count": int(record["source_view_count"]),
            "source_support_gaussian_count": int(record["support_gaussian_count"]),
            "agreeing_camera_count": len(agreeing),
            "agreeing_camera_indices": sorted(
                {int(item.camera_index) for item in agreeing}
            ),
            "agreeing_proposal_ids": [int(item.proposal_id) for item in agreeing],
            "dissenting_camera_indices": sorted(
                {int(item.camera_index) for item in dissenting}
            ),
            "dissenting_proposal_ids": [
                int(item.proposal_id) for item in dissenting
            ],
            "semantic_stability": stability,
        }
        if tier == "mixed_or_unstable":
            source_record.update(
                {
                    "status": "rejected_mixed_or_unstable_identity",
                    "retained_multiview_gaussian_count": 0,
                    "spatial_component_count": 0,
                    "accepted_spatial_core_count": 0,
                    "accepted_spatial_core_gaussian_count": 0,
                }
            )
            source_records.append(source_record)
            visualization_components.append(
                {
                    "component_id": source_component_id,
                    "accepted": False,
                    "class": str(record["class"]),
                    "status": source_record["status"],
                    "proposal_ids": [
                        int(item.proposal_id) for item in component_proposals
                    ],
                }
            )
            continue

        support_indices, support_counts = aggregate_camera_presence(agreeing)
        keep = support_counts >= thresholds.min_gaussian_camera_support
        retained_indices = support_indices[keep]
        retained_counts = support_counts[keep]
        source_record["agreeing_union_gaussian_count"] = int(support_indices.size)
        source_record["retained_multiview_gaussian_count"] = int(
            retained_indices.size
        )
        if retained_indices.size:
            points = np.column_stack(
                [
                    vertices[axis][retained_indices].astype(np.float64)
                    for axis in ("x", "y", "z")
                ]
            )
            log_scales = np.column_stack(
                [
                    vertices[axis][retained_indices].astype(np.float64)
                    for axis in ("scale_0", "scale_1", "scale_2")
                ]
            )
        else:
            points = np.empty((0, 3), dtype=np.float64)
            log_scales = np.empty((0, 3), dtype=np.float64)
        split_records, median_scale, voxel_size, geometry = (
            split_multiview_spatial_support(
                source_component_id,
                retained_indices,
                retained_counts,
                agreeing,
                points,
                log_scales,
                thresholds,
            )
        )
        accepted_count = 0
        accepted_gaussians = 0
        source_core_ids: list[int] = []
        for split_record in split_records:
            indices = split_record.pop("_indices")
            camera_counts = split_record.pop("_camera_counts")
            core_record = {
                "component_id": next_core_id,
                "class": str(record["class"]),
                "project_id": int(record["project_id"]),
                "ade_id": int(record["ade_id"]),
                "kind": str(record["kind"]),
                "consensus_tier": tier,
                "source_proposal_ids": [
                    int(item.proposal_id) for item in agreeing
                ],
                "excluded_dissenting_proposal_ids": [
                    int(item.proposal_id) for item in dissenting
                ],
                "median_gaussian_scale": median_scale,
                "voxel_size": voxel_size,
                **split_record,
            }
            core_records.append(core_record)
            source_core_ids.append(next_core_id)
            if bool(core_record["accepted"]):
                accepted_count += 1
                accepted_gaussians += int(indices.size)
                accepted_cores.append(
                    SpatialCoreSupport(
                        component_id=next_core_id,
                        source_component_id=source_component_id,
                        project_id=int(record["project_id"]),
                        indices=indices,
                        camera_counts=camera_counts,
                        record=core_record,
                    )
                )
            next_core_id += 1
        source_record.update(
            {
                "status": (
                    "retained_with_report_only_spatial_core"
                    if accepted_count
                    else "rejected_no_qualifying_spatial_core"
                ),
                "median_gaussian_scale": median_scale,
                "voxel_size": voxel_size,
                "spatial_geometry": geometry,
                "spatial_component_count": len(split_records),
                "spatial_component_ids": source_core_ids,
                "accepted_spatial_core_count": accepted_count,
                "accepted_spatial_core_gaussian_count": accepted_gaussians,
            }
        )
        source_records.append(source_record)
        visualization_components.append(
            {
                "component_id": source_component_id,
                "accepted": bool(accepted_count),
                "class": str(record["class"]),
                "status": source_record["status"],
                "proposal_ids": [int(item.proposal_id) for item in agreeing],
            }
        )
        if dissenting:
            visualization_components.append(
                {
                    "component_id": -source_component_id,
                    "accepted": False,
                    "class": str(record["class"]),
                    "status": "excluded_single_dissenting_camera",
                    "proposal_ids": [
                        int(item.proposal_id) for item in dissenting
                    ],
                }
            )

    (
        proposed,
        owner_count,
        overlap,
        cross_class_overlap,
        cross_class_tie,
        exclusive,
    ) = resolve_spatial_core_ownership(accepted_cores, vertex_count)

    assigned_by_source: dict[int, int] = {}
    class_assigned: dict[str, int] = {}
    for core in accepted_cores:
        assigned = int(core.record["assigned_gaussian_count"])
        assigned_by_source[core.source_component_id] = (
            assigned_by_source.get(core.source_component_id, 0) + assigned
        )
        class_name = str(core.record["class"])
        class_assigned[class_name] = class_assigned.get(class_name, 0) + assigned
    for record in source_records:
        assigned = assigned_by_source.get(int(record["source_component_id"]), 0)
        record["assigned_gaussian_count"] = assigned
    for record in visualization_components:
        source_component_id = abs(int(record["component_id"]))
        if bool(record["accepted"]) and assigned_by_source.get(source_component_id, 0) == 0:
            record["accepted"] = False
            record["status"] = "abstained_all_support_lost_during_overlap_resolution"

    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "proposed_core_mask.npy", proposed)
    np.save(args.output_dir / "proposed_core_owner_count.npy", owner_count)
    np.save(args.output_dir / "proposed_core_overlap_mask.npy", overlap)
    np.save(
        args.output_dir / "proposed_core_cross_class_overlap_mask.npy",
        cross_class_overlap,
    )
    np.save(
        args.output_dir / "proposed_core_cross_class_tie_mask.npy",
        cross_class_tie,
    )
    np.savez_compressed(
        args.output_dir / "proposed_core_supports.npz",
        **{
            key: value
            for component_id, (indices, counts) in sorted(exclusive.items())
            for key, value in (
                (f"component_{component_id:06d}_indices", indices),
                (f"component_{component_id:06d}_camera_counts", counts),
            )
        },
    )

    accepted_source_count = sum(
        int(record["accepted_spatial_core_count"] > 0) for record in source_records
    )
    report = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": args.scene,
        "component_report": str(args.component_report),
        "proposal_manifest": str(args.proposal_manifest),
        "source_ply": str(source_ply),
        "ontology": str(args.ontology),
        "report_only": True,
        "semantic_labels_written": False,
        "semantic_project_class_arrays_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
        "v5_used": False,
        "v5_labels_used": False,
        "v5_method_adapted": True,
        "dinov2_used": False,
        "inference_rerun": False,
        "flashsplat_rerun": False,
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "offline_threshold_sweep_used": False,
        "identity_policy": (
            "strict_unanimous_or_exactly_one_outlier_with_all_leave_one_out_"
            "winners_unchanged"
        ),
        "outlier_policy": "exclude_the_single_dissenting_camera_before_support_counting",
        "gaussian_support_policy": "one_presence_vote_per_agreeing_camera",
        "geometry_policy": "adaptive_26_neighbor_voxel_components",
        "cross_class_overlap_policy": (
            "unique_maximum_supporting_camera_count_else_abstain"
        ),
        "visualization_scope": (
            "matched_full_source_proposal_masks_for_component_qa_not_an_exact_"
            "projection_of_the_retained_3d_core"
        ),
        "parameters": vars(thresholds),
        "vertex_count": vertex_count,
        "input_proposal_count": len(proposals),
        "input_component_count": len(source_records),
        "consensus_tier_counts": dict(sorted(consensus_counts.items())),
        "retained_source_component_count": accepted_source_count,
        "one_outlier_source_component_count": int(
            consensus_counts.get("one_view_outlier_leave_one_out_stable", 0)
        ),
        "excluded_dissenting_proposal_count": sum(
            len(record["dissenting_proposal_ids"])
            for record in source_records
            if record["consensus_tier"]
            == "one_view_outlier_leave_one_out_stable"
        ),
        "spatial_component_count": len(core_records),
        "accepted_spatial_core_count": len(accepted_cores),
        "candidate_spatial_core_gaussian_count": sum(
            int(core.indices.size) for core in accepted_cores
        ),
        "proposed_core_gaussian_count": int(np.count_nonzero(proposed)),
        "proposed_core_overlap_gaussian_count": int(np.count_nonzero(overlap)),
        "cross_class_overlap_gaussian_count": int(
            np.count_nonzero(cross_class_overlap)
        ),
        "cross_class_camera_count_tie_gaussian_count": int(
            np.count_nonzero(cross_class_tie)
        ),
        "class_proposed_core_gaussian_counts": dict(sorted(class_assigned.items())),
        "source_components": source_records,
        "components": core_records,
        "visualization_components": visualization_components,
        "outputs": {
            "report_only": True,
            "semantic_labels_written": False,
            "semantic_project_class_arrays_written": False,
            "label_map_written": False,
            "semantic_ply_written": False,
            "boolean_masks_written": True,
            "owner_count_mask_written": True,
            "compressed_sparse_supports_written": True,
        },
    }
    (args.output_dir / "multiview_spatial_core_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "scene": args.scene,
        "retained_source_component_count": accepted_source_count,
        "accepted_spatial_core_count": len(accepted_cores),
        "proposed_core_gaussian_count": int(np.count_nonzero(proposed)),
        "cross_class_camera_count_tie_gaussian_count": int(
            np.count_nonzero(cross_class_tie)
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
