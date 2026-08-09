#!/usr/bin/env python3
"""Split every DINOv3 proposal graph before fusing semantic identity.

The cached 3D-first report forms class-agnostic proposal graphs but assigns one
semantic identity to each complete graph.  This report-only audit reverses the
remaining order of operations:

1. aggregate one presence vote per camera over every source graph, including
   graphs whose original identity was mixed or unstable;
2. retain multiview Gaussian support and split it into adaptive spatial cores;
3. select the independent camera proposals intersecting each spatial core;
4. fuse DINOv3 identity independently inside that core;
5. remove an optional single stable dissenting camera, recompute multiview
   support, and re-split before applying global size and camera gates.

No semantic label array, label map, project-class array, or PLY is written.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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
    classify_component_record,
    classify_numeric_consensus,
)
from scripts.task1.dinov3.audit_multiview_spatial_core import (
    SpatialCoreSupport,
    aggregate_camera_presence,
    resolve_spatial_core_ownership,
    supporting_cameras_by_spatial_component,
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


SOURCE = "dinov3_core_first_semantic_identity_audit"
CONTRACT = "report_only_dinov3_core_first_semantic_identity_v1"


@dataclass(frozen=True)
class CoreFirstThresholds:
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
            raise ValueError(
                "min_spatial_component_gaussians must be positive"
            )
        if self.min_spatial_component_cameras < 2:
            raise ValueError(
                "min_spatial_component_cameras must be at least two"
            )
        if self.voxel_scale_multiplier <= 0.0:
            raise ValueError("voxel_scale_multiplier must be positive")
        if self.min_voxel_size <= 0.0:
            raise ValueError("min_voxel_size must be positive")
        if self.max_voxel_size < 0.0:
            raise ValueError("max_voxel_size must be non-negative")
        if 0.0 < self.max_voxel_size < self.min_voxel_size:
            raise ValueError(
                "max_voxel_size must be zero or at least min_voxel_size"
            )


def component_proposals(
    record: dict[str, Any],
    proposal_by_id: dict[int, QueryProposal],
) -> list[QueryProposal]:
    proposals: list[QueryProposal] = []
    for value in record.get("proposal_ids", []):
        proposal_id = int(value)
        try:
            proposals.append(proposal_by_id[proposal_id])
        except KeyError as exc:
            raise ValueError(
                f"component references missing proposal {proposal_id}"
            ) from exc
    camera_indices = [item.camera_index for item in proposals]
    frame_files = [item.frame_file for item in proposals]
    if len(set(camera_indices)) != len(camera_indices):
        raise ValueError("source component contains multiple proposals per camera")
    if len(set(frame_files)) != len(frame_files):
        raise ValueError("source component contains multiple proposals per frame")
    return proposals


def proposals_intersecting_core(
    core_indices: np.ndarray,
    proposals: list[QueryProposal],
) -> tuple[list[QueryProposal], list[dict[str, int | float]]]:
    """Return camera proposals with exact support inside a spatial core."""

    indices = np.asarray(core_indices, dtype=np.uint32)
    selected: list[QueryProposal] = []
    contributions: list[dict[str, int | float]] = []
    for proposal in proposals:
        intersection = np.intersect1d(
            indices,
            proposal.indices,
            assume_unique=True,
        )
        if intersection.size == 0:
            continue
        selected.append(proposal)
        contributions.append(
            {
                "proposal_id": int(proposal.proposal_id),
                "camera_index": int(proposal.camera_index),
                "intersection_gaussians": int(intersection.size),
                "proposal_containment": (
                    int(intersection.size) / float(max(proposal.indices.size, 1))
                ),
                "core_coverage": (
                    int(intersection.size) / float(max(indices.size, 1))
                ),
            }
        )
    return selected, contributions


def fuse_core_identity(
    proposals: list[QueryProposal],
    ontology: Ontology,
) -> tuple[
    str,
    int,
    list[QueryProposal],
    list[QueryProposal],
    dict[str, Any],
]:
    """Fuse semantic identity after a core has already been spatially formed."""

    if not proposals:
        return "mixed_or_unstable", -1, [], [], {
            "view_count": 0,
            "winner_view_count": 0,
            "unanimous_view_winner": False,
            "leave_one_out_winners": [],
            "stable": False,
            "top_classes": [],
        }
    probabilities = np.stack(
        [item.class_probabilities for item in proposals],
        axis=0,
    )
    stability = semantic_stability(probabilities)
    winner = int(stability["winner"])
    if winner < 0 or winner >= len(ontology.classes):
        raise ValueError("core identity winner is outside the ontology")
    tier = classify_numeric_consensus(stability, winner)
    view_winners = np.asarray(stability["view_winners"], dtype=np.int64)
    agreeing = [
        proposal
        for proposal, view_winner in zip(proposals, view_winners)
        if int(view_winner) == winner
    ]
    dissenting = [
        proposal
        for proposal, view_winner in zip(proposals, view_winners)
        if int(view_winner) != winner
    ]
    if tier == "strict_unanimous" and dissenting:
        raise AssertionError("strict core contains a dissenting camera")
    if tier == "one_view_outlier_leave_one_out_stable" and len(dissenting) != 1:
        raise AssertionError("one-outlier core does not have one dissenting camera")
    if tier == "mixed_or_unstable":
        agreeing = []

    distribution = np.asarray(stability["distribution"], dtype=np.float32)
    order = np.argsort(distribution)[::-1][:8]
    serialized = {
        "winner_ade_id": winner,
        "winner_class": ontology.classes[winner].project_class,
        "winner_probability": float(distribution[winner]),
        "winner_view_count": int(stability["winner_view_count"]),
        "view_count": int(stability["view_count"]),
        "unanimous_view_winner": bool(stability["unanimous_view_winner"]),
        "view_winners": [
            ontology.classes[int(value)].project_class
            for value in view_winners
        ],
        "leave_one_out_winners": [
            ontology.classes[int(value)].project_class
            for value in stability["leave_one_out_winners"]
        ],
        "stable": bool(stability["stable"]),
        "top_classes": [
            {
                "ade_id": int(index),
                "class": ontology.classes[int(index)].project_class,
                "probability": float(distribution[index]),
            }
            for index in order
        ],
    }
    return tier, winner, agreeing, dissenting, serialized


def multiview_support_within_core(
    core_indices: np.ndarray,
    proposals: list[QueryProposal],
    *,
    min_gaussian_camera_support: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute exact agreeing-camera support inside one pre-identity core."""

    if not proposals:
        return (
            np.empty((0,), dtype=np.uint32),
            np.empty((0,), dtype=np.uint16),
        )
    support_indices, support_counts = aggregate_camera_presence(proposals)
    common, _core_positions, support_positions = np.intersect1d(
        np.asarray(core_indices, dtype=np.uint32),
        support_indices,
        assume_unique=True,
        return_indices=True,
    )
    counts = support_counts[support_positions]
    keep = counts >= min_gaussian_camera_support
    return (
        common[keep].astype(np.uint32, copy=False),
        counts[keep].astype(np.uint16, copy=False),
    )


def split_post_identity_support(
    source_component_id: int,
    pre_identity_core_id: int,
    indices: np.ndarray,
    camera_counts: np.ndarray,
    agreeing_proposals: list[QueryProposal],
    points: np.ndarray,
    voxel_size: float,
    thresholds: CoreFirstThresholds,
) -> list[dict[str, Any]]:
    """Re-split exact agreeing-camera support and apply global core gates."""

    if indices.size == 0:
        return []
    point_components, component_sizes, geometry = voxel_components(
        np.asarray(points, dtype=np.float64),
        voxel_size,
    )
    cameras = supporting_cameras_by_spatial_component(
        indices,
        point_components,
        int(component_sizes.size),
        agreeing_proposals,
    )
    records: list[dict[str, Any]] = []
    for local_component_id, size_value in enumerate(component_sizes):
        selected = point_components == local_component_id
        independent_cameras = sorted(cameras[local_component_id])
        size = int(size_value)
        reasons: list[str] = []
        if size < thresholds.min_spatial_component_gaussians:
            reasons.append(
                "support_gaussians<"
                f"{thresholds.min_spatial_component_gaussians}"
            )
        if len(independent_cameras) < thresholds.min_spatial_component_cameras:
            reasons.append(
                "independent_cameras<"
                f"{thresholds.min_spatial_component_cameras}"
            )
        accepted = not reasons
        records.append(
            {
                "source_component_id": source_component_id,
                "pre_identity_core_id": pre_identity_core_id,
                "local_post_identity_component_id": int(local_component_id),
                "accepted": accepted,
                "status": (
                    "accepted_report_only_core_first_semantic_identity"
                    if accepted
                    else "rejected_post_identity_global_core_gates"
                ),
                "reasons": reasons,
                "support_gaussian_count": size,
                "independent_camera_count": len(independent_cameras),
                "independent_camera_indices": independent_cameras,
                "spatial_geometry": geometry,
                "_indices": indices[selected],
                "_camera_counts": camera_counts[selected],
            }
        )
    return records


def pre_identity_spatial_split(
    proposals: list[QueryProposal],
    vertices: np.ndarray,
    thresholds: CoreFirstThresholds,
) -> tuple[
    list[tuple[np.ndarray, np.ndarray]],
    float,
    float,
    dict[str, Any],
    int,
]:
    """Form multiview spatial cores before consulting semantic probabilities."""

    support_indices, support_counts = aggregate_camera_presence(proposals)
    keep = support_counts >= thresholds.min_gaussian_camera_support
    retained_indices = support_indices[keep]
    retained_counts = support_counts[keep]
    if retained_indices.size == 0:
        return [], 0.0, 0.0, {
            "voxel_count": 0,
            "component_count": 0,
            "largest_component_gaussians": 0,
        }, int(support_indices.size)

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
    median_scale, voxel_size = adaptive_voxel_size(
        log_scales,
        voxel_scale_multiplier=thresholds.voxel_scale_multiplier,
        min_voxel_size=thresholds.min_voxel_size,
        max_voxel_size=thresholds.max_voxel_size,
    )
    point_components, component_sizes, geometry = voxel_components(
        points,
        voxel_size,
    )
    splits = [
        (
            retained_indices[point_components == component_id],
            retained_counts[point_components == component_id],
        )
        for component_id in range(int(component_sizes.size))
    ]
    return (
        splits,
        median_scale,
        voxel_size,
        geometry,
        int(support_indices.size),
    )


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
    thresholds = CoreFirstThresholds(
        min_gaussian_camera_support=args.min_gaussian_camera_support,
        min_spatial_component_gaussians=args.min_spatial_component_gaussians,
        min_spatial_component_cameras=args.min_spatial_component_cameras,
        voxel_scale_multiplier=args.voxel_scale_multiplier,
        min_voxel_size=args.min_voxel_size,
        max_voxel_size=args.max_voxel_size,
    )
    thresholds.validate()

    component_report = json.loads(
        args.component_report.read_text(encoding="utf-8")
    )
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
    pre_identity_records: list[dict[str, Any]] = []
    core_records: list[dict[str, Any]] = []
    accepted_cores: list[SpatialCoreSupport] = []
    source_tier_counts: dict[str, int] = {}
    core_tier_counts: dict[str, int] = {}
    next_pre_identity_core_id = 1
    next_component_id = 1
    seen_source_proposals: set[int] = set()

    for source_record_raw in component_report.get("components", []):
        source_component_id = int(source_record_raw["component_id"])
        source_proposals = component_proposals(
            source_record_raw,
            proposal_by_id,
        )
        source_proposal_ids = {item.proposal_id for item in source_proposals}
        if seen_source_proposals.intersection(source_proposal_ids):
            raise ValueError("proposal appears in multiple source components")
        seen_source_proposals.update(source_proposal_ids)
        source_tier = classify_component_record(source_record_raw)
        source_tier_counts[source_tier] = source_tier_counts.get(source_tier, 0) + 1

        (
            spatial_splits,
            median_scale,
            voxel_size,
            spatial_geometry,
            source_union_count,
        ) = pre_identity_spatial_split(
            source_proposals,
            vertices,
            thresholds,
        )
        source_pre_core_ids: list[int] = []
        source_final_core_ids: list[int] = []
        source_accepted_count = 0
        source_accepted_gaussians = 0

        for local_pre_core_id, (pre_indices, pre_counts) in enumerate(
            spatial_splits
        ):
            pre_identity_core_id = next_pre_identity_core_id
            next_pre_identity_core_id += 1
            source_pre_core_ids.append(pre_identity_core_id)
            evidence_proposals, contributions = proposals_intersecting_core(
                pre_indices,
                source_proposals,
            )
            tier, winner, agreeing, dissenting, stability = fuse_core_identity(
                evidence_proposals,
                ontology,
            )
            core_tier_counts[tier] = core_tier_counts.get(tier, 0) + 1
            pre_record: dict[str, Any] = {
                "pre_identity_core_id": pre_identity_core_id,
                "source_component_id": source_component_id,
                "source_consensus_tier": source_tier,
                "local_pre_identity_component_id": int(local_pre_core_id),
                "pre_identity_support_gaussian_count": int(pre_indices.size),
                "pre_identity_minimum_camera_support": (
                    int(pre_counts.min()) if pre_counts.size else 0
                ),
                "pre_identity_maximum_camera_support": (
                    int(pre_counts.max()) if pre_counts.size else 0
                ),
                "evidence_camera_count": len(evidence_proposals),
                "evidence_proposal_ids": [
                    int(item.proposal_id) for item in evidence_proposals
                ],
                "proposal_core_contributions": contributions,
                "consensus_tier": tier,
                "semantic_stability": stability,
                "agreeing_proposal_ids": [
                    int(item.proposal_id) for item in agreeing
                ],
                "excluded_dissenting_proposal_ids": [
                    int(item.proposal_id) for item in dissenting
                ],
                "post_identity_component_ids": [],
            }
            if tier == "mixed_or_unstable":
                pre_record.update(
                    {
                        "status": "rejected_mixed_or_unstable_core_identity",
                        "accepted_post_identity_component_count": 0,
                        "accepted_post_identity_gaussian_count": 0,
                    }
                )
                pre_identity_records.append(pre_record)
                continue

            retained_indices, retained_counts = multiview_support_within_core(
                pre_indices,
                agreeing,
                min_gaussian_camera_support=(
                    thresholds.min_gaussian_camera_support
                ),
            )
            points = np.column_stack(
                [
                    vertices[axis][retained_indices].astype(np.float64)
                    for axis in ("x", "y", "z")
                ]
            ) if retained_indices.size else np.empty((0, 3), dtype=np.float64)
            post_splits = split_post_identity_support(
                source_component_id,
                pre_identity_core_id,
                retained_indices,
                retained_counts,
                agreeing,
                points,
                voxel_size,
                thresholds,
            )
            accepted_from_pre = 0
            accepted_gaussians_from_pre = 0
            for post_record in post_splits:
                indices = post_record.pop("_indices")
                camera_counts = post_record.pop("_camera_counts")
                ontology_item = ontology.classes[winner]
                core_record = {
                    "component_id": next_component_id,
                    "class": ontology_item.project_class,
                    "project_id": ontology_item.project_id,
                    "ade_id": ontology_item.ade_id,
                    "kind": ontology_item.kind,
                    "source_consensus_tier": source_tier,
                    "consensus_tier": tier,
                    "source_proposal_ids": [
                        int(item.proposal_id) for item in evidence_proposals
                    ],
                    "agreeing_proposal_ids": [
                        int(item.proposal_id) for item in agreeing
                    ],
                    "excluded_dissenting_proposal_ids": [
                        int(item.proposal_id) for item in dissenting
                    ],
                    "semantic_stability": stability,
                    "median_gaussian_scale": median_scale,
                    "voxel_size": voxel_size,
                    **post_record,
                }
                core_records.append(core_record)
                source_final_core_ids.append(next_component_id)
                pre_record["post_identity_component_ids"].append(
                    next_component_id
                )
                if bool(core_record["accepted"]):
                    accepted_from_pre += 1
                    accepted_gaussians_from_pre += int(indices.size)
                    source_accepted_count += 1
                    source_accepted_gaussians += int(indices.size)
                    accepted_cores.append(
                        SpatialCoreSupport(
                            component_id=next_component_id,
                            source_component_id=source_component_id,
                            project_id=ontology_item.project_id,
                            indices=indices,
                            camera_counts=camera_counts,
                            record=core_record,
                        )
                    )
                next_component_id += 1
            pre_record.update(
                {
                    "status": (
                        "retained_with_post_identity_core"
                        if accepted_from_pre
                        else "rejected_no_post_identity_core_passed_global_gates"
                    ),
                    "post_identity_multiview_gaussian_count": int(
                        retained_indices.size
                    ),
                    "post_identity_spatial_component_count": len(post_splits),
                    "accepted_post_identity_component_count": accepted_from_pre,
                    "accepted_post_identity_gaussian_count": (
                        accepted_gaussians_from_pre
                    ),
                }
            )
            pre_identity_records.append(pre_record)

        source_records.append(
            {
                "source_component_id": source_component_id,
                "source_status": str(source_record_raw.get("status", "")),
                "source_class": str(source_record_raw.get("class", "")),
                "source_project_id": int(source_record_raw.get("project_id", 0)),
                "source_consensus_tier": source_tier,
                "source_view_count": len(source_proposals),
                "source_proposal_ids": [
                    int(item.proposal_id) for item in source_proposals
                ],
                "source_union_gaussian_count": source_union_count,
                "median_gaussian_scale": median_scale,
                "voxel_size": voxel_size,
                "pre_identity_spatial_geometry": spatial_geometry,
                "pre_identity_spatial_core_count": len(spatial_splits),
                "pre_identity_spatial_core_ids": source_pre_core_ids,
                "post_identity_component_ids": source_final_core_ids,
                "accepted_post_identity_component_count": source_accepted_count,
                "accepted_post_identity_gaussian_count": (
                    source_accepted_gaussians
                ),
                "status": (
                    "retained_after_core_first_identity"
                    if source_accepted_count
                    else "rejected_no_core_first_identity_support"
                ),
            }
        )

    if seen_source_proposals != set(proposal_by_id):
        raise ValueError("source components do not partition all proposals")

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
        record["assigned_gaussian_count"] = assigned_by_source.get(
            int(record["source_component_id"]),
            0,
        )

    mixed_source_ids = {
        int(record["source_component_id"])
        for record in source_records
        if record["source_consensus_tier"] == "mixed_or_unstable"
    }
    recovered_mixed_source_ids = {
        source_component_id
        for source_component_id in mixed_source_ids
        if assigned_by_source.get(source_component_id, 0) > 0
    }
    recovered_mixed_gaussians = sum(
        assigned_by_source.get(source_component_id, 0)
        for source_component_id in recovered_mixed_source_ids
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "core_first_mask.npy", proposed)
    np.save(args.output_dir / "core_first_owner_count.npy", owner_count)
    np.save(args.output_dir / "core_first_overlap_mask.npy", overlap)
    np.save(
        args.output_dir / "core_first_cross_class_overlap_mask.npy",
        cross_class_overlap,
    )
    np.save(
        args.output_dir / "core_first_cross_class_tie_mask.npy",
        cross_class_tie,
    )
    np.savez_compressed(
        args.output_dir / "core_first_supports.npz",
        **{
            key: value
            for component_id, (indices, counts) in sorted(exclusive.items())
            for key, value in (
                (f"component_{component_id:06d}_indices", indices),
                (f"component_{component_id:06d}_camera_counts", counts),
            )
        },
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
        "manual_component_decisions": False,
        "offline_threshold_sweep_used": False,
        "order_of_operations": (
            "all_source_graphs_to_multiview_spatial_cores_then_independent_"
            "core_semantic_identity"
        ),
        "identity_policy": (
            "per_core_strict_unanimity_or_exactly_one_outlier_with_all_leave_"
            "one_out_winners_unchanged"
        ),
        "outlier_policy": (
            "exclude_single_dissenting_camera_then_recompute_and_resplit_core"
        ),
        "gaussian_support_policy": (
            "one_presence_vote_per_camera_before_identity_and_again_after_"
            "dissent_removal"
        ),
        "geometry_policy": "adaptive_26_neighbor_voxel_components",
        "cross_class_overlap_policy": (
            "unique_maximum_supporting_camera_count_else_abstain"
        ),
        "visualization_scope": (
            "exact_projection_of_accepted_core_first_3d_support"
        ),
        "parameters": asdict(thresholds),
        "vertex_count": vertex_count,
        "input_proposal_count": len(proposals),
        "input_source_component_count": len(source_records),
        "source_consensus_tier_counts": dict(sorted(source_tier_counts.items())),
        "pre_identity_spatial_core_count": len(pre_identity_records),
        "per_core_consensus_tier_counts": dict(sorted(core_tier_counts.items())),
        "post_identity_spatial_component_count": len(core_records),
        "accepted_core_first_component_count": len(accepted_cores),
        "candidate_core_first_gaussian_count": sum(
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
        "mixed_source_component_count": len(mixed_source_ids),
        "mixed_source_component_with_accepted_core_count": len(
            recovered_mixed_source_ids
        ),
        "mixed_source_component_ids_with_accepted_cores": sorted(
            recovered_mixed_source_ids
        ),
        "mixed_source_assigned_gaussian_count": recovered_mixed_gaussians,
        "class_proposed_core_gaussian_counts": dict(sorted(class_assigned.items())),
        "source_components": source_records,
        "pre_identity_cores": pre_identity_records,
        "components": core_records,
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
    (
        args.output_dir / "core_first_semantic_identity_audit.json"
    ).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "scene": args.scene,
                "input_source_component_count": len(source_records),
                "mixed_source_component_count": len(mixed_source_ids),
                "mixed_source_component_with_accepted_core_count": len(
                    recovered_mixed_source_ids
                ),
                "accepted_core_first_component_count": len(accepted_cores),
                "proposed_core_gaussian_count": int(np.count_nonzero(proposed)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
