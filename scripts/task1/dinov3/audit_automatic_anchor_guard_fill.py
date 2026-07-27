#!/usr/bin/env python3
"""Audit automatic class-consistent anchor guards for incremental fills.

This stage never writes semantic labels or a PLY. For every cached proposed
fill Gaussian, it queries nearby immutable preferred-v2 labels and retains the
point only when same-class anchors are sufficiently numerous, dominate the
local labeled neighborhood, and are closer than competing-class anchors.
Retained points are spatially re-split and must again pass global size and
independent-camera gates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from scripts.task1.common.ply_utils import vertex_data_memmap
from scripts.task1.dinov3.associate_3d_query_regions import (
    QueryProposal,
    load_query_proposals,
)
from scripts.task1.dinov3.audit_incremental_spatial_core_fill import file_sha256
from scripts.task1.dinov3.audit_multiview_spatial_core import (
    supporting_cameras_by_spatial_component,
)
from scripts.task1.grounding.cluster_semantic_flashsplat_proposals import (
    voxel_components,
)


SOURCE = "dinov3_automatic_class_consistent_anchor_guard_fill_audit"
CONTRACT = "report_only_dinov3_automatic_anchor_guard_fill_v1"
UPSTREAM_CONTRACT = "report_only_dinov3_incremental_spatial_core_fill_v1"


@dataclass(frozen=True)
class AnchorProfile:
    name: str
    radius_voxel_multiplier: float


PROFILES = (
    AnchorProfile("radius_1x", 1.0),
    AnchorProfile("radius_2x", 2.0),
    AnchorProfile("radius_4x", 4.0),
)


@dataclass(frozen=True)
class AnchorGuardThresholds:
    neighbor_count: int = 16
    min_same_class_neighbors: int = 4
    min_same_class_fraction: float = 0.75
    max_same_to_competing_distance_ratio: float = 1.0
    min_spatial_component_gaussians: int = 500
    min_spatial_component_cameras: int = 2
    query_workers: int = 4

    def validate(self) -> None:
        if self.neighbor_count < 1:
            raise ValueError("neighbor_count must be positive")
        if not 1 <= self.min_same_class_neighbors <= self.neighbor_count:
            raise ValueError(
                "min_same_class_neighbors must be within the neighbor count"
            )
        if not 0.0 < self.min_same_class_fraction <= 1.0:
            raise ValueError("min_same_class_fraction must be in (0, 1]")
        if self.max_same_to_competing_distance_ratio <= 0.0:
            raise ValueError(
                "max_same_to_competing_distance_ratio must be positive"
            )
        if self.min_spatial_component_gaussians < 1:
            raise ValueError(
                "min_spatial_component_gaussians must be positive"
            )
        if self.min_spatial_component_cameras < 2:
            raise ValueError(
                "min_spatial_component_cameras must be at least two"
            )
        if self.query_workers < 1:
            raise ValueError("query_workers must be positive")


@dataclass(frozen=True)
class UpstreamFillComponent:
    component_id: int
    project_id: int
    class_name: str
    voxel_size: float
    source_proposal_ids: tuple[int, ...]
    indices: np.ndarray
    camera_counts: np.ndarray


def load_upstream_components(
    report: dict[str, Any],
    archive: Mapping[str, np.ndarray],
    *,
    vertex_count: int,
    scene: str,
) -> list[UpstreamFillComponent]:
    if report.get("contract") != UPSTREAM_CONTRACT:
        raise ValueError("incremental-fill source has the wrong contract")
    if str(report.get("scene")) != scene:
        raise ValueError("incremental-fill source scene differs from the request")
    if not bool(report.get("report_only")):
        raise ValueError("incremental-fill source must be report-only")
    if int(report.get("vertex_count", -1)) != vertex_count:
        raise ValueError("incremental-fill vertex count differs from labels")
    if not bool(report.get("preferred_labels_read_only")):
        raise ValueError("incremental-fill source did not preserve preferred labels")
    if bool(report.get("preferred_labels_modified")):
        raise ValueError("incremental-fill source reports modified preferred labels")

    components: list[UpstreamFillComponent] = []
    seen_ids: set[int] = set()
    seen_indices = np.zeros((vertex_count,), dtype=bool)
    for raw in report.get("components", []):
        if not bool(raw.get("accepted")):
            continue
        component_id = int(raw["component_id"])
        if component_id < 1 or component_id in seen_ids:
            raise ValueError("accepted source component IDs are invalid")
        seen_ids.add(component_id)
        prefix = f"component_{component_id:06d}"
        index_key = f"{prefix}_indices"
        count_key = f"{prefix}_camera_counts"
        if index_key not in archive or count_key not in archive:
            raise ValueError(f"missing source support for component {component_id}")
        indices = np.asarray(archive[index_key], dtype=np.uint32)
        camera_counts = np.asarray(archive[count_key], dtype=np.uint16)
        if indices.ndim != 1 or camera_counts.shape != indices.shape:
            raise ValueError("source indices and camera counts must align")
        if indices.size and (
            int(indices[-1]) >= vertex_count
            or np.any(indices[1:] <= indices[:-1])
        ):
            raise ValueError("source indices must be valid and strictly increasing")
        if indices.size and np.any(seen_indices[indices]):
            raise ValueError("accepted source supports overlap")
        if camera_counts.size and np.any(camera_counts < 2):
            raise ValueError("source support is not multiview")
        if int(raw["support_gaussian_count"]) != int(indices.size):
            raise ValueError("source support count differs from its report")
        voxel_size = float(raw["voxel_size"])
        if not np.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("source component voxel size must be positive")
        seen_indices[indices] = True
        components.append(
            UpstreamFillComponent(
                component_id=component_id,
                project_id=int(raw["project_id"]),
                class_name=str(raw["class"]),
                voxel_size=voxel_size,
                source_proposal_ids=tuple(
                    int(value) for value in raw.get("source_proposal_ids", [])
                ),
                indices=indices,
                camera_counts=camera_counts,
            )
        )

    expected_components = int(
        report.get("accepted_residual_component_count", -1)
    )
    expected_gaussians = int(report.get("incremental_fill_gaussian_count", -1))
    if len(components) != expected_components:
        raise ValueError("source accepted component count differs from its report")
    if sum(int(item.indices.size) for item in components) != expected_gaussians:
        raise ValueError("source sparse support total differs from its report")
    return components


def query_anchor_neighbors(
    tree: cKDTree,
    candidate_points: np.ndarray,
    *,
    neighbor_count: int,
    maximum_radius: float,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    distances, indices = tree.query(
        np.asarray(candidate_points, dtype=np.float64),
        k=neighbor_count,
        distance_upper_bound=maximum_radius,
        workers=workers,
    )
    if neighbor_count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    return (
        np.asarray(distances, dtype=np.float64),
        np.asarray(indices, dtype=np.int64),
    )


def class_consistent_anchor_mask(
    distances: np.ndarray,
    neighbor_indices: np.ndarray,
    anchor_labels: np.ndarray,
    *,
    target_project_id: int,
    radius: float,
    min_same_class_neighbors: int,
    min_same_class_fraction: float,
    max_same_to_competing_distance_ratio: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return a scene- and class-agnostic local anchor decision per point."""

    dist = np.asarray(distances, dtype=np.float64)
    neighbor_ids = np.asarray(neighbor_indices, dtype=np.int64)
    labels = np.asarray(anchor_labels)
    if dist.ndim != 2 or neighbor_ids.shape != dist.shape:
        raise ValueError("neighbor distances and indices must be aligned matrices")
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("anchor_labels must be a one-dimensional integer array")
    if radius <= 0.0 or not np.isfinite(radius):
        raise ValueError("radius must be finite and positive")
    valid = (
        np.isfinite(dist)
        & (dist <= radius)
        & (neighbor_ids >= 0)
        & (neighbor_ids < labels.size)
    )
    safe_ids = np.where(valid, neighbor_ids, 0)
    neighbor_labels = labels[safe_ids]
    same = valid & (neighbor_labels == target_project_id)
    competing = valid & (neighbor_labels != target_project_id)
    valid_count = valid.sum(axis=1, dtype=np.int32)
    same_count = same.sum(axis=1, dtype=np.int32)
    competing_count = competing.sum(axis=1, dtype=np.int32)
    same_fraction = np.divide(
        same_count,
        np.maximum(valid_count, 1),
        dtype=np.float64,
    )
    nearest_same = np.min(np.where(same, dist, np.inf), axis=1)
    nearest_competing = np.min(np.where(competing, dist, np.inf), axis=1)
    distance_consistent = (
        nearest_same
        <= nearest_competing * max_same_to_competing_distance_ratio
    )
    accepted = (
        (same_count >= min_same_class_neighbors)
        & (same_fraction >= min_same_class_fraction)
        & distance_consistent
    )
    return accepted, {
        "valid_neighbor_count": valid_count,
        "same_class_neighbor_count": same_count,
        "competing_class_neighbor_count": competing_count,
        "same_class_fraction": same_fraction,
        "nearest_same_class_distance": nearest_same,
        "nearest_competing_class_distance": nearest_competing,
    }


def _finite_quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"p10": None, "p50": None, "p90": None}
    quantiles = np.quantile(finite, [0.10, 0.50, 0.90])
    return {
        "p10": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p90": float(quantiles[2]),
    }


def split_guarded_support(
    component: UpstreamFillComponent,
    guarded_indices: np.ndarray,
    guarded_camera_counts: np.ndarray,
    guarded_points: np.ndarray,
    agreeing_proposals: list[QueryProposal],
    thresholds: AnchorGuardThresholds,
) -> list[dict[str, Any]]:
    if guarded_indices.size == 0:
        return []
    point_components, component_sizes, _geometry = voxel_components(
        np.asarray(guarded_points, dtype=np.float64),
        component.voxel_size,
    )
    cameras = supporting_cameras_by_spatial_component(
        guarded_indices,
        point_components,
        int(component_sizes.size),
        agreeing_proposals,
    )
    splits: list[dict[str, Any]] = []
    for local_component_id, size_value in enumerate(component_sizes):
        selected = point_components == local_component_id
        indices = guarded_indices[selected]
        counts = guarded_camera_counts[selected]
        independent_cameras = sorted(cameras[local_component_id])
        size = int(size_value)
        reasons: list[str] = []
        if size < thresholds.min_spatial_component_gaussians:
            reasons.append(
                f"retained_gaussians<{thresholds.min_spatial_component_gaussians}"
            )
        if len(independent_cameras) < thresholds.min_spatial_component_cameras:
            reasons.append(
                "independent_cameras<"
                f"{thresholds.min_spatial_component_cameras}"
            )
        splits.append(
            {
                "local_guarded_component_id": int(local_component_id),
                "accepted": not reasons,
                "status": (
                    "accepted_report_only_automatic_anchor_guard_fill"
                    if not reasons
                    else "rejected_after_automatic_anchor_guard"
                ),
                "reasons": reasons,
                "support_gaussian_count": size,
                "independent_camera_count": len(independent_cameras),
                "independent_camera_indices": independent_cameras,
                "_indices": indices,
                "_camera_counts": counts,
            }
        )
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incremental-fill-report", required=True, type=Path)
    parser.add_argument("--incremental-fill-supports", required=True, type=Path)
    parser.add_argument("--proposal-manifest", required=True, type=Path)
    parser.add_argument("--preferred-labels", required=True, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--neighbor-count", default=16, type=int)
    parser.add_argument("--min-same-class-neighbors", default=4, type=int)
    parser.add_argument("--min-same-class-fraction", default=0.75, type=float)
    parser.add_argument(
        "--max-same-to-competing-distance-ratio",
        default=1.0,
        type=float,
    )
    parser.add_argument("--min-spatial-component-gaussians", default=500, type=int)
    parser.add_argument("--min-spatial-component-cameras", default=2, type=int)
    parser.add_argument("--query-workers", default=4, type=int)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (
        args.incremental_fill_report,
        args.incremental_fill_supports,
        args.proposal_manifest,
        args.preferred_labels,
        args.source_ply,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    thresholds = AnchorGuardThresholds(
        neighbor_count=args.neighbor_count,
        min_same_class_neighbors=args.min_same_class_neighbors,
        min_same_class_fraction=args.min_same_class_fraction,
        max_same_to_competing_distance_ratio=(
            args.max_same_to_competing_distance_ratio
        ),
        min_spatial_component_gaussians=args.min_spatial_component_gaussians,
        min_spatial_component_cameras=args.min_spatial_component_cameras,
        query_workers=args.query_workers,
    )
    thresholds.validate()

    preferred_hash = file_sha256(args.preferred_labels)
    preferred_labels = np.load(args.preferred_labels, mmap_mode="r")
    if preferred_labels.ndim != 1 or not np.issubdtype(
        preferred_labels.dtype, np.integer
    ):
        raise ValueError("preferred labels must be a one-dimensional integer array")
    if preferred_labels.size < 1 or np.any(preferred_labels < 0):
        raise ValueError("preferred labels must be non-negative and non-empty")
    vertex_count = int(preferred_labels.size)
    preferred_nonzero = np.asarray(preferred_labels != 0)
    if not np.any(preferred_nonzero):
        raise ValueError("preferred labels contain no nonzero anchors")

    upstream_report = json.loads(
        args.incremental_fill_report.read_text(encoding="utf-8")
    )
    with np.load(args.incremental_fill_supports, allow_pickle=False) as archive:
        upstream_components = load_upstream_components(
            upstream_report,
            archive,
            vertex_count=vertex_count,
            scene=args.scene,
        )
        copied_components = [
            UpstreamFillComponent(
                component_id=item.component_id,
                project_id=item.project_id,
                class_name=item.class_name,
                voxel_size=item.voxel_size,
                source_proposal_ids=item.source_proposal_ids,
                indices=item.indices.copy(),
                camera_counts=item.camera_counts.copy(),
            )
            for item in upstream_components
        ]
    proposals, proposal_manifest = load_query_proposals(args.proposal_manifest)
    if int(proposal_manifest.get("vertex_count", -1)) != vertex_count:
        raise ValueError("proposal manifest vertex count differs from labels")
    proposal_by_id = {item.proposal_id: item for item in proposals}
    if len(proposal_by_id) != len(proposals):
        raise ValueError("proposal IDs are not unique")

    header, vertices = vertex_data_memmap(args.source_ply)
    if not header.elements or int(header.elements[0].count) != vertex_count:
        raise ValueError("source PLY vertex count differs from labels")
    if not {"x", "y", "z"}.issubset(vertices.dtype.names or ()):
        raise ValueError("source PLY lacks coordinates")
    anchor_indices = np.flatnonzero(preferred_nonzero)
    anchor_points = np.column_stack(
        [
            vertices[axis][anchor_indices].astype(np.float64)
            for axis in ("x", "y", "z")
        ]
    )
    anchor_labels = np.asarray(preferred_labels[anchor_indices], dtype=np.int32)
    anchor_tree = cKDTree(anchor_points)

    profile_states: dict[str, dict[str, Any]] = {}
    for profile in PROFILES:
        profile_states[profile.name] = {
            "profile": profile,
            "mask": np.zeros((vertex_count,), dtype=bool),
            "supports": {},
            "components": [],
            "source_components": [],
            "next_component_id": 1,
        }

    for component in copied_components:
        if np.any(preferred_nonzero[component.indices]):
            raise ValueError("upstream fill overlaps immutable preferred labels")
        candidate_points = np.column_stack(
            [
                vertices[axis][component.indices].astype(np.float64)
                for axis in ("x", "y", "z")
            ]
        )
        maximum_radius = (
            max(item.radius_voxel_multiplier for item in PROFILES)
            * component.voxel_size
        )
        distances, neighbor_ids = query_anchor_neighbors(
            anchor_tree,
            candidate_points,
            neighbor_count=thresholds.neighbor_count,
            maximum_radius=maximum_radius,
            workers=thresholds.query_workers,
        )
        agreeing_proposals: list[QueryProposal] = []
        for proposal_id in component.source_proposal_ids:
            try:
                agreeing_proposals.append(proposal_by_id[proposal_id])
            except KeyError as exc:
                raise ValueError(
                    f"source component references missing proposal {proposal_id}"
                ) from exc
        if len({item.camera_index for item in agreeing_proposals}) != len(
            agreeing_proposals
        ):
            raise ValueError(
                "source component contains multiple proposals from one camera"
            )

        for profile in PROFILES:
            state = profile_states[profile.name]
            radius = profile.radius_voxel_multiplier * component.voxel_size
            guarded, metrics = class_consistent_anchor_mask(
                distances,
                neighbor_ids,
                anchor_labels,
                target_project_id=component.project_id,
                radius=radius,
                min_same_class_neighbors=thresholds.min_same_class_neighbors,
                min_same_class_fraction=thresholds.min_same_class_fraction,
                max_same_to_competing_distance_ratio=(
                    thresholds.max_same_to_competing_distance_ratio
                ),
            )
            guarded_indices = component.indices[guarded]
            guarded_counts = component.camera_counts[guarded]
            splits = split_guarded_support(
                component,
                guarded_indices,
                guarded_counts,
                candidate_points[guarded],
                agreeing_proposals,
                thresholds,
            )
            child_ids: list[int] = []
            accepted_count = 0
            accepted_gaussians = 0
            for split in splits:
                indices = split.pop("_indices")
                counts = split.pop("_camera_counts")
                result_id = int(state["next_component_id"])
                state["next_component_id"] = result_id + 1
                result = {
                    "component_id": result_id,
                    "source_residual_component_id": component.component_id,
                    "class": component.class_name,
                    "project_id": component.project_id,
                    "voxel_size": component.voxel_size,
                    "anchor_radius": radius,
                    **split,
                }
                state["components"].append(result)
                child_ids.append(result_id)
                if bool(result["accepted"]):
                    if np.any(state["mask"][indices]):
                        raise ValueError(
                            "automatic guarded supports overlap within a profile"
                        )
                    state["mask"][indices] = True
                    prefix = f"component_{result_id:06d}"
                    state["supports"][f"{prefix}_indices"] = indices
                    state["supports"][f"{prefix}_camera_counts"] = counts
                    accepted_count += 1
                    accepted_gaussians += int(indices.size)
            same_fraction = metrics["same_class_fraction"]
            state["source_components"].append(
                {
                    "source_residual_component_id": component.component_id,
                    "class": component.class_name,
                    "project_id": component.project_id,
                    "source_gaussian_count": int(component.indices.size),
                    "voxel_size": component.voxel_size,
                    "anchor_radius": radius,
                    "point_guard_retained_gaussian_count": int(
                        np.count_nonzero(guarded)
                    ),
                    "point_guard_retained_fraction": float(np.mean(guarded)),
                    "same_class_neighbor_count_quantiles": _finite_quantiles(
                        metrics["same_class_neighbor_count"]
                    ),
                    "same_class_fraction_quantiles": _finite_quantiles(
                        same_fraction
                    ),
                    "nearest_same_class_distance_quantiles": _finite_quantiles(
                        metrics["nearest_same_class_distance"]
                    ),
                    "nearest_competing_class_distance_quantiles": (
                        _finite_quantiles(
                            metrics["nearest_competing_class_distance"]
                        )
                    ),
                    "guarded_spatial_component_count": len(splits),
                    "accepted_guarded_component_count": accepted_count,
                    "accepted_guarded_gaussian_count": accepted_gaussians,
                    "result_component_ids": child_ids,
                }
            )

    if file_sha256(args.preferred_labels) != preferred_hash:
        raise RuntimeError("preferred label source changed during the audit")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    profile_summaries: list[dict[str, Any]] = []
    source_fill_count = sum(int(item.indices.size) for item in copied_components)
    preferred_labeled_count = int(np.count_nonzero(preferred_nonzero))
    preferred_unlabeled_count = vertex_count - preferred_labeled_count
    for profile in PROFILES:
        state = profile_states[profile.name]
        profile_dir = args.output_dir / profile.name
        profile_dir.mkdir()
        fill_mask = np.asarray(state["mask"], dtype=bool)
        if np.any(fill_mask & preferred_nonzero):
            raise AssertionError("automatic fill overlaps immutable preferred labels")
        fill_count = int(np.count_nonzero(fill_mask))
        np.save(profile_dir / "automatic_fill_mask.npy", fill_mask)
        np.savez_compressed(
            profile_dir / "automatic_fill_supports.npz",
            **state["supports"],
        )
        accepted_components = sum(
            int(bool(record["accepted"])) for record in state["components"]
        )
        report = {
            "source": SOURCE,
            "contract": CONTRACT,
            "scene": args.scene,
            "profile": asdict(profile),
            "thresholds": asdict(thresholds),
            "incremental_fill_report": str(args.incremental_fill_report),
            "incremental_fill_supports": str(args.incremental_fill_supports),
            "proposal_manifest": str(args.proposal_manifest),
            "preferred_labels": str(args.preferred_labels),
            "preferred_labels_sha256": preferred_hash,
            "source_ply": str(args.source_ply),
            "report_only": True,
            "preferred_labels_read_only": True,
            "preferred_labels_modified": False,
            "scene_specific_rules": False,
            "class_specific_thresholds": False,
            "manual_component_decisions": False,
            "semantic_labels_written": False,
            "semantic_project_class_arrays_written": False,
            "label_map_written": False,
            "semantic_ply_written": False,
            "vertex_count": vertex_count,
            "preferred_labeled_gaussian_count": preferred_labeled_count,
            "source_incremental_fill_gaussian_count": source_fill_count,
            "source_incremental_fill_component_count": len(copied_components),
            "accepted_residual_component_count": accepted_components,
            "incremental_fill_gaussian_count": fill_count,
            "preferred_unlabeled_recovery_ratio": (
                fill_count / float(max(preferred_unlabeled_count, 1))
            ),
            "source_fill_retained_ratio": (
                fill_count / float(max(source_fill_count, 1))
            ),
            "combined_assigned_gaussian_count": (
                preferred_labeled_count + fill_count
            ),
            "combined_assigned_ratio": (
                (preferred_labeled_count + fill_count) / float(vertex_count)
            ),
            "source_components": state["source_components"],
            "components": state["components"],
        }
        report_path = profile_dir / "automatic_anchor_guard_audit.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        profile_summaries.append(
            {
                "profile": asdict(profile),
                "report": str(report_path),
                "accepted_component_count": accepted_components,
                "fill_gaussian_count": fill_count,
                "source_fill_retained_ratio": report[
                    "source_fill_retained_ratio"
                ],
                "preferred_unlabeled_recovery_ratio": report[
                    "preferred_unlabeled_recovery_ratio"
                ],
                "combined_assigned_ratio": report["combined_assigned_ratio"],
            }
        )

    summary = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": args.scene,
        "report_only": True,
        "preferred_labels_read_only": True,
        "preferred_labels_modified": False,
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "manual_component_decisions": False,
        "semantic_labels_written": False,
        "semantic_project_class_arrays_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
        "vertex_count": vertex_count,
        "preferred_labeled_gaussian_count": preferred_labeled_count,
        "source_incremental_fill_gaussian_count": source_fill_count,
        "anchor_gaussian_count": int(anchor_indices.size),
        "thresholds": asdict(thresholds),
        "profiles": profile_summaries,
    }
    (args.output_dir / "automatic_anchor_guard_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
