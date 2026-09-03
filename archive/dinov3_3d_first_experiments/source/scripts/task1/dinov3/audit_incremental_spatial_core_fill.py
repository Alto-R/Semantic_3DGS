#!/usr/bin/env python3
"""Audit fill-only residuals from resolved DINOv3 multiview spatial cores.

The accepted label source is immutable.  This report-only stage subtracts
every already-labeled Gaussian from the resolved spatial-core supports, splits
each unlabeled residual with the same adaptive 26-neighbor voxel rule, and
retains only residual components that pass global size and independent-camera
gates.  It writes masks, sparse supports, and a JSON report; it never writes a
semantic label array, label map, or PLY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.task1.common.ply_utils import vertex_data_memmap
from scripts.task1.dinov3.associate_3d_query_regions import (
    QueryProposal,
    load_query_proposals,
)
from scripts.task1.dinov3.audit_multiview_spatial_core import (
    supporting_cameras_by_spatial_component,
)
from scripts.task1.dinov3.propagate_dense_labels_from_region_seeds import (
    adaptive_voxel_size,
)
from scripts.task1.grounding.cluster_semantic_flashsplat_proposals import (
    voxel_components,
)


SOURCE = "dinov3_incremental_spatial_core_fill_audit"
CONTRACT = "report_only_dinov3_incremental_spatial_core_fill_v1"
SPATIAL_CORE_CONTRACT = "report_only_dinov3_multiview_spatial_core_v1"


@dataclass(frozen=True)
class IncrementalFillThresholds:
    min_spatial_component_gaussians: int = 500
    min_spatial_component_cameras: int = 2
    voxel_scale_multiplier: float = 4.0
    min_voxel_size: float = 0.01
    max_voxel_size: float = 0.20

    def validate(self) -> None:
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
class ResolvedCoreSupport:
    component_id: int
    project_id: int
    class_name: str
    source_component_id: int
    source_proposal_ids: tuple[int, ...]
    indices: np.ndarray
    camera_counts: np.ndarray
    record: dict[str, Any]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strictly_increasing(indices: np.ndarray) -> bool:
    return indices.size < 2 or bool(np.all(indices[1:] > indices[:-1]))


def load_resolved_core_supports(
    report: dict[str, Any],
    archive: Mapping[str, np.ndarray],
    vertex_count: int,
) -> list[ResolvedCoreSupport]:
    """Load the post-ownership support for every accepted spatial core."""

    if report.get("contract") != SPATIAL_CORE_CONTRACT:
        raise ValueError("spatial-core report has the wrong contract")
    if not bool(report.get("report_only")):
        raise ValueError("spatial-core source must be report-only")
    if int(report.get("vertex_count", -1)) != vertex_count:
        raise ValueError("spatial-core report vertex count differs from labels")

    supports: list[ResolvedCoreSupport] = []
    seen_component_ids: set[int] = set()
    seen_indices = np.zeros((vertex_count,), dtype=bool)
    for raw in report.get("components", []):
        if not bool(raw.get("accepted")):
            continue
        component_id = int(raw["component_id"])
        if component_id < 1 or component_id in seen_component_ids:
            raise ValueError("accepted spatial-core component IDs are invalid")
        seen_component_ids.add(component_id)
        prefix = f"component_{component_id:06d}"
        index_key = f"{prefix}_indices"
        count_key = f"{prefix}_camera_counts"
        if index_key not in archive or count_key not in archive:
            raise ValueError(f"missing resolved support for component {component_id}")
        indices = np.asarray(archive[index_key], dtype=np.uint32)
        camera_counts = np.asarray(archive[count_key], dtype=np.uint16)
        if indices.ndim != 1 or camera_counts.shape != indices.shape:
            raise ValueError("resolved indices and camera counts must align")
        if not _strictly_increasing(indices):
            raise ValueError("resolved spatial-core indices must be strictly increasing")
        if indices.size and int(indices[-1]) >= vertex_count:
            raise ValueError("resolved spatial-core index is outside the source PLY")
        if camera_counts.size and np.any(camera_counts < 2):
            raise ValueError("resolved spatial-core support is not multiview")
        if indices.size and np.any(seen_indices[indices]):
            raise ValueError("resolved spatial-core ownership is not exclusive")
        seen_indices[indices] = True
        supports.append(
            ResolvedCoreSupport(
                component_id=component_id,
                project_id=int(raw["project_id"]),
                class_name=str(raw["class"]),
                source_component_id=int(raw["source_component_id"]),
                source_proposal_ids=tuple(
                    int(value) for value in raw.get("source_proposal_ids", [])
                ),
                indices=indices,
                camera_counts=camera_counts,
                record=raw,
            )
        )
    return supports


def split_incremental_residual(
    parent_component_id: int,
    residual_indices: np.ndarray,
    camera_counts: np.ndarray,
    agreeing_proposals: list[QueryProposal],
    points: np.ndarray,
    log_scales: np.ndarray,
    thresholds: IncrementalFillThresholds,
) -> tuple[list[dict[str, Any]], float, float, dict[str, Any]]:
    """Split one unlabeled core residual and apply only size/camera gates."""

    indices = np.asarray(residual_indices, dtype=np.uint32)
    counts = np.asarray(camera_counts, dtype=np.uint16)
    if counts.shape != indices.shape:
        raise ValueError("camera counts must align with residual indices")
    if points.shape != (indices.size, 3) or log_scales.shape != (indices.size, 3):
        raise ValueError("geometry must align with residual indices")
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

    records: list[dict[str, Any]] = []
    for local_component_id, size_value in enumerate(component_sizes):
        selected = point_components == local_component_id
        local_counts = counts[selected]
        size = int(size_value)
        independent_cameras = sorted(component_cameras[local_component_id])
        large_enough = size >= thresholds.min_spatial_component_gaussians
        enough_cameras = (
            len(independent_cameras) >= thresholds.min_spatial_component_cameras
        )
        accepted = large_enough and enough_cameras
        if not large_enough:
            status = "rejected_insufficient_residual_gaussians"
        elif not enough_cameras:
            status = "rejected_insufficient_independent_cameras"
        else:
            status = "accepted_report_only_incremental_fill_residual"
        unique_counts, frequencies = np.unique(local_counts, return_counts=True)
        records.append(
            {
                "parent_spatial_core_component_id": parent_component_id,
                "local_residual_component_id": local_component_id,
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
    return records, median_scale, voxel_size, geometry


def _component_proposals(
    core: ResolvedCoreSupport,
    proposal_by_id: dict[int, QueryProposal],
) -> list[QueryProposal]:
    proposals: list[QueryProposal] = []
    for proposal_id in core.source_proposal_ids:
        try:
            proposals.append(proposal_by_id[proposal_id])
        except KeyError as exc:
            raise ValueError(
                f"spatial core references missing proposal {proposal_id}"
            ) from exc
    if len({item.camera_index for item in proposals}) != len(proposals):
        raise ValueError("spatial core contains multiple proposals from one camera")
    return proposals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial-core-report", required=True, type=Path)
    parser.add_argument("--spatial-core-supports", required=True, type=Path)
    parser.add_argument("--proposal-manifest", required=True, type=Path)
    parser.add_argument("--preferred-labels", required=True, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--min-spatial-component-gaussians", default=500, type=int)
    parser.add_argument("--min-spatial-component-cameras", default=2, type=int)
    parser.add_argument("--voxel-scale-multiplier", default=4.0, type=float)
    parser.add_argument("--min-voxel-size", default=0.01, type=float)
    parser.add_argument("--max-voxel-size", default=0.20, type=float)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (
        args.spatial_core_report,
        args.spatial_core_supports,
        args.proposal_manifest,
        args.preferred_labels,
        args.source_ply,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    thresholds = IncrementalFillThresholds(
        min_spatial_component_gaussians=args.min_spatial_component_gaussians,
        min_spatial_component_cameras=args.min_spatial_component_cameras,
        voxel_scale_multiplier=args.voxel_scale_multiplier,
        min_voxel_size=args.min_voxel_size,
        max_voxel_size=args.max_voxel_size,
    )
    thresholds.validate()

    preferred_labels_hash = file_sha256(args.preferred_labels)
    preferred_labels = np.load(args.preferred_labels, mmap_mode="r")
    if preferred_labels.ndim != 1 or not np.issubdtype(
        preferred_labels.dtype, np.integer
    ):
        raise ValueError("preferred labels must be a one-dimensional integer array")
    if preferred_labels.size < 1 or np.any(preferred_labels < 0):
        raise ValueError("preferred labels must be non-negative and non-empty")
    vertex_count = int(preferred_labels.size)

    spatial_report = json.loads(
        args.spatial_core_report.read_text(encoding="utf-8")
    )
    with np.load(args.spatial_core_supports, allow_pickle=False) as archive:
        resolved_cores = load_resolved_core_supports(
            spatial_report, archive, vertex_count
        )
    proposals, proposal_manifest = load_query_proposals(args.proposal_manifest)
    if int(proposal_manifest.get("vertex_count", -1)) != vertex_count:
        raise ValueError("proposal manifest vertex count differs from labels")
    proposal_by_id = {item.proposal_id: item for item in proposals}
    if len(proposal_by_id) != len(proposals):
        raise ValueError("proposal IDs are not unique")

    header, vertices = vertex_data_memmap(args.source_ply)
    if not header.elements or int(header.elements[0].count) != vertex_count:
        raise ValueError("source PLY vertex count differs from labels")
    required_fields = {"x", "y", "z", "scale_0", "scale_1", "scale_2"}
    if not required_fields.issubset(vertices.dtype.names or ()):
        raise ValueError("source PLY lacks coordinates or Gaussian scales")

    preferred_labeled = np.asarray(preferred_labels != 0)
    residual_candidate_mask = np.zeros((vertex_count,), dtype=bool)
    excluded_preferred_mask = np.zeros((vertex_count,), dtype=bool)
    incremental_fill_mask = np.zeros((vertex_count,), dtype=bool)
    parent_records: list[dict[str, Any]] = []
    residual_records: list[dict[str, Any]] = []
    accepted_supports: dict[str, np.ndarray] = {}
    class_fill_counts: dict[str, int] = {}
    next_residual_component_id = 1

    for core in resolved_cores:
        already_labeled = preferred_labeled[core.indices]
        excluded_indices = core.indices[already_labeled]
        residual_indices = core.indices[~already_labeled]
        residual_counts = core.camera_counts[~already_labeled]
        excluded_preferred_mask[excluded_indices] = True
        residual_candidate_mask[residual_indices] = True
        agreeing_proposals = _component_proposals(core, proposal_by_id)

        if residual_indices.size:
            points = np.column_stack(
                [
                    vertices[axis][residual_indices].astype(np.float64)
                    for axis in ("x", "y", "z")
                ]
            )
            log_scales = np.column_stack(
                [
                    vertices[axis][residual_indices].astype(np.float64)
                    for axis in ("scale_0", "scale_1", "scale_2")
                ]
            )
        else:
            points = np.empty((0, 3), dtype=np.float64)
            log_scales = np.empty((0, 3), dtype=np.float64)

        splits, median_scale, voxel_size, geometry = split_incremental_residual(
            core.component_id,
            residual_indices,
            residual_counts,
            agreeing_proposals,
            points,
            log_scales,
            thresholds,
        )
        accepted_count = 0
        accepted_gaussians = 0
        child_component_ids: list[int] = []
        for split in splits:
            indices = split.pop("_indices")
            counts = split.pop("_camera_counts")
            residual_record = {
                "component_id": next_residual_component_id,
                "class": core.class_name,
                "project_id": core.project_id,
                "source_component_id": core.source_component_id,
                "parent_spatial_core_component_id": core.component_id,
                "source_proposal_ids": list(core.source_proposal_ids),
                "median_gaussian_scale": median_scale,
                "voxel_size": voxel_size,
                **split,
            }
            residual_records.append(residual_record)
            child_component_ids.append(next_residual_component_id)
            if bool(residual_record["accepted"]):
                if np.any(incremental_fill_mask[indices]):
                    raise ValueError("accepted residual supports overlap")
                incremental_fill_mask[indices] = True
                accepted_count += 1
                accepted_gaussians += int(indices.size)
                prefix = f"component_{next_residual_component_id:06d}"
                accepted_supports[f"{prefix}_indices"] = indices
                accepted_supports[f"{prefix}_camera_counts"] = counts
                class_fill_counts[core.class_name] = (
                    class_fill_counts.get(core.class_name, 0) + int(indices.size)
                )
            next_residual_component_id += 1

        parent_records.append(
            {
                "parent_spatial_core_component_id": core.component_id,
                "source_component_id": core.source_component_id,
                "class": core.class_name,
                "project_id": core.project_id,
                "source_proposal_ids": list(core.source_proposal_ids),
                "resolved_core_gaussian_count": int(core.indices.size),
                "excluded_preferred_label_gaussian_count": int(
                    excluded_indices.size
                ),
                "unlabeled_residual_gaussian_count": int(residual_indices.size),
                "residual_spatial_component_count": len(splits),
                "accepted_residual_component_count": accepted_count,
                "accepted_incremental_fill_gaussian_count": accepted_gaussians,
                "median_gaussian_scale": median_scale,
                "voxel_size": voxel_size,
                "spatial_geometry": geometry,
                "residual_component_ids": child_component_ids,
            }
        )

    if np.any(incremental_fill_mask & preferred_labeled):
        raise AssertionError("incremental fill overlaps immutable preferred labels")
    if file_sha256(args.preferred_labels) != preferred_labels_hash:
        raise RuntimeError("preferred label source changed during the audit")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "incremental_fill_mask.npy", incremental_fill_mask)
    np.save(
        args.output_dir / "unlabeled_residual_candidate_mask.npy",
        residual_candidate_mask,
    )
    np.save(
        args.output_dir / "excluded_preferred_label_mask.npy",
        excluded_preferred_mask,
    )
    np.savez_compressed(
        args.output_dir / "incremental_fill_supports.npz",
        **accepted_supports,
    )

    preferred_labeled_count = int(np.count_nonzero(preferred_labeled))
    preferred_unlabeled_count = vertex_count - preferred_labeled_count
    fill_count = int(np.count_nonzero(incremental_fill_mask))
    combined_count = preferred_labeled_count + fill_count
    resolved_core_count = int(np.count_nonzero(
        residual_candidate_mask | excluded_preferred_mask
    ))
    report = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": args.scene,
        "spatial_core_report": str(args.spatial_core_report),
        "spatial_core_supports": str(args.spatial_core_supports),
        "proposal_manifest": str(args.proposal_manifest),
        "preferred_labels": str(args.preferred_labels),
        "preferred_labels_sha256": preferred_labels_hash,
        "source_ply": str(args.source_ply),
        "report_only": True,
        "preferred_labels_read_only": True,
        "preferred_labels_modified": False,
        "semantic_labels_written": False,
        "semantic_project_class_arrays_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
        "inference_rerun": False,
        "flashsplat_rerun": False,
        "spatial_core_rerun": False,
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "offline_threshold_sweep_used": False,
        "residual_policy": (
            "resolved_spatial_core_support_minus_all_nonzero_preferred_labels"
        ),
        "geometry_policy": "adaptive_26_neighbor_voxel_components",
        "acceptance_policy": (
            "global_minimum_residual_size_and_independent_camera_count"
        ),
        "visualization_scope": (
            "exact_projection_of_accepted_unlabeled_residual_3d_support"
        ),
        "parameters": vars(thresholds),
        "vertex_count": vertex_count,
        "preferred_labeled_gaussian_count": preferred_labeled_count,
        "preferred_unlabeled_gaussian_count": preferred_unlabeled_count,
        "preferred_assigned_ratio": preferred_labeled_count / float(vertex_count),
        "resolved_spatial_core_count": len(resolved_cores),
        "resolved_spatial_core_gaussian_count": resolved_core_count,
        "resolved_core_preferred_overlap_gaussian_count": int(
            np.count_nonzero(excluded_preferred_mask)
        ),
        "unlabeled_residual_candidate_gaussian_count": int(
            np.count_nonzero(residual_candidate_mask)
        ),
        "residual_spatial_component_count": len(residual_records),
        "accepted_residual_component_count": sum(
            int(record["accepted"]) for record in residual_records
        ),
        "incremental_fill_gaussian_count": fill_count,
        "preferred_unlabeled_recovery_ratio": (
            fill_count / float(preferred_unlabeled_count)
            if preferred_unlabeled_count
            else 0.0
        ),
        "combined_assigned_gaussian_count_if_materialized": combined_count,
        "combined_assigned_ratio_if_materialized": combined_count
        / float(vertex_count),
        "class_incremental_fill_gaussian_counts": dict(
            sorted(class_fill_counts.items())
        ),
        "parent_spatial_cores": parent_records,
        "components": residual_records,
        "outputs": {
            "report_only": True,
            "semantic_labels_written": False,
            "semantic_project_class_arrays_written": False,
            "label_map_written": False,
            "semantic_ply_written": False,
            "boolean_masks_written": True,
            "compressed_sparse_supports_written": True,
        },
    }
    (args.output_dir / "incremental_spatial_core_fill_audit.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "scene": args.scene,
                "preferred_labeled_gaussian_count": preferred_labeled_count,
                "unlabeled_residual_candidate_gaussian_count": int(
                    np.count_nonzero(residual_candidate_mask)
                ),
                "accepted_residual_component_count": report[
                    "accepted_residual_component_count"
                ],
                "incremental_fill_gaussian_count": fill_count,
                "combined_assigned_ratio_if_materialized": report[
                    "combined_assigned_ratio_if_materialized"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
