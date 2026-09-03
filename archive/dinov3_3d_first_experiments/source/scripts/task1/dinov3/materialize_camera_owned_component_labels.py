#!/usr/bin/env python3
"""Resolve DINOv3 component overlaps with independent camera support counts.

Semantic identity is still selected once per physical component by equal
camera plurality.  When accepted components of different classes overlap at a
Gaussian, the component supported there by the uniquely largest number of
source cameras owns the Gaussian.  Equal camera counts abstain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.common.ply_utils import read_ply_header
from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.dinov3.materialize_3d_component_labels import (
    ComponentSupport,
    remap_nonempty_labels,
)
from scripts.task1.dinov3.materialize_plurality_component_labels import (
    load_plurality_components,
)
from scripts.task1.qa.add_labels_from_npy import write_ply_with_labels


SOURCE = "dinov3_plurality_components_unique_camera_support_ownership"
CONTRACT = "unique_maximum_component_camera_support_per_gaussian_v1"


def aggregate_supporting_camera_counts(
    proposal_ids: list[int],
    proposal_metadata: dict[int, dict[str, Any]],
    support_dir: Path,
    expected_indices: np.ndarray,
) -> np.ndarray:
    """Return independent supporting-camera counts aligned to a component."""

    proposal_indices: list[np.ndarray] = []
    for proposal_id in proposal_ids:
        try:
            metadata = proposal_metadata[proposal_id]
        except KeyError as exc:
            raise ValueError(f"component references missing proposal {proposal_id}") from exc
        support_path = support_dir / str(metadata["support_file"])
        with np.load(support_path, allow_pickle=False) as support:
            indices = np.asarray(support["indices"], dtype=np.uint32)
        if indices.ndim != 1:
            raise ValueError(f"invalid support indices in {support_path}")
        if indices.size and np.any(indices[1:] <= indices[:-1]):
            raise ValueError(
                f"support indices are not strictly increasing: {support_path}"
            )
        proposal_indices.append(indices)
    if not proposal_indices:
        raise ValueError("accepted component has no camera proposals")

    concatenated = np.concatenate(proposal_indices)
    unique, counts = np.unique(concatenated, return_counts=True)
    expected = np.asarray(expected_indices, dtype=np.uint32)
    if not np.array_equal(unique.astype(np.uint32, copy=False), expected):
        raise ValueError("camera-count support differs from component support")
    if counts.size and int(counts.max()) > np.iinfo(np.uint16).max:
        raise ValueError("supporting camera count exceeds uint16 capacity")
    return counts.astype(np.uint16, copy=False)


def resolve_camera_support_ownership(
    components: list[ComponentSupport],
    supporting_camera_counts: list[np.ndarray],
    vertex_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve component ownership by unique maximum camera support.

    Geometric support magnitude may break ties only between components with
    the same semantic class.  It never resolves a cross-class decision.
    """

    if vertex_count < 1:
        raise ValueError("vertex_count must be positive")
    if len(components) != len(supporting_camera_counts):
        raise ValueError("camera-count vectors must match components")

    labels = np.zeros((vertex_count,), dtype=np.int32)
    project_classes = np.zeros((vertex_count,), dtype=np.int32)
    winning_camera_counts = np.zeros((vertex_count,), dtype=np.uint16)
    best_same_class_scores = np.zeros((vertex_count,), dtype=np.float32)
    first_supported_class = np.zeros((vertex_count,), dtype=np.int32)
    cross_class_overlap = np.zeros((vertex_count,), dtype=bool)
    cross_class_camera_tie = np.zeros((vertex_count,), dtype=bool)

    for label_id, (component, camera_counts_value) in enumerate(
        zip(components, supporting_camera_counts),
        start=1,
    ):
        indices = np.asarray(component.indices, dtype=np.int64)
        scores = np.asarray(component.scores, dtype=np.float32)
        camera_counts = np.asarray(camera_counts_value)
        if (
            indices.ndim != 1
            or scores.shape != indices.shape
            or camera_counts.shape != indices.shape
        ):
            raise ValueError("component support vectors must have matching shapes")
        if indices.size and (indices[0] < 0 or indices[-1] >= vertex_count):
            raise ValueError("component support index is outside the source PLY")
        if not np.issubdtype(camera_counts.dtype, np.integer):
            raise ValueError("supporting camera counts must be integers")
        source_view_count = int(component.record["source_view_count"])
        if camera_counts.size and (
            int(camera_counts.min()) < 1
            or int(camera_counts.max()) > source_view_count
        ):
            raise ValueError("supporting camera count is outside component provenance")
        camera_counts = camera_counts.astype(np.uint16, copy=False)

        first_classes = first_supported_class[indices]
        empty_first = first_classes == 0
        different_first = (~empty_first) & (
            first_classes != component.project_id
        )
        cross_class_overlap[indices[different_first]] = True
        first_supported_class[indices[empty_first]] = component.project_id

        current_counts = winning_camera_counts[indices]
        current_classes = project_classes[indices]
        current_scores = best_same_class_scores[indices]
        greater = camera_counts > current_counts
        equal = camera_counts == current_counts

        greater_indices = indices[greater]
        winning_camera_counts[greater_indices] = camera_counts[greater]
        project_classes[greater_indices] = component.project_id
        labels[greater_indices] = label_id
        best_same_class_scores[greater_indices] = scores[greater]
        cross_class_camera_tie[greater_indices] = False

        equal_different_class = (
            equal
            & (current_counts > 0)
            & (current_classes != component.project_id)
        )
        cross_class_camera_tie[indices[equal_different_class]] = True

        equal_same_class_better = (
            equal
            & (current_counts > 0)
            & (current_classes == component.project_id)
            & (scores > current_scores)
        )
        same_class_indices = indices[equal_same_class_better]
        labels[same_class_indices] = label_id
        best_same_class_scores[same_class_indices] = scores[
            equal_same_class_better
        ]

    labels[cross_class_camera_tie] = 0
    project_classes[cross_class_camera_tie] = 0
    return (
        labels,
        project_classes,
        cross_class_camera_tie,
        cross_class_overlap,
        winning_camera_counts,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-manifest", required=True, type=Path)
    parser.add_argument("--component-report", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--min-views", default=2, type=int)
    parser.add_argument("--no-semantic-ply", action="store_true")
    args = parser.parse_args()

    for path in (
        args.proposal_manifest,
        args.component_report,
        args.ontology,
        args.source_ply,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.min_views < 2:
        raise ValueError("min_views must be at least two for multiview plurality")

    ontology = load_ontology(args.ontology)
    components, selections, proposal_manifest, component_report = (
        load_plurality_components(
            args.proposal_manifest,
            args.component_report,
            ontology,
            min_views=args.min_views,
        )
    )
    vertex_count = int(proposal_manifest["vertex_count"])
    manifest_ply = Path(str(proposal_manifest["ply_path"]))
    if manifest_ply.resolve() != args.source_ply.resolve():
        raise ValueError("source PLY differs from lifted proposal provenance")
    header = read_ply_header(args.source_ply)
    if not header.elements or header.elements[0].name != "vertex":
        raise ValueError("source PLY does not begin with a vertex element")
    if int(header.elements[0].count) != vertex_count:
        raise ValueError("source PLY vertex count differs from proposal manifest")

    proposal_list = proposal_manifest.get("proposals", [])
    proposal_metadata = {
        int(metadata["proposal_id"]): metadata for metadata in proposal_list
    }
    if len(proposal_metadata) != len(proposal_list):
        raise ValueError("proposal IDs are not unique")
    support_dir = args.proposal_manifest.parent / "proposal_supports"
    camera_counts = [
        aggregate_supporting_camera_counts(
            [int(item) for item in component.record["proposal_ids"]],
            proposal_metadata,
            support_dir,
            component.indices,
        )
        for component in components
    ]
    (
        provisional,
        project_classes,
        camera_tie_mask,
        cross_class_overlap_mask,
        winning_camera_counts,
    ) = resolve_camera_support_ownership(
        components,
        camera_counts,
        vertex_count,
    )
    instance_labels, label_records = remap_nonempty_labels(
        provisional,
        components,
    )

    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "gaussian_labels.npy", instance_labels)
    np.save(
        args.output_dir / "gaussian_project_class_ids.npy",
        project_classes,
    )
    np.save(
        args.output_dir / "cross_class_overlap_mask.npy",
        cross_class_overlap_mask,
    )
    np.save(
        args.output_dir / "cross_class_camera_count_tie_mask.npy",
        camera_tie_mask,
    )
    np.save(
        args.output_dir / "maximum_component_camera_count.npy",
        winning_camera_counts,
    )

    labels_json: list[dict[str, Any]] = [
        {"id": 0, "name": "unlabeled", "class": "unlabeled", "type": "unlabeled"}
    ]
    class_ordinals: dict[str, int] = {}
    for record in label_records:
        class_name = str(record["class"])
        class_ordinals[class_name] = class_ordinals.get(class_name, 0) + 1
        labels_json.append(
            {
                **record,
                "name": f"{class_name}_{class_ordinals[class_name]:03d}",
                "type": record["kind"],
                "identity_policy": "unique_equal_camera_plurality",
                "overlap_ownership_policy": (
                    "unique_maximum_supporting_camera_count"
                ),
            }
        )
    label_map = {
        "scene": args.scene,
        "source": SOURCE,
        "ontology": str(args.ontology),
        "component_report": str(args.component_report),
        "labels": labels_json,
    }
    (args.output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2),
        encoding="utf-8",
    )

    semantic_ply = args.output_dir / "semantic_point_cloud.ply"
    if not args.no_semantic_ply:
        partial = semantic_ply.with_suffix(".ply.partial")
        write_ply_with_labels(args.source_ply, partial, instance_labels)
        partial.replace(semantic_ply)

    status_counts = {
        str(status): int(count)
        for status, count in zip(
            *np.unique(
                [record["status"] for record in selections],
                return_counts=True,
            )
        )
    }
    assigned = project_classes > 0
    class_counts = {
        ontology.by_project_id[int(project_id)].project_class: int(count)
        for project_id, count in zip(
            *np.unique(project_classes[assigned], return_counts=True)
        )
    }
    component_assignment_records: list[dict[str, Any]] = []
    for provisional_id, (component, counts) in enumerate(
        zip(components, camera_counts),
        start=1,
    ):
        unique_counts, count_frequencies = np.unique(counts, return_counts=True)
        selected = provisional == provisional_id
        component_assignment_records.append(
            {
                "component_id": component.component_id,
                "class": component.record["class"],
                "source_view_count": int(component.record["source_view_count"]),
                "support_gaussian_count": int(component.indices.shape[0]),
                "camera_support_count_histogram": {
                    str(int(count)): int(frequency)
                    for count, frequency in zip(
                        unique_counts,
                        count_frequencies,
                    )
                },
                "assigned_gaussian_count": int(np.count_nonzero(selected)),
                "assigned_cross_class_overlap_gaussian_count": int(
                    np.count_nonzero(selected & cross_class_overlap_mask)
                ),
            }
        )

    cross_class_overlap_count = int(
        np.count_nonzero(cross_class_overlap_mask)
    )
    camera_tie_count = int(np.count_nonzero(camera_tie_mask))
    summary = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": args.scene,
        "proposal_manifest": str(args.proposal_manifest),
        "component_report": str(args.component_report),
        "source_ply": str(args.source_ply),
        "ontology": str(args.ontology),
        "v5_used": False,
        "dinov2_used": False,
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "offline_threshold_sweep_used": False,
        "component_identity_policy": (
            "one_equal_argmax_vote_per_source_camera_unique_plurality"
        ),
        "cross_class_overlap_policy": (
            "unique_maximum_supporting_camera_count_else_abstain"
        ),
        "cross_class_semantic_probability_weighting_used": False,
        "cross_class_geometric_strength_weighting_used": False,
        "same_class_instance_tie_break": (
            "accumulated_flashsplat_support_only_after_class_agreement"
        ),
        "min_views": args.min_views,
        "vertex_count": vertex_count,
        "input_component_count": len(selections),
        "accepted_plurality_component_count": len(components),
        "materialized_label_count": len(label_records),
        "status_counts": status_counts,
        "plurality_differs_from_pooled_component_count": int(
            sum(record["differs_from_pooled_soft_class"] for record in selections)
        ),
        "cross_class_overlap_gaussian_count": cross_class_overlap_count,
        "cross_class_camera_count_tie_gaussian_count": camera_tie_count,
        "resolved_cross_class_overlap_gaussian_count": int(
            np.count_nonzero(cross_class_overlap_mask & assigned)
        ),
        "assigned_gaussian_count": int(np.count_nonzero(assigned)),
        "assigned_ratio": float(np.mean(assigned)),
        "unlabeled_gaussian_count": int(np.count_nonzero(~assigned)),
        "class_assigned_gaussian_counts": dict(sorted(class_counts.items())),
        "component_selections": selections,
        "component_camera_support_assignments": component_assignment_records,
        "semantic_labels_written": True,
        "semantic_ply_written": not args.no_semantic_ply,
        "input_report_outputs": component_report.get("outputs", {}),
    }
    (args.output_dir / "camera_owned_component_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
