#!/usr/bin/env python3
"""Materialize reviewed DINOv3 3D components with conflict abstention.

The input association remains the authoritative report-only result.  This
separate stage assigns an instance label only where every accepted component
supporting a Gaussian agrees on the project class.  Same-class component
overlaps are resolved by their accumulated FlashSplat support; cross-class
overlaps remain unlabeled.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.common.ply_utils import read_ply_header
from scripts.task1.dinov2.dinov2_ontology import Ontology, load_ontology
from scripts.task1.qa.add_labels_from_npy import write_ply_with_labels


@dataclass(frozen=True)
class ComponentSupport:
    component_id: int
    project_id: int
    indices: np.ndarray
    scores: np.ndarray
    record: dict[str, Any]


def aggregate_support(
    proposal_ids: list[int],
    proposal_metadata: dict[int, dict[str, Any]],
    support_dir: Path,
    source_view_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    for proposal_id in proposal_ids:
        try:
            metadata = proposal_metadata[proposal_id]
        except KeyError as exc:
            raise ValueError(f"component references missing proposal {proposal_id}") from exc
        support_path = support_dir / str(metadata["support_file"])
        with np.load(support_path, allow_pickle=False) as support:
            proposal_indices = np.asarray(support["indices"], dtype=np.uint32)
            proposal_counts = np.asarray(support["counts"], dtype=np.float32)
        if proposal_indices.ndim != 1 or proposal_counts.shape != proposal_indices.shape:
            raise ValueError(f"invalid support vectors in {support_path}")
        if proposal_indices.size and np.any(
            proposal_indices[1:] <= proposal_indices[:-1]
        ):
            raise ValueError(f"support indices are not strictly increasing: {support_path}")
        if not np.isfinite(proposal_counts).all() or np.any(proposal_counts <= 0.0):
            raise ValueError(f"support counts are not finite positive values: {support_path}")
        indices.append(proposal_indices)
        counts.append(proposal_counts)
    if not indices:
        return np.empty((0,), dtype=np.uint32), np.empty((0,), dtype=np.float32)
    concatenated_indices = np.concatenate(indices)
    concatenated_counts = np.concatenate(counts)
    unique, inverse = np.unique(concatenated_indices, return_inverse=True)
    accumulated = np.zeros((unique.shape[0],), dtype=np.float32)
    np.add.at(accumulated, inverse, concatenated_counts)
    accumulated /= np.float32(max(source_view_count, 1))
    return unique.astype(np.uint32, copy=False), accumulated


def resolve_component_assignments(
    components: list[ComponentSupport],
    vertex_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return provisional instance labels, project classes, and conflicts."""
    if vertex_count < 1:
        raise ValueError("vertex_count must be positive")
    labels = np.zeros((vertex_count,), dtype=np.int32)
    project_classes = np.zeros((vertex_count,), dtype=np.int32)
    best_scores = np.zeros((vertex_count,), dtype=np.float32)
    cross_class_conflict = np.zeros((vertex_count,), dtype=bool)

    for label_id, component in enumerate(components, start=1):
        indices = np.asarray(component.indices, dtype=np.int64)
        scores = np.asarray(component.scores, dtype=np.float32)
        if indices.ndim != 1 or scores.shape != indices.shape:
            raise ValueError("component indices and scores must be matching vectors")
        if indices.size and (indices[0] < 0 or indices[-1] >= vertex_count):
            raise ValueError("component support index is outside the source PLY")
        previous_classes = project_classes[indices]
        cross_class = (previous_classes != 0) & (
            previous_classes != component.project_id
        )
        cross_class_conflict[indices[cross_class]] = True
        empty = previous_classes == 0
        project_classes[indices[empty]] = component.project_id

        # The winner matters only for same-class overlap; cross-class vertices
        # are cleared after every component has contributed.
        eligible = empty | (previous_classes == component.project_id)
        selected = eligible & (scores > best_scores[indices])
        selected_indices = indices[selected]
        best_scores[selected_indices] = scores[selected]
        labels[selected_indices] = label_id

    labels[cross_class_conflict] = 0
    project_classes[cross_class_conflict] = 0
    return labels, project_classes, cross_class_conflict


def remap_nonempty_labels(
    labels: np.ndarray,
    components: list[ComponentSupport],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    remapped = np.zeros_like(labels, dtype=np.int32)
    output_records: list[dict[str, Any]] = []
    next_label = 1
    for provisional_id, component in enumerate(components, start=1):
        selected = labels == provisional_id
        count = int(np.count_nonzero(selected))
        if count == 0:
            continue
        remapped[selected] = next_label
        output_records.append(
            {
                "id": next_label,
                "component_id": component.component_id,
                "project_id": component.project_id,
                "assigned_gaussian_count": count,
                **{
                    key: component.record[key]
                    for key in (
                        "class",
                        "ade_id",
                        "kind",
                        "probability",
                        "proposal_ids",
                        "source_frames",
                        "source_view_count",
                        "support_gaussian_count",
                    )
                },
            }
        )
        next_label += 1
    return remapped, output_records


def load_accepted_components(
    proposal_manifest_path: Path,
    component_report_path: Path,
    ontology: Ontology,
) -> tuple[list[ComponentSupport], dict[str, Any], dict[str, Any]]:
    proposal_manifest = json.loads(
        proposal_manifest_path.read_text(encoding="utf-8")
    )
    component_report = json.loads(component_report_path.read_text(encoding="utf-8"))
    if component_report.get("contract") != "report_only_class_agnostic_3d_association_v1":
        raise ValueError("component report has the wrong contract")
    if component_report.get("v5_used") is not False:
        raise ValueError("component report is not independent of v5")
    if component_report.get("dinov2_used") is not False:
        raise ValueError("component report is not independent of DINOv2")
    if component_report.get("semantic_identity_hardened_before_3d") is not False:
        raise ValueError("component semantics were hardened before 3D association")
    outputs = component_report.get("outputs", {})
    if outputs.get("report_only") is not True:
        raise ValueError("component report is not the reviewed report-only result")

    proposal_list = proposal_manifest.get("proposals", [])
    if int(component_report.get("proposal_count", -1)) != len(proposal_list):
        raise ValueError("proposal count differs between report and manifest")
    proposal_metadata = {
        int(metadata["proposal_id"]): metadata for metadata in proposal_list
    }
    if len(proposal_metadata) != len(proposal_list):
        raise ValueError("proposal IDs are not unique")
    support_dir = proposal_manifest_path.parent / "proposal_supports"
    accepted: list[ComponentSupport] = []
    for record in component_report.get("components", []):
        if not bool(record.get("accepted", False)):
            continue
        if record.get("status") != "accepted_stable_multiview_identity":
            raise ValueError("accepted component does not have stable identity status")
        project_id = int(record["project_id"])
        try:
            ontology_class = ontology.by_project_id[project_id]
        except KeyError as exc:
            raise ValueError(f"unknown project class ID {project_id}") from exc
        if (
            ontology_class.project_class != str(record["class"])
            or ontology_class.ade_id != int(record["ade_id"])
            or ontology_class.kind != str(record["kind"])
        ):
            raise ValueError("component identity does not match the ontology")
        indices, scores = aggregate_support(
            [int(item) for item in record["proposal_ids"]],
            proposal_metadata,
            support_dir,
            int(record["source_view_count"]),
        )
        if int(record["support_gaussian_count"]) != int(indices.shape[0]):
            raise ValueError("reconstructed component support differs from report")
        accepted.append(
            ComponentSupport(
                component_id=int(record["component_id"]),
                project_id=project_id,
                indices=indices,
                scores=scores,
                record=record,
            )
        )
    if len(accepted) != int(component_report.get("accepted_component_count", -1)):
        raise ValueError("accepted component count differs from report")
    return accepted, proposal_manifest, component_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-manifest", required=True, type=Path)
    parser.add_argument("--component-report", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene")
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

    ontology = load_ontology(args.ontology)
    components, proposal_manifest, component_report = load_accepted_components(
        args.proposal_manifest,
        args.component_report,
        ontology,
    )
    vertex_count = int(proposal_manifest["vertex_count"])
    manifest_ply = Path(str(proposal_manifest["ply_path"]))
    if manifest_ply.resolve() != args.source_ply.resolve():
        raise ValueError("source PLY differs from the lifted proposal provenance")
    ply_header = read_ply_header(args.source_ply)
    if not ply_header.elements or ply_header.elements[0].name != "vertex":
        raise ValueError("source PLY does not begin with a vertex element")
    if int(ply_header.elements[0].count) != vertex_count:
        raise ValueError("source PLY vertex count differs from proposal manifest")

    provisional, project_classes, conflict_mask = resolve_component_assignments(
        components,
        vertex_count,
    )
    labels, label_records = remap_nonempty_labels(provisional, components)
    clean_count = int(np.count_nonzero(labels))
    conflict_count = int(np.count_nonzero(conflict_mask))
    supported_any = clean_count + conflict_count

    args.output_dir.mkdir(parents=True)
    labels_path = args.output_dir / "gaussian_labels.npy"
    classes_path = args.output_dir / "gaussian_project_class_ids.npy"
    conflicts_path = args.output_dir / "cross_class_conflict_mask.npy"
    label_map_path = args.output_dir / "label_map.json"
    summary_path = args.output_dir / "materialization_summary.json"
    semantic_ply_path = args.output_dir / "semantic_point_cloud.ply"

    np.save(labels_path, labels)
    np.save(classes_path, project_classes)
    np.save(conflicts_path, conflict_mask)
    labels_json = [
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
            }
        )
    label_map = {
        "scene": args.scene or args.source_ply.parents[2].name,
        "source": "dinov3_mask2former_3d_first_conflict_abstain",
        "ontology": str(args.ontology),
        "component_report": str(args.component_report),
        "labels": labels_json,
    }
    label_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    if not args.no_semantic_ply:
        partial_ply = semantic_ply_path.with_suffix(".ply.partial")
        write_ply_with_labels(args.source_ply, partial_ply, labels)
        partial_ply.replace(semantic_ply_path)

    class_counts = {
        ontology.by_project_id[int(project_id)].project_class: int(count)
        for project_id, count in zip(
            *np.unique(project_classes[project_classes > 0], return_counts=True)
        )
    }
    summary = {
        "source": "dinov3_mask2former_3d_first_conflict_abstain",
        "contract": "reviewed_3d_components_cross_class_conflict_abstain_v1",
        "scene": label_map["scene"],
        "proposal_manifest": str(args.proposal_manifest),
        "component_report": str(args.component_report),
        "source_ply": str(args.source_ply),
        "ontology": str(args.ontology),
        "v5_used": False,
        "dinov2_used": False,
        "semantic_identity_hardened_before_3d": False,
        "assignment_policy": {
            "same_class_overlap": "highest_accumulated_flashsplat_support",
            "cross_class_overlap": "unlabeled_abstain",
            "scene_specific_rules": False,
            "class_specific_thresholds": False,
        },
        "vertex_count": vertex_count,
        "accepted_component_count": len(components),
        "materialized_label_count": len(label_records),
        "supported_any_gaussian_count": supported_any,
        "cross_class_conflict_gaussian_count": conflict_count,
        "assigned_gaussian_count": clean_count,
        "assigned_ratio": clean_count / float(vertex_count),
        "unlabeled_gaussian_count": vertex_count - clean_count,
        "class_assigned_gaussian_counts": dict(sorted(class_counts.items())),
        "semantic_labels_written": True,
        "label_map_written": True,
        "semantic_ply_written": not args.no_semantic_ply,
        "component_report_outputs": component_report.get("outputs", {}),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
