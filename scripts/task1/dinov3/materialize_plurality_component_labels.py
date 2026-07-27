#!/usr/bin/env python3
"""Materialize class-agnostic 3D components using equal camera plurality.

Every source camera contributes exactly one semantic winner to its physical
component.  Components with fewer than two source cameras or a tied plurality
abstain.  Soft probability magnitudes never weight one camera above another.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.common.ply_utils import read_ply_header
from scripts.task1.dinov2.dinov2_ontology import Ontology, load_ontology
from scripts.task1.dinov3.materialize_3d_component_labels import (
    ComponentSupport,
    aggregate_support,
    remap_nonempty_labels,
    resolve_component_assignments,
)
from scripts.task1.qa.add_labels_from_npy import write_ply_with_labels


SOURCE = "dinov3_class_agnostic_components_camera_plurality"
CONTRACT = "unique_equal_camera_plurality_minimum_two_views_v1"


def unique_camera_plurality(
    view_winners: list[str] | tuple[str, ...],
    *,
    min_views: int = 2,
) -> tuple[str | None, int, dict[str, int]]:
    """Return a unique plurality winner, its votes, and the full vote counts."""

    if min_views < 1:
        raise ValueError("min_views must be positive")
    normalized = [str(value).strip() for value in view_winners]
    if any(not value for value in normalized):
        raise ValueError("camera semantic winners must be non-empty")
    counts = Counter(normalized)
    ordered_counts = dict(sorted(counts.items()))
    if len(normalized) < min_views or not counts:
        return None, 0, ordered_counts
    largest = max(counts.values())
    winners = sorted(name for name, count in counts.items() if count == largest)
    if len(winners) != 1:
        return None, largest, ordered_counts
    return winners[0], largest, ordered_counts


def validate_report(
    proposal_manifest: dict[str, Any],
    component_report: dict[str, Any],
) -> None:
    if component_report.get("contract") != "report_only_class_agnostic_3d_association_v1":
        raise ValueError("component report has the wrong contract")
    if component_report.get("v5_used") is not False:
        raise ValueError("component report is not independent of v5")
    if component_report.get("dinov2_used") is not False:
        raise ValueError("component report is not independent of DINOv2")
    if component_report.get("semantic_identity_hardened_before_3d") is not False:
        raise ValueError("component semantics were hardened before 3D association")
    if component_report.get("outputs", {}).get("report_only") is not True:
        raise ValueError("component source is not report-only")
    proposals = proposal_manifest.get("proposals", [])
    if int(component_report.get("proposal_count", -1)) != len(proposals):
        raise ValueError("proposal count differs between report and manifest")


def load_plurality_components(
    proposal_manifest_path: Path,
    component_report_path: Path,
    ontology: Ontology,
    *,
    min_views: int,
) -> tuple[
    list[ComponentSupport],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    proposal_manifest = json.loads(
        proposal_manifest_path.read_text(encoding="utf-8")
    )
    component_report = json.loads(component_report_path.read_text(encoding="utf-8"))
    validate_report(proposal_manifest, component_report)

    proposal_list = proposal_manifest.get("proposals", [])
    proposal_metadata = {
        int(metadata["proposal_id"]): metadata for metadata in proposal_list
    }
    if len(proposal_metadata) != len(proposal_list):
        raise ValueError("proposal IDs are not unique")
    support_dir = proposal_manifest_path.parent / "proposal_supports"
    ontology_by_name = {item.project_class: item for item in ontology.classes}

    accepted: list[ComponentSupport] = []
    selection_records: list[dict[str, Any]] = []
    seen_component_ids: set[int] = set()
    for record in component_report.get("components", []):
        component_id = int(record["component_id"])
        if component_id in seen_component_ids:
            raise ValueError("component IDs are not unique")
        seen_component_ids.add(component_id)
        source_view_count = int(record["source_view_count"])
        proposal_ids = [int(item) for item in record["proposal_ids"]]
        source_frames = [str(item) for item in record["source_frames"]]
        stability = record.get("semantic_stability", {})
        view_winners = [str(value) for value in stability.get("view_winners", [])]
        if not (
            len(proposal_ids)
            == len(source_frames)
            == len(view_winners)
            == source_view_count
        ):
            raise ValueError("component does not contain exactly one vote per camera")
        if len(set(source_frames)) != source_view_count:
            raise ValueError("component source frames are not unique")
        try:
            manifest_frames = [
                str(proposal_metadata[proposal_id]["frame_file"])
                for proposal_id in proposal_ids
            ]
        except KeyError as exc:
            raise ValueError("component references a missing proposal") from exc
        if sorted(manifest_frames) != sorted(source_frames):
            raise ValueError("component source frames differ from proposal provenance")
        winner, winner_votes, vote_counts = unique_camera_plurality(
            view_winners,
            min_views=min_views,
        )
        if source_view_count < min_views:
            status = "abstained_insufficient_camera_views"
        elif winner is None:
            status = "abstained_tied_camera_plurality"
        else:
            status = "accepted_unique_camera_plurality"

        selection = {
            "component_id": component_id,
            "status": status,
            "accepted": winner is not None,
            "plurality_class": winner,
            "plurality_vote_count": winner_votes,
            "source_view_count": source_view_count,
            "plurality_vote_ratio": (
                winner_votes / float(source_view_count) if winner is not None else 0.0
            ),
            "camera_vote_counts": vote_counts,
            "camera_winners": view_winners,
            "pooled_soft_class": str(record["class"]),
            "pooled_soft_probability": float(record["probability"]),
            "differs_from_pooled_soft_class": (
                winner is not None and winner != str(record["class"])
            ),
            "proposal_ids": proposal_ids,
            "source_frames": source_frames,
            "support_gaussian_count": int(record["support_gaussian_count"]),
        }
        selection_records.append(selection)
        if winner is None:
            continue
        try:
            ontology_class = ontology_by_name[winner]
        except KeyError as exc:
            raise ValueError(f"unknown plurality class {winner}") from exc
        indices, scores = aggregate_support(
            selection["proposal_ids"],
            proposal_metadata,
            support_dir,
            source_view_count,
        )
        if int(record["support_gaussian_count"]) != int(indices.shape[0]):
            raise ValueError("reconstructed component support differs from report")
        accepted.append(
            ComponentSupport(
                component_id=component_id,
                project_id=ontology_class.project_id,
                indices=indices,
                scores=scores,
                record={
                    "class": ontology_class.project_class,
                    "ade_id": ontology_class.ade_id,
                    "kind": ontology_class.kind,
                    "probability": selection["plurality_vote_ratio"],
                    "proposal_ids": selection["proposal_ids"],
                    "source_frames": selection["source_frames"],
                    "source_view_count": source_view_count,
                    "support_gaussian_count": int(record["support_gaussian_count"]),
                },
            )
        )
    return accepted, selection_records, proposal_manifest, component_report


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

    provisional, project_classes, conflict_mask = resolve_component_assignments(
        components,
        vertex_count,
    )
    instance_labels, label_records = remap_nonempty_labels(provisional, components)
    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "gaussian_labels.npy", instance_labels)
    np.save(args.output_dir / "gaussian_project_class_ids.npy", project_classes)
    np.save(args.output_dir / "cross_class_conflict_mask.npy", conflict_mask)

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
        json.dumps(label_map, indent=2), encoding="utf-8"
    )

    semantic_ply = args.output_dir / "semantic_point_cloud.ply"
    if not args.no_semantic_ply:
        partial = semantic_ply.with_suffix(".ply.partial")
        write_ply_with_labels(args.source_ply, partial, instance_labels)
        partial.replace(semantic_ply)

    status_counts = {
        str(status): int(count)
        for status, count in zip(
            *np.unique([record["status"] for record in selections], return_counts=True)
        )
    }
    assigned = project_classes > 0
    class_counts = {
        ontology.by_project_id[int(project_id)].project_class: int(count)
        for project_id, count in zip(
            *np.unique(project_classes[assigned], return_counts=True)
        )
    }
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
        "camera_vote_policy": "one_equal_argmax_vote_per_source_camera",
        "component_acceptance_policy": "minimum_two_views_and_unique_plurality",
        "soft_probability_weighting_used": False,
        "cross_class_overlap_policy": "unlabeled_abstain",
        "min_views": args.min_views,
        "vertex_count": vertex_count,
        "input_component_count": len(selections),
        "accepted_plurality_component_count": len(components),
        "materialized_label_count": len(label_records),
        "status_counts": status_counts,
        "plurality_differs_from_pooled_component_count": int(
            sum(record["differs_from_pooled_soft_class"] for record in selections)
        ),
        "cross_class_conflict_gaussian_count": int(np.count_nonzero(conflict_mask)),
        "assigned_gaussian_count": int(np.count_nonzero(assigned)),
        "assigned_ratio": float(np.mean(assigned)),
        "unlabeled_gaussian_count": int(np.count_nonzero(~assigned)),
        "class_assigned_gaussian_counts": dict(sorted(class_counts.items())),
        "component_selections": selections,
        "semantic_labels_written": True,
        "semantic_ply_written": not args.no_semantic_ply,
        "input_report_outputs": component_report.get("outputs", {}),
    }
    (args.output_dir / "plurality_component_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
