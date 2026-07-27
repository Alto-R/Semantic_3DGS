#!/usr/bin/env python3
"""Materialize an explicitly reviewed subset of incremental DINOv3 fills.

Every nonzero preferred-v2 label is immutable. Accepted residual components
fill only preferred-unlabeled Gaussians, while exact-QA exclusions are supplied
by a scene review configuration. The result is a fresh candidate-v3 output;
the preferred-v2 files are never modified.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.task1.common.ply_utils import read_ply_header
from scripts.task1.dinov2.dinov2_ontology import Ontology, load_ontology
from scripts.task1.dinov3.audit_incremental_spatial_core_fill import file_sha256
from scripts.task1.qa.add_labels_from_npy import write_ply_with_labels


SOURCE = "dinov3_reviewed_incremental_spatial_core_fill_candidate"
CONTRACT = "immutable_preferred_v2_reviewed_incremental_fill_candidate_v3"
AUDIT_CONTRACT = "report_only_dinov3_incremental_spatial_core_fill_v1"


@dataclass(frozen=True)
class FillComponent:
    component_id: int
    project_id: int
    class_name: str
    indices: np.ndarray
    camera_counts: np.ndarray
    record: dict[str, Any]


@dataclass(frozen=True)
class ExclusionDecision:
    component_id: int
    class_name: str
    reason: str


def load_review_config(
    path: Path,
    *,
    scene: str,
) -> tuple[dict[int, ExclusionDecision], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if str(raw.get("scene")) != scene:
        raise ValueError("review config scene differs from the requested scene")
    if str(raw.get("source_contract")) != AUDIT_CONTRACT:
        raise ValueError("review config source contract is invalid")
    if str(raw.get("default_decision")) != "accept":
        raise ValueError("review config default_decision must be accept")
    decisions: dict[int, ExclusionDecision] = {}
    for item in raw.get("excluded_components", []):
        component_id = int(item["component_id"])
        class_name = str(item["class"])
        reason = str(item["reason"]).strip()
        if component_id < 1 or component_id in decisions:
            raise ValueError("review exclusions contain invalid component IDs")
        if not class_name or not reason:
            raise ValueError("review exclusions require class and reason")
        decisions[component_id] = ExclusionDecision(
            component_id=component_id,
            class_name=class_name,
            reason=reason,
        )
    return decisions, raw


def load_fill_components(
    report: dict[str, Any],
    archive: Mapping[str, np.ndarray],
    *,
    scene: str,
    vertex_count: int,
    ontology: Ontology,
) -> list[FillComponent]:
    if report.get("contract") != AUDIT_CONTRACT:
        raise ValueError("incremental-fill audit has the wrong contract")
    if str(report.get("scene")) != scene:
        raise ValueError("incremental-fill audit scene differs from the request")
    if not bool(report.get("report_only")):
        raise ValueError("incremental-fill source must be report-only")
    if int(report.get("vertex_count", -1)) != vertex_count:
        raise ValueError("incremental-fill vertex count differs from preferred labels")
    if not bool(report.get("preferred_labels_read_only")):
        raise ValueError("incremental-fill audit did not preserve preferred labels")
    if bool(report.get("preferred_labels_modified")):
        raise ValueError("incremental-fill audit reports modified preferred labels")

    components: list[FillComponent] = []
    seen_ids: set[int] = set()
    seen_indices = np.zeros((vertex_count,), dtype=bool)
    for record in report.get("components", []):
        if not bool(record.get("accepted")):
            continue
        component_id = int(record["component_id"])
        if component_id < 1 or component_id in seen_ids:
            raise ValueError("accepted residual component IDs are invalid")
        seen_ids.add(component_id)
        project_id = int(record["project_id"])
        try:
            ontology_item = ontology.by_project_id[project_id]
        except KeyError as exc:
            raise ValueError(f"unknown project class ID {project_id}") from exc
        class_name = str(record["class"])
        if class_name != ontology_item.project_class:
            raise ValueError("residual component class differs from the ontology")
        prefix = f"component_{component_id:06d}"
        index_key = f"{prefix}_indices"
        count_key = f"{prefix}_camera_counts"
        if index_key not in archive or count_key not in archive:
            raise ValueError(f"missing sparse support for component {component_id}")
        indices = np.asarray(archive[index_key], dtype=np.uint32)
        camera_counts = np.asarray(archive[count_key], dtype=np.uint16)
        if indices.ndim != 1 or camera_counts.shape != indices.shape:
            raise ValueError("component indices and camera counts must align")
        if indices.size and (
            int(indices[-1]) >= vertex_count
            or np.any(indices[1:] <= indices[:-1])
        ):
            raise ValueError("component indices are invalid")
        if indices.size and np.any(seen_indices[indices]):
            raise ValueError("accepted incremental-fill components overlap")
        if int(record["support_gaussian_count"]) != int(indices.size):
            raise ValueError("component support count differs from the audit")
        seen_indices[indices] = True
        components.append(
            FillComponent(
                component_id=component_id,
                project_id=project_id,
                class_name=class_name,
                indices=indices,
                camera_counts=camera_counts,
                record=record,
            )
        )

    if len(components) != int(report.get("accepted_residual_component_count", -1)):
        raise ValueError("accepted component count differs from the audit")
    if sum(int(item.indices.size) for item in components) != int(
        report.get("incremental_fill_gaussian_count", -1)
    ):
        raise ValueError("accepted sparse support total differs from the audit")
    return components


def validate_preferred_label_map(
    label_map: dict[str, Any],
    labels: np.ndarray,
    *,
    scene: str,
) -> dict[int, dict[str, Any]]:
    if str(label_map.get("scene")) != scene:
        raise ValueError("preferred label-map scene differs from the request")
    records = label_map.get("labels")
    if not isinstance(records, list):
        raise ValueError("preferred label map has no labels list")
    by_id: dict[int, dict[str, Any]] = {}
    for raw in records:
        label_id = int(raw["id"])
        if label_id < 0 or label_id in by_id:
            raise ValueError("preferred label map contains invalid IDs")
        record = dict(raw)
        by_id[label_id] = record
        if label_id > 0 and int(record.get("project_id", -1)) != label_id:
            raise ValueError(
                "preferred labels are not identity-mapped project classes"
            )
    if 0 not in by_id:
        raise ValueError("preferred label map lacks the unlabeled ID")
    used_ids = {int(value) for value in np.unique(labels)}
    missing = used_ids.difference(by_id)
    if missing:
        raise ValueError(f"preferred labels are absent from the label map: {missing}")
    return by_id


def build_candidate_labels(
    preferred_labels: np.ndarray,
    components: list[FillComponent],
    exclusions: dict[int, ExclusionDecision],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[FillComponent],
    list[FillComponent],
]:
    preferred = np.asarray(preferred_labels)
    if preferred.ndim != 1 or not np.issubdtype(preferred.dtype, np.integer):
        raise ValueError("preferred labels must be a one-dimensional integer array")
    if np.any(preferred < 0):
        raise ValueError("preferred labels must be non-negative")

    component_by_id = {item.component_id: item for item in components}
    unknown_exclusions = set(exclusions).difference(component_by_id)
    if unknown_exclusions:
        raise ValueError(
            f"review config excludes unknown components: {sorted(unknown_exclusions)}"
        )
    for component_id, decision in exclusions.items():
        component = component_by_id[component_id]
        if decision.class_name != component.class_name:
            raise ValueError(
                f"reviewed class differs for component {component_id}"
            )

    final_labels = preferred.astype(np.int32, copy=True)
    fill_mask = np.zeros(preferred.shape, dtype=bool)
    excluded_mask = np.zeros(preferred.shape, dtype=bool)
    component_ids = np.zeros(preferred.shape, dtype=np.int32)
    included: list[FillComponent] = []
    excluded: list[FillComponent] = []
    for component in components:
        if component.component_id in exclusions:
            excluded_mask[component.indices] = True
            excluded.append(component)
            continue
        if np.any(preferred[component.indices] != 0):
            raise ValueError("incremental fill overlaps a preferred nonzero label")
        if np.any(fill_mask[component.indices]):
            raise ValueError("reviewed incremental-fill components overlap")
        final_labels[component.indices] = component.project_id
        fill_mask[component.indices] = True
        component_ids[component.indices] = component.component_id
        included.append(component)
    if np.any(final_labels[preferred != 0] != preferred[preferred != 0]):
        raise AssertionError("candidate changed a nonzero preferred label")
    if np.any(fill_mask & excluded_mask):
        raise AssertionError("included and excluded reviewed supports overlap")
    return (
        final_labels,
        fill_mask,
        excluded_mask,
        component_ids,
        included,
        excluded,
    )


def build_candidate_label_map(
    preferred_label_map: dict[str, Any],
    existing_by_id: dict[int, dict[str, Any]],
    included: list[FillComponent],
    ontology: Ontology,
    *,
    scene: str,
    preferred_label_map_path: Path,
    review_config_path: Path,
) -> dict[str, Any]:
    records = {label_id: dict(record) for label_id, record in existing_by_id.items()}
    for project_id in sorted({item.project_id for item in included}):
        ontology_item = ontology.by_project_id[project_id]
        existing = records.get(project_id)
        if existing is not None:
            if (
                str(existing.get("class")) != ontology_item.project_class
                or int(existing.get("project_id", -1)) != project_id
            ):
                raise ValueError("existing label-map identity differs from ontology")
            continue
        records[project_id] = {
            "id": project_id,
            "name": ontology_item.project_class,
            "class": ontology_item.project_class,
            "project_id": project_id,
            "ade_id": ontology_item.ade_id,
            "type": ontology_item.kind,
        }
    return {
        "scene": scene,
        "source": SOURCE,
        "contract": CONTRACT,
        "preferred_label_map": str(preferred_label_map_path),
        "preferred_source": preferred_label_map.get("source", ""),
        "review_config": str(review_config_path),
        "labels": [records[label_id] for label_id in sorted(records)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preferred-labels", required=True, type=Path)
    parser.add_argument("--preferred-label-map", required=True, type=Path)
    parser.add_argument("--incremental-fill-report", required=True, type=Path)
    parser.add_argument("--incremental-fill-supports", required=True, type=Path)
    parser.add_argument("--review-config", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--no-semantic-ply", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (
        args.preferred_labels,
        args.preferred_label_map,
        args.incremental_fill_report,
        args.incremental_fill_supports,
        args.review_config,
        args.ontology,
        args.source_ply,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    preferred_hash = file_sha256(args.preferred_labels)
    preferred_labels = np.load(args.preferred_labels, mmap_mode="r")
    if preferred_labels.ndim != 1:
        raise ValueError("preferred labels must be one-dimensional")
    vertex_count = int(preferred_labels.size)
    preferred_label_map = json.loads(
        args.preferred_label_map.read_text(encoding="utf-8")
    )
    existing_by_id = validate_preferred_label_map(
        preferred_label_map,
        preferred_labels,
        scene=args.scene,
    )
    audit_report = json.loads(
        args.incremental_fill_report.read_text(encoding="utf-8")
    )
    ontology = load_ontology(args.ontology)
    with np.load(args.incremental_fill_supports, allow_pickle=False) as archive:
        components = load_fill_components(
            audit_report,
            archive,
            scene=args.scene,
            vertex_count=vertex_count,
            ontology=ontology,
        )
    exclusions, review_config = load_review_config(
        args.review_config,
        scene=args.scene,
    )
    (
        final_labels,
        fill_mask,
        excluded_mask,
        component_ids,
        included,
        excluded,
    ) = build_candidate_labels(preferred_labels, components, exclusions)

    ply_header = read_ply_header(args.source_ply)
    if not ply_header.elements or ply_header.elements[0].name != "vertex":
        raise ValueError("source PLY does not begin with a vertex element")
    if int(ply_header.elements[0].count) != vertex_count:
        raise ValueError("source PLY vertex count differs from preferred labels")

    label_map = build_candidate_label_map(
        preferred_label_map,
        existing_by_id,
        included,
        ontology,
        scene=args.scene,
        preferred_label_map_path=args.preferred_label_map,
        review_config_path=args.review_config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "gaussian_labels.npy", final_labels)
    np.save(args.output_dir / "gaussian_project_class_ids.npy", final_labels)
    np.save(args.output_dir / "incremental_fill_mask.npy", fill_mask)
    np.save(args.output_dir / "excluded_review_component_mask.npy", excluded_mask)
    np.save(args.output_dir / "incremental_fill_component_ids.npy", component_ids)
    np.savez_compressed(
        args.output_dir / "reviewed_incremental_fill_supports.npz",
        **{
            key: value
            for component in included
            for key, value in (
                (
                    f"component_{component.component_id:06d}_indices",
                    component.indices,
                ),
                (
                    f"component_{component.component_id:06d}_camera_counts",
                    component.camera_counts,
                ),
            )
        },
    )
    (args.output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2),
        encoding="utf-8",
    )
    semantic_ply_path = args.output_dir / "semantic_point_cloud.ply"
    if not args.no_semantic_ply:
        partial = semantic_ply_path.with_suffix(".ply.partial")
        write_ply_with_labels(args.source_ply, partial, final_labels)
        partial.replace(semantic_ply_path)

    if file_sha256(args.preferred_labels) != preferred_hash:
        raise RuntimeError("preferred label source changed during materialization")
    preferred_labeled_count = int(np.count_nonzero(preferred_labels))
    fill_count = int(np.count_nonzero(fill_mask))
    excluded_count = int(np.count_nonzero(excluded_mask))
    final_count = int(np.count_nonzero(final_labels))
    class_added_counts: dict[str, int] = {}
    for component in included:
        class_added_counts[component.class_name] = (
            class_added_counts.get(component.class_name, 0)
            + int(component.indices.size)
        )
    summary = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": args.scene,
        "candidate_only": True,
        "preferred_labels": str(args.preferred_labels),
        "preferred_labels_sha256": preferred_hash,
        "preferred_labels_read_only": True,
        "preferred_nonzero_labels_preserved": True,
        "preferred_label_map": str(args.preferred_label_map),
        "incremental_fill_report": str(args.incremental_fill_report),
        "incremental_fill_supports": str(args.incremental_fill_supports),
        "review_config": str(args.review_config),
        "review_config_contents": review_config,
        "ontology": str(args.ontology),
        "source_ply": str(args.source_ply),
        "assignment_policy": "fill_only_preferred_zero_gaussians",
        "review_policy": "explicit_exact_projection_component_exclusions",
        "vertex_count": vertex_count,
        "preferred_labeled_gaussian_count": preferred_labeled_count,
        "source_incremental_fill_gaussian_count": int(
            audit_report["incremental_fill_gaussian_count"]
        ),
        "source_accepted_component_count": len(components),
        "reviewed_included_component_count": len(included),
        "reviewed_excluded_component_count": len(excluded),
        "reviewed_fill_gaussian_count": fill_count,
        "reviewed_excluded_gaussian_count": excluded_count,
        "final_assigned_gaussian_count": final_count,
        "final_assigned_ratio": final_count / float(vertex_count),
        "preferred_unlabeled_recovery_ratio": fill_count
        / float(vertex_count - preferred_labeled_count),
        "class_added_gaussian_counts": dict(sorted(class_added_counts.items())),
        "included_components": [
            {
                "component_id": item.component_id,
                "class": item.class_name,
                "project_id": item.project_id,
                "gaussian_count": int(item.indices.size),
            }
            for item in included
        ],
        "excluded_components": [
            {
                "component_id": item.component_id,
                "class": item.class_name,
                "project_id": item.project_id,
                "gaussian_count": int(item.indices.size),
                "reason": exclusions[item.component_id].reason,
            }
            for item in excluded
        ],
        "semantic_labels_written": True,
        "semantic_project_class_arrays_written": True,
        "label_map_written": True,
        "semantic_ply_written": not args.no_semantic_ply,
    }
    (args.output_dir / "reviewed_incremental_fill_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
