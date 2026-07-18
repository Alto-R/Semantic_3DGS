#!/usr/bin/env python3
"""Fuse exact DINOv2 multi-view votes and form semantic/instance labels."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from add_labels_from_npy import write_ply_with_labels
from cluster_semantic_flashsplat_proposals import voxel_components
from dinov2_ontology import OntologyClass, load_ontology
from dinov2_voting import (
    accumulate_vote_arrays,
    semantic_evidence_fractions,
    semantic_winner_metrics,
    supporting_view_counts,
    threshold_winners,
    winner_metrics,
)
from ply_utils import read_ply_header, resolve_semantic_ply_output, vertex_data_memmap


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONTOLOGY = PROJECT_ROOT / "configs" / "ade20k_to_project.json"


def point_cloud_path(model_path: Path, iteration: int) -> Path:
    path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


@dataclass
class CandidateGroup:
    temporary_id: int
    ontology_class: OntologyClass
    indices: np.ndarray
    voxel_size: float | None
    component_index: int
    source_frames: set[str] = field(default_factory=set)
    assigned_threshold: int = 0
    final_id: int = 0


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "labels": output_dir / "gaussian_labels.npy",
        "project_classes": output_dir / "gaussian_project_class_ids.npy",
        "raw_winners": output_dir / "raw_project_class_ids.npy",
        "agreements": output_dir / "winner_agreements.npy",
        "semantic_evidence": output_dir / "winner_semantic_evidence.npy",
        "supporting_views": output_dir / "winner_supporting_views.npy",
        "label_map": output_dir / "label_map.json",
        "summary": output_dir / "dinov2_vote_summary.json",
    }


def calculate_voxel_size(
    vertex_data: np.memmap,
    indices: np.ndarray,
    multiplier: float,
    minimum: float,
    maximum: float,
) -> float:
    required = {"scale_0", "scale_1", "scale_2"}
    missing = sorted(required - set(vertex_data.dtype.names or ()))
    if missing:
        raise ValueError(f"PLY is missing instance scale properties: {missing}")
    log_scales = np.column_stack(
        [vertex_data[name][indices].astype(np.float64) for name in sorted(required)]
    )
    scales = np.exp(np.clip(log_scales.max(axis=1), -20.0, 5.0))
    finite = scales[np.isfinite(scales)]
    if finite.shape[0] == 0:
        raise ValueError("Could not derive a finite Gaussian scale")
    size = max(minimum, float(np.median(finite)) * multiplier)
    return min(size, maximum) if maximum > 0 else size


def build_candidate_groups(
    project_classes: np.ndarray,
    ontology_by_id: dict[int, OntologyClass],
    vertex_data: np.memmap,
    voxel_multiplier: float,
    min_voxel_size: float,
    max_voxel_size: float,
    min_component_gaussians: int,
    min_component_ratio: float,
) -> tuple[list[CandidateGroup], np.ndarray, list[dict[str, Any]]]:
    required = {"x", "y", "z"}
    missing = sorted(required - set(vertex_data.dtype.names or ()))
    if missing:
        raise ValueError(f"PLY is missing instance position properties: {missing}")

    candidates: list[CandidateGroup] = []
    temporary_labels = np.zeros(project_classes.shape, dtype=np.int32)
    component_reports: list[dict[str, Any]] = []
    for project_id in sorted(int(value) for value in np.unique(project_classes) if value > 0):
        ontology_class = ontology_by_id[project_id]
        indices = np.flatnonzero(project_classes == project_id)
        if ontology_class.kind == "stuff":
            temporary_id = len(candidates) + 1
            candidate = CandidateGroup(
                temporary_id=temporary_id,
                ontology_class=ontology_class,
                indices=indices,
                voxel_size=None,
                component_index=0,
            )
            candidates.append(candidate)
            temporary_labels[indices] = temporary_id
            component_reports.append(
                {
                    "project_id": project_id,
                    "class": ontology_class.project_class,
                    "type": "stuff",
                    "input_gaussians": int(indices.shape[0]),
                    "component_count": 1,
                }
            )
            continue

        voxel_size = calculate_voxel_size(
            vertex_data,
            indices,
            voxel_multiplier,
            min_voxel_size,
            max_voxel_size,
        )
        points = np.column_stack(
            [vertex_data[axis][indices].astype(np.float64) for axis in ("x", "y", "z")]
        )
        point_components, component_sizes, stats = voxel_components(points, voxel_size)
        keep_threshold = max(
            max(1, min_component_gaussians),
            int(math.ceil(int(component_sizes.max()) * max(0.0, min_component_ratio))),
        )
        kept_components = np.flatnonzero(component_sizes >= keep_threshold)
        if kept_components.shape[0] == 0:
            kept_components = np.asarray([int(np.argmax(component_sizes))], dtype=np.int64)

        order = np.argsort(point_components, kind="stable")
        sorted_components = point_components[order]
        boundaries = np.flatnonzero(np.diff(sorted_components)) + 1
        component_chunks = np.split(indices[order], boundaries)
        component_ids = sorted_components[np.r_[0, boundaries]]
        chunks_by_id = {
            int(component_id): chunk
            for component_id, chunk in zip(component_ids, component_chunks)
        }
        ordered_chunks = sorted(
            (chunks_by_id[int(component)] for component in kept_components),
            key=lambda chunk: int(chunk.min()),
        )
        for component_index, component_indices in enumerate(ordered_chunks, start=1):
            temporary_id = len(candidates) + 1
            candidate = CandidateGroup(
                temporary_id=temporary_id,
                ontology_class=ontology_class,
                indices=component_indices,
                voxel_size=voxel_size,
                component_index=component_index,
            )
            candidates.append(candidate)
            temporary_labels[component_indices] = temporary_id
        component_reports.append(
            {
                "project_id": project_id,
                "class": ontology_class.project_class,
                "type": "thing",
                "input_gaussians": int(indices.shape[0]),
                "voxel_size": voxel_size,
                "component_keep_threshold": keep_threshold,
                "kept_component_count": int(kept_components.shape[0]),
                "spatially_pruned_gaussians": int(
                    indices.shape[0] - component_sizes[kept_components].sum()
                ),
                **stats,
            }
        )
    return candidates, temporary_labels, component_reports


def populate_source_frames(
    candidates: list[CandidateGroup],
    temporary_labels: np.ndarray,
    frames: list[dict[str, Any]],
    vote_root: Path,
) -> None:
    candidates_by_id = {candidate.temporary_id: candidate for candidate in candidates}
    group_project_ids = np.zeros((len(candidates) + 1,), dtype=np.uint16)
    for candidate in candidates:
        group_project_ids[candidate.temporary_id] = candidate.ontology_class.project_id
    for frame in frames:
        vote_path = vote_root / str(frame["vote_file"])
        with np.load(vote_path) as data:
            indices = data["indices"].astype(np.int64, copy=False)
            class_ids = data["class_ids"].astype(np.uint16, copy=False)
        group_ids = temporary_labels[indices]
        selected = group_ids > 0
        if not selected.any():
            continue
        expected_classes = group_project_ids[group_ids[selected]]
        matched_groups = np.unique(group_ids[selected][class_ids[selected] == expected_classes])
        for group_id in matched_groups:
            candidates_by_id[int(group_id)].source_frames.add(str(frame["file"]))


def adaptive_threshold(
    candidate: CandidateGroup,
    total_views: int,
    minimum: int,
    minimum_thing: int,
    minimum_stuff: int,
    stuff_min_view_ratio: float,
    stuff_threshold_ratio: float,
    thing_floor_ratio: float,
) -> int:
    threshold = max(0, minimum)
    view_ratio = len(candidate.source_frames) / float(max(total_views, 1))
    if candidate.ontology_class.kind == "stuff":
        threshold = max(threshold, max(0, minimum_stuff))
        if total_views > 0 and view_ratio >= stuff_min_view_ratio:
            threshold = max(minimum, math.ceil(threshold * stuff_threshold_ratio))
    else:
        threshold = max(threshold, max(0, minimum_thing))
        if total_views > 0:
            ratio = max(thing_floor_ratio, 1.0 - min(1.0, view_ratio))
            threshold = max(minimum, math.ceil(threshold * ratio))
    return threshold


def finalize_groups(
    candidates: list[CandidateGroup],
    total_views: int,
    minimum: int,
    minimum_thing: int,
    minimum_stuff: int,
    stuff_min_view_ratio: float,
    stuff_threshold_ratio: float,
    thing_floor_ratio: float,
    gaussian_count: int,
) -> tuple[np.ndarray, list[CandidateGroup], list[CandidateGroup]]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.ontology_class.project_id,
            int(candidate.indices.min()),
        ),
    )
    kept: list[CandidateGroup] = []
    pruned: list[CandidateGroup] = []
    labels = np.zeros((gaussian_count,), dtype=np.int32)
    for candidate in ordered:
        candidate.assigned_threshold = adaptive_threshold(
            candidate,
            total_views,
            minimum,
            minimum_thing,
            minimum_stuff,
            stuff_min_view_ratio,
            stuff_threshold_ratio,
            thing_floor_ratio,
        )
        if candidate.indices.shape[0] < candidate.assigned_threshold:
            pruned.append(candidate)
            continue
        candidate.final_id = len(kept) + 1
        labels[candidate.indices] = candidate.final_id
        kept.append(candidate)
    return labels, kept, pruned


def group_record(candidate: CandidateGroup, total_views: int) -> dict[str, Any]:
    return {
        "id": candidate.final_id,
        "project_class_id": candidate.ontology_class.project_id,
        "ade_id": candidate.ontology_class.ade_id,
        "class": candidate.ontology_class.project_class,
        "type": candidate.ontology_class.kind,
        "component_index": candidate.component_index,
        "gaussian_count": int(candidate.indices.shape[0]),
        "source_view_count": len(candidate.source_frames),
        "source_view_ratio": len(candidate.source_frames) / float(max(total_views, 1)),
        "assigned_threshold": candidate.assigned_threshold,
        "voxel_size": candidate.voxel_size,
        "minimum_gaussian_index": int(candidate.indices.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--vote-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--min-views", default=2, type=int)
    parser.add_argument("--min-agreement", default=0.5, type=float)
    parser.add_argument(
        "--fusion-mode",
        choices=("joint", "semantic_only", "separate_abstain"),
        default="joint",
    )
    parser.add_argument("--min-semantic-evidence", default=0.5, type=float)
    parser.add_argument("--tie-epsilon", default=0.0, type=float)
    parser.add_argument("--fusion-chunk-size", default=100_000, type=int)
    parser.add_argument("--voxel-scale-multiplier", default=4.0, type=float)
    parser.add_argument("--min-voxel-size", default=0.01, type=float)
    parser.add_argument("--max-voxel-size", default=0.20, type=float)
    parser.add_argument("--min-component-gaussians", default=500, type=int)
    parser.add_argument("--min-component-ratio", default=0.01, type=float)
    parser.add_argument("--min-assigned-gaussians", default=0, type=int)
    parser.add_argument("--min-assigned-thing-gaussians", default=5000, type=int)
    parser.add_argument("--min-assigned-stuff-gaussians", default=10000, type=int)
    parser.add_argument("--adaptive-stuff-min-view-ratio", default=0.5, type=float)
    parser.add_argument("--adaptive-stuff-threshold-ratio", default=0.75, type=float)
    parser.add_argument("--adaptive-thing-threshold-floor-ratio", default=0.5, type=float)
    parser.add_argument("--semantic-ply-name")
    parser.add_argument("--semantic-ply-path", type=Path)
    parser.add_argument("--no-semantic-ply", action="store_true")
    parser.add_argument("--scene", default="")
    parser.add_argument("--baseline-labels", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.min_views < 0:
        raise ValueError("min-views must be non-negative")
    if args.tie_epsilon < 0.0:
        raise ValueError("tie-epsilon must be non-negative")
    if not 0.0 <= args.min_agreement <= 1.0:
        raise ValueError("min-agreement must be between zero and one")
    if not 0.0 <= args.min_semantic_evidence <= 1.0:
        raise ValueError("min-semantic-evidence must be between zero and one")
    if args.fusion_chunk_size <= 0:
        raise ValueError("fusion-chunk-size must be positive")
    if not 0.0 <= args.adaptive_stuff_min_view_ratio <= 1.0:
        raise ValueError("adaptive-stuff-min-view-ratio must be between zero and one")
    if not 0.0 < args.adaptive_stuff_threshold_ratio <= 1.0:
        raise ValueError("adaptive-stuff-threshold-ratio must be in (0, 1]")
    if not 0.0 < args.adaptive_thing_threshold_floor_ratio <= 1.0:
        raise ValueError("adaptive-thing-threshold-floor-ratio must be in (0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(args.output_dir)
    semantic_ply = resolve_semantic_ply_output(
        args.output_dir,
        semantic_ply_path=args.semantic_ply_path,
        semantic_ply_name=args.semantic_ply_name,
        disabled=args.no_semantic_ply,
    )
    output_files = [*paths.values(), *([semantic_ply] if semantic_ply is not None else [])]
    for path in output_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    vote_manifest_path = args.vote_dir / "vote_manifest.json"
    vote_manifest = json.loads(vote_manifest_path.read_text(encoding="utf-8"))
    frames = vote_manifest["frames"]
    vote_files = [args.vote_dir / str(frame["vote_file"]) for frame in frames]
    if not vote_files:
        raise ValueError("Vote manifest contains no views")
    ontology = load_ontology(args.ontology)
    class_count = ontology.class_count
    ply_path = point_cloud_path(args.model_path, args.iteration)
    header = read_ply_header(ply_path)
    vertex = header.element("vertex")
    if vertex is None:
        raise ValueError(f"{ply_path} has no vertex element")
    gaussian_count = vertex.count
    if int(vote_manifest["gaussian_count"]) != gaussian_count:
        raise ValueError("Vote manifest Gaussian count does not match the PLY")

    temp_path = args.output_dir / ".dinov2_vote_matrix.float32"
    if temp_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Temporary vote matrix already exists: {temp_path}")
        temp_path.unlink()
    vote_matrix: np.memmap | None = None
    try:
        vote_matrix = np.memmap(
            temp_path,
            dtype=np.float32,
            mode="w+",
            shape=(class_count + 1, gaussian_count),
        )
        vote_matrix[:] = 0.0
        for path in vote_files:
            with np.load(path) as data:
                accumulate_vote_arrays(
                    vote_matrix,
                    data["indices"],
                    data["class_ids"],
                    data["weights"],
                )
        vote_matrix.flush()

        raw_winners = np.zeros((gaussian_count,), dtype=np.uint16)
        agreements = np.zeros((gaussian_count,), dtype=np.float32)
        semantic_evidence = np.zeros((gaussian_count,), dtype=np.float32)
        for start in range(0, gaussian_count, args.fusion_chunk_size):
            end = min(start + args.fusion_chunk_size, gaussian_count)
            chunk = vote_matrix[:, start:end]
            if args.fusion_mode == "joint":
                raw, _winner_scores, _second_scores, agreement = winner_metrics(
                    chunk,
                    tie_epsilon=args.tie_epsilon,
                )
                evidence = semantic_evidence_fractions(chunk)
            else:
                (
                    raw,
                    _winner_scores,
                    _second_scores,
                    agreement,
                    evidence,
                ) = semantic_winner_metrics(
                    chunk,
                    tie_epsilon=args.tie_epsilon,
                )
            raw_winners[start:end] = raw
            agreements[start:end] = agreement
            semantic_evidence[start:end] = evidence
        supporting_views = supporting_view_counts(vote_files, raw_winners)
        applied_min_semantic_evidence = (
            args.min_semantic_evidence
            if args.fusion_mode == "separate_abstain"
            else 0.0
        )
        project_classes = threshold_winners(
            raw_winners,
            agreements,
            supporting_views,
            args.min_views,
            args.min_agreement,
            semantic_evidence=semantic_evidence,
            min_semantic_evidence=applied_min_semantic_evidence,
        )
    finally:
        if vote_matrix is not None:
            vote_matrix.flush()
            del vote_matrix
        if temp_path.exists():
            temp_path.unlink()

    _header, vertex_data = vertex_data_memmap(ply_path)
    candidates, temporary_labels, component_reports = build_candidate_groups(
        project_classes,
        ontology.by_project_id,
        vertex_data,
        args.voxel_scale_multiplier,
        args.min_voxel_size,
        args.max_voxel_size,
        args.min_component_gaussians,
        args.min_component_ratio,
    )
    populate_source_frames(candidates, temporary_labels, frames, args.vote_dir)
    labels, kept, pruned = finalize_groups(
        candidates,
        len(frames),
        args.min_assigned_gaussians,
        args.min_assigned_thing_gaussians,
        args.min_assigned_stuff_gaussians,
        args.adaptive_stuff_min_view_ratio,
        args.adaptive_stuff_threshold_ratio,
        args.adaptive_thing_threshold_floor_ratio,
        gaussian_count,
    )

    raw_unlabeled = float(np.mean(raw_winners == 0))
    thresholded_unlabeled = float(np.mean(project_classes == 0))
    final_unlabeled = float(np.mean(labels == 0))
    baseline_comparison = None
    if args.baseline_labels is not None:
        baseline_labels = np.load(args.baseline_labels)
        if baseline_labels.shape != labels.shape:
            raise ValueError(
                f"Baseline labels shape {baseline_labels.shape} does not match {labels.shape}"
            )
        baseline_unlabeled = float(np.mean(baseline_labels == 0))
        baseline_comparison = {
            "labels_path": str(args.baseline_labels),
            "baseline_unlabeled_ratio": baseline_unlabeled,
            "dinov2_final_unlabeled_ratio": final_unlabeled,
            "unlabeled_ratio_reduction": baseline_unlabeled - final_unlabeled,
        }

    np.save(paths["labels"], labels)
    np.save(paths["project_classes"], project_classes)
    np.save(paths["raw_winners"], raw_winners)
    np.save(paths["agreements"], agreements)
    np.save(paths["semantic_evidence"], semantic_evidence)
    np.save(paths["supporting_views"], supporting_views)
    if semantic_ply is not None:
        write_ply_with_labels(ply_path, semantic_ply, labels)

    class_totals: dict[int, int] = {}
    for candidate in kept:
        project_id = candidate.ontology_class.project_id
        class_totals[project_id] = class_totals.get(project_id, 0) + 1
    class_ordinals: dict[int, int] = {}
    labels_json: list[dict[str, Any]] = [
        {"id": 0, "name": "unlabeled", "class": "unlabeled", "type": "unlabeled"}
    ]
    for candidate in kept:
        project_id = candidate.ontology_class.project_id
        class_ordinals[project_id] = class_ordinals.get(project_id, 0) + 1
        class_name = candidate.ontology_class.project_class
        name = (
            class_name
            if candidate.ontology_class.kind == "stuff" and class_totals[project_id] == 1
            else f"{class_name}_{class_ordinals[project_id]:03d}"
        )
        labels_json.append({"name": name, **group_record(candidate, len(frames))})
    label_map = {
        "scene": args.scene or args.model_path.name,
        "source": "dinov2_vitl14_ade20k_linear_multiview_voting",
        "ontology": str(args.ontology),
        "labels": labels_json,
    }
    paths["label_map"].write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    histogram = {
        str(label): int(count)
        for label, count in zip(*np.unique(labels, return_counts=True))
    }
    summary = {
        "source": "dinov2_vitl14_ade20k_linear_multiview_voting",
        "stage": "exact_vote_fusion_and_instance_postprocessing",
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "vote_manifest": str(vote_manifest_path),
        "ontology": str(args.ontology),
        "gaussian_count": gaussian_count,
        "source_view_count": len(frames),
        "vote_accumulator": "temporary_disk_backed_float32_exact_class_by_gaussian_sum",
        "parameters": {
            "fusion_mode": args.fusion_mode,
            "min_views": args.min_views,
            "min_agreement": args.min_agreement,
            "agreement_denominator": (
                "all_vote_mass_including_abstain"
                if args.fusion_mode == "joint"
                else "semantic_vote_mass_only"
            ),
            "winner_domain": (
                "all_rows_including_abstain"
                if args.fusion_mode == "joint"
                else "semantic_rows_only"
            ),
            "min_semantic_evidence": applied_min_semantic_evidence,
            "semantic_evidence_definition": (
                "semantic_vote_mass/(semantic_vote_mass+abstain_vote_mass)"
            ),
            "tie_epsilon": args.tie_epsilon,
            "voxel_scale_multiplier": args.voxel_scale_multiplier,
            "min_voxel_size": args.min_voxel_size,
            "max_voxel_size": args.max_voxel_size,
            "min_component_gaussians": args.min_component_gaussians,
            "min_component_ratio": args.min_component_ratio,
            "min_assigned_gaussians": args.min_assigned_gaussians,
            "min_assigned_thing_gaussians": args.min_assigned_thing_gaussians,
            "min_assigned_stuff_gaussians": args.min_assigned_stuff_gaussians,
            "adaptive_stuff_min_view_ratio": args.adaptive_stuff_min_view_ratio,
            "adaptive_stuff_threshold_ratio": args.adaptive_stuff_threshold_ratio,
            "adaptive_thing_threshold_floor_ratio": args.adaptive_thing_threshold_floor_ratio,
        },
        "coverage": {
            "raw_argmax_unlabeled_ratio": raw_unlabeled,
            "thresholded_unlabeled_ratio": thresholded_unlabeled,
            "final_post_pruning_unlabeled_ratio": final_unlabeled,
            "groundingdino_baseline_comparison": baseline_comparison,
        },
        "candidate_group_count": len(candidates),
        "final_group_count": len(kept),
        "pruned_group_count": len(pruned),
        "label_histogram": histogram,
        "components": component_reports,
        "groups": [group_record(candidate, len(frames)) for candidate in kept],
        "pruned_groups": [group_record(candidate, len(frames)) for candidate in pruned],
        "gaze_hit_coverage": {
            "status": "not_run",
            "reason": "Task 2 gaze-to-scene coordinate alignment is not yet established",
        },
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if semantic_ply is not None:
        print(f"wrote {semantic_ply}")
    else:
        print("semantic PLY disabled; retained labels and label map only")
    print(json.dumps(summary["coverage"], sort_keys=True))


if __name__ == "__main__":
    main()
