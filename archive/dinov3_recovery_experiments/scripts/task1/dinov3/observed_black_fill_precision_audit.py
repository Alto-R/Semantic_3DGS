#!/usr/bin/env python3
"""Held-out precision audit for component-graph black-spot fills.

The round-trip validator measures whole-scene agreement, which is dominated by
the immutable anchors and barely moves when a few thousand black Gaussians are
labeled.  This audit isolates the fills: for every original baseline camera it
rebuilds the same leave-one-camera-out component-graph candidate, renders ONLY
the newly resolved black Gaussians into the held-out camera, and measures how
often those fills agree with the cached DINOv3 map, per class and per camera.
It is cache-only and report-only and writes no labels or PLY.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from scripts.task1.common.flashsplat_cameras import (
    background_tensor,
    default_pipeline,
    load_cameras,
    load_flashsplat,
    load_gaussians,
    make_camera,
    point_cloud_path,
)
from scripts.task1.dinov3.observed_black_component_graph_round_trip_validation import (
    build_fold_candidate,
    prepare_component_validation,
)
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    quantile_summary,
    render_binary_project_ids,
    save_visuals,
)


SOURCE = "dinov3_observed_black_fill_precision_audit"
CONTRACT = "report_only_leave_one_camera_out_fill_precision_v1"


def fill_labels(
    baseline_labels: np.ndarray,
    candidate_labels: np.ndarray,
) -> np.ndarray:
    """Keep only black Gaussians that the candidate newly resolved."""

    baseline = np.asarray(baseline_labels, dtype=np.uint16)
    candidate = np.asarray(candidate_labels, dtype=np.uint16)
    if baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate label arrays have different shapes")
    return np.where((baseline == 0) & (candidate > 0), candidate, 0).astype(np.uint16)


def _fill_metrics() -> Dict[str, int]:
    return {
        "fill_pixels": 0,
        "fill_source_pixels": 0,
        "fill_agreed": 0,
        "fill_wrong": 0,
        "fill_unverifiable_pixels": 0,
    }


def update_fill_metrics(
    values: Dict[str, int],
    predicted: np.ndarray,
    valid: np.ndarray,
    source: np.ndarray,
) -> None:
    fill_pixels = int(np.count_nonzero(valid))
    source_pixels = valid & (source > 0)
    unverifiable = valid & (source == 0)
    agreed = source_pixels & (predicted == source)
    values["fill_pixels"] += fill_pixels
    values["fill_source_pixels"] += int(np.count_nonzero(source_pixels))
    values["fill_agreed"] += int(np.count_nonzero(agreed))
    values["fill_wrong"] += int(
        np.count_nonzero(source_pixels & (predicted != source))
    )
    values["fill_unverifiable_pixels"] += int(np.count_nonzero(unverifiable))


def update_fill_per_class(
    values: Dict[int, Dict[str, int]],
    predicted: np.ndarray,
    valid: np.ndarray,
    source: np.ndarray,
) -> None:
    for project_id in np.unique(source):
        key = int(project_id)
        if key <= 0:
            continue
        row = values.setdefault(
            key,
            {"source_pixels": 0, "agreed": 0, "wrong": 0, "resolved_gaussians": 0},
        )
        selected = valid & (source == project_id)
        row["source_pixels"] += int(np.count_nonzero(selected))
        row["agreed"] += int(
            np.count_nonzero(selected & (predicted == project_id))
        )
        row["wrong"] += int(
            np.count_nonzero(selected & (predicted != project_id))
        )


def main() -> None:
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source-view-dir", required=True, type=Path)
    parser.add_argument("--selected-cache-report", required=True, type=Path)
    parser.add_argument("--baseline-vote-manifest", required=True, type=Path)
    parser.add_argument("--additional-vote-manifest", required=True, type=Path)
    parser.add_argument("--hard-audit-report", required=True, type=Path)
    parser.add_argument("--hard-diagnostics", required=True, type=Path)
    parser.add_argument("--hard-confusion", required=True, type=Path)
    parser.add_argument("--recovery-report", required=True, type=Path)
    parser.add_argument("--candidate-labels", required=True, type=Path)
    parser.add_argument("--recovery-source-codes", required=True, type=Path)
    parser.add_argument("--component-audit-report", required=True, type=Path)
    parser.add_argument("--component-diagnostics", required=True, type=Path)
    parser.add_argument("--dinov2-vote-manifest", default=None, type=Path)
    parser.add_argument("--runner-up-cap", default=None, type=float)
    parser.add_argument("--class-aware", action="store_true")
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--flashsplat-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--chunk-size", default=100_000, type=int)
    parser.add_argument("--query-workers", default=-1, type=int)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    context = prepare_component_validation(args)
    cache_report = context["cache_report"]
    dino_manifest = context["dino_manifest"]
    baseline_manifest = context["baseline_manifest"]
    component_report = context["component_report"]
    baseline_indices = context["baseline_indices"]
    gaussian_count = context["gaussian_count"]
    class_count = context["class_count"]
    ontology = context["ontology"]
    baseline_evidence = context["baseline_evidence"]
    additional_evidence = context["additional_evidence"]
    dinov2_evidence = context["dinov2_evidence"]
    runner_up_cap = context["runner_up_cap"]
    class_aware = context["class_aware"]
    counts = context["counts"]
    temporary = context["temporary"]
    baseline_statistics = context["baseline_statistics"]
    status = context["status"]
    class_reliability = context["class_reliability"]
    vertices = context["vertices"]
    black_indices = context["black_indices"]
    anchor_indices = context["anchor_indices"]
    anchor_tree = context["anchor_tree"]

    modules = load_flashsplat(args.flashsplat_root)
    cameras = load_cameras(args.model_path)
    ply_path = point_cloud_path(args.model_path, args.iteration)
    if ply_path.resolve() != args.source_ply.resolve():
        raise ValueError("model PLY differs from the component audit source PLY")
    gaussians = load_gaussians(modules, ply_path, args.sh_degree)
    if int(gaussians.get_xyz.shape[0]) != gaussian_count:
        raise ValueError("model Gaussian count differs from vote manifest")
    pipeline = default_pipeline()
    background = background_tensor(False)
    rgb_dir = args.source_view_dir / "rgb_renders"
    lookup = ontology.ade_to_project
    values = _fill_metrics()
    per_class = {}
    newly_resolved = np.zeros((black_indices.size,), dtype=bool)
    newly_resolved_by_class: Dict[int, int] = {}
    per_camera: List[Dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=False)
    overlay_dir = args.output_dir / "fill_heldout_overlays"
    disagreement_dir = args.output_dir / "fill_heldout_disagreement"
    overlay_dir.mkdir()
    disagreement_dir.mkdir()

    for frame in baseline_manifest["frames"]:
        camera_index = int(frame["camera_index"])
        fold = build_fold_candidate(
            counts=counts,
            baseline_statistics=baseline_statistics,
            status=status,
            baseline_evidence=baseline_evidence,
            additional_evidence=additional_evidence,
            dinov2_evidence=dinov2_evidence,
            runner_up_cap=runner_up_cap,
            class_aware=class_aware,
            vertices=vertices,
            black_indices=black_indices,
            anchor_indices=anchor_indices,
            anchor_tree=anchor_tree,
            class_reliability=class_reliability,
            component_report=component_report,
            camera_index=camera_index,
            gaussian_count=gaussian_count,
            class_count=class_count,
            chunk_size=args.chunk_size,
            query_workers=args.query_workers,
        )
        baseline_labels = fold["baseline_labels"]
        candidate = fold["candidate"]
        fold_newly = fold["fold_newly"]
        newly_resolved |= fold_newly
        fold_classes = fold["fold_values"]["candidate_project_id"][fold_newly]
        for project_id in np.unique(fold_classes):
            key = int(project_id)
            if key <= 0:
                continue
            newly_resolved_by_class[key] = newly_resolved_by_class.get(key, 0) + int(
                np.count_nonzero(fold_classes == project_id)
            )
        fills = fill_labels(baseline_labels, candidate)
        camera = make_camera(
            cameras[camera_index], modules, int(cache_report["render"]["max_width"])
        )
        fill_predicted, fill_valid, fill_margin = render_binary_project_ids(
            fills, camera, gaussians, modules, pipeline, background,
            class_count=class_count,
        )
        dino_frame = next(
            item for item in dino_manifest["frames"]
            if int(item["camera_index"]) == camera_index
        )
        with np.load(
            args.source_view_dir / str(dino_frame["segment_file"]), allow_pickle=False
        ) as segment:
            source_project = lookup[np.asarray(segment["class_id"], dtype=np.uint8)]
        if source_project.shape != fill_predicted.shape:
            raise ValueError("held-out DINO map and fill projection shapes differ")
        update_fill_metrics(values, fill_predicted, fill_valid, source_project)
        update_fill_per_class(per_class, fill_predicted, fill_valid, source_project)
        base_rgb = np.asarray(
            Image.open(rgb_dir / str(frame["file"])).convert("RGB"), dtype=np.uint8
        )
        save_visuals(
            base_rgb,
            fill_predicted,
            fill_valid,
            source_project,
            ontology,
            overlay_dir / str(frame["file"]),
            disagreement_dir / str(frame["file"]),
        )
        per_camera.append(
            {
                "camera_index": camera_index,
                "camera_id": int(frame["camera_id"]),
                "file": str(frame["file"]),
                "newly_resolved_gaussian_count": int(np.count_nonzero(fold_newly)),
                "fill_pixels": int(np.count_nonzero(fill_valid)),
                "fill_source_pixels": int(np.count_nonzero(fill_valid & (source_project > 0))),
                "fill_agreed": int(
                    np.count_nonzero(
                        fill_valid & (source_project > 0) & (fill_predicted == source_project)
                    )
                ),
                "fill_wrong": int(
                    np.count_nonzero(
                        fill_valid & (source_project > 0) & (fill_predicted != source_project)
                    )
                ),
                "fill_unverifiable_pixels": int(
                    np.count_nonzero(fill_valid & (source_project == 0))
                ),
                "fill_precision": (
                    int(
                        np.count_nonzero(
                            fill_valid & (source_project > 0) & (fill_predicted == source_project)
                        )
                    )
                    / int(np.count_nonzero(fill_valid & (source_project > 0)))
                    if np.any(fill_valid & (source_project > 0))
                    else 0.0
                ),
                "fill_binary_margin": quantile_summary(fill_margin[fill_valid]),
            }
        )
        print(
            "held out camera %s: fill_pixels=%d agreed=%d precision=%.4f"
            % (
                camera_index,
                per_camera[-1]["fill_pixels"],
                per_camera[-1]["fill_agreed"],
                per_camera[-1]["fill_precision"],
            )
        )

    per_class_recovery = []
    for project_id in sorted(per_class):
        row = per_class[project_id]
        item = ontology.by_project_id[project_id]
        source_pixels = row["source_pixels"]
        per_class_recovery.append(
            {
                "project_id": project_id,
                "class": item.project_class,
                "type": item.kind,
                "source_pixels": source_pixels,
                "agreed": row["agreed"],
                "wrong": row["wrong"],
                "precision_of_source": (
                    row["agreed"] / source_pixels if source_pixels else 0.0
                ),
                "newly_resolved_gaussians": newly_resolved_by_class.get(project_id, 0),
            }
        )
    fill_precision = (
        values["fill_agreed"] / values["fill_source_pixels"]
        if values["fill_source_pixels"]
        else 0.0
    )
    report = {
        "source": SOURCE,
        "contract": CONTRACT,
        "scene": args.scene,
        "report_only": True,
        "gaussian_count": gaussian_count,
        "camera_count": len(baseline_indices),
        "camera_indices": baseline_indices,
        "heldout_policy": "exclude_each_original_baseline_camera_from_baseline_and_component_graph_evidence",
        "component_audit_report": str(args.component_audit_report),
        "component_diagnostics": str(args.component_diagnostics),
        "dinov2_vote_manifest": (
            str(context["dinov2_vote_manifest"])
            if context["dinov2_vote_manifest"] is not None
            else None
        ),
        "dinov2_agreement_gate_used": bool(dinov2_evidence),
        "runner_up_cap": runner_up_cap,
        "class_aware": class_aware,
        "candidate_rule": (
            "class_aware_plurality_runner_up_cap"
            if (runner_up_cap is not None and class_aware)
            else (
                "plurality_runner_up_cap"
                if runner_up_cap is not None
                else "strict_component_graph"
            )
        ),
        "full_evidence_candidate_reproduced": True,
        "fill_metrics": values,
        "fill_precision_of_source": fill_precision,
        "per_class_fill": per_class_recovery,
        "per_camera": per_camera,
        "candidate_recovered_count": int(np.count_nonzero(newly_resolved)),
        "candidate_recovered_ratio": (
            float(np.mean(newly_resolved)) if newly_resolved.size else 0.0
        ),
        "immutable_anchor_labels_changed": 0,
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "manual_class_selection_used": False,
        "scene_specific_rules": False,
        "accepted_gaussian_labels_written": False,
        "gaussian_project_class_array_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
    }
    (args.output_dir / "fill_precision_audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output_dir / "experiment_mode.txt").write_text(
        "mode=report_only_leave_one_camera_out_fill_precision\n"
        "accepted_gaussian_labels_written=0\n"
        "semantic_ply_written=0\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "scene", "camera_count", "fill_metrics", "fill_precision_of_source",
        "candidate_recovered_count",
    )}, indent=2))
    del counts
    temporary.cleanup()


if __name__ == "__main__":
    main()
