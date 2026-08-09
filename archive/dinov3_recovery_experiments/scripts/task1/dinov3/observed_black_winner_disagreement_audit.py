#!/usr/bin/env python3
"""Held-out decomposition of the raw-vs-weighted component disagreement.

Most camera-observed black Gaussians are blocked because the raw component
winner and the reliability-weighted component winner disagree.  For every
original baseline camera this audit rebuilds the same leave-one-camera-out
component-graph candidate, then asks whether the held-out camera's own
component vote supports the raw winner, the weighted winner, both, or neither.
It also renders the newly resolved fills and builds a predicted-vs-source
confusion matrix.  Cache-only and report-only; no labels or PLY are written.
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
from scripts.task1.dinov3.observed_black_component_graph_audit import (
    DECISION_COMPONENT_CONFLICT,
)
from scripts.task1.dinov3.observed_black_component_graph_round_trip_validation import (
    build_fold_candidate,
    prepare_component_validation,
)
from scripts.task1.dinov3.observed_black_fill_precision_audit import (
    fill_labels,
)
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    render_binary_project_ids,
)


SOURCE = "dinov3_observed_black_winner_disagreement_audit"
CONTRACT = "report_only_leave_one_camera_out_winner_disagreement_v1"

RAW_TIED = "raw_tied"
WINNERS_DISAGREE = "raw_weighted_winners_disagree"
WEIGHTED_NOT_ACCEPTED = "weighted_not_accepted"


def camera_component_votes(
    component_ids: np.ndarray,
    observed_mask: np.ndarray,
    winners: np.ndarray,
    masses: np.ndarray,
    black_indices: np.ndarray,
    *,
    class_count: int,
) -> Dict[str, np.ndarray]:
    """Aggregate one held-out camera's votes per component.

    Mirrors the per-camera normalization used by the component-graph audit:
    per component, sum each class's winning mass, require a unique winner with
    more than half of the component's mass, otherwise abstain.
    """

    components = np.asarray(component_ids, dtype=np.int64)[observed_mask]
    selected = np.asarray(black_indices, dtype=np.int64)[observed_mask]
    local_winner = np.asarray(winners, dtype=np.uint16)[selected]
    local_mass = np.asarray(masses, dtype=np.float32)[selected]
    supported = local_winner > 0
    component_count = int(components.max() + 1) if components.size else 0
    totals = np.zeros((component_count, class_count + 1), dtype=np.float32)
    if np.any(supported):
        keys = (
            components[supported] * np.int64(class_count + 1)
            + local_winner[supported].astype(np.int64)
        )
        np.add.at(totals.reshape(-1), keys, local_mass[supported])
    semantic = totals[:, 1:]
    maximum = semantic.max(axis=1)
    winner = semantic.argmax(axis=1).astype(np.uint16) + np.uint16(1)
    tied = (semantic == maximum[:, None]).sum(axis=1) > 1
    total = semantic.sum(axis=1)
    accepted = (maximum > 0.0) & ~tied & (maximum * 2.0 > total)
    return {
        "winner": np.where(accepted, winner, 0).astype(np.uint16),
        "accepted": accepted,
        "tied": tied,
        "share": np.divide(
            maximum,
            total,
            out=np.zeros_like(maximum),
            where=total > 0.0,
        ),
    }


def disagreement_subtype(
    raw_winner: int,
    weighted_winner: int,
    raw_accepted: bool,
    weighted_accepted: bool,
    raw_tied: bool,
) -> str:
    if raw_tied:
        return RAW_TIED
    if raw_winner != weighted_winner:
        return WINNERS_DISAGREE
    if not weighted_accepted:
        return WEIGHTED_NOT_ACCEPTED
    if not raw_accepted:
        return RAW_TIED
    return WINNERS_DISAGREE


def update_fill_confusion(
    matrix: np.ndarray,
    predicted: np.ndarray,
    valid: np.ndarray,
    source: np.ndarray,
) -> None:
    selected = valid & (source > 0) & (predicted > 0)
    rows = predicted[selected].astype(np.int64)
    columns = source[selected].astype(np.int64)
    np.add.at(matrix, (rows, columns), 1)


def _counters() -> Dict[str, Dict[str, int]]:
    return {
        name: {
            "component_count": 0,
            "gaussian_count": 0,
            "raw_matches_heldout": 0,
            "weighted_matches_heldout": 0,
            "both_match": 0,
            "neither_match": 0,
            "heldout_abstained": 0,
        }
        for name in (RAW_TIED, WINNERS_DISAGREE, WEIGHTED_NOT_ACCEPTED)
    }


def main() -> None:
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
    lookup = ontology.ade_to_project
    counters = _counters()
    raw_confusion = np.zeros((class_count + 1, class_count + 1), dtype=np.uint64)
    weighted_confusion = np.zeros((class_count + 1, class_count + 1), dtype=np.uint64)
    fill_confusion = np.zeros((class_count + 1, class_count + 1), dtype=np.uint64)
    fill_pixels = 0
    fill_source_pixels = 0
    fill_agreed = 0
    newly_resolved = np.zeros((black_indices.size,), dtype=bool)
    per_camera: List[Dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=False)

    for frame in baseline_manifest["frames"]:
        camera_index = int(frame["camera_index"])
        fold = build_fold_candidate(
            counts=counts,
            baseline_statistics=baseline_statistics,
            status=status,
            baseline_evidence=baseline_evidence,
            additional_evidence=additional_evidence,
            dinov2_evidence=dinov2_evidence,
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
        evidence = fold["evidence"]
        baseline_labels = fold["baseline_labels"]
        candidate = fold["candidate"]
        fold_values = fold["fold_values"]
        fold_newly = fold["fold_newly"]
        newly_resolved |= fold_newly

        conflict_nodes = (
            np.asarray(fold_values["decision_code"]) == DECISION_COMPONENT_CONFLICT
        )
        conflict_components = np.unique(
            np.asarray(fold_values["component_id"])[conflict_nodes]
        )
        conflict_components = conflict_components[conflict_components >= 0]
        heldout_votes = camera_component_votes(
            np.asarray(fold_values["component_id"], dtype=np.int64),
            np.asarray(fold_values["observed_mask"], dtype=bool),
            evidence["winners"],
            evidence["mass"],
            black_indices,
            class_count=class_count,
        )
        camera_counter = {name: 0 for name in counters}
        camera_gaussians = {name: 0 for name in counters}
        camera_raw = {name: 0 for name in counters}
        camera_weighted = {name: 0 for name in counters}
        camera_both = {name: 0 for name in counters}
        camera_neither = {name: 0 for name in counters}
        camera_abstained = {name: 0 for name in counters}
        for component_id in conflict_components:
            nodes = (
                np.asarray(fold_values["component_id"]) == component_id
            ) & conflict_nodes
            if not np.any(nodes):
                continue
            size = int(np.count_nonzero(nodes))
            raw_winner = int(np.asarray(fold_values["raw_winner"])[nodes][0])
            weighted_winner = int(
                np.asarray(fold_values["weighted_winner"])[nodes][0]
            )
            raw_accepted = bool(
                np.asarray(fold_values["raw_accepted"])[nodes][0]
            )
            weighted_accepted = bool(
                np.asarray(fold_values["weighted_accepted"])[nodes][0]
            )
            raw_tied = bool(np.asarray(fold_values["raw_tied"])[nodes][0])
            name = disagreement_subtype(
                raw_winner,
                weighted_winner,
                raw_accepted,
                weighted_accepted,
                raw_tied,
            )
            heldout_winner = int(heldout_votes["winner"][component_id])
            raw_matches = (
                raw_winner > 0
                and heldout_winner > 0
                and raw_winner == heldout_winner
            )
            weighted_matches = (
                weighted_winner > 0
                and heldout_winner > 0
                and weighted_winner == heldout_winner
            )
            counters[name]["component_count"] += 1
            counters[name]["gaussian_count"] += size
            camera_counter[name] += 1
            camera_gaussians[name] += size
            if heldout_winner == 0:
                counters[name]["heldout_abstained"] += 1
                camera_abstained[name] += 1
            else:
                if raw_matches:
                    counters[name]["raw_matches_heldout"] += 1
                    camera_raw[name] += 1
                if weighted_matches:
                    counters[name]["weighted_matches_heldout"] += 1
                    camera_weighted[name] += 1
                if raw_matches and weighted_matches:
                    counters[name]["both_match"] += 1
                    camera_both[name] += 1
                if not raw_matches and not weighted_matches:
                    counters[name]["neither_match"] += 1
                    camera_neither[name] += 1
            if raw_winner > 0 and heldout_winner > 0:
                raw_confusion[raw_winner, heldout_winner] += np.uint64(size)
            if weighted_winner > 0 and heldout_winner > 0:
                weighted_confusion[weighted_winner, heldout_winner] += np.uint64(size)

        fills = fill_labels(baseline_labels, candidate)
        camera = make_camera(
            cameras[camera_index], modules, int(cache_report["render"]["max_width"])
        )
        fill_predicted, fill_valid, _ = render_binary_project_ids(
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
        update_fill_confusion(
            fill_confusion, fill_predicted, fill_valid, source_project
        )
        fill_pixels += int(np.count_nonzero(fill_valid))
        fill_source_pixels += int(
            np.count_nonzero(fill_valid & (source_project > 0))
        )
        fill_agreed += int(
            np.count_nonzero(
                fill_valid
                & (source_project > 0)
                & (fill_predicted == source_project)
            )
        )
        per_camera.append(
            {
                "camera_index": camera_index,
                "camera_id": int(frame["camera_id"]),
                "file": str(frame["file"]),
                "conflict_component_count": int(len(conflict_components)),
                "conflict_gaussian_count": int(
                    np.count_nonzero(conflict_nodes)
                ),
                "by_subtype": {
                    name: {
                        "component_count": camera_counter[name],
                        "gaussian_count": camera_gaussians[name],
                        "raw_matches_heldout": camera_raw[name],
                        "weighted_matches_heldout": camera_weighted[name],
                        "both_match": camera_both[name],
                        "neither_match": camera_neither[name],
                        "heldout_abstained": camera_abstained[name],
                    }
                    for name in counters
                },
                "fill_pixels": int(np.count_nonzero(fill_valid)),
                "fill_agreed": int(
                    np.count_nonzero(
                        fill_valid
                        & (source_project > 0)
                        & (fill_predicted == source_project)
                    )
                ),
            }
        )
        print(
            "held out camera %s: conflict_components=%d raw_matches=%d weighted_matches=%d"
            % (
                camera_index,
                len(conflict_components),
                sum(camera_raw.values()),
                sum(camera_weighted.values()),
            )
        )

    def confusion_rows(matrix: np.ndarray) -> List[Dict[str, Any]]:
        rows = []
        for predicted_id, source_id in zip(*np.nonzero(matrix)):
            if int(predicted_id) == 0 or int(source_id) == 0:
                continue
            rows.append(
                {
                    "predicted_project_id": int(predicted_id),
                    "predicted_class": ontology.by_project_id[
                        int(predicted_id)
                    ].project_class,
                    "source_project_id": int(source_id),
                    "source_class": ontology.by_project_id[int(source_id)].project_class,
                    "gaussian_count": int(matrix[predicted_id, source_id]),
                }
            )
        rows.sort(key=lambda row: -row["gaussian_count"])
        return rows

    def fill_confusion_rows(matrix: np.ndarray) -> List[Dict[str, Any]]:
        rows = []
        for predicted_id, source_id in zip(*np.nonzero(matrix)):
            if int(predicted_id) == 0 or int(source_id) == 0:
                continue
            rows.append(
                {
                    "predicted_project_id": int(predicted_id),
                    "predicted_class": ontology.by_project_id[
                        int(predicted_id)
                    ].project_class,
                    "source_project_id": int(source_id),
                    "source_class": ontology.by_project_id[int(source_id)].project_class,
                    "pixels": int(matrix[predicted_id, source_id]),
                }
            )
        rows.sort(key=lambda row: -row["pixels"])
        return rows

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
        "full_evidence_candidate_reproduced": True,
        "disagreement_by_subtype": counters,
        "raw_winner_vs_heldout_confusion": confusion_rows(raw_confusion),
        "weighted_winner_vs_heldout_confusion": confusion_rows(weighted_confusion),
        "fill_confusion": fill_confusion_rows(fill_confusion),
        "fill_metrics": {
            "fill_pixels": fill_pixels,
            "fill_source_pixels": fill_source_pixels,
            "fill_agreed": fill_agreed,
            "fill_precision_of_source": (
                fill_agreed / fill_source_pixels if fill_source_pixels else 0.0
            ),
        },
        "candidate_recovered_count": int(np.count_nonzero(newly_resolved)),
        "per_camera": per_camera,
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
    (args.output_dir / "winner_disagreement_audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output_dir / "experiment_mode.txt").write_text(
        "mode=report_only_leave_one_camera_out_winner_disagreement\n"
        "accepted_gaussian_labels_written=0\n"
        "semantic_ply_written=0\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "scene": args.scene,
                "disagreement_by_subtype": counters,
                "fill_metrics": report["fill_metrics"],
                "candidate_recovered_count": report["candidate_recovered_count"],
            },
            indent=2,
        )
    )
    del counts
    temporary.cleanup()


if __name__ == "__main__":
    main()
