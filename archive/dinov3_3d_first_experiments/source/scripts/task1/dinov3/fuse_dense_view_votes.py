#!/usr/bin/env python3
"""Fuse complete dense DINOv3 FlashSplat votes at every Gaussian."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.common.ply_utils import read_ply_header
from scripts.task1.dinov2.dinov2_ontology import load_ontology
from scripts.task1.qa.add_labels_from_npy import write_ply_with_labels


REASON_ACCEPTED = 0
REASON_UNOBSERVED = 1
REASON_INSUFFICIENT_VIEWS = 2
REASON_EXACT_TIE = 3

REASON_NAMES = {
    REASON_ACCEPTED: "accepted",
    REASON_UNOBSERVED: "unobserved",
    REASON_INSUFFICIENT_VIEWS: "insufficient_views",
    REASON_EXACT_TIE: "exact_tie",
}


def add_view_votes(
    vote_matrix: np.ndarray,
    view_counts: np.ndarray,
    indices: np.ndarray,
    class_ids: np.ndarray,
    weights: np.ndarray,
) -> None:
    """Add one normalized sparse camera distribution to global accumulators."""

    idx = np.asarray(indices)
    classes = np.asarray(class_ids)
    values = np.asarray(weights, dtype=np.float32)
    if not (idx.shape == classes.shape == values.shape) or idx.ndim != 1:
        raise ValueError("view vote arrays must be matching one-dimensional arrays")
    if idx.size == 0:
        return
    if np.any(idx < 0) or int(idx.max()) >= vote_matrix.shape[1]:
        raise ValueError("view vote references a Gaussian outside the vote matrix")
    if np.any(classes <= 0) or int(classes.max()) >= vote_matrix.shape[0]:
        raise ValueError("view vote references an invalid project class")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("view vote weights must be finite and positive")

    totals = np.zeros((vote_matrix.shape[1],), dtype=np.float32)
    np.add.at(totals, idx.astype(np.int64, copy=False), values)
    observed = totals > 0.0
    if not np.allclose(totals[observed], 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("each camera must contribute unit mass per visible Gaussian")
    view_counts[observed] += np.uint16(1)
    for project_id in np.unique(classes):
        selected = classes == project_id
        np.add.at(
            vote_matrix[int(project_id)],
            idx[selected].astype(np.int64, copy=False),
            values[selected],
        )


def decide_dense_labels(
    semantic_votes: np.ndarray,
    view_counts: np.ndarray,
    *,
    min_views: int,
    tie_epsilon: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose the unique soft class winner for every sufficiently seen Gaussian."""

    votes = np.asarray(semantic_votes, dtype=np.float32)
    views = np.asarray(view_counts)
    if votes.ndim != 2 or votes.shape[0] < 2:
        raise ValueError("semantic_votes must have shape classes x gaussians")
    if views.shape != (votes.shape[1],):
        raise ValueError("view_counts must have one value per Gaussian")
    if min_views < 1:
        raise ValueError("min_views must be positive")
    if tie_epsilon < 0.0:
        raise ValueError("tie_epsilon must be non-negative")

    winner_offsets = np.argmax(votes, axis=0)
    columns = np.arange(votes.shape[1], dtype=np.int64)
    winner_scores = votes[winner_offsets, columns]
    if votes.shape[0] == 1:
        runner_up_scores = np.zeros_like(winner_scores)
    else:
        runner_up_scores = np.partition(votes, -2, axis=0)[-2]
    totals = votes.sum(axis=0, dtype=np.float32)
    winner_share = np.divide(
        winner_scores,
        totals,
        out=np.zeros_like(winner_scores, dtype=np.float32),
        where=totals > 0.0,
    )
    winner_margin = np.divide(
        winner_scores - runner_up_scores,
        totals,
        out=np.zeros_like(winner_scores, dtype=np.float32),
        where=totals > 0.0,
    )

    reasons = np.full(views.shape, REASON_UNOBSERVED, dtype=np.uint8)
    observed = views > 0
    insufficient = observed & (views < min_views)
    reasons[insufficient] = REASON_INSUFFICIENT_VIEWS
    enough = views >= min_views
    unique = (winner_scores - runner_up_scores) > np.float32(tie_epsilon)
    reasons[enough & ~unique] = REASON_EXACT_TIE
    accepted = enough & unique
    reasons[accepted] = REASON_ACCEPTED

    winners = np.zeros(views.shape, dtype=np.uint16)
    winners[accepted] = winner_offsets[accepted].astype(np.uint16) + np.uint16(1)
    return winners, reasons, winner_share, winner_margin, runner_up_scores


def load_vote_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("source") != "dinov3_dense_pixel_flashsplat_votes":
        raise ValueError("vote manifest is not dense DINOv3 evidence")
    if manifest.get("contract") != (
        "complete_dense_argmax_pixels_normalized_per_camera_v1"
    ):
        raise ValueError("unsupported dense DINOv3 vote contract")
    for field in (
        "query_region_filtering_used",
        "confidence_threshold_used",
        "v5_used",
        "dinov2_used",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"dense DINOv3 manifest violates {field}=false")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vote-manifest", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene")
    parser.add_argument("--min-views", default=1, type=int)
    parser.add_argument("--tie-epsilon", default=0.0, type=float)
    parser.add_argument("--chunk-size", default=100_000, type=int)
    parser.add_argument("--no-semantic-ply", action="store_true")
    args = parser.parse_args()

    for path in (args.vote_manifest, args.ontology, args.source_ply):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.min_views < 1:
        raise ValueError("min_views must be positive")
    if args.tie_epsilon < 0.0:
        raise ValueError("tie_epsilon must be non-negative")
    if args.chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    manifest = load_vote_manifest(args.vote_manifest)
    ontology = load_ontology(args.ontology)
    vertex_count = int(manifest["gaussian_count"])
    manifest_ply = Path(str(manifest["ply_path"]))
    if manifest_ply.resolve() != args.source_ply.resolve():
        raise ValueError("source PLY differs from the dense vote provenance")
    header = read_ply_header(args.source_ply)
    if not header.elements or header.elements[0].name != "vertex":
        raise ValueError("source PLY does not begin with a vertex element")
    if int(header.elements[0].count) != vertex_count:
        raise ValueError("source PLY vertex count differs from the vote manifest")

    args.output_dir.mkdir(parents=True)
    view_counts = np.zeros((vertex_count,), dtype=np.uint16)
    winners = np.zeros((vertex_count,), dtype=np.uint16)
    reasons = np.zeros((vertex_count,), dtype=np.uint8)
    winner_share = np.zeros((vertex_count,), dtype=np.float16)
    winner_margin = np.zeros((vertex_count,), dtype=np.float16)
    runner_up_score = np.zeros((vertex_count,), dtype=np.float16)

    vote_matrix: np.memmap | None = None
    with tempfile.TemporaryDirectory(prefix="dense_vote_cache_", dir=args.output_dir) as cache:
        vote_matrix = np.memmap(
            Path(cache) / "class_votes.float32",
            mode="w+",
            dtype=np.float32,
            shape=(ontology.class_count + 1, vertex_count),
        )
        vote_matrix[:] = 0.0
        for frame in manifest["frames"]:
            vote_path = args.vote_manifest.parent / str(frame["vote_file"])
            with np.load(vote_path) as data:
                add_view_votes(
                    vote_matrix,
                    view_counts,
                    data["indices"],
                    data["class_ids"],
                    data["weights"],
                )
            print(f"accumulated {frame['file']}")
        vote_matrix.flush()

        for start in range(0, vertex_count, args.chunk_size):
            end = min(start + args.chunk_size, vertex_count)
            (
                winners[start:end],
                reasons[start:end],
                share,
                margin,
                runner,
            ) = decide_dense_labels(
                np.asarray(vote_matrix[1:, start:end]),
                view_counts[start:end],
                min_views=args.min_views,
                tie_epsilon=args.tie_epsilon,
            )
            winner_share[start:end] = share.astype(np.float16)
            winner_margin[start:end] = margin.astype(np.float16)
            runner_up_score[start:end] = runner.astype(np.float16)
        vote_matrix.flush()
        del vote_matrix
        vote_matrix = None

    accepted = winners > 0
    labels = winners.astype(np.int32)
    outputs = {
        "gaussian_labels.npy": labels,
        "gaussian_project_class_ids.npy": labels,
        "supporting_views.npy": view_counts,
        "winner_share.npy": winner_share,
        "winner_margin.npy": winner_margin,
        "runner_up_score.npy": runner_up_score,
        "abstain_reason_codes.npy": reasons,
    }
    for filename, array in outputs.items():
        np.save(args.output_dir / filename, array)

    present_ids = np.unique(winners[accepted])
    labels_json: list[dict[str, Any]] = [
        {"id": 0, "name": "unlabeled", "class": "unlabeled", "type": "unlabeled"}
    ]
    for project_id in present_ids:
        item = ontology.by_project_id[int(project_id)]
        labels_json.append(
            {
                "id": int(project_id),
                "name": item.project_class,
                "class": item.project_class,
                "project_id": item.project_id,
                "ade_id": item.ade_id,
                "type": item.kind,
            }
        )
    scene = args.scene or args.source_ply.parents[2].name
    label_map = {
        "scene": scene,
        "source": "dinov3_dense_pixel_flashsplat_soft_class_mass",
        "ontology": str(args.ontology),
        "vote_manifest": str(args.vote_manifest),
        "labels": labels_json,
    }
    (args.output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2), encoding="utf-8"
    )

    semantic_ply = args.output_dir / "semantic_point_cloud.ply"
    if not args.no_semantic_ply:
        partial = semantic_ply.with_suffix(".ply.partial")
        write_ply_with_labels(args.source_ply, partial, labels)
        partial.replace(semantic_ply)

    reason_counts = {
        REASON_NAMES[int(code)]: int(count)
        for code, count in zip(*np.unique(reasons, return_counts=True))
    }
    class_counts = {
        ontology.by_project_id[int(project_id)].project_class: int(count)
        for project_id, count in zip(*np.unique(winners[accepted], return_counts=True))
    }
    support_histogram = {
        str(int(count)): int(frequency)
        for count, frequency in zip(*np.unique(view_counts, return_counts=True))
    }
    summary = {
        "source": "dinov3_dense_pixel_flashsplat_soft_class_mass",
        "contract": "complete_dense_pixels_unique_soft_argmax_v1",
        "scene": scene,
        "vote_manifest": str(args.vote_manifest),
        "source_ply": str(args.source_ply),
        "ontology": str(args.ontology),
        "v5_used": False,
        "dinov2_used": False,
        "query_region_filtering_used": False,
        "confidence_threshold_used": False,
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "camera_vote_policy": "complete_dense_pixels_normalized_once_per_camera",
        "semantic_vote_policy": "unique_soft_class_mass_argmax",
        "min_views": args.min_views,
        "tie_epsilon": args.tie_epsilon,
        "camera_count": int(manifest["camera_count"]),
        "vertex_count": vertex_count,
        "gaussians_with_any_view": int(np.count_nonzero(view_counts)),
        "gaussians_with_multiple_views": int(np.count_nonzero(view_counts >= 2)),
        "assigned_gaussian_count": int(np.count_nonzero(accepted)),
        "assigned_ratio": float(np.mean(accepted)),
        "unlabeled_gaussian_count": int(np.count_nonzero(~accepted)),
        "supporting_view_histogram": support_histogram,
        "abstain_reason_counts": reason_counts,
        "class_assigned_gaussian_counts": dict(sorted(class_counts.items())),
        "semantic_labels_written": True,
        "label_map_written": True,
        "semantic_ply_written": not args.no_semantic_ply,
    }
    (args.output_dir / "dense_vote_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
