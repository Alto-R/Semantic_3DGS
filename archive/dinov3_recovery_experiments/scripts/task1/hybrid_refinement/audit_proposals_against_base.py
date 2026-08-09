#!/usr/bin/env python3
"""Report how hybrid 3D proposals overlap an immutable base label set.

This module writes JSON only. It never mutates the base, writes candidate
Gaussian labels, or exports a semantic PLY.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.common.semantic_palette import normalize_class_name


def load_label_classes(path: Path) -> dict[int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    classes = {
        int(item["id"]): normalize_class_name(
            item.get(
                "class",
                "unlabeled" if int(item["id"]) == 0 else item.get("name", "unknown"),
            )
        )
        for item in data.get("labels", [])
    }
    classes.setdefault(0, "unlabeled")
    return classes


def load_support_indices(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        indices = data["indices"].astype(np.int64)
    if indices.ndim != 1:
        raise ValueError(f"{path} indices must be one-dimensional")
    return indices


def summarize_class_overlap(
    target_class: str,
    candidate_indices: np.ndarray,
    base_labels: np.ndarray,
    base_classes: dict[int, str],
) -> dict[str, Any]:
    target_class = normalize_class_name(target_class)
    candidate_indices = np.unique(np.asarray(candidate_indices, dtype=np.int64))
    if candidate_indices.size and (
        int(candidate_indices.min()) < 0
        or int(candidate_indices.max()) >= int(base_labels.shape[0])
    ):
        raise IndexError("Candidate Gaussian index is outside the base label array")
    selected = base_labels[candidate_indices] if candidate_indices.size else np.zeros(0, dtype=np.int64)
    label_ids, counts = np.unique(selected, return_counts=True)
    transitions: list[dict[str, Any]] = []
    unlabeled = 0
    same_class = 0
    other_class = 0
    for label_id, count in zip(label_ids, counts):
        label_id = int(label_id)
        count = int(count)
        base_class = base_classes.get(label_id, "unknown")
        if label_id == 0 or base_class == "unlabeled":
            category = "base_unlabeled"
            unlabeled += count
        elif base_class == target_class:
            category = "base_same_class"
            same_class += count
        else:
            category = "base_other_class"
            other_class += count
        transitions.append(
            {
                "base_label_id": label_id,
                "base_class": base_class,
                "candidate_class": target_class,
                "gaussians": count,
                "category": category,
            }
        )
    total = int(candidate_indices.size)
    return {
        "class": target_class,
        "candidate_gaussians": total,
        "base_unlabeled_gaussians": unlabeled,
        "base_same_class_gaussians": same_class,
        "base_other_class_gaussians": other_class,
        "base_unlabeled_ratio": unlabeled / float(max(total, 1)),
        "base_same_class_ratio": same_class / float(max(total, 1)),
        "base_other_class_ratio": other_class / float(max(total, 1)),
        "transitions": transitions,
    }


def summarize_cross_class_pair(
    left_class: str,
    left_indices: np.ndarray,
    right_class: str,
    right_indices: np.ndarray,
    conflict_containment_threshold: float,
) -> dict[str, Any]:
    left_indices = np.unique(np.asarray(left_indices, dtype=np.int64))
    right_indices = np.unique(np.asarray(right_indices, dtype=np.int64))
    intersection = int(
        np.intersect1d(left_indices, right_indices, assume_unique=True).shape[0]
    )
    union = int(left_indices.size + right_indices.size - intersection)
    containment = intersection / float(
        max(min(int(left_indices.size), int(right_indices.size)), 1)
    )
    return {
        "left_class": normalize_class_name(left_class),
        "right_class": normalize_class_name(right_class),
        "left_gaussians": int(left_indices.size),
        "right_gaussians": int(right_indices.size),
        "intersection_gaussians": intersection,
        "iou": intersection / float(max(union, 1)),
        "containment": containment,
        "identity_conflict": containment >= conflict_containment_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-dir", required=True, type=Path)
    parser.add_argument("--base-labels", required=True, type=Path)
    parser.add_argument("--base-label-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--conflict-containment-threshold",
        default=0.10,
        type=float,
    )
    parser.add_argument("--fail-on-conflict", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.conflict_containment_threshold <= 1.0:
        raise ValueError("conflict-containment-threshold must be between 0 and 1")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
    proposal_manifest_path = args.proposal_dir / "proposal_manifest.json"
    support_dir = args.proposal_dir / "proposal_supports"
    manifest = json.loads(proposal_manifest_path.read_text(encoding="utf-8"))
    base_labels = np.load(args.base_labels, allow_pickle=False).astype(np.int64)
    if base_labels.ndim != 1:
        raise ValueError("Base labels must be one-dimensional")
    base_classes = load_label_classes(args.base_label_map)

    indices_by_class: dict[str, list[np.ndarray]] = defaultdict(list)
    proposal_counts: dict[str, int] = defaultdict(int)
    for proposal in manifest.get("proposals", []):
        class_name = normalize_class_name(
            proposal.get("class_name", proposal.get("class", "unknown"))
        )
        support_file = support_dir / str(proposal["support_file"])
        indices_by_class[class_name].append(load_support_indices(support_file))
        proposal_counts[class_name] += 1

    union_indices_by_class = {
        class_name: np.unique(np.concatenate(class_indices))
        for class_name, class_indices in indices_by_class.items()
    }
    class_reports: list[dict[str, Any]] = []
    for class_name in sorted(union_indices_by_class):
        candidate_indices = union_indices_by_class[class_name]
        report = summarize_class_overlap(
            class_name,
            candidate_indices,
            base_labels,
            base_classes,
        )
        report["proposal_count"] = proposal_counts[class_name]
        class_reports.append(report)

    cross_class_pairs: list[dict[str, Any]] = []
    class_names = sorted(union_indices_by_class)
    for left_position, left_class in enumerate(class_names):
        for right_class in class_names[left_position + 1 :]:
            cross_class_pairs.append(
                summarize_cross_class_pair(
                    left_class,
                    union_indices_by_class[left_class],
                    right_class,
                    union_indices_by_class[right_class],
                    args.conflict_containment_threshold,
                )
            )
    conflicts = [pair for pair in cross_class_pairs if pair["identity_conflict"]]

    output = {
        "source": "hybrid_proposal_base_overlap_audit",
        "proposal_manifest": str(proposal_manifest_path),
        "base_labels": str(args.base_labels),
        "base_label_map": str(args.base_label_map),
        "proposal_count": int(sum(proposal_counts.values())),
        "class_count": len(class_reports),
        "classes": class_reports,
        "cross_class_identity": {
            "conflict_containment_threshold": args.conflict_containment_threshold,
            "pair_count": len(cross_class_pairs),
            "conflict_count": len(conflicts),
            "pairs": cross_class_pairs,
        },
        "base_modified": False,
        "semantic_labels_written": False,
        "semantic_ply_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    if args.fail_on_conflict and conflicts:
        raise RuntimeError(
            f"{len(conflicts)} cross-class 3D identity conflict(s) exceeded "
            f"containment {args.conflict_containment_threshold}"
        )


if __name__ == "__main__":
    main()
