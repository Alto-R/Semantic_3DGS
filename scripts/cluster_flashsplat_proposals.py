#!/usr/bin/env python3
"""Cluster lifted FlashSplat mask proposals into automatic 3D object groups."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from add_labels_from_npy import write_ply_with_labels
from flashsplat_cameras import point_cloud_path
from ply_utils import read_ply_header


@dataclass
class Proposal:
    proposal_id: int
    support_file: str
    gaussian_count: int
    metadata: Dict[str, Any]
    indices: Optional[np.ndarray] = None


@dataclass
class Group:
    group_id: int
    indices: np.ndarray
    proposal_ids: List[int] = field(default_factory=list)
    source_frames: set[str] = field(default_factory=set)

    @property
    def gaussian_count(self) -> int:
        return int(self.indices.shape[0])

    @property
    def proposal_count(self) -> int:
        return len(self.proposal_ids)


def load_indices(path: Path) -> np.ndarray:
    with np.load(path) as data:
        indices = data["indices"].astype(np.uint32)
    return np.unique(indices)


def intersection_count(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.intersect1d(left, right, assume_unique=True).shape[0])


def merge_score(proposal_indices: np.ndarray, group_indices: np.ndarray) -> tuple[float, float, int]:
    inter = intersection_count(proposal_indices, group_indices)
    if inter == 0:
        return 0.0, 0.0, 0
    union = proposal_indices.shape[0] + group_indices.shape[0] - inter
    iou = inter / float(max(union, 1))
    containment = inter / float(max(min(proposal_indices.shape[0], group_indices.shape[0]), 1))
    return iou, containment, inter


def load_proposals(manifest_path: Path, support_dir: Path, min_gaussians: int, max_gaussians: int) -> List[Proposal]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proposals: List[Proposal] = []
    for item in manifest["proposals"]:
        count = int(item["gaussian_count"])
        if count < min_gaussians:
            continue
        if max_gaussians > 0 and count > max_gaussians:
            continue
        proposals.append(
            Proposal(
                proposal_id=int(item["proposal_id"]),
                support_file=str(item["support_file"]),
                gaussian_count=count,
                metadata=item,
                indices=load_indices(support_dir / str(item["support_file"])),
            )
        )
    proposals.sort(
        key=lambda proposal: (
            float(proposal.metadata.get("predicted_iou", 0.0))
            * float(proposal.metadata.get("stability_score", 0.0)),
            proposal.gaussian_count,
        ),
        reverse=True,
    )
    return proposals


def cluster_proposals(
    proposals: List[Proposal],
    merge_iou: float,
    containment: float,
) -> List[Group]:
    groups: List[Group] = []
    for proposal in proposals:
        if proposal.indices is None:
            continue
        best_group = None
        best_score = (0.0, 0.0, 0)
        for group in groups:
            score = merge_score(proposal.indices, group.indices)
            if score[0] > best_score[0] or score[1] > best_score[1]:
                best_group = group
                best_score = score

        if best_group is not None and (best_score[0] >= merge_iou or best_score[1] >= containment):
            best_group.indices = np.union1d(best_group.indices, proposal.indices)
            best_group.proposal_ids.append(proposal.proposal_id)
            best_group.source_frames.add(str(proposal.metadata.get("frame_file", "")))
        else:
            group = Group(group_id=len(groups) + 1, indices=proposal.indices)
            group.proposal_ids.append(proposal.proposal_id)
            group.source_frames.add(str(proposal.metadata.get("frame_file", "")))
            groups.append(group)
    return groups


def filter_groups(
    groups: List[Group],
    min_group_gaussians: int,
    min_group_proposals: int,
    max_groups: int,
) -> List[Group]:
    kept = [
        group
        for group in groups
        if group.gaussian_count >= min_group_gaussians
        and group.proposal_count >= min_group_proposals
    ]
    kept.sort(key=lambda group: (group.proposal_count, group.gaussian_count), reverse=True)
    if max_groups > 0:
        kept = kept[:max_groups]
    for label_id, group in enumerate(kept, start=1):
        group.group_id = label_id
    return kept


def assign_labels(vertex_count: int, groups: List[Group]) -> np.ndarray:
    labels = np.zeros((vertex_count,), dtype=np.int32)
    for group in groups:
        unassigned = labels[group.indices] == 0
        labels[group.indices[unassigned]] = group.group_id
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--proposal-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--min-proposal-gaussians", default=500, type=int)
    parser.add_argument("--max-proposal-gaussians", default=0, type=int)
    parser.add_argument("--merge-iou", default=0.35, type=float)
    parser.add_argument("--containment-threshold", default=0.70, type=float)
    parser.add_argument("--min-group-gaussians", default=1000, type=int)
    parser.add_argument("--min-group-proposals", default=2, type=int)
    parser.add_argument("--max-groups", default=128, type=int)
    parser.add_argument("--scene", default="")
    parser.add_argument("--semantic-ply-name", default="semantic_point_cloud_auto.ply")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = args.proposal_dir / "proposal_manifest.json"
    support_dir = args.proposal_dir / "proposal_supports"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not support_dir.exists():
        raise FileNotFoundError(support_dir)

    ply_path = point_cloud_path(args.model_path, args.iteration)
    vertex = read_ply_header(ply_path).element("vertex")
    if vertex is None:
        raise ValueError(f"{ply_path} has no vertex element")

    proposals = load_proposals(
        manifest_path,
        support_dir,
        args.min_proposal_gaussians,
        args.max_proposal_gaussians,
    )
    groups = cluster_proposals(proposals, args.merge_iou, args.containment_threshold)
    groups = filter_groups(
        groups,
        args.min_group_gaussians,
        args.min_group_proposals,
        args.max_groups,
    )
    labels = assign_labels(vertex.count, groups)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_dir / "gaussian_labels_auto.npy"
    np.save(labels_path, labels)

    scene_name = args.scene or args.model_path.name
    label_map = {
        "scene": scene_name,
        "labels": [{"id": 0, "name": "unlabeled", "class": "unlabeled"}],
    }
    for group in groups:
        label_map["labels"].append(
            {
                "id": group.group_id,
                "name": f"object_group_{group.group_id:03d}",
                "class": "object_candidate",
                "proposal_count": group.proposal_count,
                "gaussian_count": group.gaussian_count,
                "source_view_count": len(group.source_frames),
            }
        )
    (args.output_dir / "label_map_auto.json").write_text(
        json.dumps(label_map, indent=2),
        encoding="utf-8",
    )

    summary = {
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "proposal_dir": str(args.proposal_dir),
        "proposal_count": len(proposals),
        "group_count": len(groups),
        "label_histogram": {
            str(label): int(count)
            for label, count in zip(*np.unique(labels, return_counts=True))
        },
        "groups": [
            {
                "id": group.group_id,
                "gaussian_count": group.gaussian_count,
                "proposal_count": group.proposal_count,
                "source_view_count": len(group.source_frames),
                "proposal_ids": group.proposal_ids,
            }
            for group in groups
        ],
    }
    (args.output_dir / "auto_group_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    semantic_ply = args.output_dir / args.semantic_ply_name
    if semantic_ply.exists() and not args.overwrite:
        raise FileExistsError(f"{semantic_ply} exists; pass --overwrite to replace it")
    write_ply_with_labels(ply_path, semantic_ply, labels)
    print(f"wrote {semantic_ply}")
    print(json.dumps(summary["label_histogram"], sort_keys=True))


if __name__ == "__main__":
    main()
