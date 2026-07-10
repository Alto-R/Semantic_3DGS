#!/usr/bin/env python3
"""Fuse semantic FlashSplat proposals into final per-Gaussian labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from add_labels_from_npy import write_ply_with_labels
from ply_utils import read_ply_header


DEFAULT_STUFF_CLASSES = {"ground", "road", "sidewalk", "sky", "vegetation", "terrain"}


@dataclass
class SemanticProposal:
    proposal_id: int
    class_name: str
    support_file: str
    gaussian_count: int
    score: float
    metadata: dict[str, Any]
    indices: np.ndarray


@dataclass
class SemanticGroup:
    group_id: int
    class_name: str
    indices: np.ndarray
    proposal_ids: list[int] = field(default_factory=list)
    source_frames: set[str] = field(default_factory=set)
    phrases: Counter[str] = field(default_factory=Counter)
    scores: list[float] = field(default_factory=list)
    is_stuff: bool = False
    assigned_count: int = 0
    assignment_reliability: float = 0.0
    assignment_peak_quality: float = 0.0

    @property
    def gaussian_count(self) -> int:
        return int(self.indices.shape[0])

    @property
    def proposal_count(self) -> int:
        return len(self.proposal_ids)

    @property
    def score(self) -> float:
        return max(self.scores) if self.scores else 0.0

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0


def normalize_class_name(value: Any) -> str:
    class_name = str(value or "").strip().lower().replace(" ", "_")
    return class_name or "unknown"


def point_cloud_path(model_path: Path, iteration: int) -> Path:
    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)
    return ply_path


def proposal_score(item: dict[str, Any]) -> float:
    for key in ["confidence", "grounding_score", "sam_score", "predicted_iou"]:
        if key in item:
            try:
                return float(item[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def load_indices(path: Path) -> np.ndarray:
    with np.load(path) as data:
        indices = data["indices"].astype(np.uint32)
    return np.unique(indices)


def load_stuff_classes(path: Path | None, override: str) -> set[str]:
    stuff = set(DEFAULT_STUFF_CLASSES)
    if path is not None and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        classes = data.get("classes", data) if isinstance(data, dict) else data
        for item in classes:
            if isinstance(item, dict) and str(item.get("type", "thing")).lower() == "stuff":
                stuff.add(normalize_class_name(item.get("class", item.get("name", ""))))
    if override:
        stuff = {normalize_class_name(item) for item in override.split(",") if item.strip()}
    return stuff


def load_assignment_priorities(path: Path | None, override: str) -> dict[str, int]:
    names: list[str] = []
    if override:
        names = [normalize_class_name(item) for item in override.split(",") if item.strip()]
    elif path is not None and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            names = [normalize_class_name(item) for item in data.get("assignment_priority", [])]
    return {class_name: len(names) - index for index, class_name in enumerate(names)}


def group_order_key(group: SemanticGroup, priorities: dict[str, int]) -> tuple[Any, ...]:
    return (
        0 if group.is_stuff else 1,
        priorities.get(group.class_name, 0),
        group.score,
        group.proposal_count,
        group.gaussian_count,
    )


def load_proposals(
    manifest_path: Path,
    support_dir: Path,
    min_gaussians: int,
    max_gaussians: int,
    require_class: bool,
) -> list[SemanticProposal]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proposals: list[SemanticProposal] = []
    for item in manifest["proposals"]:
        count = int(item["gaussian_count"])
        if count < min_gaussians:
            continue
        if max_gaussians > 0 and count > max_gaussians:
            continue
        class_name = normalize_class_name(item.get("class_name", item.get("class", "")))
        if require_class and class_name in {"", "unknown", "object_candidate"}:
            continue
        proposals.append(
            SemanticProposal(
                proposal_id=int(item["proposal_id"]),
                class_name=class_name,
                support_file=str(item["support_file"]),
                gaussian_count=count,
                score=proposal_score(item),
                metadata=item,
                indices=load_indices(support_dir / str(item["support_file"])),
            )
        )
    proposals.sort(key=lambda proposal: (proposal.score, proposal.gaussian_count), reverse=True)
    return proposals


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


def add_proposal_to_group(group: SemanticGroup, proposal: SemanticProposal) -> None:
    group.indices = np.union1d(group.indices, proposal.indices)
    group.proposal_ids.append(proposal.proposal_id)
    group.source_frames.add(str(proposal.metadata.get("frame_file", "")))
    phrase = str(proposal.metadata.get("phrase", proposal.class_name)).strip()
    if phrase:
        group.phrases[phrase] += 1
    group.scores.append(proposal.score)


def cluster_class_proposals(
    proposals: list[SemanticProposal],
    stuff_classes: set[str],
    merge_iou: float,
    containment: float,
) -> list[SemanticGroup]:
    groups: list[SemanticGroup] = []
    for proposal in proposals:
        best_group: SemanticGroup | None = None
        best_score = (0.0, 0.0, 0)
        for group in groups:
            if group.class_name != proposal.class_name:
                continue
            score = merge_score(proposal.indices, group.indices)
            if score[0] > best_score[0] or score[1] > best_score[1]:
                best_group = group
                best_score = score

        if best_group is not None and (best_score[0] >= merge_iou or best_score[1] >= containment):
            add_proposal_to_group(best_group, proposal)
        else:
            group = SemanticGroup(
                group_id=0,
                class_name=proposal.class_name,
                indices=proposal.indices,
                is_stuff=proposal.class_name in stuff_classes,
            )
            add_proposal_to_group(group, proposal)
            groups.append(group)
    return groups


def merge_stuff_groups(groups: list[SemanticGroup], stuff_classes: set[str]) -> list[SemanticGroup]:
    merged_by_class: dict[str, SemanticGroup] = {}
    kept: list[SemanticGroup] = []
    for group in groups:
        if group.class_name not in stuff_classes:
            kept.append(group)
            continue
        target = merged_by_class.get(group.class_name)
        if target is None:
            target = SemanticGroup(
                group_id=0,
                class_name=group.class_name,
                indices=group.indices,
                is_stuff=True,
            )
            merged_by_class[group.class_name] = target
        else:
            target.indices = np.union1d(target.indices, group.indices)
        target.proposal_ids.extend(group.proposal_ids)
        target.source_frames.update(group.source_frames)
        target.phrases.update(group.phrases)
        target.scores.extend(group.scores)
    kept.extend(merged_by_class.values())
    return kept


def filter_groups(
    groups: list[SemanticGroup],
    min_group_gaussians: int,
    min_group_proposals: int,
    max_groups: int,
    priorities: dict[str, int],
) -> list[SemanticGroup]:
    kept = [
        group
        for group in groups
        if group.gaussian_count >= min_group_gaussians
        and group.proposal_count >= min_group_proposals
    ]
    kept.sort(key=lambda group: group_order_key(group, priorities), reverse=True)
    if max_groups > 0:
        kept = kept[:max_groups]
    for label_id, group in enumerate(kept, start=1):
        group.group_id = label_id
    return kept


def final_label_name(group: SemanticGroup, class_counts: dict[str, int], class_ordinals: dict[str, int]) -> str:
    if group.is_stuff:
        return group.class_name
    class_ordinals[group.class_name] += 1
    return f"{group.class_name}_{class_ordinals[group.class_name]:02d}"


def frame_supports_for_group(
    group: SemanticGroup,
    proposals_by_id: dict[int, SemanticProposal],
) -> list[tuple[np.ndarray, float]]:
    proposals_by_frame: dict[str, list[SemanticProposal]] = defaultdict(list)
    for proposal_id in group.proposal_ids:
        proposal = proposals_by_id[proposal_id]
        frame_file = str(proposal.metadata.get("frame_file", proposal_id))
        proposals_by_frame[frame_file].append(proposal)

    frame_supports: list[tuple[np.ndarray, float]] = []
    for frame_proposals in proposals_by_frame.values():
        if len(frame_proposals) == 1:
            indices = frame_proposals[0].indices
        else:
            indices = np.unique(np.concatenate([proposal.indices for proposal in frame_proposals]))
        frame_supports.append((indices, max(proposal.score for proposal in frame_proposals)))
    return frame_supports


def assign_labels(
    vertex_count: int,
    groups: list[SemanticGroup],
    proposals: list[SemanticProposal],
    priorities: dict[str, int],
    reliability_views: float,
    min_quality: float,
) -> np.ndarray:
    labels = np.zeros((vertex_count,), dtype=np.int32)
    best_quality = np.zeros((vertex_count,), dtype=np.float32)
    proposals_by_id = {proposal.proposal_id: proposal for proposal in proposals}

    for group in groups:
        frame_supports = frame_supports_for_group(group, proposals_by_id)
        if not frame_supports:
            continue
        evidence = np.zeros((vertex_count,), dtype=np.float32)
        total_weight = sum(max(weight, 1e-6) for _, weight in frame_supports)
        for indices, weight in frame_supports:
            evidence[indices] += max(weight, 1e-6)

        view_count = len(frame_supports)
        reliability = 1.0 if reliability_views <= 0 else view_count / (view_count + reliability_views)
        evidence *= reliability / max(total_weight, 1e-6)
        group.assignment_reliability = float(reliability)
        group.assignment_peak_quality = float(evidence.max())

        # Priority resolves exact numerical ties only; multi-view evidence owns the decision.
        tie_break = priorities.get(group.class_name, 0) * 1e-7
        candidate_quality = evidence + np.float32(tie_break)
        selected = (evidence >= min_quality) & (candidate_quality > best_quality)
        labels[selected] = group.group_id
        best_quality[selected] = candidate_quality[selected]
    return labels


def prune_and_compact_groups(labels: np.ndarray, groups: list[SemanticGroup]) -> tuple[np.ndarray, list[SemanticGroup]]:
    histogram = {int(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))}
    active_groups = [group for group in groups if histogram.get(group.group_id, 0) > 0]
    remap = {group.group_id: new_id for new_id, group in enumerate(active_groups, start=1)}
    if len(active_groups) == len(groups) and all(remap[group.group_id] == group.group_id for group in active_groups):
        for group in active_groups:
            group.assigned_count = histogram.get(group.group_id, 0)
        return labels, active_groups

    compacted = np.zeros(labels.shape, dtype=np.int32)
    for group in active_groups:
        old_id = group.group_id
        new_id = remap[old_id]
        compacted[labels == old_id] = new_id
        group.group_id = new_id
        group.assigned_count = histogram.get(old_id, 0)
    return compacted, active_groups


def assigned_prune_threshold(
    group: SemanticGroup,
    min_assigned_gaussians: int,
    min_assigned_thing_gaussians: int,
    min_assigned_stuff_gaussians: int,
) -> int:
    threshold = max(0, min_assigned_gaussians)
    if group.is_stuff:
        threshold = max(threshold, max(0, min_assigned_stuff_gaussians))
    else:
        threshold = max(threshold, max(0, min_assigned_thing_gaussians))
    return threshold


def prune_assigned_groups(
    labels: np.ndarray,
    groups: list[SemanticGroup],
    min_assigned_gaussians: int,
    min_assigned_thing_gaussians: int,
    min_assigned_stuff_gaussians: int,
    min_label_score: float,
) -> tuple[np.ndarray, list[SemanticGroup], list[dict[str, Any]]]:
    kept: list[SemanticGroup] = []
    pruned: list[dict[str, Any]] = []
    prune_ids: list[int] = []
    for group in groups:
        threshold = assigned_prune_threshold(
            group,
            min_assigned_gaussians,
            min_assigned_thing_gaussians,
            min_assigned_stuff_gaussians,
        )
        reasons: list[str] = []
        if threshold > 0 and group.assigned_count < threshold:
            reasons.append(f"assigned_gaussians<{threshold}")
        if min_label_score > 0.0 and group.score < min_label_score:
            reasons.append(f"score<{min_label_score}")

        if not reasons:
            kept.append(group)
            continue

        prune_ids.append(group.group_id)
        pruned.append(
            {
                "id": group.group_id,
                "class": group.class_name,
                "is_stuff": group.is_stuff,
                "assigned_gaussian_count": group.assigned_count,
                "support_gaussian_count": group.gaussian_count,
                "proposal_count": group.proposal_count,
                "source_view_count": len(group.source_frames),
                "score": group.score,
                "mean_score": group.mean_score,
                "assigned_threshold": threshold,
                "reasons": reasons,
                "proposal_ids": group.proposal_ids,
                "phrases": dict(group.phrases),
            }
        )

    if prune_ids:
        for label_id in prune_ids:
            labels[labels == label_id] = 0
    return labels, kept, pruned


def build_label_map(scene: str, groups: list[SemanticGroup]) -> tuple[dict[str, Any], dict[int, str]]:
    class_counts = Counter(group.class_name for group in groups)
    class_ordinals: dict[str, int] = defaultdict(int)
    names_by_id: dict[int, str] = {}
    label_map: dict[str, Any] = {
        "scene": scene,
        "labels": [{"id": 0, "name": "unlabeled", "class": "unlabeled"}],
    }
    for group in sorted(groups, key=lambda item: item.group_id):
        name = final_label_name(group, class_counts, class_ordinals)
        names_by_id[group.group_id] = name
        label_map["labels"].append(
            {
                "id": group.group_id,
                "name": name,
                "class": group.class_name,
                "source": "groundingdino_sam_flashsplat",
                "proposal_count": group.proposal_count,
                "gaussian_count": group.assigned_count or group.gaussian_count,
                "support_gaussian_count": group.gaussian_count,
                "source_view_count": len(group.source_frames),
                "score": group.score,
                "mean_score": group.mean_score,
                "phrases": [phrase for phrase, _ in group.phrases.most_common(5)],
            }
        )
    return label_map, names_by_id


def group_summary(group: SemanticGroup, name: str) -> dict[str, Any]:
    return {
        "id": group.group_id,
        "name": name,
        "class": group.class_name,
        "is_stuff": group.is_stuff,
        "gaussian_count": group.assigned_count or group.gaussian_count,
        "support_gaussian_count": group.gaussian_count,
        "proposal_count": group.proposal_count,
        "source_view_count": len(group.source_frames),
        "score": group.score,
        "mean_score": group.mean_score,
        "assignment_reliability": group.assignment_reliability,
        "assignment_peak_quality": group.assignment_peak_quality,
        "proposal_ids": group.proposal_ids,
        "phrases": dict(group.phrases),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--proposal-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--class-config", type=Path)
    parser.add_argument("--stuff-classes", default="")
    parser.add_argument("--class-priority", default="")
    parser.add_argument("--assignment-reliability-views", default=2.0, type=float)
    parser.add_argument("--assignment-min-quality", default=0.03, type=float)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--min-proposal-gaussians", default=500, type=int)
    parser.add_argument("--max-proposal-gaussians", default=0, type=int)
    parser.add_argument("--merge-iou", default=0.35, type=float)
    parser.add_argument("--containment-threshold", default=0.70, type=float)
    parser.add_argument("--min-group-gaussians", default=1000, type=int)
    parser.add_argument("--min-group-proposals", default=1, type=int)
    parser.add_argument("--max-groups", default=128, type=int)
    parser.add_argument("--min-assigned-gaussians", default=0, type=int)
    parser.add_argument("--min-assigned-thing-gaussians", default=0, type=int)
    parser.add_argument("--min-assigned-stuff-gaussians", default=0, type=int)
    parser.add_argument("--min-label-score", default=0.0, type=float)
    parser.add_argument("--scene", default="")
    parser.add_argument("--semantic-ply-name", default="semantic_point_cloud.ply")
    parser.add_argument("--labels-path", type=Path)
    parser.add_argument("--label-map-path", type=Path)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--semantic-ply-path", type=Path)
    parser.add_argument("--require-class", action="store_true", default=True)
    parser.add_argument("--allow-unknown-class", dest="require_class", action="store_false")
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

    stuff_classes = load_stuff_classes(args.class_config, args.stuff_classes)
    assignment_priorities = load_assignment_priorities(args.class_config, args.class_priority)
    proposals = load_proposals(
        manifest_path,
        support_dir,
        args.min_proposal_gaussians,
        args.max_proposal_gaussians,
        args.require_class,
    )
    groups = cluster_class_proposals(proposals, stuff_classes, args.merge_iou, args.containment_threshold)
    groups = merge_stuff_groups(groups, stuff_classes)
    groups = filter_groups(
        groups,
        args.min_group_gaussians,
        args.min_group_proposals,
        args.max_groups,
        assignment_priorities,
    )
    labels = assign_labels(
        vertex.count,
        groups,
        proposals,
        assignment_priorities,
        args.assignment_reliability_views,
        args.assignment_min_quality,
    )
    labels, groups = prune_and_compact_groups(labels, groups)
    labels, groups, pruned_groups = prune_assigned_groups(
        labels,
        groups,
        args.min_assigned_gaussians,
        args.min_assigned_thing_gaussians,
        args.min_assigned_stuff_gaussians,
        args.min_label_score,
    )
    labels, groups = prune_and_compact_groups(labels, groups)
    if pruned_groups:
        labels = assign_labels(
            vertex.count,
            groups,
            proposals,
            assignment_priorities,
            args.assignment_reliability_views,
            args.assignment_min_quality,
        )
        labels, groups = prune_and_compact_groups(labels, groups)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.labels_path or args.output_dir / "gaussian_labels.npy"
    label_map_path = args.label_map_path or args.output_dir / "label_map.json"
    summary_path = args.summary_path or args.output_dir / "semantic_group_summary.json"
    semantic_ply = args.semantic_ply_path or args.output_dir / args.semantic_ply_name
    for path in [labels_path, label_map_path, summary_path, semantic_ply]:
        path.parent.mkdir(parents=True, exist_ok=True)
    for path in [labels_path, label_map_path, summary_path, semantic_ply]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    np.save(labels_path, labels)
    scene_name = args.scene or args.model_path.name
    label_map, names_by_id = build_label_map(scene_name, groups)
    label_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    histogram = {str(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))}
    summary = {
        "source": "groundingdino_sam_flashsplat",
        "stage": "semantic_fusion_and_pruning",
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "proposal_dir": str(args.proposal_dir),
        "parameters": {
            "min_proposal_gaussians": args.min_proposal_gaussians,
            "max_proposal_gaussians": args.max_proposal_gaussians,
            "merge_iou": args.merge_iou,
            "containment_threshold": args.containment_threshold,
            "min_group_gaussians": args.min_group_gaussians,
            "min_group_proposals": args.min_group_proposals,
            "max_groups": args.max_groups,
            "min_assigned_gaussians": args.min_assigned_gaussians,
            "min_assigned_thing_gaussians": args.min_assigned_thing_gaussians,
            "min_assigned_stuff_gaussians": args.min_assigned_stuff_gaussians,
            "min_label_score": args.min_label_score,
            "assignment_priority": assignment_priorities,
            "assignment_mode": "confidence_weighted_multiview",
            "assignment_reliability_views": args.assignment_reliability_views,
            "assignment_min_quality": args.assignment_min_quality,
        },
        "proposal_count": len(proposals),
        "group_count": len(groups),
        "pruning": {
            "pruned_group_count": len(pruned_groups),
            "pruned_groups": pruned_groups,
            "reassigned_after_pruning": bool(pruned_groups),
        },
        "stuff_classes": sorted(stuff_classes),
        "label_histogram": histogram,
        "groups": [group_summary(group, names_by_id[group.group_id]) for group in groups],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_ply_with_labels(ply_path, semantic_ply, labels)

    print(f"wrote {semantic_ply}")
    print(json.dumps(histogram, sort_keys=True))


if __name__ == "__main__":
    main()
