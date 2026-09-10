"""S3: associate per-view SAM3 masks into global instances via 3D overlap.

The selected reconstruction cameras are not in capture order, so association
never relies on temporal tracking. Two masks belong to the same instance when
their lifted Gaussian supports overlap strongly. Masks only associate within
one concept; duplicate concepts from synonym prompts are handled later by the
hierarchy stage.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


ASSOCIATION_SOURCE = "sam3_support_overlap_association"
ASSOCIATION_CONTRACT = "per_concept_weighted_jaccard_union_v1"
MAX_INSTANCE_ID = 65535


@dataclass(frozen=True)
class MaskSupport:
    """One per-view mask with its lifted sparse Gaussian support."""

    view: str
    mask_index: int
    concept: str
    score: float
    indices: np.ndarray
    weights: np.ndarray


class _UnionFind:
    def __init__(self, size: int):
        self._parent = list(range(size))

    def find(self, node: int) -> int:
        while self._parent[node] != node:
            self._parent[node] = self._parent[self._parent[node]]
            node = self._parent[node]
        return node

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_b] = root_a


def weighted_jaccard(a: MaskSupport, b: MaskSupport) -> float:
    """sum(min) / sum(max) over the union of both sparse supports."""

    common_a, in_a, in_b = np.intersect1d(
        a.indices, b.indices, assume_unique=True, return_indices=True
    )
    if common_a.size == 0:
        return 0.0
    common_min = float(np.minimum(a.weights[in_a], b.weights[in_b]).sum())
    total = float(a.weights.sum()) + float(b.weights.sum()) - common_min
    if total <= 0.0:
        return 0.0
    return common_min / total


def associate_masks(
    masks: list[MaskSupport], threshold: float
) -> list[list[int]]:
    """Union masks of one concept across different views by support overlap.

    Returns groups of positions into ``masks`` (singletons included). Pairs
    from the same view never union directly: two masks in one image are two
    distinct instances by SAM3's own evidence.
    """

    union = _UnionFind(len(masks))
    by_concept: dict[str, list[int]] = {}
    for position, mask in enumerate(masks):
        by_concept.setdefault(mask.concept, []).append(position)

    for positions in by_concept.values():
        inverted: dict[int, list[int]] = {}
        for position in positions:
            for gaussian in masks[position].indices.tolist():
                inverted.setdefault(gaussian, []).append(position)
        candidate_pairs: set[tuple[int, int]] = set()
        for bucket in inverted.values():
            for left, right in combinations(bucket, 2):
                if masks[left].view != masks[right].view:
                    candidate_pairs.add((left, right))
        for left, right in sorted(candidate_pairs):
            if weighted_jaccard(masks[left], masks[right]) >= threshold:
                union.union(left, right)

    groups: dict[int, list[int]] = {}
    for position in range(len(masks)):
        groups.setdefault(union.find(position), []).append(position)
    return [sorted(group) for group in groups.values()]


def build_instance_registry(
    masks: list[MaskSupport],
    groups: list[list[int]],
    threshold: float,
) -> dict[str, Any]:
    """Assign 1-based global instance ids to associated mask groups."""

    def group_key(group: list[int]) -> tuple[str, str, int]:
        first = min(group, key=lambda p: (masks[p].view, masks[p].mask_index))
        return (masks[first].concept, masks[first].view, masks[first].mask_index)

    ordered = sorted(groups, key=group_key)
    if len(ordered) > MAX_INSTANCE_ID:
        raise ValueError(
            f"{len(ordered)} instances exceed the uint16 id space"
        )

    instances: list[dict[str, Any]] = []
    conflict_groups = 0
    for instance_id, group in enumerate(ordered, start=1):
        views = [masks[position].view for position in group]
        distinct_views = set(views)
        if len(distinct_views) != len(views):
            conflict_groups += 1
        members = sorted(
            (
                {
                    "view": masks[position].view,
                    "mask_index": masks[position].mask_index,
                    "score": masks[position].score,
                }
                for position in group
            ),
            key=lambda member: (member["view"], member["mask_index"]),
        )
        instances.append(
            {
                "instance_id": instance_id,
                "concept": masks[group[0]].concept,
                "members": members,
                "supporting_camera_count": len(distinct_views),
            }
        )

    return {
        "source": ASSOCIATION_SOURCE,
        "contract": ASSOCIATION_CONTRACT,
        "threshold": float(threshold),
        "mask_count": len(masks),
        "instance_count": len(instances),
        "same_view_conflict_groups": conflict_groups,
        "instances": instances,
    }


def validate_instance_registry(registry: dict[str, Any]) -> None:
    if registry.get("source") != ASSOCIATION_SOURCE:
        raise ValueError(f"unsupported registry source: {registry.get('source')!r}")
    if registry.get("contract") != ASSOCIATION_CONTRACT:
        raise ValueError(
            f"unsupported registry contract: {registry.get('contract')!r}"
        )
    if not isinstance(registry.get("instances"), list):
        raise ValueError("instance registry must list its instances")


def load_mask_supports(
    masks_manifest: dict[str, Any],
    votes_manifest: dict[str, Any],
    votes_dir: Path,
) -> list[MaskSupport]:
    """Join S1 mask metadata with S2 sparse supports, one MaskSupport per mask."""

    from scripts.task1.sam3.segment_views_core import stem_index

    meta: dict[str, dict[int, tuple[str, float]]] = {}
    for stem, frame in stem_index(
        masks_manifest["frames"], "masks manifest"
    ).items():
        meta[stem] = {
            int(mask["mask_index"]): (str(mask["concept"]), float(mask["score"]))
            for mask in frame["masks"]
        }

    supports: list[MaskSupport] = []
    for frame in votes_manifest["frames"]:
        stem = Path(str(frame["file"])).stem
        frame_meta = meta.get(stem)
        if frame_meta is None:
            raise ValueError(f"masks manifest does not know view {stem}")
        with np.load(votes_dir / str(frame["vote_file"]), allow_pickle=False) as data:
            indices = data["indices"]
            mask_ids = data["mask_ids"]
            weights = data["weights"]
        for raw_index in np.unique(mask_ids):
            mask_index = int(raw_index)
            if mask_index not in frame_meta:
                raise ValueError(
                    f"view {stem} vote references unknown mask {mask_index}"
                )
            concept, score = frame_meta[mask_index]
            rows = mask_ids == raw_index
            supports.append(
                MaskSupport(
                    view=stem,
                    mask_index=mask_index,
                    concept=concept,
                    score=score,
                    indices=indices[rows].astype(np.uint32),
                    weights=weights[rows].astype(np.float32),
                )
            )
    return supports


def main(argv: list[str] | None = None) -> None:
    from scripts.task1.sam3.lift_mask_view_votes import validate_votes_manifest
    from scripts.task1.sam3.segment_views_core import validate_masks_manifest

    parser = argparse.ArgumentParser()
    parser.add_argument("--masks-manifest", required=True, type=Path)
    parser.add_argument("--votes-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", default=0.3, type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    masks_manifest = json.loads(args.masks_manifest.read_text(encoding="utf-8"))
    validate_masks_manifest(masks_manifest)
    votes_manifest = json.loads(args.votes_manifest.read_text(encoding="utf-8"))
    validate_votes_manifest(votes_manifest)

    supports = load_mask_supports(
        masks_manifest, votes_manifest, args.votes_manifest.parent
    )
    groups = associate_masks(supports, args.threshold)
    registry = build_instance_registry(supports, groups, args.threshold)
    registry["masks_manifest"] = str(args.masks_manifest)
    registry["votes_manifest"] = str(args.votes_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(
        f"associated {registry['mask_count']} masks into "
        f"{registry['instance_count']} instances"
    )


if __name__ == "__main__":
    main()
