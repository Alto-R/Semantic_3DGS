"""S3: associate per-view SAM3 masks into global instances via 3D overlap.

The selected reconstruction cameras are not in capture order, so association
never relies on temporal tracking. Two masks belong to the same instance when
their lifted Gaussian supports overlap strongly. Masks only associate within
one concept; duplicate concepts from synonym prompts are handled later by the
hierarchy stage.
"""

from __future__ import annotations

from dataclasses import dataclass
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
            for i, left in enumerate(bucket):
                for right in bucket[i + 1 :]:
                    if masks[left].view == masks[right].view:
                        continue
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
        if len(set(views)) != len(views):
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
                "supporting_camera_count": len(set(views)),
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
