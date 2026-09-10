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
CONSTRAINED_ASSOCIATION_CONTRACT = "per_concept_constrained_weighted_jaccard_union_v2"
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
    masks: list[MaskSupport], threshold: float,
    cannot_links: dict[int, set[int]] | None = None,
) -> list[list[int]]:
    """Union masks of one concept across different views by support overlap.

    Returns groups of positions into ``masks`` (singletons included). Pairs
    from the same view never union directly: two masks in one image are two
    distinct instances by SAM3's own evidence.
    """

    union = _UnionFind(len(masks))
    members = {i: {i} for i in range(len(masks))} if cannot_links else None
    forbidden = {i: set(cannot_links.get(i, ())) for i in range(len(masks))} if cannot_links else None
    by_concept: dict[str, list[int]] = {}
    for position, mask in enumerate(masks):
        by_concept.setdefault(mask.concept, []).append(position)

    for positions in by_concept.values():
        # Candidate pairs are ranked by intersection count so heavily
        # overlapping masks union first; pairs already connected are then
        # skipped without computing their Jaccard. Neither ranking nor
        # skipping can change the final connectivity.
        for left, right in _candidate_pairs(masks, positions):
            root_l, root_r = union.find(left), union.find(right)
            if root_l == root_r:
                continue
            if forbidden is not None and (members[root_l] & forbidden[root_r]
                                          or members[root_r] & forbidden[root_l]):
                continue
            if weighted_jaccard(masks[left], masks[right]) >= threshold:
                union.union(left, right)
                if members is not None:
                    members[root_l] |= members.pop(root_r)
                    forbidden[root_l] |= forbidden.pop(root_r)

    groups: dict[int, list[int]] = {}
    for position in range(len(masks)):
        groups.setdefault(union.find(position), []).append(position)
    return [sorted(group) for group in groups.values()]


def same_view_cannot_links(masks, manifest, masks_dir, max_iou, concepts):
    """Disallow disjoint same-concept 2D detections, including transitive unions.

    Overlapping duplicate/synonym detections remain eligible for association.
    Context concepts may intentionally be excluded by the caller.
    """
    if not 0 <= max_iou <= 1:
        raise ValueError('cannot-link IoU must be in [0, 1]')
    position = {(m.view, m.mask_index): i for i, m in enumerate(masks)}
    result = {}
    for frame in manifest['frames']:
        stem = Path(frame['file']).stem
        groups = {}
        for m in frame['masks']:
            if m['concept'] in concepts and (stem, m['mask_index']) in position:
                groups.setdefault(m['concept'], []).append(m['mask_index'])
        with np.load(masks_dir / frame['mask_file']) as z:
            stack = z['mask_stack']
        boxes, areas = {}, {}
        for indices in groups.values():
            for i in indices:
                yy, xx = np.nonzero(stack[i])
                areas[i] = len(xx)
                boxes[i] = (int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1) if len(xx) else (0,0,0,0)
            for a, b in combinations(indices, 2):
                ba, bb = boxes[a], boxes[b]
                x0,y0,x1,y1 = max(ba[0],bb[0]),max(ba[1],bb[1]),min(ba[2],bb[2]),min(ba[3],bb[3])
                overlap = int(np.count_nonzero((stack[a,y0:y1,x0:x1]>0) & (stack[b,y0:y1,x0:x1]>0))) if x1>x0 and y1>y0 else 0
                area = areas[a]+areas[b]-overlap
                iou = overlap/area if area else 0
                if iou <= max_iou:
                    pa,pb = position[(stem,a)],position[(stem,b)]
                    result.setdefault(pa,set()).add(pb)
                    result.setdefault(pb,set()).add(pa)
    return result


def _candidate_pairs(
    masks: list[MaskSupport], positions: list[int]
) -> list[tuple[int, int]]:
    """Cross-view mask pairs sharing support, ranked by intersection count.

    Stuff concepts produce hundreds of masks with hundreds of thousands of
    supporting Gaussians each; the pair search runs as a sparse matrix
    product so its cost stays in C. Falls back to a pure-Python inverted
    index when scipy is unavailable.
    """

    try:
        from scipy import sparse
    except ImportError:  # pragma: no cover - scipy present in project envs
        return _candidate_pairs_python(masks, positions)

    lengths = np.array([masks[p].indices.size for p in positions], np.int64)
    if lengths.sum() == 0:
        return []
    stacked = np.concatenate([masks[p].indices for p in positions])
    _, compact = np.unique(stacked, return_inverse=True)
    indptr = np.zeros(len(positions) + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    matrix = sparse.csr_matrix(
        (np.ones(stacked.size, np.int64), compact, indptr),
        shape=(len(positions), int(compact.max()) + 1),
    )
    overlap = (matrix @ matrix.T).tocoo()

    views = np.array([masks[p].view for p in positions])
    keep = (overlap.row < overlap.col) & (views[overlap.row] != views[overlap.col])
    rows, cols, counts = overlap.row[keep], overlap.col[keep], overlap.data[keep]
    order = np.lexsort((cols, rows, -counts))
    return [
        (positions[int(row)], positions[int(col)])
        for row, col in zip(rows[order], cols[order])
    ]


def _candidate_pairs_python(
    masks: list[MaskSupport], positions: list[int]
) -> list[tuple[int, int]]:
    inverted: dict[int, list[int]] = {}
    for position in positions:
        for gaussian in masks[position].indices.tolist():
            inverted.setdefault(gaussian, []).append(position)
    candidate_pairs: set[tuple[int, int]] = set()
    for bucket in inverted.values():
        for left, right in combinations(bucket, 2):
            if masks[left].view != masks[right].view:
                candidate_pairs.add((left, right))
    return sorted(candidate_pairs)


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
    if registry.get("contract") not in (ASSOCIATION_CONTRACT, CONSTRAINED_ASSOCIATION_CONTRACT):
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
    parser.add_argument("--cannot-link-vocabulary", type=Path,
                        help="Enable same-view disjoint-mask constraints for gnn_node concepts in this vocabulary")
    parser.add_argument("--cannot-link-max-iou", type=float, default=0.1)
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
    constraints = None
    if args.cannot_link_vocabulary:
        vocabulary = json.loads(args.cannot_link_vocabulary.read_text())
        constrained_concepts = {p['phrase'] for p in vocabulary['phrases'] if p['role']=='gnn_node'}
        constraints = same_view_cannot_links(supports, masks_manifest, args.masks_manifest.parent/'masks',
                                           args.cannot_link_max_iou, constrained_concepts)
        print('CANNOT_LINK_PAIRS', sum(map(len,constraints.values()))//2, flush=True)
    groups = associate_masks(supports, args.threshold, constraints)
    registry = build_instance_registry(supports, groups, args.threshold)
    from scripts.task1.sam3.provenance import sha256_file
    if args.cannot_link_vocabulary:
        registry.update({'contract': CONSTRAINED_ASSOCIATION_CONTRACT,
            'cannot_link_policy': 'same_view_same_concept_mask_iou_at_most_threshold',
            'cannot_link_max_iou': args.cannot_link_max_iou,
            'cannot_link_concepts': sorted(constrained_concepts),
            'cannot_link_pairs': sum(map(len,constraints.values()))//2,
            'cannot_link_vocabulary_sha256': sha256_file(args.cannot_link_vocabulary)})

    registry["masks_manifest"] = str(args.masks_manifest)
    registry["votes_manifest"] = str(args.votes_manifest)
    registry["masks_manifest_sha256"] = sha256_file(args.masks_manifest)
    registry["votes_manifest_sha256"] = sha256_file(args.votes_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(
        f"associated {registry['mask_count']} masks into "
        f"{registry['instance_count']} instances"
    )


if __name__ == "__main__":
    main()
