"""S5: classify instance overlaps and derive the scene graph.

Support-set overlap between accepted instances has three distinct causes:
residual duplicates (mutual containment within one concept), true hierarchy
(asymmetric containment across concepts, giving part_of edges), and boundary
noise (everything else). All relations come from post-vote 3D evidence;
configured expectations are used only as QA checks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.sam3.instance_membership import STATUS_ACCEPTED


HIERARCHY_SOURCE = "sam3_instance_hierarchy"
HIERARCHY_CONTRACT = "support_containment_classification_v1"
SCENE_GRAPH_SOURCE = "sam3_instance_scene_graph"
SCENE_GRAPH_CONTRACT = "membership_derived_nodes_part_of_edges_v1"


@dataclass(frozen=True)
class OverlapThresholds:
    part_of_child: float = 0.6
    part_of_parent: float = 0.3
    duplicate_mutual: float = 0.8


@dataclass(frozen=True)
class OverlapClassification:
    merges: list[tuple[int, int]] = field(default_factory=list)
    part_of_edges: list[dict[str, Any]] = field(default_factory=list)
    noise_pairs: list[tuple[int, int]] = field(default_factory=list)


def containment(a: np.ndarray, b: np.ndarray) -> float:
    """|A intersect B| / |A| over sparse index arrays."""

    left = np.asarray(a)
    if left.size == 0:
        return 0.0
    common = np.intersect1d(left, np.asarray(b), assume_unique=True)
    return float(common.size) / float(left.size)


def classify_overlaps(
    instances: list[dict[str, Any]],
    thresholds: OverlapThresholds,
) -> OverlapClassification:
    """Split every overlapping instance pair into merge, part_of, or noise."""

    merges: list[tuple[int, int]] = []
    edges: list[dict[str, Any]] = []
    noise: list[tuple[int, int]] = []

    for position, left in enumerate(instances):
        for right in instances[position + 1 :]:
            c_left = containment(left["support"], right["support"])
            c_right = containment(right["support"], left["support"])
            if c_left == 0.0 and c_right == 0.0:
                continue
            pair = tuple(sorted((int(left["instance_id"]), int(right["instance_id"]))))

            if left["concept"] == right["concept"]:
                if (
                    c_left >= thresholds.duplicate_mutual
                    and c_right >= thresholds.duplicate_mutual
                ):
                    smaller, larger = sorted(
                        (left, right),
                        key=lambda inst: (
                            inst["support"].size,
                            -int(inst["instance_id"]),
                        ),
                    )
                    merges.append(
                        (int(smaller["instance_id"]), int(larger["instance_id"]))
                    )
                else:
                    noise.append(pair)
                continue

            if (
                c_left >= thresholds.part_of_child
                and c_right <= thresholds.part_of_parent
            ):
                child, parent = left, right
                child_in_parent, parent_in_child = c_left, c_right
            elif (
                c_right >= thresholds.part_of_child
                and c_left <= thresholds.part_of_parent
            ):
                child, parent = right, left
                child_in_parent, parent_in_child = c_right, c_left
            else:
                noise.append(pair)
                continue
            edges.append(
                {
                    "child": int(child["instance_id"]),
                    "parent": int(parent["instance_id"]),
                    "containment_child_in_parent": child_in_parent,
                    "containment_parent_in_child": parent_in_child,
                }
            )

    return OverlapClassification(merges=merges, part_of_edges=edges, noise_pairs=noise)


def instance_geometry(
    indices: np.ndarray, scores: np.ndarray, xyz: np.ndarray
) -> dict[str, list[float]]:
    """Score-weighted centroid and axis-aligned bbox of a support set."""

    support = np.asarray(indices, dtype=np.int64)
    weight = np.asarray(scores, dtype=np.float64)
    if support.shape != weight.shape or support.size == 0:
        raise ValueError("instance geometry needs aligned, non-empty support")
    points = np.asarray(xyz, dtype=np.float64)[support]
    centroid = (points * weight[:, None]).sum(axis=0) / weight.sum()
    return {
        "centroid": [float(v) for v in centroid],
        "bbox_min": [float(v) for v in points.min(axis=0)],
        "bbox_max": [float(v) for v in points.max(axis=0)],
    }


def _merge_supports(
    target: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    indices = np.concatenate([target["support"], source["support"]])
    scores = np.concatenate([target["scores"], source["scores"]])
    order = np.lexsort((-scores, indices))
    indices, scores = indices[order], scores[order]
    first = np.ones(indices.size, dtype=bool)
    first[1:] = indices[1:] != indices[:-1]
    merged = dict(target)
    merged["support"] = indices[first]
    merged["scores"] = scores[first]
    return merged


def _require_acyclic(node_ids: set[int], edges: set[tuple[int, int]]) -> None:
    """Kahn's algorithm over the deduplicated child -> parent edge set."""

    outgoing: dict[int, set[int]] = {node: set() for node in node_ids}
    incoming_count: dict[int, int] = {node: 0 for node in node_ids}
    for child, parent in edges:
        outgoing[child].add(parent)
        incoming_count[parent] += 1
    frontier = [node for node in node_ids if incoming_count[node] == 0]
    visited = 0
    while frontier:
        node = frontier.pop()
        visited += 1
        for parent in outgoing[node]:
            incoming_count[parent] -= 1
            if incoming_count[parent] == 0:
                frontier.append(parent)
    if visited != len(node_ids):
        raise ValueError("part_of cycle detected in the instance hierarchy")


def build_scene_graph(
    instances: list[dict[str, Any]],
    part_of_edges: list[dict[str, Any]],
    merges: list[tuple[int, int]],
    xyz: np.ndarray,
    expected_part_of: list[tuple[str, str]],
) -> dict[str, Any]:
    """Assemble the GNN-facing scene graph and validate part_of acyclicity."""

    by_id: dict[int, dict[str, Any]] = {
        int(instance["instance_id"]): dict(instance) for instance in instances
    }
    alias = _resolve_aliases(merges)
    for source_id, target_id in alias.items():
        if source_id in by_id and target_id in by_id:
            by_id[target_id] = _merge_supports(
                by_id[target_id], by_id.pop(source_id)
            )

    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[int, int]] = set()
    for edge in part_of_edges:
        child = alias.get(int(edge["child"]), int(edge["child"]))
        parent = alias.get(int(edge["parent"]), int(edge["parent"]))
        if child == parent or (child, parent) in seen_edges:
            continue
        seen_edges.add((child, parent))
        edges.append(
            {
                "child": child,
                "parent": parent,
                "type": "part_of",
                "containment_child_in_parent": edge["containment_child_in_parent"],
                "containment_parent_in_child": edge["containment_parent_in_child"],
            }
        )

    _require_acyclic(set(by_id), seen_edges)

    nodes = []
    for instance_id in sorted(by_id):
        instance = by_id[instance_id]
        geometry = instance_geometry(instance["support"], instance["scores"], xyz)
        nodes.append(
            {
                "instance_id": instance_id,
                "concept": instance["concept"],
                "gaussian_count": int(instance["support"].size),
                **geometry,
            }
        )

    concept_of = {node["instance_id"]: node["concept"] for node in nodes}
    edge_concepts = {
        (concept_of[edge["child"]], concept_of[edge["parent"]]) for edge in edges
    }
    qa = [
        {"child": child, "parent": parent, "found": (child, parent) in edge_concepts}
        for child, parent in expected_part_of
    ]

    return {
        "source": SCENE_GRAPH_SOURCE,
        "contract": SCENE_GRAPH_CONTRACT,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "merged_instance_count": len(alias),
        "nodes": nodes,
        "edges": edges,
        "expected_part_of_found": qa,
    }


def _accepted_entries(
    membership: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand accepted CSR entries into gaussian, instance id, score columns."""

    counts = np.diff(membership.indptr)
    gaussian_of_entry = np.repeat(
        np.arange(counts.shape[0], dtype=np.int64), counts
    )
    accepted = membership.status == STATUS_ACCEPTED
    return (
        gaussian_of_entry[accepted],
        membership.instance_ids[accepted].astype(np.int64),
        membership.scores[accepted],
    )


def accepted_instances(
    membership: Any, registry: dict[str, Any]
) -> list[dict[str, Any]]:
    """Group accepted membership entries into per-instance support sets."""

    concept_by_id = {
        int(instance["instance_id"]): str(instance["concept"])
        for instance in registry["instances"]
    }
    gaussians, ids, scores = _accepted_entries(membership)

    order = np.lexsort((gaussians, ids))
    gaussians, ids, scores = gaussians[order], ids[order], scores[order]
    boundaries = np.flatnonzero(np.diff(ids)) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [ids.shape[0]]])

    instances: list[dict[str, Any]] = []
    for start, stop in zip(starts, stops):
        if start == stop:
            continue
        instance_id = int(ids[start])
        if instance_id not in concept_by_id:
            raise ValueError(f"membership references unknown instance {instance_id}")
        instances.append(
            {
                "instance_id": instance_id,
                "concept": concept_by_id[instance_id],
                "support": gaussians[start:stop].astype(np.uint32),
                "scores": scores[start:stop].astype(np.float32),
            }
        )
    return instances


def _resolve_aliases(merges: list[tuple[int, int]]) -> dict[int, int]:
    """Flatten merge chains so every source maps to its terminal target.

    Merges may retarget earlier targets ((2,1), (1,3), (5,2) must all land on
    3), so both ends are chain-resolved on insertion and the map is
    path-compressed afterwards.
    """

    alias: dict[int, int] = {}
    for source_id, target_id in merges:
        while source_id in alias:
            source_id = alias[source_id]
        while target_id in alias:
            target_id = alias[target_id]
        if source_id != target_id:
            alias[source_id] = target_id
    for source_id, target_id in alias.items():
        while target_id in alias:
            target_id = alias[target_id]
        alias[source_id] = target_id
    return alias


def flat_instance_labels(
    membership: Any,
    merges: list[tuple[int, int]],
    size_by_id: dict[int, int],
    gaussian_count: int,
) -> np.ndarray:
    """Derive the one-instance-per-Gaussian visualization view.

    Among accepted memberships the top score wins; score ties resolve to the
    most specific (smallest) instance so nested labels stay visible, then to
    the smaller id. Gaussians without accepted membership stay 0.
    """

    alias = _resolve_aliases(merges)
    gaussians, ids, scores = _accepted_entries(membership)
    labels = np.zeros(gaussian_count, dtype=np.int32)
    if gaussians.size == 0:
        return labels
    ids = np.array(
        [alias.get(value, value) for value in ids.tolist()], dtype=np.int64
    )
    sizes = np.array(
        [size_by_id[value] for value in ids.tolist()], dtype=np.int64
    )

    order = np.lexsort((ids, sizes, -scores.astype(np.float64), gaussians))
    gaussians, ids = gaussians[order], ids[order]
    first = np.ones(gaussians.shape[0], dtype=bool)
    first[1:] = gaussians[1:] != gaussians[:-1]
    labels[gaussians[first]] = ids[first].astype(np.int32)
    return labels


def main(argv: list[str] | None = None) -> None:
    from scripts.task1.common.ply_utils import read_vertex_xyz
    from scripts.task1.sam3.associate_instances import validate_instance_registry
    from scripts.task1.sam3.instance_membership import load_membership
    from scripts.task1.sam3.vocabulary import load_vocabulary

    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--source-ply", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--part-of-child", default=0.6, type=float)
    parser.add_argument("--part-of-parent", default=0.3, type=float)
    parser.add_argument("--duplicate-mutual", default=0.8, type=float)
    args = parser.parse_args(argv)

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    membership = load_membership(args.membership)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    validate_instance_registry(registry)
    vocabulary = load_vocabulary(args.vocabulary)
    xyz = read_vertex_xyz(args.source_ply)
    gaussian_count = membership.indptr.shape[0] - 1
    if xyz.shape[0] != gaussian_count:
        raise ValueError(
            f"PLY has {xyz.shape[0]} vertices; membership covers {gaussian_count}"
        )

    thresholds = OverlapThresholds(
        part_of_child=args.part_of_child,
        part_of_parent=args.part_of_parent,
        duplicate_mutual=args.duplicate_mutual,
    )
    instances = accepted_instances(membership, registry)
    classification = classify_overlaps(instances, thresholds)
    graph = build_scene_graph(
        instances,
        classification.part_of_edges,
        classification.merges,
        xyz,
        list(vocabulary.expected_part_of),
    )
    size_by_id = {node["instance_id"]: node["gaussian_count"] for node in graph["nodes"]}
    labels = flat_instance_labels(
        membership, classification.merges, size_by_id, gaussian_count
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    hierarchy = {
        "source": HIERARCHY_SOURCE,
        "contract": HIERARCHY_CONTRACT,
        "membership": str(args.membership),
        "registry": str(args.registry),
        "thresholds": {
            "part_of_child": thresholds.part_of_child,
            "part_of_parent": thresholds.part_of_parent,
            "duplicate_mutual": thresholds.duplicate_mutual,
        },
        "merges": [list(pair) for pair in classification.merges],
        "part_of_edges": classification.part_of_edges,
        "noise_pairs": [list(pair) for pair in classification.noise_pairs],
    }
    (args.output_dir / "hierarchy.json").write_text(
        json.dumps(hierarchy, indent=2), encoding="utf-8"
    )
    (args.output_dir / "scene_graph.json").write_text(
        json.dumps(graph, indent=2), encoding="utf-8"
    )
    np.save(args.output_dir / "gaussian_instances.npy", labels)
    print(
        f"scene graph: {graph['node_count']} nodes, {graph['edge_count']} "
        f"part_of edges, {len(classification.merges)} merges"
    )


if __name__ == "__main__":
    main()
