"""S5: classify instance overlaps and derive the scene graph.

Support-set overlap between accepted instances has three distinct causes:
residual duplicates (mutual containment within one concept), true hierarchy
(asymmetric containment across concepts, giving part_of edges), and boundary
noise (everything else). All relations come from post-vote 3D evidence;
configured expectations are used only as QA checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
                if c_left >= thresholds.duplicate_mutual and (
                    c_right >= thresholds.duplicate_mutual
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

            if c_left >= thresholds.part_of_child and (
                c_right <= thresholds.part_of_parent
            ):
                child, parent = left, right
                child_in_parent, parent_in_child = c_left, c_right
            elif c_right >= thresholds.part_of_child and (
                c_left <= thresholds.part_of_parent
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
    alias: dict[int, int] = {}
    for source_id, target_id in merges:
        source_id = alias.get(source_id, source_id)
        target_id = alias.get(target_id, target_id)
        if source_id == target_id or source_id not in by_id:
            continue
        by_id[target_id] = _merge_supports(by_id[target_id], by_id.pop(source_id))
        alias[source_id] = target_id

    def resolve(instance_id: int) -> int:
        while instance_id in alias:
            instance_id = alias[instance_id]
        return instance_id

    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[int, int]] = set()
    for edge in part_of_edges:
        child = resolve(int(edge["child"]))
        parent = resolve(int(edge["parent"]))
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


def _require_acyclic(node_ids: set[int], edges: set[tuple[int, int]]) -> None:
    outgoing: dict[int, set[int]] = {node: set() for node in node_ids}
    incoming_count: dict[int, int] = {node: 0 for node in node_ids}
    for child, parent in edges:
        if parent not in outgoing[child]:
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
