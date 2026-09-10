"""Tests for overlap classification, hierarchy, and the scene graph."""

import numpy as np
import pytest

from scripts.task1.sam3.instance_hierarchy import (
    OverlapThresholds,
    build_scene_graph,
    classify_overlaps,
    containment,
    instance_geometry,
)


def make_instance(instance_id, concept, start, stop):
    support = np.arange(start, stop, dtype=np.uint32)
    return {
        "instance_id": instance_id,
        "concept": concept,
        "support": support,
        "scores": np.ones(support.size, dtype=np.float32),
    }


def test_containment():
    a = np.array([1, 2, 3], np.uint32)
    b = np.array([2, 3, 4, 5], np.uint32)
    assert containment(a, b) == 2 / 3


def test_classify_part_of_and_noise():
    building = make_instance(1, "building", 0, 100)
    window = make_instance(2, "window", 10, 20)
    car = make_instance(3, "car", 98, 130)
    result = classify_overlaps(
        [building, window, car],
        OverlapThresholds(
            part_of_child=0.6, part_of_parent=0.3, duplicate_mutual=0.8
        ),
    )
    assert result.part_of_edges == [
        {
            "child": 2,
            "parent": 1,
            "containment_child_in_parent": 1.0,
            "containment_parent_in_child": 0.1,
        }
    ]
    assert result.merges == []
    assert result.noise_pairs == [(1, 3)]


def test_classify_duplicates_merge_same_concept():
    a = make_instance(1, "car", 0, 10)
    b = make_instance(2, "car", 0, 9)
    result = classify_overlaps([a, b], OverlapThresholds())
    assert result.merges == [(2, 1)]  # smaller merges into larger
    assert result.part_of_edges == []


def test_same_concept_never_emits_part_of():
    big = make_instance(1, "car", 0, 100)
    small = make_instance(2, "car", 10, 20)
    result = classify_overlaps([big, small], OverlapThresholds())
    assert result.part_of_edges == []
    assert result.noise_pairs == [(1, 2)]


def test_disjoint_instances_are_unrelated():
    a = make_instance(1, "car", 0, 10)
    b = make_instance(2, "tree", 50, 60)
    result = classify_overlaps([a, b], OverlapThresholds())
    assert result.part_of_edges == []
    assert result.merges == []
    assert result.noise_pairs == []


def test_instance_geometry_weighted():
    xyz = np.array([[0, 0, 0], [2, 0, 0], [9, 9, 9]], np.float32)
    geometry = instance_geometry(
        np.array([0, 1], np.uint32), np.array([1.0, 3.0], np.float32), xyz
    )
    np.testing.assert_allclose(geometry["centroid"], [1.5, 0.0, 0.0])
    np.testing.assert_allclose(geometry["bbox_min"], [0, 0, 0])
    np.testing.assert_allclose(geometry["bbox_max"], [2, 0, 0])


def test_scene_graph_nodes_edges_and_qa():
    xyz = np.zeros((200, 3), np.float32)
    xyz[:, 0] = np.arange(200, dtype=np.float32)
    building = make_instance(1, "building", 0, 100)
    window = make_instance(2, "window", 10, 20)
    edges = [
        {
            "child": 2,
            "parent": 1,
            "containment_child_in_parent": 1.0,
            "containment_parent_in_child": 0.1,
        }
    ]
    graph = build_scene_graph(
        [building, window],
        part_of_edges=edges,
        merges=[],
        xyz=xyz,
        expected_part_of=[("window", "building")],
    )
    assert graph["source"] == "sam3_instance_scene_graph"
    assert graph["contract"] == "membership_derived_nodes_part_of_edges_v1"
    nodes = {node["instance_id"]: node for node in graph["nodes"]}
    assert nodes[1]["concept"] == "building"
    assert nodes[1]["gaussian_count"] == 100
    np.testing.assert_allclose(nodes[2]["centroid"], [14.5, 0.0, 0.0])
    np.testing.assert_allclose(nodes[2]["bbox_min"], [10.0, 0.0, 0.0])
    np.testing.assert_allclose(nodes[2]["bbox_max"], [19.0, 0.0, 0.0])
    assert graph["edges"] == [
        {"child": 2, "parent": 1, "type": "part_of",
         "containment_child_in_parent": 1.0,
         "containment_parent_in_child": 0.1}
    ]
    assert graph["expected_part_of_found"] == [
        {"child": "window", "parent": "building", "found": True}
    ]


def test_scene_graph_applies_merges():
    xyz = np.zeros((20, 3), np.float32)
    a = make_instance(1, "car", 0, 10)
    b = make_instance(2, "car", 5, 12)
    graph = build_scene_graph(
        [a, b], part_of_edges=[], merges=[(2, 1)], xyz=xyz, expected_part_of=[]
    )
    ids = [node["instance_id"] for node in graph["nodes"]]
    assert ids == [1]
    assert graph["nodes"][0]["gaussian_count"] == 12  # union of both supports


def test_scene_graph_survives_merge_chains():
    # Regression: alias chains from cascading same-concept duplicates.
    # merges retarget earlier targets: (2,1), (1,3), (5,2) must all land on 3.
    xyz = np.zeros((40, 3), np.float32)
    instances = [
        make_instance(1, "car", 0, 12),
        make_instance(2, "car", 0, 11),
        make_instance(3, "car", 0, 14),
        make_instance(5, "car", 1, 12),
    ]
    graph = build_scene_graph(
        instances,
        part_of_edges=[],
        merges=[(2, 1), (1, 3), (5, 2)],
        xyz=xyz,
        expected_part_of=[],
    )
    assert [node["instance_id"] for node in graph["nodes"]] == [3]
    assert graph["nodes"][0]["gaussian_count"] == 14  # union of all supports
    assert graph["merged_instance_count"] == 3


def test_flat_labels_resolve_merge_chains():
    from scripts.task1.sam3.instance_hierarchy import flat_instance_labels
    from scripts.task1.sam3.instance_membership import accumulate_membership

    # instance 2 wins gaussians 0..3 in both cameras; merges chain 2 -> 1 -> 3,
    # so the flat labels must resolve to the surviving id 3.
    events = [
        (np.arange(4, dtype=np.uint32), np.full(4, 2, np.uint16)),
        (np.arange(4, dtype=np.uint32), np.full(4, 2, np.uint16)),
    ]
    membership = accumulate_membership(
        events, np.full(6, 2, np.uint16), gaussian_count=6
    )
    labels = flat_instance_labels(
        membership,
        merges=[(2, 1), (1, 3)],
        size_by_id={3: 10},
        gaussian_count=6,
    )
    assert labels.tolist() == [3, 3, 3, 3, 0, 0]


def test_scene_graph_cycle_raises():
    xyz = np.zeros((20, 3), np.float32)
    a = make_instance(1, "building", 0, 10)
    b = make_instance(2, "window", 0, 10)
    edges = [
        {"child": 2, "parent": 1, "containment_child_in_parent": 1.0,
         "containment_parent_in_child": 1.0},
        {"child": 1, "parent": 2, "containment_child_in_parent": 1.0,
         "containment_parent_in_child": 1.0},
    ]
    with pytest.raises(ValueError, match="cycle"):
        build_scene_graph(
            [a, b], part_of_edges=edges, merges=[], xyz=xyz, expected_part_of=[]
        )
