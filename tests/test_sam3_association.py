"""Tests for cross-view instance association by 3D support overlap."""

import numpy as np

from scripts.task1.sam3.associate_instances import (
    ASSOCIATION_CONTRACT,
    ASSOCIATION_SOURCE,
    MaskSupport,
    associate_masks,
    build_instance_registry,
    weighted_jaccard,
)


def sup(view, mask_index, concept, pairs, score=0.9):
    idx = np.array([pair[0] for pair in pairs], np.uint32)
    wts = np.array([pair[1] for pair in pairs], np.float32)
    return MaskSupport(
        view=view,
        mask_index=mask_index,
        concept=concept,
        score=score,
        indices=idx,
        weights=wts,
    )


def test_weighted_jaccard():
    a = sup("v0", 0, "car", [(1, 1.0), (2, 0.5)])
    b = sup("v1", 0, "car", [(2, 0.5), (3, 1.0)])
    assert weighted_jaccard(a, b) == 0.5 / 2.5


def test_weighted_jaccard_disjoint_is_zero():
    a = sup("v0", 0, "car", [(1, 1.0)])
    b = sup("v1", 0, "car", [(2, 1.0)])
    assert weighted_jaccard(a, b) == 0.0


def test_same_object_across_views_merges():
    car_v0 = sup("v0", 0, "car", [(i, 1.0) for i in range(10)])
    car_v1 = sup("v1", 0, "car", [(i, 1.0) for i in range(2, 12)])
    other = sup("v1", 1, "car", [(i, 1.0) for i in range(100, 110)])
    groups = associate_masks([car_v0, car_v1, other], threshold=0.3)
    assert sorted(len(group) for group in groups) == [1, 2]


def test_concepts_never_merge():
    car = sup("v0", 0, "car", [(i, 1.0) for i in range(10)])
    tree = sup("v1", 0, "tree", [(i, 1.0) for i in range(10)])
    assert len(associate_masks([car, tree], threshold=0.1)) == 2


def test_same_view_pairs_do_not_union_directly():
    a = sup("v0", 0, "car", [(i, 1.0) for i in range(10)])
    b = sup("v0", 1, "car", [(i, 1.0) for i in range(10)])
    assert len(associate_masks([a, b], threshold=0.1)) == 2


def test_associate_masks_matches_bruteforce_reference():
    rng = np.random.default_rng(7)
    masks = []
    for view in range(6):
        for mask_index in range(4):
            size = int(rng.integers(3, 30))
            indices = np.sort(
                rng.choice(200, size=size, replace=False)
            ).astype(np.uint32)
            weights = (rng.random(size) + 0.1).astype(np.float32)
            masks.append(
                MaskSupport(
                    view=f"v{view}",
                    mask_index=mask_index,
                    concept="car",
                    score=0.9,
                    indices=indices,
                    weights=weights,
                )
            )
    threshold = 0.08
    groups = associate_masks(masks, threshold)

    parent = list(range(len(masks)))

    def find(node):
        while parent[node] != node:
            node = parent[node]
        return node

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if masks[i].view == masks[j].view:
                continue
            if weighted_jaccard(masks[i], masks[j]) >= threshold:
                parent[find(j)] = find(i)
    expected: dict = {}
    for i in range(len(masks)):
        expected.setdefault(find(i), []).append(i)

    assert sorted(sorted(g) for g in groups) == sorted(
        sorted(g) for g in expected.values()
    )
    assert 1 < len(groups) < len(masks)  # the case is neither trivial nor total


def test_registry_shape_and_conflict_count():
    a = sup("v0", 0, "car", [(i, 1.0) for i in range(10)])
    b = sup("v1", 0, "car", [(i, 1.0) for i in range(10)])
    c = sup("v0", 1, "car", [(i, 1.0) for i in range(5, 15)])
    masks = [a, b, c]
    registry = build_instance_registry(
        masks, associate_masks(masks, threshold=0.3), threshold=0.3
    )
    assert registry["source"] == ASSOCIATION_SOURCE
    assert registry["contract"] == ASSOCIATION_CONTRACT
    assert registry["threshold"] == 0.3
    instances = registry["instances"]
    assert instances[0]["instance_id"] == 1
    assert [inst["instance_id"] for inst in instances] == list(
        range(1, len(instances) + 1)
    )
    merged = max(instances, key=lambda inst: len(inst["members"]))
    assert merged["concept"] == "car"
    assert merged["supporting_camera_count"] == 2
    assert merged["members"] == [
        {"view": "v0", "mask_index": 0, "score": 0.9},
        {"view": "v0", "mask_index": 1, "score": 0.9},
        {"view": "v1", "mask_index": 0, "score": 0.9},
    ]
    # a and c share view v0 inside one merged group -> one conflict group
    assert registry["same_view_conflict_groups"] == 1


def test_registry_without_conflicts():
    a = sup("v0", 0, "car", [(i, 1.0) for i in range(10)])
    b = sup("v1", 0, "car", [(i, 1.0) for i in range(10)])
    registry = build_instance_registry(
        [a, b], associate_masks([a, b], threshold=0.3), threshold=0.3
    )
    assert registry["same_view_conflict_groups"] == 0
    assert len(registry["instances"]) == 1
    members = registry["instances"][0]["members"]
    assert members == [
        {"view": "v0", "mask_index": 0, "score": 0.9},
        {"view": "v1", "mask_index": 0, "score": 0.9},
    ]
