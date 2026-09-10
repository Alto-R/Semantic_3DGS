"""Tests for the per-concept membership consensus stage."""

import numpy as np
import pytest

from scripts.task1.sam3.instance_membership import (
    STATUS_ACCEPTED,
    STATUS_SINGLE_CAMERA,
    STATUS_WEAK_MAJORITY,
    accumulate_membership,
    load_membership,
    save_membership,
    view_concept_winners,
)


def test_view_concept_winners_require_dominant_unique_mass():
    indices = np.array([0, 0, 1, 2], np.uint32)
    mask_ids = np.array([4, 7, 4, 7], np.uint16)
    weights = np.array([0.8, 0.2, 0.4, 0.5], np.float32)
    winners_g, winners_m = view_concept_winners(
        indices, mask_ids, weights, min_weight=0.5
    )
    # gaussian 0: mask 4 wins with 0.8; gaussian 1: 0.4 below min_weight;
    # gaussian 2: exactly 0.5 counts
    assert winners_g.tolist() == [0, 2]
    assert winners_m.tolist() == [4, 7]


def test_view_concept_winners_tie_abstains():
    indices = np.array([0, 0], np.uint32)
    mask_ids = np.array([4, 7], np.uint16)
    weights = np.array([0.5, 0.5], np.float32)
    winners_g, _ = view_concept_winners(indices, mask_ids, weights, min_weight=0.5)
    assert winners_g.size == 0


def test_view_concept_winners_empty_input():
    winners_g, winners_m = view_concept_winners(
        np.zeros(0, np.uint32),
        np.zeros(0, np.uint16),
        np.zeros(0, np.float32),
        min_weight=0.5,
    )
    assert winners_g.size == 0
    assert winners_m.size == 0


def test_accumulate_membership_statuses():
    # gaussian 0 observed by 3 cameras, supported by 2 -> accepted
    # gaussian 1 observed by 3, supported by 1 -> single camera
    # gaussian 2 observed by 4, supported by 2 -> weak majority
    events = [
        (np.array([0, 1, 2], np.uint32), np.array([1, 1, 1], np.uint16)),
        (np.array([0, 2], np.uint32), np.array([1, 1], np.uint16)),
    ]
    observe_counts = np.array([3, 3, 4, 9], np.uint16)
    csr = accumulate_membership(events, observe_counts, gaussian_count=4)
    assert csr.indptr.tolist() == [0, 1, 2, 3, 3]
    assert csr.instance_ids.tolist() == [1, 1, 1]
    assert csr.support_counts.tolist() == [2, 1, 2]
    assert csr.observe_counts.tolist() == [3, 3, 4]
    np.testing.assert_allclose(csr.scores, [2 / 3, 1 / 3, 2 / 4])
    assert csr.status.tolist() == [
        STATUS_ACCEPTED,
        STATUS_SINGLE_CAMERA,
        STATUS_WEAK_MAJORITY,
    ]


def test_accumulate_membership_multiple_instances_sorted_per_gaussian():
    events = [
        (np.array([0, 0], np.uint32), np.array([2, 5], np.uint16)),
        (np.array([0], np.uint32), np.array([2], np.uint16)),
    ]
    observe_counts = np.array([3], np.uint16)
    csr = accumulate_membership(events, observe_counts, gaussian_count=1)
    assert csr.instance_ids.tolist() == [2, 5]
    assert csr.support_counts.tolist() == [2, 1]


def test_single_camera_support_never_upgrades_to_accepted():
    # support == 1 with observe == 1 satisfies the strict-majority predicate
    # but must stay SINGLE_CAMERA, mirroring the audited abstention policy.
    events = [(np.array([0], np.uint32), np.array([1], np.uint16))]
    csr = accumulate_membership(
        events, np.array([1], np.uint16), gaussian_count=1
    )
    assert csr.status.tolist() == [STATUS_SINGLE_CAMERA]


def test_accumulate_membership_rejects_support_exceeding_observation():
    # two votes on a Gaussian observed once is a driver bug, not weak data
    events = [
        (np.array([0], np.uint32), np.array([1], np.uint16)),
        (np.array([0], np.uint32), np.array([1], np.uint16)),
    ]
    with pytest.raises(RuntimeError):
        accumulate_membership(events, np.array([1], np.uint16), gaussian_count=1)


def test_accumulate_membership_rejects_vote_without_visibility():
    events = [(np.array([0], np.uint32), np.array([1], np.uint16))]
    observe_counts = np.array([0], np.uint16)
    with pytest.raises(RuntimeError):
        accumulate_membership(events, observe_counts, gaussian_count=1)


def test_membership_npz_roundtrip(tmp_path):
    events = [
        (np.array([0, 1], np.uint32), np.array([1, 2], np.uint16)),
        (np.array([0], np.uint32), np.array([1], np.uint16)),
    ]
    observe_counts = np.array([2, 3], np.uint16)
    csr = accumulate_membership(events, observe_counts, gaussian_count=2)
    save_membership(tmp_path / "membership.npz", csr)
    loaded = load_membership(tmp_path / "membership.npz")
    np.testing.assert_array_equal(loaded.indptr, csr.indptr)
    np.testing.assert_array_equal(loaded.instance_ids, csr.instance_ids)
    np.testing.assert_array_equal(loaded.support_counts, csr.support_counts)
    np.testing.assert_array_equal(loaded.observe_counts, csr.observe_counts)
    np.testing.assert_allclose(loaded.scores, csr.scores)
    np.testing.assert_array_equal(loaded.status, csr.status)
