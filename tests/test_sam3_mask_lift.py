"""Tests for the per-concept FlashSplat mask lift core."""

import numpy as np
import pytest

from scripts.task1.sam3.lift_mask_votes_core import (
    concept_index_map,
    mask_membership_votes,
    observed_gaussians,
)


def test_concept_index_map_scores_break_overlaps():
    a = np.zeros((4, 4), np.uint8)
    a[0:2, 0:2] = 1
    b = np.zeros((4, 4), np.uint8)
    b[1:3, 1:3] = 1
    index_map = concept_index_map(np.stack([a, b]), np.array([0.6, 0.9]))
    assert index_map.dtype == np.float32
    assert index_map[0, 0] == 1.0  # only mask a
    assert index_map[1, 1] == 2.0  # overlap -> higher score wins
    assert index_map[2, 2] == 2.0  # only mask b
    assert index_map[3, 3] == 0.0  # background


def test_concept_index_map_rejects_mismatched_scores():
    stack = np.zeros((2, 4, 4), np.uint8)
    with pytest.raises(ValueError):
        concept_index_map(stack, np.array([0.5]))


def test_membership_votes_are_per_mask_fractions():
    # rows: background + 2 masks, 5 gaussians
    used = np.array(
        [
            [0.0, 1.0, 0.5, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.0],
        ],
        np.float32,
    )
    visibility = used.sum(axis=0)
    indices, mask_ids, weights = mask_membership_votes(
        used, visibility, view_mask_indices=np.array([4, 7], np.uint16)
    )
    assert indices.dtype == np.uint32
    assert mask_ids.dtype == np.uint16
    assert weights.dtype == np.float32
    assert indices.tolist() == [0, 1, 2]
    assert mask_ids.tolist() == [4, 4, 7]
    np.testing.assert_allclose(weights, [1.0, 0.5, 0.5])


def test_membership_votes_rejects_row_mismatch():
    with pytest.raises(ValueError):
        mask_membership_votes(
            np.zeros((1, 4), np.float32),
            np.zeros(4, np.float32),
            np.array([1], np.uint16),
        )  # needs background + 1 mask row


def test_membership_votes_rejects_negative_support():
    used = np.array([[0.0, 1.0], [-0.5, 0.0]], np.float32)
    with pytest.raises(ValueError):
        mask_membership_votes(
            used, np.abs(used).sum(axis=0), np.array([3], np.uint16)
        )


def test_observed_gaussians():
    observed = observed_gaussians(np.array([0.0, 0.4, 0.0, 2.0], np.float32))
    assert observed.dtype == np.uint32
    assert observed.tolist() == [1, 3]
