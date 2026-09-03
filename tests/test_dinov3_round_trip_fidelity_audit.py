from __future__ import annotations

import unittest

import numpy as np

from scripts.task1.dinov3.round_trip_fidelity_audit import (
    STATUS_ACCEPTED,
    STATUS_EXACT_TIE,
    STATUS_NO_STRICT_MAJORITY,
    STATUS_SINGLE_CAMERA,
    STATUS_UNOBSERVED,
    boundary_mask,
    collapse_camera_distribution,
    consensus_statistics,
    leave_one_out_consensus,
    validate_provenance,
)


class RoundTripFidelityAuditTest(unittest.TestCase):
    def test_collapses_distribution_to_one_unique_camera_identity(self) -> None:
        winners, summary = collapse_camera_distribution(
            indices=np.asarray([0, 1, 0, 1, 2, 2], dtype=np.uint32),
            class_ids=np.asarray([1, 1, 2, 2, 1, 2], dtype=np.uint16),
            weights=np.asarray([0.7, 0.5, 0.3, 0.5, 0.2, 0.8], dtype=np.float32),
            gaussian_count=4,
            class_count=3,
        )
        np.testing.assert_array_equal(winners, [1, 0, 2, 0])
        self.assertEqual(summary["visible_gaussian_count"], 3)
        self.assertEqual(summary["unique_camera_winner_count"], 2)
        self.assertEqual(summary["exact_tie_count"], 1)

    def test_rejects_camera_mass_that_is_not_normalized(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to one"):
            collapse_camera_distribution(
                indices=np.asarray([0, 0], dtype=np.uint32),
                class_ids=np.asarray([1, 2], dtype=np.uint16),
                weights=np.asarray([0.6, 0.3], dtype=np.float32),
                gaussian_count=1,
                class_count=2,
            )

    def test_consensus_requires_two_cameras_unique_strict_majority(self) -> None:
        # Columns: accepted 3/4, tie 2/2, plurality 2/1/1, single, unseen.
        counts = np.zeros((5, 5), dtype=np.uint8)
        counts[1:, 0] = [3, 1, 0, 0]
        counts[1:, 1] = [2, 2, 0, 0]
        counts[1:, 2] = [2, 1, 1, 0]
        counts[1:, 3] = [1, 0, 0, 0]
        stats = consensus_statistics(counts, chunk_size=2)
        np.testing.assert_array_equal(
            stats["status"],
            [
                STATUS_ACCEPTED,
                STATUS_EXACT_TIE,
                STATUS_NO_STRICT_MAJORITY,
                STATUS_SINGLE_CAMERA,
                STATUS_UNOBSERVED,
            ],
        )

    def test_consensus_supports_more_than_255_cameras(self) -> None:
        # A scene can have more than 255 cameras; per-class counts must not
        # wrap at uint8 (Kitchen has 279 cameras).
        counts = np.zeros((4, 3), dtype=np.uint16)
        counts[1:, 0] = [300, 0, 0]    # unanimous 300/300 -> accepted
        counts[1:, 1] = [150, 150, 0]  # exact tie -> tie
        counts[1:, 2] = [160, 140, 0]  # 160/300 -> accepted
        stats = consensus_statistics(counts)
        np.testing.assert_array_equal(
            stats["status"],
            [STATUS_ACCEPTED, STATUS_EXACT_TIE, STATUS_ACCEPTED],
        )
        self.assertEqual(int(stats["maximum"][0]), 300)
        self.assertEqual(int(stats["maximum"][1]), 150)

    def test_leave_one_out_removes_target_before_consensus(self) -> None:
        counts = np.zeros((4, 4), dtype=np.uint8)
        counts[1:, 0] = [3, 1, 0]  # remove class 1 -> 2/1, still accepted
        counts[1:, 1] = [2, 1, 0]  # remove class 1 -> 1/1, abstain
        counts[1:, 2] = [2, 2, 0]  # remove tied class 1 -> unique class 2
        counts[1:, 3] = [1, 1, 1]  # remove one tie -> still tied
        stats = consensus_statistics(counts)
        heldout = np.asarray([1, 1, 1, 1], dtype=np.uint16)
        labels, remaining = leave_one_out_consensus(stats, heldout)
        np.testing.assert_array_equal(labels, [1, 0, 2, 0])
        np.testing.assert_array_equal(remaining, [3, 2, 3, 2])

    def test_boundary_mask_marks_both_sides_of_four_connected_change(self) -> None:
        values = np.asarray([[1, 1, 2], [1, 1, 2]], dtype=np.uint16)
        np.testing.assert_array_equal(
            boundary_mask(values),
            [[False, True, True], [False, True, True]],
        )

    def test_provenance_rejects_manual_or_reordered_cache(self) -> None:
        cache = {
            "contract": "threshold_selected_dinov3_view_cache_v1",
            "camera_indices": [4, 8],
            "camera_ids": [40, 80],
            "render": {"max_width": 960},
            "manual_camera_selection_used": False,
            "semantic_vote_lifting_run": False,
            "semantic_fusion_run": False,
            "semantic_labels_written": False,
            "label_map_written": False,
            "semantic_ply_written": False,
        }
        frames = [
            {"camera_index": 4, "camera_id": 40, "file": "00000_cam0040.png"},
            {"camera_index": 8, "camera_id": 80, "file": "00001_cam0080.png"},
        ]
        view = {"max_width": 960, "frames": frames}
        dino = {"frames": frames}
        vote = {
            "source": "dinov3_dense_pixel_flashsplat_votes",
            "contract": "complete_dense_argmax_pixels_normalized_per_camera_v1",
            "frames": frames,
            "query_region_filtering_used": False,
            "confidence_threshold_used": False,
            "one_normalized_vote_per_camera": True,
            "v5_used": False,
            "dinov2_used": False,
            "render_max_width": 960,
        }
        self.assertEqual(validate_provenance(cache, view, dino, vote), [4, 8])
        cache["manual_camera_selection_used"] = True
        with self.assertRaisesRegex(ValueError, "manual_camera_selection_used"):
            validate_provenance(cache, view, dino, vote)

if __name__ == "__main__":
    unittest.main()
