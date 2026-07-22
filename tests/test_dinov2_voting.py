from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


from scripts.task1.dinov2.dinov2_voting import (
    accumulate_vote_arrays,
    flashsplat_class_rows,
    mean_class_confidences,
    semantic_evidence_fractions,
    semantic_winner_metrics,
    sparse_view_votes,
    supporting_view_counts,
    threshold_winners,
    winner_metrics,
)


class FlashSplatClassRowsTest(unittest.TestCase):
    def test_strips_unused_zero_sentinel_row(self) -> None:
        used = np.asarray(
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
            dtype=np.float32,
        )

        rows = flashsplat_class_rows(used, class_count=2, gaussian_count=2)

        np.testing.assert_array_equal(rows, used[:2])

    def test_rejects_nonzero_sentinel_row(self) -> None:
        used = np.asarray(
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.1]],
            dtype=np.float32,
        )

        with self.assertRaisesRegex(ValueError, "sentinel row"):
            flashsplat_class_rows(used, class_count=2, gaussian_count=2)

    def test_accepts_exact_class_axis(self) -> None:
        used = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        rows = flashsplat_class_rows(used, class_count=2, gaussian_count=2)

        np.testing.assert_array_equal(rows, used)


class SparseViewVoteTest(unittest.TestCase):
    def test_abstain_uses_rejected_pixel_confidence(self) -> None:
        means = mean_class_confidences(
            np.asarray([[0, 0, 4], [8, 4, 8]], dtype=np.uint16),
            np.asarray([[0.2, 0.4, 0.8], [0.7, 0.6, 0.9]], dtype=np.float32),
            np.asarray([0, 4, 8], dtype=np.uint16),
        )

        np.testing.assert_allclose(means, [0.3, 0.7, 0.8], rtol=0.0, atol=1e-6)

    def test_absent_abstain_pixels_have_zero_confidence(self) -> None:
        means = mean_class_confidences(
            np.asarray([[4, 4]], dtype=np.uint16),
            np.asarray([[0.6, 0.8]], dtype=np.float32),
            np.asarray([0, 4], dtype=np.uint16),
        )

        np.testing.assert_allclose(means, [0.0, 0.7], rtol=0.0, atol=1e-6)

    def test_confidence_inputs_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            mean_class_confidences(
                np.asarray([0, 4], dtype=np.uint16),
                np.asarray([0.2], dtype=np.float32),
                np.asarray([0, 4], dtype=np.uint16),
            )

    def test_votes_are_normalized_by_per_gaussian_visibility(self) -> None:
        used = np.asarray(
            [
                [0.0, 1.0, 1.0],
                [3.0, 3.0, 1.0],
                [1.0, 0.0, 2.0],
            ],
            dtype=np.float32,
        )
        indices, classes, weights = sparse_view_votes(
            used,
            np.asarray([0, 4, 8], dtype=np.uint16),
            np.asarray([1.0, 0.8, 0.5], dtype=np.float32),
            view_quality=1.0,
            support_threshold=0.0,
        )

        records = {(int(c), int(i)): float(w) for i, c, w in zip(indices, classes, weights)}
        self.assertAlmostEqual(records[(4, 0)], 0.75 * 0.8)
        self.assertAlmostEqual(records[(8, 0)], 0.25 * 0.5)
        self.assertAlmostEqual(records[(0, 1)], 0.25)
        self.assertAlmostEqual(records[(4, 1)], 0.75 * 0.8)

    def test_support_threshold_drops_negligible_rows(self) -> None:
        indices, classes, _ = sparse_view_votes(
            np.asarray([[1.0], [0.04]], dtype=np.float32),
            np.asarray([0, 1], dtype=np.uint16),
            np.asarray([1.0, 0.9], dtype=np.float32),
            view_quality=1.0,
            support_threshold=0.05,
        )

        self.assertEqual(indices.tolist(), [0])
        self.assertEqual(classes.tolist(), [0])


class MultiviewFusionTest(unittest.TestCase):
    def test_abstain_mass_is_included_in_agreement(self) -> None:
        votes = np.asarray([[0.6], [0.4], [0.0]], dtype=np.float32)
        raw, _, _, agreement = winner_metrics(votes)

        self.assertEqual(raw.tolist(), [0])
        self.assertAlmostEqual(float(agreement[0]), 0.6)

    def test_exact_nonzero_tie_stays_unlabeled(self) -> None:
        votes = np.asarray([[0.0], [1.0], [1.0]], dtype=np.float32)
        raw, _, _, _ = winner_metrics(votes)

        self.assertEqual(raw.tolist(), [0])

    def test_semantic_winner_separates_abstain_from_class_agreement(self) -> None:
        votes = np.asarray(
            [
                [0.60, 0.35, 0.00],
                [0.40, 0.45, 1.00],
                [0.00, 0.10, 1.00],
            ],
            dtype=np.float32,
        )

        raw, winner, second, agreement, evidence = semantic_winner_metrics(votes)

        self.assertEqual(raw.tolist(), [1, 1, 0])
        np.testing.assert_allclose(winner, [0.4, 0.45, 1.0], atol=1e-6)
        np.testing.assert_allclose(second, [0.0, 0.1, 1.0], atol=1e-6)
        np.testing.assert_allclose(agreement, [1.0, 0.45 / 0.55, 0.5], atol=1e-6)
        np.testing.assert_allclose(evidence, [0.4, 0.55 / 0.9, 1.0], atol=1e-6)

    def test_semantic_evidence_fraction_handles_empty_votes(self) -> None:
        evidence = semantic_evidence_fractions(
            np.asarray([[0.0, 0.5], [0.0, 0.5]], dtype=np.float32)
        )

        np.testing.assert_allclose(evidence, [0.0, 0.5], atol=1e-6)

    def test_separate_abstain_gate_requires_semantic_evidence(self) -> None:
        thresholded = threshold_winners(
            np.asarray([1, 1, 1], dtype=np.uint16),
            np.asarray([0.8, 0.8, 0.8], dtype=np.float32),
            np.asarray([2, 2, 2], dtype=np.uint16),
            min_views=2,
            min_agreement=0.5,
            semantic_evidence=np.asarray([0.49, 0.50, 0.70], dtype=np.float32),
            min_semantic_evidence=0.5,
        )

        self.assertEqual(thresholded.tolist(), [0, 1, 1])

    def test_positive_semantic_evidence_threshold_requires_array(self) -> None:
        with self.assertRaisesRegex(ValueError, "semantic_evidence is required"):
            threshold_winners(
                np.asarray([1], dtype=np.uint16),
                np.asarray([0.8], dtype=np.float32),
                np.asarray([2], dtype=np.uint16),
                min_views=2,
                min_agreement=0.5,
                min_semantic_evidence=0.5,
            )

    def test_repeating_consistent_views_preserves_agreement(self) -> None:
        once = np.asarray([[0.2], [0.8], [0.0]], dtype=np.float32)
        repeated = once * 50.0
        raw_once, _, _, agreement_once = winner_metrics(once)
        raw_repeated, _, _, agreement_repeated = winner_metrics(repeated)

        self.assertEqual(raw_once.tolist(), raw_repeated.tolist())
        self.assertAlmostEqual(float(agreement_once[0]), float(agreement_repeated[0]))

    def test_accumulation_and_distinct_view_threshold(self) -> None:
        matrix = np.zeros((3, 2), dtype=np.float32)
        accumulate_vote_arrays(
            matrix,
            np.asarray([0, 1], dtype=np.uint32),
            np.asarray([1, 2], dtype=np.uint16),
            np.asarray([0.7, 0.8], dtype=np.float32),
        )
        raw, _, _, agreement = winner_metrics(matrix)
        thresholded = threshold_winners(
            raw,
            agreement,
            np.asarray([2, 1], dtype=np.uint16),
            min_views=2,
            min_agreement=0.5,
        )

        self.assertEqual(raw.tolist(), [1, 2])
        self.assertEqual(thresholded.tolist(), [1, 0])

    def test_supporting_views_count_each_file_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(2):
                path = Path(directory) / f"view_{index}.npz"
                np.savez_compressed(
                    path,
                    indices=np.asarray([0, 0, 1], dtype=np.uint32),
                    class_ids=np.asarray([1, 2, 2], dtype=np.uint16),
                    weights=np.ones((3,), dtype=np.float32),
                )
                paths.append(path)

            counts = supporting_view_counts(paths, np.asarray([1, 2], dtype=np.uint16))

        self.assertEqual(counts.tolist(), [2, 2])


if __name__ == "__main__":
    unittest.main()
