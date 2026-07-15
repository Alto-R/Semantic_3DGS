from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "task1"))

from dinov2_voting import (  # noqa: E402
    accumulate_vote_arrays,
    flashsplat_class_rows,
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
