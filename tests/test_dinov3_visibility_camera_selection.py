from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.select_visibility_cameras import (
    POPCOUNT,
    build_threshold_report,
    compact_threshold_report,
    greedy_multicover_order,
    packed_intersection_counts,
    packed_intersection_popcount,
    validate_visibility_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_visibility_camera_selection_scene.sbatch"
)


def pack(rows: list[list[int]]) -> np.ndarray:
    return np.packbits(np.asarray(rows, dtype=np.uint8), axis=1, bitorder="little")


class VisibilityCameraSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = np.asarray(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
            ],
            dtype=np.uint8,
        )
        self.packed = np.packbits(self.rows, axis=1, bitorder="little")
        self.view_count = self.rows.sum(axis=0, dtype=np.uint16)
        self.camera_indices = np.asarray([10, 20, 30, 40], dtype=np.int32)

    def test_intersection_popcount(self) -> None:
        np.testing.assert_array_equal(
            POPCOUNT[[0, 1, 3, 7, 255]],
            [0, 1, 2, 3, 8],
        )
        self.assertEqual(
            packed_intersection_popcount(self.packed[0], self.packed[1]),
            1,
        )
        np.testing.assert_array_equal(
            packed_intersection_counts(self.packed, self.packed[1]),
            [1, 2, 1, 0],
        )

    def test_greedy_order_prioritizes_single_then_second_coverage(self) -> None:
        result = greedy_multicover_order(
            self.packed,
            gaussian_count=4,
            camera_indices=self.camera_indices,
            global_view_count=self.view_count,
        )
        np.testing.assert_array_equal(result["camera_rows"], [0, 2, 1, 3])
        np.testing.assert_array_equal(result["camera_indices"], [10, 30, 20, 40])
        self.assertEqual(
            result["phases"],
            ["single_coverage", "single_coverage", "second_coverage", "second_coverage"],
        )
        np.testing.assert_array_equal(result["single_covered"], [2, 4, 4, 4])
        np.testing.assert_array_equal(result["double_covered"], [0, 0, 2, 3])
        np.testing.assert_array_equal(result["zero_remaining"], [2, 0, 0, 0])

    def test_thresholds_report_observable_and_multiview_denominators(self) -> None:
        result = greedy_multicover_order(
            self.packed,
            gaussian_count=4,
            camera_indices=self.camera_indices,
            global_view_count=self.view_count,
        )
        thresholds = build_threshold_report(result)
        single = thresholds["single_view_coverage_of_observable"]
        double_observed = thresholds["two_view_coverage_of_observable"]
        double_capable = thresholds["two_view_coverage_of_multiview_capable"]
        self.assertEqual(single["100"]["selected_camera_count"], 2)
        self.assertEqual(single["100"]["camera_indices"], [10, 30])
        self.assertIsNone(double_observed["90"]["selected_camera_count"])
        self.assertEqual(double_capable["90"]["selected_camera_count"], 4)

    def test_ties_preserve_source_row_order(self) -> None:
        rows = pack([[1, 0], [0, 1]])
        result = greedy_multicover_order(
            rows,
            gaussian_count=2,
            camera_indices=np.asarray([8, 3], dtype=np.int32),
            global_view_count=np.asarray([1, 1], dtype=np.uint16),
        )
        np.testing.assert_array_equal(result["camera_indices"], [8, 3])

    def test_rejects_padding_and_view_count_mismatches(self) -> None:
        bad_padding = self.packed.copy()
        bad_padding[0, -1] |= np.uint8(0b10000000)
        with self.assertRaisesRegex(ValueError, "padding"):
            validate_visibility_matrix(
                bad_padding,
                gaussian_count=4,
                camera_indices=self.camera_indices,
                global_view_count=self.view_count,
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_visibility_matrix(
                self.packed,
                gaussian_count=4,
                camera_indices=self.camera_indices,
                global_view_count=np.asarray([1, 1, 1, 1], dtype=np.uint16),
            )

    def test_scheduler_is_automatic_and_report_only(self) -> None:
        source = SCHEDULER.read_text(encoding="utf-8")
        self.assertIn("single_coverage_first_then_second_coverage", source)
        self.assertIn("manual_camera_selection_used=0", source)
        self.assertIn("semantic_inference_used=0", source)
        self.assertIn("renderer_used=0", source)
        self.assertIn("test ! -e \"${SELECTION_DIR}/gaussian_labels.npy\"", source)
        self.assertIn("unexpectedly wrote a PLY", source)

    def test_saved_threshold_prefixes_are_not_required_in_console_summary(self) -> None:
        result = greedy_multicover_order(
            self.packed,
            gaussian_count=4,
            camera_indices=self.camera_indices,
            global_view_count=self.view_count,
        )
        thresholds = build_threshold_report(result)
        compact = compact_threshold_report(thresholds)
        self.assertNotIn("camera_indices", str(compact))
        self.assertEqual(
            compact["single_view_coverage_of_observable"]["100"][
                "selected_camera_count"
            ],
            2,
        )


if __name__ == "__main__":
    unittest.main()
