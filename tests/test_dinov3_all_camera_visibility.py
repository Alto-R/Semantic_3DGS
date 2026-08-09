from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.audit_all_camera_visibility import (
    summarize_visibility,
    update_visibility_state,
    visibility_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_all_camera_visibility_scene.sbatch"
)


class VisibilitySupportTest(unittest.TestCase):
    def test_collapses_all_flashsplat_rows(self) -> None:
        used = np.asarray([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(
            visibility_support(used, 3),
            np.asarray([1.0, 3.0, 2.0], dtype=np.float32),
        )

    def test_rejects_invalid_support(self) -> None:
        with self.assertRaisesRegex(ValueError, "wrong Gaussian"):
            visibility_support(np.zeros((2, 2), dtype=np.float32), 3)
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            visibility_support(np.asarray([0.0, -1.0]), 2)

    def test_updates_counts_and_keeps_first_best_camera(self) -> None:
        count = np.zeros(3, dtype=np.uint16)
        total = np.zeros(3, dtype=np.float32)
        maximum = np.zeros(3, dtype=np.float32)
        first = np.full(3, -1, dtype=np.int32)
        best = np.full(3, -1, dtype=np.int32)
        first_visible = update_visibility_state(
            np.asarray([0.0, 2.0, 1.0], dtype=np.float32),
            camera_index=4,
            support_threshold=0.0,
            view_count=count,
            total_support=total,
            max_support=maximum,
            first_camera_index=first,
            best_camera_index=best,
        )
        second_visible = update_visibility_state(
            np.asarray([3.0, 1.0, 4.0], dtype=np.float32),
            camera_index=7,
            support_threshold=0.0,
            view_count=count,
            total_support=total,
            max_support=maximum,
            first_camera_index=first,
            best_camera_index=best,
        )
        np.testing.assert_array_equal(first_visible, [False, True, True])
        np.testing.assert_array_equal(second_visible, [True, True, True])
        np.testing.assert_array_equal(count, [1, 2, 2])
        np.testing.assert_array_equal(first, [7, 4, 4])
        np.testing.assert_array_equal(best, [7, 4, 7])
        np.testing.assert_array_equal(total, [3.0, 3.0, 5.0])

    def test_threshold_is_strictly_greater(self) -> None:
        count = np.zeros(2, dtype=np.uint16)
        total = np.zeros(2, dtype=np.float32)
        maximum = np.zeros(2, dtype=np.float32)
        first = np.full(2, -1, dtype=np.int32)
        best = np.full(2, -1, dtype=np.int32)
        visible = update_visibility_state(
            np.asarray([0.05, 0.051], dtype=np.float32),
            camera_index=0,
            support_threshold=0.05,
            view_count=count,
            total_support=total,
            max_support=maximum,
            first_camera_index=first,
            best_camera_index=best,
        )
        np.testing.assert_array_equal(visible, [False, True])
        np.testing.assert_allclose(total, [0.0, 0.051])


class VisibilitySummaryTest(unittest.TestCase):
    def test_reports_zero_one_and_multiview_coverage(self) -> None:
        summary = summarize_visibility(
            np.asarray([0, 1, 2, 4], dtype=np.uint16),
            np.asarray([0.0, 1.0, 2.0, 5.0], dtype=np.float32),
            np.asarray([0.0, 1.0, 1.5, 3.0], dtype=np.float32),
        )
        self.assertEqual(summary["observed_gaussian_count"], 3)
        self.assertEqual(summary["unobserved_gaussian_count"], 1)
        self.assertEqual(summary["single_view_gaussian_count"], 1)
        self.assertEqual(summary["multiview_gaussian_count"], 2)
        self.assertEqual(summary["view_count_histogram"], {"0": 1, "1": 1, "2": 1, "4": 1})

    def test_scheduler_is_all_camera_and_report_only(self) -> None:
        source = SCHEDULER.read_text(encoding="utf-8")
        self.assertIn("all_reconstruction_cameras_in_cameras_json_order", source)
        self.assertIn("semantic_inference_used=0", source)
        self.assertIn("semantic_labels_written=0", source)
        self.assertIn("test ! -e \"${AUDIT_DIR}/gaussian_labels.npy\"", source)
        self.assertIn("unexpectedly wrote a PLY", source)


if __name__ == "__main__":
    unittest.main()
