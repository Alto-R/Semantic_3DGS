from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.merge_additional_vote_manifests import (
    merge_manifests,
)
from scripts.task1.dinov3.select_black_evidence_cameras import (
    black_evidence_requirements,
    greedy_black_evidence_selection,
)


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "scripts" / "task1" / "dinov3" / "select_black_evidence_cameras.py"
MERGER = ROOT / "scripts" / "task1" / "dinov3" / "merge_additional_vote_manifests.py"
SCHEDULER = ROOT / "scripts" / "slurm" / (
    "slurm_task1_dinov3_black_evidence_expansion_scene.sbatch"
)


def packed_rows(rows: list[np.ndarray], gaussian_count: int) -> np.ndarray:
    width = (gaussian_count + 7) // 8
    matrix = np.zeros((len(rows), width), dtype=np.uint8)
    for row, bits in enumerate(rows):
        matrix[row] = np.packbits(bits, bitorder="little")
    return matrix


class BlackEvidenceRequirementTest(unittest.TestCase):
    def test_requirements_cover_black_gaussians_only(self) -> None:
        total = np.asarray([0, 1, 2, 2, 3], dtype=np.uint16)
        maximum = np.asarray([0, 1, 1, 2, 2], dtype=np.uint8)
        black = np.asarray([True, True, True, False, True])
        required = black_evidence_requirements(total, maximum, black)
        np.testing.assert_array_equal(required, [2, 1, 1, 0, 0])

    def test_rejects_misaligned_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "aligned"):
            black_evidence_requirements(
                np.zeros((3,), dtype=np.uint16),
                np.zeros((2,), dtype=np.uint8),
                np.zeros((3,), dtype=bool),
            )


class GreedyBlackEvidenceSelectionTest(unittest.TestCase):
    def test_excludes_used_cameras_and_reaches_coverage(self) -> None:
        gaussian_count = 8
        # Camera 0 covers Gaussians 0-3, camera 1 covers 2-5, camera 2 covers 4-7.
        rows = [
            np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
            np.asarray([0, 0, 1, 1, 1, 1, 0, 0], dtype=bool),
            np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=bool),
        ]
        matrix = packed_rows(rows, gaussian_count)
        cameras = [
            {
                "id": index,
                "position": [float(index), 0.0, 0.0],
                "rotation": np.eye(3),
            }
            for index in range(3)
        ]
        required = np.asarray([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint16)
        result = greedy_black_evidence_selection(
            matrix,
            gaussian_count=gaussian_count,
            camera_indices=np.asarray([0, 1, 2], dtype=np.int32),
            used_camera_indices=np.asarray([0], dtype=np.int32),
            required=required,
            cameras=cameras,
            target_coverage=0.99,
        )
        selected = result["selected_camera_indices"].tolist()
        self.assertNotIn(0, selected)
        self.assertEqual(result["achieved_units"], result["target_units"])

    def test_rejects_no_unused_cameras(self) -> None:
        gaussian_count = 4
        matrix = packed_rows(
            [np.asarray([1, 1, 1, 1], dtype=bool)], gaussian_count
        )
        cameras = [{"id": 0, "position": [0.0, 0.0, 0.0], "rotation": np.eye(3)}]
        with self.assertRaisesRegex(ValueError, "no unused reconstruction cameras"):
            greedy_black_evidence_selection(
                matrix,
                gaussian_count=gaussian_count,
                camera_indices=np.asarray([0], dtype=np.int32),
                used_camera_indices=np.asarray([0], dtype=np.int32),
                required=np.ones((gaussian_count,), dtype=np.uint16),
                cameras=cameras,
            )


class MergeManifestTest(unittest.TestCase):
    def test_merges_frames_and_verifies_common_fields(self) -> None:
        existing = {
            "source": "dinov3_dense_pixel_flashsplat_votes",
            "contract": "complete_dense_argmax_pixels_normalized_per_camera_v1",
            "gaussian_count": 10,
            "ply_path": "/same.ply",
            "render_max_width": 960,
            "camera_count": 1,
            "frames": [
                {"camera_index": 1, "camera_id": 1, "vote_file": "view_votes/a.npz"}
            ],
        }
        new = {
            "source": "dinov3_dense_pixel_flashsplat_votes",
            "contract": "complete_dense_argmax_pixels_normalized_per_camera_v1",
            "gaussian_count": 10,
            "ply_path": "/same.ply",
            "render_max_width": 960,
            "camera_count": 1,
            "frames": [
                {"camera_index": 2, "camera_id": 2, "vote_file": "view_votes/b.npz"}
            ],
        }
        merged = merge_manifests(existing, new)
        self.assertEqual(len(merged["frames"]), 2)
        self.assertEqual(merged["camera_count"], 2)

    def test_rejects_overlapping_cameras(self) -> None:
        existing = {
            "source": "dinov3_dense_pixel_flashsplat_votes",
            "contract": "complete_dense_argmax_pixels_normalized_per_camera_v1",
            "gaussian_count": 10,
            "ply_path": "/same.ply",
            "render_max_width": 960,
            "camera_count": 1,
            "frames": [
                {"camera_index": 1, "camera_id": 1, "vote_file": "view_votes/a.npz"}
            ],
        }
        new = {
            "source": "dinov3_dense_pixel_flashsplat_votes",
            "contract": "complete_dense_argmax_pixels_normalized_per_camera_v1",
            "gaussian_count": 10,
            "ply_path": "/same.ply",
            "render_max_width": 960,
            "camera_count": 1,
            "frames": [
                {"camera_index": 1, "camera_id": 1, "vote_file": "view_votes/b.npz"}
            ],
        }
        with self.assertRaisesRegex(ValueError, "repeats an existing additional camera"):
            merge_manifests(existing, new)


class ExpansionSchedulerContractTest(unittest.TestCase):
    def test_scheduler_is_automatic_report_only_and_reuses_caches(self) -> None:
        source = SCHEDULER.read_text(encoding="utf-8")
        for expected in (
            "Manual camera selection is not accepted",
            "camera_selection=automatic_current_black_visibility_and_pose_diversity",
            "coverage_target=0.99",
            "accepted_gaussian_labels_written=0",
            "semantic_ply_written=0",
            "SOURCE_HARD_AUDIT_OUTPUT_NAME",
            "SOURCE_VISIBILITY_OUTPUT_NAME",
            "SOURCE_RECOVERY_OUTPUT_NAME",
            "select_black_evidence_cameras",
            "merge_additional_vote_manifests",
            "recover_detected_abstentions",
            "observed_black_component_graph_audit",
        ):
            self.assertIn(expected, source)
        disallowed = "v" + "5"
        for path in (SELECTOR, MERGER, SCHEDULER):
            self.assertNotIn(disallowed, path.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
