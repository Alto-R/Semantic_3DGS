from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.task1.dinov3.propagate_dense_labels_from_region_seeds import (
    COMPONENT_ACCEPTED,
    COMPONENT_CONFLICTING,
    COMPONENT_UNSEEDED,
    adaptive_voxel_size,
    component_seed_guard,
    main,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_seeded_dense_scene.sbatch"
)


class SeedGuardTest(unittest.TestCase):
    def test_matching_seed_accepts_component_and_fills_only_unlabeled(self) -> None:
        components = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
        seeds = np.asarray([4, 0, 4, 0, 0], dtype=np.int32)
        statuses, fill, matching, conflicting = component_seed_guard(
            components,
            seeds,
            candidate_class_id=4,
        )
        np.testing.assert_array_equal(
            statuses, [COMPONENT_ACCEPTED, COMPONENT_UNSEEDED]
        )
        np.testing.assert_array_equal(fill, [False, True, False, False, False])
        np.testing.assert_array_equal(matching, [2, 0])
        np.testing.assert_array_equal(conflicting, [0, 0])

    def test_any_conflicting_seed_rejects_the_whole_component(self) -> None:
        components = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
        seeds = np.asarray([4, 0, 9, 4, 0], dtype=np.int32)
        statuses, fill, matching, conflicting = component_seed_guard(
            components,
            seeds,
            candidate_class_id=4,
        )
        np.testing.assert_array_equal(
            statuses, [COMPONENT_CONFLICTING, COMPONENT_ACCEPTED]
        )
        np.testing.assert_array_equal(fill, [False, False, False, False, True])
        np.testing.assert_array_equal(matching, [1, 1])
        np.testing.assert_array_equal(conflicting, [1, 0])

    def test_component_count_can_include_empty_component_ids(self) -> None:
        statuses, fill, *_ = component_seed_guard(
            np.asarray([0, 2], dtype=np.int64),
            np.asarray([3, 0], dtype=np.int32),
            candidate_class_id=3,
            component_count=4,
        )
        np.testing.assert_array_equal(
            statuses,
            [
                COMPONENT_ACCEPTED,
                COMPONENT_UNSEEDED,
                COMPONENT_UNSEEDED,
                COMPONENT_UNSEEDED,
            ],
        )
        np.testing.assert_array_equal(fill, [False, False])


class AdaptiveGeometryTest(unittest.TestCase):
    def test_voxel_size_is_scale_adaptive_and_globally_bounded(self) -> None:
        median, voxel = adaptive_voxel_size(
            np.log(np.asarray([[0.02, 0.01, 0.01], [0.04, 0.01, 0.01]])),
            voxel_scale_multiplier=4.0,
            min_voxel_size=0.01,
            max_voxel_size=0.20,
        )
        self.assertAlmostEqual(median, 0.03)
        self.assertAlmostEqual(voxel, 0.12)

        _, capped = adaptive_voxel_size(
            np.log(np.asarray([[1.0, 0.5, 0.25]])),
            voxel_scale_multiplier=4.0,
            min_voxel_size=0.01,
            max_voxel_size=0.20,
        )
        self.assertAlmostEqual(capped, 0.20)


class SeededDenseEndToEndTest(unittest.TestCase):
    def test_main_preserves_all_seeds_and_fills_only_accepted_component(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_value:
            temp_dir = Path(temp_dir_value)
            source_ply = temp_dir / "point_cloud.ply"
            fields = ["x", "y", "z", "scale_0", "scale_1", "scale_2"]
            header = [
                "ply",
                "format binary_little_endian 1.0",
                "element vertex 6",
                *[f"property float {name}" for name in fields],
                "end_header",
                "",
            ]
            log_scale = float(np.log(0.01))
            coordinates = [
                (0.000, 0.0, 0.0),
                (0.005, 0.0, 0.0),
                (0.010, 0.0, 0.0),
                (1.000, 0.0, 0.0),
                (1.005, 0.0, 0.0),
                (2.000, 0.0, 0.0),
            ]
            payload = b"".join(
                struct.pack("<6f", x, y, z, log_scale, log_scale, log_scale)
                for x, y, z in coordinates
            )
            source_ply.write_bytes("\n".join(header).encode("ascii") + payload)

            seed_labels_path = temp_dir / "seed.npy"
            dense_labels_path = temp_dir / "dense.npy"
            np.save(seed_labels_path, np.asarray([4, 0, 0, 0, 9, 7], dtype=np.int32))
            np.save(dense_labels_path, np.asarray([4, 4, 4, 4, 4, 0], dtype=np.int32))

            base_summary = {
                "scene": "synthetic",
                "source_ply": str(source_ply),
                "vertex_count": 6,
                "v5_used": False,
                "dinov2_used": False,
            }
            seed_summary = temp_dir / "region_vote_summary.json"
            seed_summary.write_text(
                json.dumps(
                    {
                        **base_summary,
                        "source": "dinov3_query_regions_direct_multiview_majority",
                        "contract": "per_gaussian_one_vote_per_camera_strict_majority_v1",
                    }
                ),
                encoding="utf-8",
            )
            dense_summary = temp_dir / "dense_vote_summary.json"
            dense_summary.write_text(
                json.dumps(
                    {
                        **base_summary,
                        "source": "dinov3_dense_pixel_flashsplat_soft_class_mass",
                        "contract": "complete_dense_pixels_unique_soft_argmax_v1",
                    }
                ),
                encoding="utf-8",
            )
            output_dir = temp_dir / "output"
            argv = [
                "propagate_dense_labels_from_region_seeds",
                "--seed-labels",
                str(seed_labels_path),
                "--seed-summary",
                str(seed_summary),
                "--dense-labels",
                str(dense_labels_path),
                "--dense-summary",
                str(dense_summary),
                "--ontology",
                str(PROJECT_ROOT / "configs" / "ade20k_to_project.json"),
                "--source-ply",
                str(source_ply),
                "--output-dir",
                str(output_dir),
                "--scene",
                "synthetic",
                "--no-semantic-ply",
            ]
            with mock.patch.object(sys, "argv", argv):
                main()

            np.testing.assert_array_equal(
                np.load(output_dir / "gaussian_project_class_ids.npy"),
                [4, 4, 4, 0, 9, 7],
            )
            np.testing.assert_array_equal(
                np.load(output_dir / "propagated_fill_mask.npy"),
                [False, True, True, False, False, False],
            )
            summary = json.loads(
                (output_dir / "propagation_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["region_seed_gaussian_count"], 3)
            self.assertEqual(summary["added_gaussian_count"], 2)
            self.assertEqual(summary["accepted_component_count"], 1)
            self.assertEqual(summary["conflicting_component_count"], 1)


class SeededDenseSchedulerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCHEDULER.read_text(encoding="utf-8")

    def test_scheduler_uses_only_completed_dinov3_label_sources(self) -> None:
        self.assertIn("REGION_OUTPUT_NAME", self.text)
        self.assertIn("DENSE_OUTPUT_NAME", self.text)
        self.assertIn("region_vote_summary.json", self.text)
        self.assertIn("dense_vote_summary.json", self.text)
        self.assertIn("propagate_dense_labels_from_region_seeds", self.text)
        self.assertNotIn("dinov3_segment_views", self.text)
        self.assertNotIn("lift_dense_view_votes", self.text)
        self.assertNotIn("fuse_dense_view_votes", self.text)

    def test_scheduler_exposes_global_geometry_without_component_size_gate(self) -> None:
        self.assertIn('VOXEL_SCALE_MULTIPLIER="${VOXEL_SCALE_MULTIPLIER:-4.0}"', self.text)
        self.assertIn('MIN_VOXEL_SIZE="${MIN_VOXEL_SIZE:-0.01}"', self.text)
        self.assertIn('MAX_VOXEL_SIZE="${MAX_VOXEL_SIZE:-0.20}"', self.text)
        self.assertNotIn("MIN_COMPONENT", self.text)

    def test_scheduler_writes_end_to_end_review_artifacts(self) -> None:
        self.assertIn("semantic_point_cloud_supersplat_debug.ply", self.text)
        self.assertIn("render_auto_label_overlays", self.text)
        self.assertIn("visible_overlay_coverage.json", self.text)
        self.assertIn("semantic_labels.png", self.text)


if __name__ == "__main__":
    unittest.main()
