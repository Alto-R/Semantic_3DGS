from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.recover_dinov3_abstentions_pipeline import (
    camera_reliabilities,
    load_dinov3_votes,
    recover_weighted,
)
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    STATUS_ACCEPTED,
    consensus_statistics,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_end_to_end_recovery_scene.sbatch"
)


class Dinov3RecoveryPipelineTest(unittest.TestCase):
    def test_recovery_only_fills_multicamera_weighted_majority(self) -> None:
        gaussian_count = 6
        class_count = 5
        winners = np.asarray(
            [
                [1, 1, 1, 1, 1, 0],   # camera 0
                [1, 2, 0, 1, 1, 0],   # camera 1
                [1, 2, 0, 2, 1, 0],   # camera 2
            ],
            dtype=np.uint16,
        )
        mass = np.ones_like(winners, dtype=np.float32)
        evidence = [
            {
                "camera_index": camera,
                "camera_id": camera,
                "winners": winners[camera],
                "mass": mass[camera],
            }
            for camera in range(3)
        ]
        counts = np.zeros((class_count + 1, gaussian_count), dtype=np.uint16)
        for item in evidence:
            supported = np.flatnonzero(item["winners"])
            counts[
                item["winners"][supported].astype(np.int64), supported
            ] += np.uint16(1)
        statistics = consensus_statistics(counts, chunk_size=10)
        status = statistics["status"]
        self.assertEqual(int(status[0]), STATUS_ACCEPTED)
        reliabilities = camera_reliabilities(statistics, evidence, z=1.96)
        for value in reliabilities.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        recovered, source_codes = recover_weighted(
            evidence,
            reliabilities,
            status,
            gaussian_count=gaussian_count,
            class_count=class_count,
        )
        np.testing.assert_array_equal(source_codes[[0, 3, 4]], [1, 1, 1])
        self.assertEqual(int(source_codes[5]), 0)
        self.assertEqual(int(source_codes[2]), 0)
        self.assertEqual(int(recovered[2]), 0)
        self.assertEqual(int(recovered[5]), 0)

    def test_loads_dense_dinov3_votes(self) -> None:
        gaussian_count = 3
        class_count = 3
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vote_path = root / "vote.npz"
            np.savez_compressed(
                vote_path,
                indices=np.asarray([0, 1, 0], dtype=np.uint32),
                class_ids=np.asarray([1, 2, 2], dtype=np.uint16),
                weights=np.asarray([0.6, 1.0, 0.4], dtype=np.float32),
            )
            manifest_path = root / "vote_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "source": "dinov3_dense_pixel_flashsplat_votes",
                        "contract": "complete_dense_argmax_pixels_normalized_per_camera_v1",
                        "gaussian_count": gaussian_count,
                        "query_region_filtering_used": False,
                        "confidence_threshold_used": False,
                        "one_normalized_vote_per_camera": True,
                        "v5_used": False,
                        "dinov2_used": False,
                        "frames": [
                            {
                                "file": "cam.png",
                                "camera_index": 3,
                                "camera_id": 30,
                                "vote_file": "vote.npz",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            evidence, counts = load_dinov3_votes(
                manifest_path,
                gaussian_count=gaussian_count,
                class_count=class_count,
            )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["camera_index"], 3)
        np.testing.assert_array_equal(evidence[0]["winners"], [1, 2, 0])
        np.testing.assert_array_equal(counts[1], [1, 0, 0])
        np.testing.assert_array_equal(counts[2], [0, 1, 0])

    def test_load_rejects_wrong_vote_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "vote_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "source": "not_dinov3",
                        "contract": "complete_dense_argmax_pixels_normalized_per_camera_v1",
                        "gaussian_count": 2,
                        "query_region_filtering_used": False,
                        "confidence_threshold_used": False,
                        "one_normalized_vote_per_camera": True,
                        "v5_used": False,
                        "dinov2_used": False,
                        "frames": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "wrong contract"):
                load_dinov3_votes(manifest_path, gaussian_count=2, class_count=2)

    def test_scheduler_is_general_and_end_to_end(self) -> None:
        source = SCHEDULER.read_text(encoding="utf-8")
        self.assertNotIn("playroom or drjohnson", source)
        for marker in (
            "render_task1_views",
            "dinov3_segment_views",
            "lift_dense_view_votes",
            "recover_dinov3_abstentions_pipeline",
            "validate_task1_outputs",
            "CONFIG_ONLY",
        ):
            self.assertIn(marker, source)


if __name__ == "__main__":
    unittest.main()
