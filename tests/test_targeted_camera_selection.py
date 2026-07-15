from __future__ import annotations

import sys
import argparse
import json
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "task1"))

from select_targeted_cameras import (  # noqa: E402
    seed_command,
    coverage_by_camera,
    projection_rejection_reasons,
    projection_thresholds,
    select_low_coverage_diverse_views,
)


def camera(index: int, x: float) -> dict:
    return {
        "id": index,
        "img_name": f"camera_{index}",
        "position": [x, 0.0, 0.0],
        "rotation": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "width": 100,
        "height": 100,
        "fx": 50.0,
        "fy": 50.0,
    }


class TargetedCameraSelectionTest(unittest.TestCase):
    def test_seed_selection_is_even_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "view_manifest.json"
            output = root / "seed_manifest.json"
            indices = root / "seed_indices.txt"
            source.write_text(
                json.dumps(
                    {
                        "frames": [
                            {"file": f"{index:05d}.png", "camera_index": index}
                            for index in range(10)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            seed_command(
                argparse.Namespace(
                    source_manifest=source,
                    output=output,
                    indices_output=indices,
                    count=4,
                )
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["selected_camera_indices"], [0, 3, 6, 9])
            self.assertEqual(indices.read_text(encoding="utf-8").strip(), "0,3,6,9")

    def test_projection_screen_is_anchored_to_known_safe_baseline(self) -> None:
        baseline = [
            {
                "front_ratio": 0.8,
                "inside_ratio": 0.3,
                "z_q01": 0.2,
                "scale_over_z_q99": 0.04,
                "scale_over_z_max": 0.5,
            },
            {
                "front_ratio": 0.9,
                "inside_ratio": 0.4,
                "z_q01": 0.3,
                "scale_over_z_q99": 0.05,
                "scale_over_z_max": 0.4,
            },
        ]
        thresholds = projection_thresholds(baseline, margin=2.0)

        self.assertAlmostEqual(thresholds["min_front_ratio"], 0.4)
        self.assertAlmostEqual(thresholds["max_scale_over_z_q99"], 0.1)
        unsafe = {
            "front_ratio": 0.7,
            "inside_ratio": 0.2,
            "z_q01": 0.2,
            "scale_over_z_q99": 0.2,
            "scale_over_z_max": 0.6,
        }
        self.assertEqual(projection_rejection_reasons(unsafe, thresholds), ["scale_over_z_q99"])

    def test_low_coverage_candidates_are_preferred(self) -> None:
        cameras = [camera(index, float(index)) for index in range(4)]
        selected, records = select_low_coverage_diverse_views(
            {1: 0.20, 2: 0.25, 3: 0.90},
            baseline_indices=[0],
            cameras=cameras,
            additional_count=2,
            low_coverage_quantile=0.67,
            coverage_weight=0.8,
            novelty_weight=0.2,
        )

        self.assertEqual(set(selected), {1, 2})
        self.assertEqual({int(record["camera_index"]) for record in records}, {1, 2})

    def test_pose_novelty_breaks_equal_coverage_tie(self) -> None:
        cameras = [camera(0, 0.0), camera(1, 0.1), camera(2, 4.0)]
        selected, _records = select_low_coverage_diverse_views(
            {1: 0.5, 2: 0.5},
            baseline_indices=[0],
            cameras=cameras,
            additional_count=1,
            low_coverage_quantile=1.0,
            coverage_weight=0.0,
            novelty_weight=1.0,
        )

        self.assertEqual(selected, [2])

    def test_coverage_frames_map_back_to_camera_indices(self) -> None:
        manifest = {
            "frames": [
                {"file": "00000_cam0007.png", "camera_index": 4},
                {"file": "00001_cam0009.png", "camera_index": 8},
            ]
        }
        coverage = {
            "frames": [
                {"file": "00000_cam0007.png", "overlay_changed_pixel_ratio": 0.3},
                {"file": "00001_cam0009.png", "overlay_changed_pixel_ratio": 0.7},
            ]
        }

        self.assertEqual(coverage_by_camera(manifest, coverage), {4: 0.3, 8: 0.7})


if __name__ == "__main__":
    unittest.main()
