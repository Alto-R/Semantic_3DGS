from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.dinov2_second_source import (
    SOURCE,
    align_dinov2_evidence,
    collapse_dinov2_camera,
    exclude_dinov2_camera,
    load_dinov2_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def write_dinov2_manifest(
    root: Path,
    frames: list[dict],
    *,
    gaussian_count: int,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": SOURCE,
        "gaussian_count": gaussian_count,
        "camera_count": len(frames),
        "frames": frames,
    }
    path = root / "vote_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class CollapseDinov2CameraTest(unittest.TestCase):
    def test_unique_winner_and_mass_without_normalization(self) -> None:
        indices = np.asarray([0, 0, 1, 1, 2], dtype=np.uint32)
        classes = np.asarray([15, 20, 15, 15, 0], dtype=np.uint16)
        weights = np.asarray([0.4, 0.3, 0.5, 0.2, 0.9], dtype=np.float32)
        winner, mass = collapse_dinov2_camera(
            indices,
            classes,
            weights,
            gaussian_count=4,
            class_count=25,
        )
        np.testing.assert_array_equal(winner, [15, 15, 0, 0])
        np.testing.assert_allclose(mass, [0.4, 0.7, 0.0, 0.0])

    def test_exact_tie_abstains(self) -> None:
        winner, mass = collapse_dinov2_camera(
            np.asarray([0, 0], dtype=np.uint32),
            np.asarray([15, 20], dtype=np.uint16),
            np.asarray([0.5, 0.5], dtype=np.float32),
            gaussian_count=1,
            class_count=25,
        )
        self.assertEqual(int(winner[0]), 0)
        self.assertEqual(float(mass[0]), 0.0)

    def test_rejects_invalid_arrays(self) -> None:
        with self.assertRaisesRegex(ValueError, "aligned"):
            collapse_dinov2_camera(
                np.asarray([0], dtype=np.uint32),
                np.asarray([15, 20], dtype=np.uint16),
                np.asarray([0.5, 0.5], dtype=np.float32),
                gaussian_count=1,
                class_count=25,
            )
        with self.assertRaisesRegex(ValueError, "outside the model"):
            collapse_dinov2_camera(
                np.asarray([4], dtype=np.uint32),
                np.asarray([15], dtype=np.uint16),
                np.asarray([0.5], dtype=np.float32),
                gaussian_count=4,
                class_count=25,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            collapse_dinov2_camera(
                np.asarray([0], dtype=np.uint32),
                np.asarray([15], dtype=np.uint16),
                np.asarray([-0.5], dtype=np.float32),
                gaussian_count=1,
                class_count=25,
            )


class LoadAndAlignTest(unittest.TestCase):
    def test_load_and_align_by_camera_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez_compressed(
                root / "camera_5.npz",
                indices=np.asarray([0], dtype=np.uint32),
                class_ids=np.asarray([15], dtype=np.uint16),
                weights=np.asarray([0.5], dtype=np.float32),
            )
            manifest_path = write_dinov2_manifest(
                root,
                [
                    {
                        "camera_index": 5,
                        "camera_id": 5,
                        "file": "camera_5.png",
                        "vote_file": "camera_5.npz",
                    }
                ],
                gaussian_count=2,
            )
            dinov2 = load_dinov2_evidence(
                manifest_path, gaussian_count=2, class_count=25
            )
            self.assertEqual(int(dinov2[0]["camera_id"]), 5)
            aligned = align_dinov2_evidence(
                [
                    {
                        "camera_index": 5,
                        "camera_id": 5,
                        "winners": np.zeros((2,), dtype=np.uint16),
                    }
                ],
                dinov2,
            )
            self.assertEqual(int(aligned[0]["camera_id"]), 5)

    def test_alignment_rejects_missing_counterpart(self) -> None:
        with self.assertRaisesRegex(ValueError, "no DINOv2 counterpart"):
            align_dinov2_evidence(
                [{"camera_index": 7, "camera_id": 7}],
                [{"camera_index": 5, "camera_id": 5}],
            )

    def test_exclude_dinov2_camera_removes_heldout_row(self) -> None:
        rows = [
            {"camera_index": 1, "camera_id": 1},
            {"camera_index": 2, "camera_id": 2},
            {"camera_index": 3, "camera_id": 3},
        ]
        remaining = exclude_dinov2_camera(rows, 2)
        self.assertEqual(
            [int(row["camera_id"]) for row in remaining], [1, 3]
        )


class SchedulerContractTest(unittest.TestCase):
    def test_no_v5_references_in_second_source_module(self) -> None:
        module = ROOT / "scripts" / "task1" / "dinov3" / "dinov2_second_source.py"
        self.assertNotIn("v" + "5", module.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
