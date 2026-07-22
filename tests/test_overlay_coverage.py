from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


from scripts.task1.qa.measure_overlay_coverage import measure_directory


class OverlayCoverageTest(unittest.TestCase):
    def test_excludes_debug_border_and_counts_changed_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rgb_dir = root / "rgb"
            overlay_dir = root / "overlay"
            rgb_dir.mkdir()
            overlay_dir.mkdir()

            rgb = np.zeros((6, 6, 3), dtype=np.uint8)
            overlay = rgb.copy()
            overlay[[0, -1], :, :] = 255
            overlay[:, [0, -1], :] = 255
            overlay[1:3, 1:3, 0] = 100
            Image.fromarray(rgb).save(rgb_dir / "frame.png")
            Image.fromarray(overlay).save(overlay_dir / "frame.png")

            report = measure_directory(rgb_dir, overlay_dir, difference_threshold=2, exclude_border=1)

            self.assertEqual(report["frame_count"], 1)
            self.assertEqual(report["pixel_count"], 16)
            self.assertEqual(report["overlay_changed_pixel_count"], 4)
            self.assertEqual(report["overlay_changed_pixel_ratio"], 0.25)


if __name__ == "__main__":
    unittest.main()
