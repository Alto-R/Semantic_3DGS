from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_grounded_sam_masks import ClassSpec, class_from_phrase  # noqa: E402


class GroundedClassResolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.specs = [
            ClassSpec(name="guitar", prompts=("guitar", "acoustic guitar")),
            ClassSpec(name="indoor_plant", prompts=("indoor plant", "potted plant")),
            ClassSpec(name="chair", prompts=("chair", "armchair")),
        ]

    def test_exact_prompt_inside_phrase(self) -> None:
        self.assertEqual(class_from_phrase("wooden chair", self.specs), "chair")

    def test_partial_multiword_prompt_maps_to_configured_class(self) -> None:
        self.assertEqual(class_from_phrase("acoustic", self.specs), "guitar")
        self.assertEqual(class_from_phrase("indoor", self.specs), "indoor_plant")

    def test_unconfigured_phrase_is_rejected(self) -> None:
        self.assertEqual(class_from_phrase("television", self.specs), "unknown")


if __name__ == "__main__":
    unittest.main()
