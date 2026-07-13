from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "task1"))

from generate_grounded_sam_masks import ClassSpec, class_from_phrase  # noqa: E402


class GroundedClassResolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.specs = [
            ClassSpec(name="guitar", prompts=("guitar", "acoustic guitar")),
            ClassSpec(name="indoor_plant", prompts=("indoor plant", "potted plant")),
            ClassSpec(name="chair", prompts=("chair", "armchair")),
            ClassSpec(name="sofa", prompts=("sofa", "couch")),
            ClassSpec(name="television", prompts=("television", "tv")),
            ClassSpec(name="media_console", prompts=("television stand", "entertainment center")),
            ClassSpec(name="table", prompts=("table", "desk")),
            ClassSpec(name="speaker", prompts=("speaker", "floor speaker")),
            ClassSpec(name="floor", prompts=("floor",)),
        ]

    def test_exact_prompt_inside_phrase(self) -> None:
        self.assertEqual(class_from_phrase("wooden chair", self.specs), "chair")

    def test_partial_multiword_prompt_maps_to_configured_class(self) -> None:
        self.assertEqual(class_from_phrase("acoustic", self.specs), "guitar")
        self.assertEqual(class_from_phrase("indoor", self.specs), "indoor_plant")

    def test_unconfigured_phrase_is_rejected(self) -> None:
        self.assertEqual(class_from_phrase("wardrobe", self.specs), "unknown")

    def test_unrelated_exact_classes_are_rejected_as_ambiguous(self) -> None:
        self.assertEqual(class_from_phrase("chair armchair sofa couch", self.specs), "unknown")
        self.assertEqual(class_from_phrase("television stand table desk", self.specs), "unknown")

    def test_shorter_prompt_inside_specific_prompt_is_ignored(self) -> None:
        self.assertEqual(class_from_phrase("floor speaker", self.specs), "speaker")
        self.assertEqual(class_from_phrase("television stand", self.specs), "media_console")


if __name__ == "__main__":
    unittest.main()
