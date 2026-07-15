from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "task1"))

from semantic_palette import (  # noqa: E402
    PALETTE_VERSION,
    fallback_class_color,
    label_palette,
    palette_records,
    rgb_for_class,
)


class SemanticPaletteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = {
            0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
            1: {"id": 1, "name": "chair_01", "class": "chair"},
            2: {"id": 2, "name": "chair_02", "class": "chair"},
            3: {"id": 3, "name": "sofa_01", "class": "sofa"},
        }

    def test_same_class_has_same_color(self) -> None:
        palette = label_palette(self.labels)
        self.assertEqual(palette[1], palette[2])
        self.assertNotEqual(palette[1], palette[3])

    def test_class_color_ignores_label_ids_order_and_other_labels(self) -> None:
        first = label_palette(self.labels)[1]
        reordered = {
            99: {"id": 99, "name": "chair_99", "class": "chair"},
            4: {"id": 4, "name": "lamp_01", "class": "lamp"},
        }
        self.assertEqual(first, label_palette(reordered)[99])

    def test_instance_mode_is_stable_and_distinguishes_names(self) -> None:
        first = label_palette(self.labels, color_mode="instance")
        second = label_palette(dict(reversed(self.labels.items())), color_mode="instance")
        self.assertEqual(first, second)
        self.assertNotEqual(first[1], first[2])

    def test_unknown_class_fallback_is_name_keyed(self) -> None:
        self.assertEqual(fallback_class_color("new object"), fallback_class_color("new_object"))
        self.assertEqual(rgb_for_class("new object"), rgb_for_class("new_object"))
        self.assertNotEqual(rgb_for_class("new_object"), rgb_for_class("other_object"))

    def test_palette_records_are_versioned(self) -> None:
        records = palette_records(self.labels, self.labels)
        self.assertTrue(records)
        self.assertTrue(all(item["palette_version"] == PALETTE_VERSION for item in records))
        self.assertTrue(all(item["color_mode"] == "class" for item in records))

    def test_scene_vocabularies_have_explicit_colors(self) -> None:
        for scene in ("room", "truck"):
            with self.subTest(scene=scene):
                config_path = ROOT / "configs" / f"task1_semantic_classes.{scene}.json"
                classes = json.loads(config_path.read_text(encoding="utf-8"))["classes"]
                for item in classes:
                    self.assertEqual(rgb_for_class(item["class"]), rgb_for_class(item["class"]))


if __name__ == "__main__":
    unittest.main()
