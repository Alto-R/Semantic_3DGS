from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from semantic_palette import MIN_LABEL_DELTA_E, color_distance, label_palette  # noqa: E402


class SemanticPaletteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = {
            0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
            1: {"id": 1, "name": "chair_01", "class": "chair"},
            2: {"id": 2, "name": "chair_02", "class": "chair"},
            3: {"id": 3, "name": "sofa_01", "class": "sofa"},
            4: {"id": 4, "name": "curtain", "class": "curtain"},
            5: {"id": 5, "name": "television_01", "class": "television"},
            6: {"id": 6, "name": "window", "class": "window"},
            7: {"id": 7, "name": "rug", "class": "rug"},
        }

    def test_every_label_has_a_distinct_color(self) -> None:
        palette = label_palette(self.labels)
        label_ids = sorted(palette)
        for index, first_id in enumerate(label_ids):
            for second_id in label_ids[index + 1 :]:
                self.assertGreaterEqual(
                    color_distance(palette[first_id], palette[second_id]),
                    MIN_LABEL_DELTA_E,
                    msg=f"labels {first_id} and {second_id} are too similar",
                )

    def test_palette_is_deterministic(self) -> None:
        self.assertEqual(label_palette(self.labels), label_palette(dict(reversed(self.labels.items()))))

    def test_scene_vocabularies_meet_separation_target(self) -> None:
        for scene in ("room", "truck"):
            with self.subTest(scene=scene):
                config_path = ROOT / "configs" / f"task1_semantic_classes.{scene}.json"
                classes = json.loads(config_path.read_text(encoding="utf-8"))["classes"]
                labels = {0: {"id": 0, "name": "unlabeled", "class": "unlabeled"}}
                labels.update(
                    {
                        label_id: {
                            "id": label_id,
                            "name": f"{item['class']}_01",
                            "class": item["class"],
                        }
                        for label_id, item in enumerate(classes, 1)
                    }
                )
                palette = label_palette(labels)
                distances = [
                    color_distance(palette[first_id], palette[second_id])
                    for first_index, first_id in enumerate(sorted(palette))
                    for second_id in sorted(palette)[first_index + 1 :]
                ]
                self.assertGreaterEqual(min(distances), MIN_LABEL_DELTA_E)


if __name__ == "__main__":
    unittest.main()
