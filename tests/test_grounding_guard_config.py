from __future__ import annotations

import unittest

from scripts.task1.grounding.build_grounding_guard_config import build_guard_config


class GroundingGuardConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dino_label_map = {
            "scene": "room",
            "labels": [
                {"id": 0, "class": "unlabeled", "type": "stuff", "gaussian_count": 200},
                {
                    "id": 1,
                    "name": "television_receiver_01",
                    "class": "television_receiver",
                    "type": "thing",
                    "gaussian_count": 100,
                },
                {"id": 2, "class": "chair", "type": "thing", "gaussian_count": 80},
                {"id": 3, "class": "stool", "type": "thing", "gaussian_count": 30},
                {"id": 4, "class": "floor", "type": "stuff", "gaussian_count": 150},
                {"id": 5, "class": "clock", "type": "thing", "gaussian_count": 20},
            ],
        }
        self.scene_config = {
            "assignment_priority": ["piano", "television", "speaker", "chair", "floor"],
            "classes": [
                {"class": "piano", "type": "thing", "prompts": ["piano", "upright piano"]},
                {
                    "class": "television",
                    "type": "thing",
                    "prompts": ["television", "tv", "television screen"],
                },
                {
                    "class": "speaker",
                    "type": "thing",
                    "prompts": ["speaker", "floor speaker"],
                },
                {"class": "chair", "type": "thing", "prompts": ["chair", "stool"]},
                {"class": "floor", "type": "stuff", "prompts": ["floor", "flooring"]},
            ],
        }
        self.extension_config = {
            "scene": "room",
            "candidate_classes": ["piano", "speaker"],
            "default_enabled_classes": ["piano", "speaker"],
        }

    def build(self, include_classes: str = "", max_prompt_words: int = 100) -> dict[str, object]:
        return build_guard_config(
            self.dino_label_map,
            self.scene_config,
            self.extension_config,
            include_classes,
            max_prompt_words,
        )

    def test_present_dino_classes_become_non_extension_guards(self) -> None:
        config = self.build()
        classes = {item["class"]: item for item in config["classes"]}

        self.assertEqual(config["selected_extension_classes"], ["piano", "speaker"])
        self.assertEqual(classes["piano"]["role"], "extension")
        self.assertEqual(classes["speaker"]["role"], "extension")
        self.assertEqual(classes["television"]["role"], "guard")
        self.assertEqual(classes["television"]["dinov2_classes"], ["television_receiver"])
        self.assertEqual(
            classes["television"]["prompts"],
            ["television", "tv", "television screen"],
        )
        self.assertEqual(classes["chair"]["dinov2_classes"], ["chair", "stool"])
        self.assertEqual(classes["chair"]["gaussian_count"], 110)
        self.assertEqual(classes["clock"]["source"], "dinov2_label_map")
        self.assertNotIn("unlabeled", classes)
        priorities = config["assignment_priority"]
        self.assertLess(priorities.index("television"), priorities.index("piano"))
        self.assertLess(priorities.index("chair"), priorities.index("speaker"))

    def test_override_changes_extensions_but_preserves_guards(self) -> None:
        config = self.build(include_classes="speaker")
        classes = {item["class"]: item for item in config["classes"]}

        self.assertEqual(config["selected_extension_classes"], ["speaker"])
        self.assertNotIn("piano", classes)
        self.assertEqual(classes["television"]["role"], "guard")
        self.assertEqual(classes["chair"]["role"], "guard")

    def test_prompt_budget_never_drops_extensions(self) -> None:
        extension_only_words = 6
        config = self.build(max_prompt_words=extension_only_words)
        classes = {item["class"]: item for item in config["classes"]}

        self.assertIn("piano", classes)
        self.assertIn("speaker", classes)
        self.assertEqual(config["prompt_word_count"], extension_only_words)
        self.assertEqual(
            set(config["dropped_guard_classes"]),
            {"television", "chair", "floor", "clock"},
        )

    def test_scene_mismatch_is_rejected(self) -> None:
        extension_config = dict(self.extension_config, scene="train")
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_guard_config(
                self.dino_label_map,
                self.scene_config,
                extension_config,
                "",
                100,
            )

    def test_unknown_extension_override_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not candidates"):
            self.build(include_classes="guitar")


if __name__ == "__main__":
    unittest.main()
