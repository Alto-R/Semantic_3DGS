from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from scripts.task1.common.semantic_palette import CLASS_COLORS
from scripts.task1.qa.summarize_task1_semantic_run import build_summary


class SceneConfigurationTest(unittest.TestCase):
    def assert_scene_config(self, scene: str) -> dict[str, object]:
        path = ROOT / "configs" / f"task1_semantic_classes.{scene}.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        classes = config["classes"]
        class_names = [item["class"] for item in classes]
        self.assertEqual(len(class_names), len(set(class_names)))
        self.assertEqual(set(config["assignment_priority"]), set(class_names))
        for item in classes:
            self.assertIn(item["type"], {"thing", "stuff"})
            self.assertTrue(item["prompts"])
        return config

    def test_train_class_config_is_complete_and_prioritized(self) -> None:
        config = self.assert_scene_config("train")
        classes = config["classes"]
        class_names = [item["class"] for item in classes]

        self.assertIn("train", class_names)
        self.assertIn("railroad_track", class_names)
        self.assertNotIn("railway_platform", class_names)
        track = next(item for item in classes if item["class"] == "railroad_track")
        self.assertIn("railway platform", track["prompts"])
        class_types = {item["class"]: item["type"] for item in classes}
        self.assertEqual(class_types["building"], "stuff")

    def test_room_class_config_and_palette_are_complete(self) -> None:
        config = self.assert_scene_config("room")
        classes = config["classes"]
        class_names = {item["class"] for item in classes}
        self.assertIn("sofa", class_names)
        self.assertIn("table", class_names)
        self.assertIn("guitar", class_names)
        self.assertTrue(class_names.issubset(CLASS_COLORS))

    def test_room_hybrid_extension_config_has_reviewed_defaults(self) -> None:
        path = ROOT / "configs" / "task1_hybrid_extensions.room.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["scene"], "room")
        self.assertEqual(
            config["candidate_classes"],
            ["piano", "speaker", "guitar"],
        )
        self.assertEqual(config["default_enabled_classes"], ["piano", "speaker"])
        self.assertTrue(
            set(config["default_enabled_classes"]).issubset(config["candidate_classes"])
        )

    def test_reviewed_extension_configs_cover_selected_scene_classes(self) -> None:
        expected = {
            "playroom": {
                "candidates": ["stroller"],
                "defaults": ["stroller"],
            },
            "train": {
                "candidates": ["train", "railroad_track"],
                "defaults": ["train"],
            },
        }
        for scene, values in expected.items():
            with self.subTest(scene=scene):
                path = ROOT / "configs" / f"task1_hybrid_extensions.{scene}.json"
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(config["version"], 1)
                self.assertEqual(config["scene"], scene)
                self.assertEqual(config["candidate_classes"], values["candidates"])
                self.assertEqual(config["default_enabled_classes"], values["defaults"])
                self.assertTrue(
                    set(config["default_enabled_classes"]).issubset(
                        config["candidate_classes"]
                    )
                )

    def test_prominent_extension_configs_match_visual_review(self) -> None:
        expected = {
            "bonsai": {
                "bonsai_tree": [
                    "bonsai tree",
                    "bonsai plant",
                    "ornamental bonsai tree",
                ],
                "electronic_keyboard": [
                    "electronic keyboard",
                    "digital piano",
                    "keyboard instrument",
                ],
            },
            "counter": {
                "oven_mitt": ["oven mitt", "oven glove", "kitchen mitt"],
                "mixing_bowl": [
                    "mixing bowl",
                    "metal mixing bowl",
                    "stainless steel bowl",
                ],
                "water_filter_pitcher": [
                    "water filter pitcher",
                    "water filter jug",
                    "water pitcher",
                ],
                "onion": ["onion", "yellow onion", "red onion"],
            },
            "kitchen": {
                "toy_bulldozer": [
                    "toy bulldozer",
                    "LEGO bulldozer",
                    "toy construction vehicle",
                    "toy front loader",
                ],
            },
        }
        reviewed_defaults = {
            "bonsai": ["bonsai_tree"],
            "counter": ["mixing_bowl"],
            "kitchen": [],
        }
        ontology = json.loads(
            (ROOT / "configs" / "ade20k_to_project.json").read_text(encoding="utf-8")
        )
        ontology_classes = {item["project_class"] for item in ontology["classes"]}
        excluded_classes = {
            "tablecloth",
            "rolling_pin",
            "egg_carton",
            "placemat",
            "countertop",
        }

        for scene, expected_prompts in expected.items():
            with self.subTest(scene=scene):
                semantic = self.assert_scene_config(scene)
                classes = semantic["classes"]
                class_names = [item["class"] for item in classes]
                self.assertEqual(class_names, list(expected_prompts))
                self.assertEqual(semantic["assignment_priority"], class_names)
                self.assertTrue(all(item["type"] == "thing" for item in classes))
                self.assertEqual(
                    {item["class"]: item["prompts"] for item in classes},
                    expected_prompts,
                )
                self.assertTrue(set(class_names).isdisjoint(ontology_classes))
                self.assertTrue(set(class_names).isdisjoint(excluded_classes))

                extension_path = (
                    ROOT / "configs" / f"task1_hybrid_extensions.{scene}.json"
                )
                extension = json.loads(extension_path.read_text(encoding="utf-8"))
                self.assertEqual(extension["version"], 1)
                self.assertEqual(extension["scene"], scene)
                self.assertEqual(extension["candidate_classes"], class_names)
                self.assertEqual(
                    extension["default_enabled_classes"], reviewed_defaults[scene]
                )

        for scene in ("flowers", "garden"):
            with self.subTest(no_active_extension_scene=scene):
                self.assertFalse(
                    (
                        ROOT / "configs" / f"task1_semantic_classes.{scene}.json"
                    ).exists()
                )
                self.assertFalse(
                    (
                        ROOT / "configs" / f"task1_hybrid_extensions.{scene}.json"
                    ).exists()
                )

    def test_custom_v2_configs_are_non_ade_and_disabled_pending_review(self) -> None:
        expected = {
            "bonsai": {
                "electronic_keyboard": [
                    "electronic musical keyboard",
                    "digital piano with black and white keys",
                    "full-size piano keyboard",
                    "electronic keyboard on a keyboard stand",
                ],
            },
            "counter": {
                "oven_mitt": [
                    "oven mitt with thumb",
                    "padded oven glove",
                    "quilted heat-resistant oven mitt",
                    "pair of oven mitts",
                ],
                "cutlery": [
                    "metal cutlery",
                    "silverware",
                    "forks knives and spoons",
                    "table cutlery",
                ],
            },
            "kitchen": {
                "oven_mitt": [
                    "oven mitt with thumb",
                    "padded oven glove",
                    "quilted heat-resistant oven mitt",
                    "pair of oven mitts",
                ],
                "toy_bulldozer": [
                    "toy bulldozer with front blade",
                    "tracked toy bulldozer",
                    "small model bulldozer",
                    "construction-set bulldozer",
                ],
            },
        }
        ontology = json.loads(
            (ROOT / "configs" / "ade20k_to_project.json").read_text(encoding="utf-8")
        )
        ontology_classes = {item["project_class"] for item in ontology["classes"]}
        skipped_native_or_rejected = {
            "bench",
            "chair",
            "food",
            "garden_statue",
            "plant_pot",
            "pot",
            "produce",
            "snack_package",
        }

        for scene, expected_prompts in expected.items():
            with self.subTest(scene=scene):
                semantic_path = (
                    ROOT / "configs" / f"task1_semantic_classes.{scene}.v2.json"
                )
                semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
                classes = semantic["classes"]
                class_names = [item["class"] for item in classes]
                self.assertEqual(class_names, list(expected_prompts))
                self.assertEqual(semantic["assignment_priority"], class_names)
                self.assertEqual(
                    {item["class"]: item["prompts"] for item in classes},
                    expected_prompts,
                )
                self.assertTrue(all(item["type"] == "thing" for item in classes))
                self.assertTrue(set(class_names).isdisjoint(ontology_classes))
                self.assertTrue(set(class_names).isdisjoint(skipped_native_or_rejected))

                extension_path = (
                    ROOT / "configs" / f"task1_hybrid_extensions.{scene}.v2.json"
                )
                extension = json.loads(extension_path.read_text(encoding="utf-8"))
                self.assertEqual(extension["version"], 2)
                self.assertEqual(extension["scene"], scene)
                self.assertEqual(extension["candidate_classes"], class_names)
                self.assertEqual(extension["default_enabled_classes"], [])

        self.assertFalse(
            (ROOT / "configs" / "task1_semantic_classes.garden.v2.json").exists()
        )
        self.assertFalse(
            (ROOT / "configs" / "task1_hybrid_extensions.garden.v2.json").exists()
        )

    def test_train_track_unions_both_cached_source_masks(self) -> None:
        path = ROOT / "configs" / "task1_hybrid_extensions.train.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            config["source_class_unions"],
            {"railroad_track": ["railroad_track", "railway_platform"]},
        )

    def test_train_track_is_an_explicit_extension(self) -> None:
        path = ROOT / "configs" / "task1_hybrid_extensions.train.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("railroad_track", config["candidate_classes"])
        self.assertNotIn("railroad_track", config["default_enabled_classes"])

    def test_rejected_extensions_are_not_selectable(self) -> None:
        rejected = {
            "room": "media_console",
            "train": "railway_platform",
        }
        for scene, rejected_class in rejected.items():
            with self.subTest(scene=scene):
                path = ROOT / "configs" / f"task1_hybrid_extensions.{scene}.json"
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertNotIn(rejected_class, config["candidate_classes"])

    def test_truck_class_config_is_complete_and_prioritized(self) -> None:
        config = self.assert_scene_config("truck")
        class_names = {item["class"] for item in config["classes"]}
        self.assertTrue({"truck", "wheel", "building", "ground", "sky"}.issubset(class_names))
        class_types = {item["class"]: item["type"] for item in config["classes"]}
        self.assertEqual(class_types["truck"], "thing")
        self.assertEqual(class_types["wheel"], "thing")
        self.assertEqual(class_types["building"], "stuff")

    def test_remaining_eyenavgs_scene_configs_are_complete(self) -> None:
        expected_classes = {
            "drjohnson": {"chair", "table", "radiator"},
            "playroom": {"toy", "table", "monitor", "stroller", "staircase"},
            "stump": {"tree"},
            "treehill": {"tree", "bench"},
        }

        for scene, required_classes in expected_classes.items():
            with self.subTest(scene=scene):
                config = self.assert_scene_config(scene)
                class_names = {item["class"] for item in config["classes"]}
                self.assertTrue(required_classes.issubset(class_names))
                self.assertTrue(class_names.issubset(CLASS_COLORS))
                if scene == "stump":
                    self.assertTrue({"tree_stump", "log"}.isdisjoint(class_names))

    def test_train_focus_classes_have_stable_palette_colors(self) -> None:
        self.assertIn("train", CLASS_COLORS)
        self.assertIn("railroad_track", CLASS_COLORS)

    def test_pipeline_summary_uses_scene_neutral_focus_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            summary = build_summary(Path(temporary_directory), "train_vs_track")

        exports = summary["stages"]["exports"]
        self.assertNotIn("debug_color_ply", exports)
        self.assertEqual(exports["focus_name"], "train_vs_track")
        self.assertTrue(exports["focus_debug_ply"]["path"].endswith("train_vs_track_supersplat_debug.ply"))
        self.assertTrue(exports["focus_overlay_dir"]["path"].endswith("train_vs_track"))


if __name__ == "__main__":
    unittest.main()
