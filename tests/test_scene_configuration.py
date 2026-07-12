from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from semantic_palette import CLASS_COLORS  # noqa: E402
from summarize_task1_semantic_run import build_summary  # noqa: E402


class SceneConfigurationTest(unittest.TestCase):
    def test_train_class_config_is_complete_and_prioritized(self) -> None:
        path = ROOT / "configs" / "task1_semantic_classes.train.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        classes = config["classes"]
        class_names = [item["class"] for item in classes]

        self.assertEqual(len(class_names), len(set(class_names)))
        self.assertEqual(set(config["assignment_priority"]), set(class_names))
        self.assertIn("train", class_names)
        self.assertIn("railroad_track", class_names)
        thresholds = {
            item["class"]: item["min_assigned_gaussians"]
            for item in classes
            if "min_assigned_gaussians" in item
        }
        self.assertEqual(thresholds, {"building": 8000, "sky": 8000})
        for item in classes:
            self.assertIn(item["type"], {"thing", "stuff"})
            self.assertTrue(item["prompts"])

    def test_train_focus_classes_have_stable_palette_colors(self) -> None:
        self.assertIn("train", CLASS_COLORS)
        self.assertIn("railroad_track", CLASS_COLORS)

    def test_pipeline_summary_uses_scene_neutral_focus_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            summary = build_summary(Path(temporary_directory), "train_vs_track")

        exports = summary["stages"]["exports"]
        self.assertEqual(exports["focus_name"], "train_vs_track")
        self.assertTrue(exports["focus_debug_ply"]["path"].endswith("train_vs_track_supersplat_debug.ply"))
        self.assertTrue(exports["focus_overlay_dir"]["path"].endswith("train_vs_track"))


if __name__ == "__main__":
    unittest.main()
