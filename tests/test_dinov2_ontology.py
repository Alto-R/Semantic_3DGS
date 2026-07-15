from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "task1"))

from dinov2_ontology import load_ontology  # noqa: E402


class Dinov2OntologyTest(unittest.TestCase):
    def test_tracked_ontology_is_complete_and_identity_preserving(self) -> None:
        ontology = load_ontology(ROOT / "configs" / "ade20k_to_project.json")

        self.assertEqual(ontology.class_count, 150)
        self.assertEqual([item.ade_id for item in ontology.classes], list(range(150)))
        self.assertEqual([item.project_id for item in ontology.classes], list(range(1, 151)))
        self.assertEqual(ontology.ade_to_project[0], 1)
        self.assertEqual(ontology.ade_to_project[149], 150)
        self.assertEqual(ontology.ade_to_project[255], 0)
        self.assertEqual(ontology.classes[127].project_class, "bicycle")
        expected_stuff_ids = {
            0, 1, 2, 3, 4, 5, 6, 9, 11, 13, 16, 17, 21, 25, 26, 28, 29,
            34, 40, 46, 48, 51, 52, 54, 59, 60, 61, 63, 68, 77, 79, 84,
            91, 94, 96, 99, 100, 101, 105, 106, 109, 113, 114, 117, 122,
            128, 131, 140, 141, 145,
        }
        actual_stuff_ids = {item.ade_id for item in ontology.classes if item.kind == "stuff"}
        self.assertEqual(actual_stuff_ids, expected_stuff_ids)

    def test_missing_ade_id_is_rejected(self) -> None:
        source = ROOT / "configs" / "ade20k_to_project.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        data["classes"] = data["classes"][:-1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ontology.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "every ADE20K id"):
                load_ontology(path)


if __name__ == "__main__":
    unittest.main()
