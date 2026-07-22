from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np


from scripts.task1.merge.merge_semantic_extensions import (
    merge_extensions,
    resolve_selected_classes,
    resolve_source_classes,
    validate_label_array,
)


class HybridMergeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_labels = np.asarray([0, 1, 1, 2, 0, 2], dtype=np.int32)
        self.base_items = {
            0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
            1: {"id": 1, "name": "wall", "class": "wall"},
            2: {"id": 2, "name": "floor", "class": "floor"},
        }
        self.extension_labels = np.asarray([0, 3, 0, 4, 4, 0], dtype=np.int32)
        self.extension_items = {
            0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
            3: {"id": 3, "name": "piano_01", "class": "piano"},
            4: {"id": 4, "name": "speaker_01", "class": "speaker"},
        }

    def test_selected_groups_override_only_their_masks(self) -> None:
        merged, appended, report = merge_extensions(
            self.base_labels,
            self.base_items,
            self.extension_labels,
            self.extension_items,
            ["piano"],
        )
        self.assertEqual(merged.tolist(), [0, 3, 1, 2, 0, 2])
        self.assertEqual([(item["source_label_id"], item["id"]) for item in appended], [(3, 3)])
        self.assertEqual(report["newly_labeled_count"], 0)
        self.assertEqual(report["relabeled_count"], 1)
        self.assertTrue(report["unchanged_outside_extension_masks"])

    def test_multiple_classes_append_in_requested_order(self) -> None:
        merged, appended, report = merge_extensions(
            self.base_labels,
            self.base_items,
            self.extension_labels,
            self.extension_items,
            ["speaker", "piano"],
        )
        self.assertEqual(
            [(item["class"], item["id"], item["source_label_id"]) for item in appended],
            [("speaker", 3, 4), ("piano", 4, 3)],
        )
        self.assertEqual(merged.tolist(), [0, 4, 1, 3, 3, 2])
        self.assertEqual(report["newly_labeled_count"], 1)
        self.assertEqual(report["relabeled_count"], 2)

    def test_source_class_union_becomes_one_output_group(self) -> None:
        extension_labels = np.asarray([0, 3, 5, 4, 5, 0], dtype=np.int32)
        extension_items = {
            0: {"id": 0, "name": "unlabeled", "class": "unlabeled"},
            3: {"id": 3, "name": "railroad_track", "class": "railroad_track"},
            4: {"id": 4, "name": "speaker_01", "class": "speaker"},
            5: {"id": 5, "name": "railway_platform", "class": "railway_platform"},
        }
        merged, appended, report = merge_extensions(
            self.base_labels,
            self.base_items,
            extension_labels,
            extension_items,
            ["railroad_track"],
            {"railroad_track": ["railroad_track", "railway_platform"]},
        )
        self.assertEqual(merged.tolist(), [0, 3, 3, 2, 3, 2])
        self.assertEqual(len(appended), 1)
        self.assertEqual(appended[0]["class"], "railroad_track")
        self.assertEqual(appended[0]["source_label_ids"], [3, 5])
        self.assertEqual(report["merged_group_count"], 1)
        self.assertEqual(report["changed_gaussian_count"], 3)
        self.assertEqual(report["newly_labeled_count"], 1)
        self.assertEqual(report["relabeled_count"], 2)

    def test_missing_final_group_is_a_noop(self) -> None:
        merged, appended, report = merge_extensions(
            self.base_labels,
            self.base_items,
            self.extension_labels,
            self.extension_items,
            ["guitar"],
        )
        np.testing.assert_array_equal(merged, self.base_labels)
        self.assertEqual(appended, [])
        self.assertEqual(report["no_final_group_classes"], ["guitar"])

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            merge_extensions(
                self.base_labels,
                self.base_items,
                self.extension_labels[:-1],
                self.extension_items,
                ["piano"],
            )

    def test_config_defaults_and_override_are_separate(self) -> None:
        config = {
            "candidate_classes": ["piano", "speaker"],
            "default_enabled_classes": ["piano"],
            "source_class_unions": {"speaker": ["speaker", "loudspeaker"]},
        }
        self.assertEqual(resolve_selected_classes(config, ""), ["piano"])
        self.assertEqual(resolve_selected_classes(config, "speaker"), ["speaker"])
        self.assertEqual(
            resolve_source_classes(config, ["speaker"]),
            ["speaker", "loudspeaker"],
        )
        with self.assertRaisesRegex(ValueError, "not candidates"):
            resolve_selected_classes(config, "chair")

    def test_noninteger_labels_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer labels"):
            validate_label_array(
                np.asarray([0.0, 1.0], dtype=np.float32),
                self.base_items,
                Path("labels.npy"),
            )


if __name__ == "__main__":
    unittest.main()
