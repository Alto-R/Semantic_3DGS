from __future__ import annotations

import unittest

import numpy as np

from scripts.task1.dinov3.compose_oov_multiclass_votes import (
    camera_winners,
    choose_oov_replacements,
    compose_oov_pixel_ids,
    compose_project_class_map,
    resolve_oov_label_ids,
)


class Dinov3OovMulticlassTest(unittest.TestCase):
    def test_composed_map_only_overrides_oov_pixels(self) -> None:
        lookup = np.arange(256, dtype=np.uint16)
        raw = np.asarray([[1, 2], [3, 4]], dtype=np.uint8)
        oov = np.asarray([[0, 151], [0, 0]], dtype=np.uint16)
        result = compose_project_class_map(raw, lookup, oov)
        np.testing.assert_array_equal(result, [[1, 151], [3, 4]])

    def test_overlapping_masks_use_higher_confidence_class(self) -> None:
        masks = np.asarray(
            [
                [[1, 1, 0], [0, 0, 0]],
                [[0, 1, 1], [0, 0, 0]],
            ],
            dtype=bool,
        )
        pixel_ids, counts = compose_oov_pixel_ids(
            masks,
            {
                "masks": [
                    {"class": "shutter", "confidence": 0.4},
                    {"class": "screen", "confidence": 0.8},
                ]
            },
            {"shutter": 151, "screen": 152},
        )
        np.testing.assert_array_equal(pixel_ids, [[151, 152, 152], [0, 0, 0]])
        self.assertEqual(counts, {"shutter": 1, "screen": 2})

    def test_camera_winner_tie_is_deterministic(self) -> None:
        indices, classes, weights = camera_winners(
            np.asarray([0, 0, 1, 1], dtype=np.uint32),
            np.asarray([2, 1, 2, 1], dtype=np.uint16),
            np.asarray([0.4, 0.4, 0.2, 0.8], dtype=np.float32),
            2,
        )
        np.testing.assert_array_equal(indices, [0, 1])
        np.testing.assert_array_equal(classes, [1, 1])
        np.testing.assert_allclose(weights, [0.4, 0.8])

    def test_oov_majority_replaces_only_qualifying_base_labels(self) -> None:
        labels, evidence = choose_oov_replacements(
            np.asarray([15, 0, 9], dtype=np.int32),
            np.asarray([4, 4, 4], dtype=np.uint16),
            np.asarray([[3, 1, 2]], dtype=np.uint16),
            np.asarray([[3, 1, 2]], dtype=np.uint16),
            np.asarray([[2.4, 0.7, 1.6]], dtype=np.float32),
            np.asarray([1, 3, 1], dtype=np.uint16),
            np.asarray([151], dtype=np.uint16),
            min_visible_views=3,
            min_oov_winner_views=2,
            min_oov_winner_share=0.50,
            min_oov_mass_share=0.35,
            min_oov_positive_views=2,
            winner_margin=0.05,
        )
        np.testing.assert_array_equal(labels, [151, 0, 151])
        np.testing.assert_array_equal(evidence["fill"], [True, False, True])

    def test_oov_ids_are_automatic_and_noncolliding(self) -> None:
        manifest = {
            "target_class": "window shutter",
            "frames": [{"masks": [{"class": "window_shutter"}]}],
        }
        result = resolve_oov_label_ids(
            {
                0: {"class": "unlabeled"},
                15: {"class": "door"},
            },
            manifest,
            "",
            "",
        )
        self.assertEqual(result, {"window_shutter": 16})


if __name__ == "__main__":
    unittest.main()
