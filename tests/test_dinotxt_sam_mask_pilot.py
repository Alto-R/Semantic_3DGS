from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.dinotxt_sam_mask_pilot import (
    aggregate_prompt_logits,
    flashsplat_mask_metadata,
    load_config,
    mask_iou,
    scale_cosine_logits,
    score_sam_masks,
    select_frames,
    select_evenly_spaced_frames,
    select_target_masks,
)


class DinotxtSamMaskPilotTest(unittest.TestCase):
    def test_scale_cosine_logits_applies_learned_temperature(self) -> None:
        cosine = np.asarray([-0.25, 0.0, 0.25], dtype=np.float32)
        np.testing.assert_allclose(
            scale_cosine_logits(cosine, 20.0),
            np.asarray([-5.0, 0.0, 5.0], dtype=np.float32),
        )
        for invalid in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                scale_cosine_logits(cosine, invalid)

    def test_topk_prompt_aggregation_keeps_the_strongest_aliases(self) -> None:
        import torch

        prompt_logits = torch.tensor(
            [1.0, 3.0, 2.0, 4.0, 5.0], dtype=torch.float32
        ).reshape(5, 1, 1)
        result = aggregate_prompt_logits(
            prompt_logits,
            (3, 2),
            "topk_mean",
            2,
        )
        self.assertEqual(tuple(result.shape), (2, 1, 1))
        self.assertAlmostEqual(float(result[0, 0, 0]), 2.5)
        self.assertAlmostEqual(float(result[1, 0, 0]), 4.5)

    def config(self, directory: str):
        path = Path(directory) / "config.json"
        path.write_text(
            json.dumps(
                {
                    "target_class": "new object",
                    "classes": [
                        {"class": "new object", "prompts": ["new object"]},
                        {"class": "door", "prompts": ["door"]},
                        {"class": "wall", "prompts": ["wall"]},
                    ],
                    "selection": {
                        "min_target_probability": 0.4,
                        "min_competitor_margin": 0.1,
                        "min_target_win_fraction": 0.5,
                        "max_selected_masks": 2,
                        "containment_threshold": 0.9,
                        "nms_iou": 0.7,
                    },
                }
            ),
            encoding="utf-8",
        )
        return load_config(path)

    def test_camera_selection_preserves_requested_order(self) -> None:
        manifest = {
            "frames": [
                {"camera_index": 103, "file": "103.png"},
                {"camera_index": 91, "file": "91.png"},
            ]
        }
        selected = select_frames(manifest, [91, 103])
        self.assertEqual([frame["file"] for frame in selected], ["91.png", "103.png"])

    def test_missing_camera_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing camera indices"):
            select_frames({"frames": []}, [239])

    def test_evenly_spaced_selection_uses_manifest_order(self) -> None:
        manifest = {
            "frames": [
                {"camera_index": index, "file": f"{index}.png"}
                for index in range(10)
            ]
        }
        selected = select_evenly_spaced_frames(manifest, 4)
        self.assertEqual([frame["camera_index"] for frame in selected], [0, 3, 6, 9])

    def test_selected_masks_export_flashsplat_metadata(self) -> None:
        selected = [
            {
                "area": 12,
                "bbox_xywh": [1, 2, 3, 4],
                "sam_predicted_iou": 0.91,
                "sam_stability_score": 0.95,
                "target_probability": 0.8,
                "target_margin": 0.3,
                "target_win_fraction": 0.7,
                "best_competitor": "door",
            }
        ]
        metadata = flashsplat_mask_metadata(selected, "window_shutter")
        self.assertEqual(metadata[0]["class_name"], "window_shutter")
        self.assertEqual(metadata[0]["bbox"], [1, 2, 3, 4])
        self.assertEqual(metadata[0]["confidence"], 0.8)

    def test_mask_scores_use_complete_mask_and_competitor_margin(self) -> None:
        probabilities = np.asarray(
            [
                [[0.8, 0.7], [0.1, 0.1]],
                [[0.1, 0.2], [0.8, 0.7]],
                [[0.1, 0.1], [0.1, 0.2]],
            ],
            dtype=np.float32,
        )
        masks = [
            {
                "segmentation": np.asarray([[1, 1], [0, 0]], dtype=bool),
                "bbox": [0, 0, 2, 1],
                "predicted_iou": 0.95,
                "stability_score": 0.96,
            },
            {
                "segmentation": np.asarray([[0, 0], [1, 1]], dtype=bool),
                "bbox": [0, 1, 2, 1],
                "predicted_iou": 0.94,
                "stability_score": 0.95,
            },
        ]
        scores = score_sam_masks(masks, probabilities, ["new_object", "door", "wall"], "new_object")
        self.assertEqual(scores[0]["mask_id"], 0)
        self.assertAlmostEqual(scores[0]["target_probability"], 0.75)
        self.assertAlmostEqual(scores[0]["target_margin"], 0.60)
        self.assertEqual(scores[0]["best_competitor"], "door")
        self.assertEqual(scores[0]["target_win_fraction"], 1.0)

    def test_selection_applies_thresholds_and_nms(self) -> None:
        masks = [
            {"segmentation": np.asarray([[1, 1], [0, 0]], dtype=bool)},
            {"segmentation": np.asarray([[1, 1], [0, 0]], dtype=bool)},
            {"segmentation": np.asarray([[0, 0], [1, 1]], dtype=bool)},
        ]
        scores = [
            {"mask_id": 0, "target_probability": 0.8, "target_margin": 0.5, "target_win_fraction": 1.0},
            {"mask_id": 1, "target_probability": 0.7, "target_margin": 0.4, "target_win_fraction": 1.0},
            {"mask_id": 2, "target_probability": 0.3, "target_margin": 0.2, "target_win_fraction": 1.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            selected = select_target_masks(masks, scores, self.config(directory))
        self.assertEqual([item["mask_id"] for item in selected], [0])
        self.assertEqual(mask_iou(masks[0]["segmentation"], masks[1]["segmentation"]), 1.0)

    def test_selection_keeps_larger_passing_mask_over_contained_submask(self) -> None:
        masks = [
            {"segmentation": np.asarray([[1, 0, 0], [0, 0, 0]], dtype=bool)},
            {"segmentation": np.asarray([[1, 1, 0], [1, 1, 0]], dtype=bool)},
            {"segmentation": np.asarray([[0, 0, 1], [0, 0, 1]], dtype=bool)},
        ]
        scores = [
            {"mask_id": 0, "target_probability": 0.9, "target_margin": 0.8, "target_win_fraction": 1.0},
            {"mask_id": 2, "target_probability": 0.7, "target_margin": 0.4, "target_win_fraction": 0.8},
            {"mask_id": 1, "target_probability": 0.6, "target_margin": 0.2, "target_win_fraction": 0.6},
        ]
        with tempfile.TemporaryDirectory() as directory:
            selected = select_target_masks(masks, scores, self.config(directory))
        self.assertEqual([item["mask_id"] for item in selected], [2, 1])

    def test_config_requires_target_in_competitive_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                '{"target_class":"shutter","classes":[{"class":"door","prompts":["door"]}]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "target_class"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
