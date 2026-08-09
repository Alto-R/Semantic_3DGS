from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.selected_view_cache import (
    CACHE_CONTRACT,
    compare_hard_caches,
    finalize_selected_cache,
    resolve_selected_prefix,
    validate_probability_resume,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_selected_view_cache_scene.sbatch"
)


class SelectedViewCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.selection = self.root / "selection"
        self.output = self.root / "output"
        self.model.mkdir()
        self.selection.mkdir()
        cameras = [{"id": value} for value in range(4)]
        (self.model / "cameras.json").write_text(json.dumps(cameras), encoding="utf-8")
        order = np.asarray([2, 0, 3, 1], dtype=np.int32)
        rows = np.asarray([2, 0, 3, 1], dtype=np.int32)
        single = np.asarray([5, 8, 9, 10], dtype=np.int64)
        double = np.asarray([1, 7, 10, 10], dtype=np.int64)
        np.save(self.selection / "greedy_camera_indices.npy", order)
        np.savez_compressed(
            self.selection / "coverage_curve.npz",
            selected_camera_count=np.arange(1, 5, dtype=np.int32),
            camera_rows=rows,
            camera_indices=order,
            single_covered=single,
            double_covered=double,
        )
        steps = [
            {
                "selection_step": offset + 1,
                "visibility_row": int(rows[offset]),
                "camera_index": int(order[offset]),
                "camera_id": int(order[offset]),
                "single_covered": int(single[offset]),
                "double_covered": int(double[offset]),
            }
            for offset in range(4)
        ]
        report = {
            "contract": "deterministic_greedy_visibility_multicover_v1",
            "camera_count": 4,
            "source_max_width": 960,
            "globally_observable_gaussian_count": 10,
            "globally_multiview_capable_gaussian_count": 10,
            "manual_camera_selection_used": False,
            "thresholds": {
                "two_view_coverage_of_multiview_capable": {
                    "99": {
                        "target_percent": 99,
                        "required_gaussian_count": 10,
                        "selected_camera_count": 3,
                        "camera_indices": [2, 0, 3],
                    }
                }
            },
            "selection_steps": steps,
        }
        (self.selection / "camera_selection_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_resolves_first_threshold_reaching_prefix(self) -> None:
        resolved = resolve_selected_prefix(self.selection, self.model)
        self.assertEqual(resolved["selected_camera_count"], 3)
        self.assertEqual(resolved["camera_indices"], [2, 0, 3])
        self.assertEqual(resolved["achieved_double_covered"], 10)
        self.assertFalse(resolved["manual_camera_selection_used"])

    def test_rejects_tampered_threshold_prefix(self) -> None:
        path = self.selection / "camera_selection_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["thresholds"]["two_view_coverage_of_multiview_capable"]["99"][
            "camera_indices"
        ] = [2, 3, 0]
        path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match ranking"):
            resolve_selected_prefix(self.selection, self.model)

    def test_rejects_nonminimal_prefix(self) -> None:
        path = self.selection / "camera_selection_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        record = report["thresholds"]["two_view_coverage_of_multiview_capable"]["99"]
        record["selected_camera_count"] = 4
        record["camera_indices"] = [2, 0, 3, 1]
        path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not the first"):
            resolve_selected_prefix(self.selection, self.model)

    def test_rejects_selection_from_a_different_render_width(self) -> None:
        path = self.selection / "camera_selection_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["source_max_width"] = 0
        path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "expected render width"):
            resolve_selected_prefix(self.selection, self.model)

    def _write_completed_cache(self, *, probabilities: bool = False) -> None:
        view_dir = self.output / "stages" / "01_real_camera_views"
        rgb_dir = view_dir / "rgb_renders"
        segment_dir = view_dir / "dinov3_segments"
        overlay_dir = view_dir / "dinov3_overlays"
        probability_dir = view_dir / "dinov3_probabilities"
        directories = [rgb_dir, segment_dir, overlay_dir]
        if probabilities:
            directories.append(probability_dir)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        frames = []
        dino_frames = []
        for output_index, camera_index in enumerate((2, 0, 3)):
            filename = f"{output_index:05d}_cam{camera_index:04d}.png"
            segment_file = f"dinov3_segments/{Path(filename).stem}.npz"
            overlay_file = f"dinov3_overlays/{filename}"
            (rgb_dir / filename).write_bytes(b"rgb")
            shape = (3, 5)
            np.savez_compressed(
                view_dir / segment_file,
                class_id=np.zeros(shape, dtype=np.uint8),
                confidence=np.full(shape, 0.5, dtype=np.float16),
                max_softmax_probability=np.full(shape, 0.6, dtype=np.float16),
                top1_top2_margin=np.full(shape, 0.2, dtype=np.float16),
                normalized_entropy_confidence=np.full(
                    shape, 0.4, dtype=np.float16
                ),
            )
            (view_dir / overlay_file).write_bytes(b"overlay")
            frame = {
                "file": filename,
                "camera_index": camera_index,
                "camera_id": camera_index,
                "render_height": shape[0],
                "render_width": shape[1],
            }
            frames.append(frame)
            dino_frames.append(
                {**frame, "segment_file": segment_file, "overlay_file": overlay_file}
            )
            if probabilities:
                probability_file = f"dinov3_probabilities/{Path(filename).stem}.npy"
                values = np.zeros((150, *shape), dtype=np.float16)
                values[0] = 1.0
                np.save(view_dir / probability_file, values, allow_pickle=False)
                dino_frames[-1]["probability_file"] = probability_file
        (view_dir / "view_manifest.json").write_text(
            json.dumps({"camera_count": 3, "max_width": 960, "frames": frames}),
            encoding="utf-8",
        )
        (view_dir / "dinov3_manifest.json").write_text(
            json.dumps(
                {
                    "contract": (
                        "raw_ade20k_class_probabilities_and_relative_margin_confidence_v3"
                        if probabilities
                        else "raw_ade20k_class_and_relative_margin_confidence_v2"
                    ),
                    "model": {
                        "precision": "bfloat16",
                        "crop_size": 512,
                        "stride": 384,
                        "checkpoint_loading": {"mode": "local_mmap"},
                    },
                    "frames": dino_frames,
                }
            ),
            encoding="utf-8",
        )

    def test_finalize_validates_exact_cache_and_writes_provenance(self) -> None:
        self._write_completed_cache()
        report = finalize_selected_cache(self.selection, self.model, self.output)
        self.assertEqual(report["contract"], CACHE_CONTRACT)
        self.assertEqual(report["camera_indices"], [2, 0, 3])
        self.assertFalse(report["semantic_vote_lifting_run"])
        self.assertTrue(
            (self.output / "selection" / "selected_view_cache_report.json").is_file()
        )
        np.testing.assert_array_equal(
            np.load(self.output / "selection" / "selected_camera_indices.npy"),
            [2, 0, 3],
        )

    def test_finalize_rejects_wrong_render_order(self) -> None:
        self._write_completed_cache()
        path = self.output / "stages" / "01_real_camera_views" / "view_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["frames"][0]["camera_index"] = 0
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match selected prefix"):
            finalize_selected_cache(self.selection, self.model, self.output)

    def test_finalize_validates_complete_probability_cache(self) -> None:
        self._write_completed_cache(probabilities=True)
        report = finalize_selected_cache(
            self.selection,
            self.model,
            self.output,
            require_probabilities=True,
        )
        dense = report["dense_probability_cache"]
        self.assertTrue(dense["available"])
        self.assertEqual(dense["class_count"], 150)
        self.assertEqual(dense["float16_argmax_disagreement_count"], 0)

    def test_compares_prior_hard_cache_without_modifying_it(self) -> None:
        self._write_completed_cache(probabilities=True)
        view_dir = self.output / "stages" / "01_real_camera_views"
        report = compare_hard_caches(
            view_dir,
            view_dir,
            self.root / "comparison.json",
        )
        self.assertTrue(report["pixel_exact_match"])
        self.assertEqual(report["hard_class_disagreement_count"], 0)
        self.assertFalse(report["manual_correction_used"])

    def test_resume_revalidates_completed_probability_and_hard_caches(self) -> None:
        self._write_completed_cache(probabilities=True)
        finalize_selected_cache(
            self.selection,
            self.model,
            self.output,
            require_probabilities=True,
        )
        view_dir = self.output / "stages" / "01_real_camera_views"
        compare_hard_caches(
            view_dir,
            view_dir,
            self.output / "selection" / "hard_cache_reproducibility.json",
        )
        result = validate_probability_resume(
            self.selection,
            self.model,
            self.output,
            view_dir,
        )
        self.assertEqual(
            result["contract"], "validated_existing_probability_cache_resume_v1"
        )
        self.assertTrue(result["hard_cache_pixel_exact_match"])
        self.assertFalse(result["artifacts_rewritten"])

    def test_resume_rejects_a_tampered_selected_cache_report(self) -> None:
        self._write_completed_cache(probabilities=True)
        finalize_selected_cache(
            self.selection,
            self.model,
            self.output,
            require_probabilities=True,
        )
        view_dir = self.output / "stages" / "01_real_camera_views"
        compare_hard_caches(
            view_dir,
            view_dir,
            self.output / "selection" / "hard_cache_reproducibility.json",
        )
        path = self.output / "selection" / "selected_view_cache_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["selected_camera_count"] = 2
        path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match current artifacts"):
            validate_probability_resume(
                self.selection,
                self.model,
                self.output,
                view_dir,
            )

    def test_scheduler_is_automatic_fixed_and_report_only(self) -> None:
        source = SCHEDULER.read_text(encoding="utf-8")
        self.assertIn("TARGET_TWO_VIEW_PERCENT=\"${TARGET_TWO_VIEW_PERCENT:-99}\"", source)
        self.assertIn("requires TARGET_TWO_VIEW_PERCENT=99", source)
        self.assertIn("Manual CAMERA_INDICES and VIEW_COUNT are not accepted", source)
        self.assertIn("python -m scripts.task1.dinov3.selected_view_cache resolve", source)
        self.assertIn("REPORT_ONLY=1", source)
        self.assertIn("WRITE_SEMANTIC_PLY=0", source)
        self.assertIn("DINOV3_CROP_SIZE=512", source)
        self.assertIn("DINOV3_STRIDE=384", source)
        self.assertIn("DINOV3_CHECKPOINT_LOAD_MODE=local_mmap", source)
        self.assertIn("DINOV3_MIN_HOST_AVAILABLE_GIB=", source)
        self.assertIn("MIN_PIXEL_CONFIDENCE=0.0", source)
        self.assertIn("semantic_fusion_run=0", source)
        self.assertIn("unexpectedly wrote a PLY", source)


if __name__ == "__main__":
    unittest.main()
