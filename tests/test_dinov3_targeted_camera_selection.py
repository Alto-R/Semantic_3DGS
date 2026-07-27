from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "scripts" / "slurm" / "slurm_task1_dinov3_targeted_50plus20_scene.sbatch"
REPORT_SCHEDULER = ROOT / "scripts" / "slurm" / "slurm_task1_dinov3_3d_first_scene.sbatch"


class TargetedDINOv3SchedulerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCHEDULER.read_text(encoding="utf-8")
        cls.report_text = REPORT_SCHEDULER.read_text(encoding="utf-8")

    def test_reuses_the_historical_50_plus_20_policy(self) -> None:
        for expected in (
            'BASE_VIEW_COUNT="${BASE_VIEW_COUNT:-50}"',
            'ADDITIONAL_VIEW_COUNT="${ADDITIONAL_VIEW_COUNT:-20}"',
            'CANDIDATE_VIEW_COUNT="${CANDIDATE_VIEW_COUNT:-100}"',
            'PROJECTION_SAMPLE_COUNT="${PROJECTION_SAMPLE_COUNT:-50000}"',
            'PROJECTION_MARGIN="${PROJECTION_MARGIN:-1.5}"',
            'LOW_COVERAGE_QUANTILE="${LOW_COVERAGE_QUANTILE:-0.35}"',
            'COVERAGE_WEIGHT="${COVERAGE_WEIGHT:-0.8}"',
            'POSE_NOVELTY_WEIGHT="${POSE_NOVELTY_WEIGHT:-0.2}"',
        ):
            self.assertIn(expected, self.text)

    def test_selection_stages_are_projection_safe_coverage_aware_and_pose_diverse(self) -> None:
        seed = self.text.index("select_targeted_cameras seed")
        screen = self.text.index("select_targeted_cameras screen")
        render = self.text.index("render_auto_label_overlays")
        measure = self.text.index("measure_overlay_coverage")
        select = self.text.index("select_targeted_cameras select")
        report = self.text.rindex("slurm_task1_dinov3_3d_first_scene.sbatch")
        self.assertLess(seed, screen)
        self.assertLess(screen, render)
        self.assertLess(render, measure)
        self.assertLess(measure, select)
        self.assertLess(select, report)

    def test_prior_labels_are_selection_only(self) -> None:
        self.assertIn("camera_selection_used_prior_labels=1", self.text)
        self.assertIn("semantic_inference_used_prior_labels=0", self.text)
        self.assertIn("v5_used=0", self.text)
        self.assertIn("dinov2_used=0", self.text)
        self.assertIn('CAMERA_INDICES="${FINAL_CAMERA_INDICES}"', self.text)
        self.assertIn('VIEW_COUNT="${FINAL_VIEW_COUNT}"', self.text)
        self.assertIn("prior_semantic_labels_used=0", self.report_text)
        self.assertNotIn("SELECTION_LABELS", self.report_text)

    def test_selection_output_is_report_only_and_fresh(self) -> None:
        self.assertIn('if [[ "${RESET_OUTPUT}" != "0" ]]', self.text)
        self.assertIn("semantic_labels_written=0", self.text)
        self.assertIn("semantic_ply_written=0", self.text)
        self.assertIn('test ! -e "${SELECTION_DIR}/gaussian_labels.npy"', self.text)
        self.assertIn("Camera-selection stage unexpectedly wrote a PLY", self.text)

    def test_report_scheduler_records_selection_provenance(self) -> None:
        self.assertIn("CAMERA_SELECTION_POLICY", self.report_text)
        self.assertIn("CAMERA_SELECTION_USED_PRIOR_LABELS", self.report_text)
        self.assertIn("camera_selection_used_prior_labels", self.report_text)


if __name__ == "__main__":
    unittest.main()
