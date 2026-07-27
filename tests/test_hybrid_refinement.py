from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from scripts.task1.hybrid_refinement.refine_sam_with_mask2former_regions import (
    RefinementThresholds,
    match_metrics,
    refine_frame_masks,
)
from scripts.task1.hybrid_refinement.audit_proposals_against_base import (
    summarize_class_overlap,
    summarize_cross_class_pair,
)
from scripts.task1.hybrid_refinement.propagate_regions_from_3d_seeds import (
    PropagationThresholds,
    build_multiview_seeds,
    select_propagated_regions,
)
from scripts.task1.grounding.cluster_semantic_flashsplat_proposals import (
    SemanticGroup,
    prune_thing_label_islands,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_sam_mask2former_hybrid_scene.sbatch"
)
FUSER_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "task1"
    / "grounding"
    / "cluster_semantic_flashsplat_proposals.py"
)
PROPOSAL_LIFTER_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "task1"
    / "grounding"
    / "run_flashsplat_mask_proposals.py"
)


def thresholds(**overrides: float | int) -> RefinementThresholds:
    values: dict[str, float | int] = {
        "min_iou": 0.45,
        "min_containment": 0.80,
        "min_sam_coverage": 0.10,
        "min_region_confidence": 0.25,
        "min_match_score": 0.50,
        "min_area_ratio": 0.50,
        "max_area_ratio": 2.00,
        "identity_margin": 0.10,
        "max_cross_class_overlap": 0.10,
        "min_overlap_pixels": 1,
    }
    values.update(overrides)
    return RefinementThresholds(**values)


def propagation_thresholds(
    **overrides: float | int,
) -> PropagationThresholds:
    values: dict[str, float | int] = {
        "projection_threshold": 0.50,
        "min_projection_pixels": 1,
        "min_projection_coverage": 0.05,
        "min_region_coverage": 0.20,
        "min_containment": 0.50,
        "min_region_confidence": 0.25,
        "min_match_score": 0.45,
        "existing_same_class_containment": 0.50,
        "identity_margin": 0.10,
        "max_cross_class_overlap": 0.10,
        "max_masks_per_class_per_view": 4,
    }
    values.update(overrides)
    return PropagationThresholds(**values)


class RegionMetricTest(unittest.TestCase):
    def test_perfect_match(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True
        metrics = match_metrics(mask, mask, np.ones((5, 5), dtype=np.float32))
        self.assertEqual(metrics["intersection"], 9)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["containment"], 1.0)
        self.assertEqual(metrics["area_ratio"], 1.0)


class HybridFrameRefinementTest(unittest.TestCase):
    def test_sam_identity_is_snapped_to_larger_matching_region(self) -> None:
        sam = np.zeros((1, 8, 8), dtype=bool)
        sam[0, 2:6, 2:6] = True
        region_id = np.zeros((8, 8), dtype=np.uint16)
        region_id[1:7, 1:7] = 4
        region_confidence = np.full((8, 8), 0.90, dtype=np.float32)

        refined, audits, counts = refine_frame_masks(
            sam,
            [{"class": "window"}],
            region_id,
            region_confidence,
            thresholds(max_area_ratio=3.0),
        )

        self.assertEqual(int(refined[0].sum()), 36)
        self.assertEqual(audits[0]["selected_region_id"], 4)
        self.assertEqual(audits[0]["status"], "snapped_to_mask2former_region")
        self.assertFalse(audits[0]["mask2former_semantic_class_used"])
        self.assertEqual(counts["snapped_to_mask2former_region"], 1)

    def test_mask2former_only_region_never_creates_a_semantic_mask(self) -> None:
        sam = np.zeros((1, 8, 8), dtype=bool)
        sam[0, 0:2, 0:2] = True
        region_id = np.zeros((8, 8), dtype=np.uint16)
        region_id[5:8, 5:8] = 1

        refined, audits, _counts = refine_frame_masks(
            sam,
            [{"class": "door"}],
            region_id,
            np.ones((8, 8), dtype=np.float32),
            thresholds(),
        )

        np.testing.assert_array_equal(refined, sam)
        self.assertEqual(audits[0]["status"], "unchanged_no_region_match")
        self.assertEqual(refined.shape[0], 1)

    def test_competing_identities_make_region_ambiguous(self) -> None:
        masks = np.zeros((2, 8, 8), dtype=bool)
        masks[:, 2:6, 2:6] = True
        region_id = np.zeros((8, 8), dtype=np.uint16)
        region_id[2:6, 2:6] = 7

        refined, audits, counts = refine_frame_masks(
            masks,
            [{"class": "window"}, {"class": "door"}],
            region_id,
            np.ones((8, 8), dtype=np.float32),
            thresholds(),
        )

        np.testing.assert_array_equal(refined, masks)
        self.assertEqual(
            [item["status"] for item in audits],
            ["unchanged_identity_conflict", "unchanged_identity_conflict"],
        )
        self.assertEqual(counts["unchanged_identity_conflict"], 2)

    def test_cross_class_overlap_blocks_region_expansion(self) -> None:
        masks = np.zeros((2, 8, 8), dtype=bool)
        masks[0, 2:5, 2:5] = True
        masks[1, 5, 5] = True
        region_id = np.zeros((8, 8), dtype=np.uint16)
        region_id[1:6, 1:6] = 3

        refined, audits, _counts = refine_frame_masks(
            masks,
            [{"class": "window"}, {"class": "door"}],
            region_id,
            np.ones((8, 8), dtype=np.float32),
            thresholds(max_area_ratio=3.0),
        )

        np.testing.assert_array_equal(refined, masks)
        self.assertEqual(audits[0]["status"], "unchanged_cross_class_overlap")

    def test_area_ratio_gate_rejects_unrelated_large_region(self) -> None:
        sam = np.zeros((1, 10, 10), dtype=bool)
        sam[0, 4:6, 4:6] = True
        region_id = np.ones((10, 10), dtype=np.uint16)

        refined, audits, _counts = refine_frame_masks(
            sam,
            [{"class": "window"}],
            region_id,
            np.ones((10, 10), dtype=np.float32),
            thresholds(max_area_ratio=2.0),
        )

        np.testing.assert_array_equal(refined, sam)
        self.assertEqual(audits[0]["status"], "unchanged_no_region_match")


class BaseOverlapAuditTest(unittest.TestCase):
    def test_overlap_report_separates_unlabeled_same_and_other_classes(self) -> None:
        base_labels = np.asarray([0, 4, 4, 7, 7, 9], dtype=np.int32)
        base_classes = {0: "unlabeled", 4: "window", 7: "door", 9: "wall"}
        report = summarize_class_overlap(
            "window",
            np.asarray([0, 1, 2, 3, 5], dtype=np.int64),
            base_labels,
            base_classes,
        )
        self.assertEqual(report["candidate_gaussians"], 5)
        self.assertEqual(report["base_unlabeled_gaussians"], 1)
        self.assertEqual(report["base_same_class_gaussians"], 2)
        self.assertEqual(report["base_other_class_gaussians"], 2)

    def test_overlap_report_rejects_out_of_range_indices(self) -> None:
        with self.assertRaises(IndexError):
            summarize_class_overlap(
                "window",
                np.asarray([3], dtype=np.int64),
                np.zeros(3, dtype=np.int32),
                {0: "unlabeled"},
            )

    def test_cross_class_containment_reports_identity_conflict(self) -> None:
        report = summarize_cross_class_pair(
            "window",
            np.asarray([1, 2, 3, 4], dtype=np.int64),
            "door",
            np.asarray([3, 4, 5], dtype=np.int64),
            0.50,
        )
        self.assertEqual(report["intersection_gaussians"], 2)
        self.assertAlmostEqual(report["containment"], 2.0 / 3.0)
        self.assertTrue(report["identity_conflict"])


class GlobalComponentAuditTest(unittest.TestCase):
    @staticmethod
    def vertex_data() -> np.ndarray:
        dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
        ]
        vertex = np.zeros(4, dtype=dtype)
        vertex["x"] = np.asarray([0.00, 0.03, 0.06, 5.00], dtype=np.float32)
        for name in ("scale_0", "scale_1", "scale_2"):
            vertex[name] = -3.0
        return vertex

    def test_stuff_group_is_audited_only_when_global_flag_is_enabled(self) -> None:
        labels = np.ones(4, dtype=np.int32)
        stuff_group = SemanticGroup(
            group_id=1,
            class_name="window",
            indices=np.arange(4, dtype=np.int64),
            is_stuff=True,
        )
        unchanged, reports = prune_thing_label_islands(
            labels.copy(),
            [stuff_group],
            self.vertex_data(),
            4.0,
            0.01,
            0.20,
            2,
            0.01,
            False,
        )
        np.testing.assert_array_equal(unchanged, labels)
        self.assertEqual(reports, [])

        global_labels, reports = prune_thing_label_islands(
            labels.copy(),
            [stuff_group],
            self.vertex_data(),
            4.0,
            0.01,
            0.20,
            2,
            0.01,
            True,
        )
        np.testing.assert_array_equal(global_labels, np.asarray([1, 1, 1, 0]))
        self.assertEqual(reports[0]["class"], "window")
        self.assertEqual(reports[0]["removed_gaussians"], 1)


class MultiviewPropagationTest(unittest.TestCase):
    @staticmethod
    def region_arrays() -> tuple[np.ndarray, np.ndarray]:
        region_id = np.zeros((8, 8), dtype=np.uint16)
        region_id[2:6, 2:6] = 3
        region_confidence = np.full((8, 8), 0.90, dtype=np.float32)
        return region_id, region_confidence

    @staticmethod
    def projection() -> np.ndarray:
        projection = np.zeros((8, 8), dtype=np.float32)
        projection[2:6, 2:6] = 1.0
        return projection

    def test_seed_support_counts_distinct_frames_not_proposals(self) -> None:
        with TemporaryDirectory() as temp_dir:
            support_dir = Path(temp_dir)
            np.savez_compressed(support_dir / "a.npz", indices=np.asarray([1, 2]))
            np.savez_compressed(support_dir / "b.npz", indices=np.asarray([1, 3]))
            np.savez_compressed(support_dir / "c.npz", indices=np.asarray([1, 2]))
            manifest = {
                "proposals": [
                    {
                        "class": "window",
                        "frame_file": "view_a.png",
                        "support_file": "a.npz",
                    },
                    {
                        "class": "window",
                        "frame_file": "view_a.png",
                        "support_file": "b.npz",
                    },
                    {
                        "class": "window",
                        "frame_file": "view_b.png",
                        "support_file": "c.npz",
                    },
                ]
            }

            seeds, reports = build_multiview_seeds(
                manifest,
                support_dir,
                gaussian_count=5,
                min_seed_views=2,
                min_seed_gaussians=1,
            )

        np.testing.assert_array_equal(seeds["window"], np.asarray([1, 2]))
        self.assertEqual(reports["window"]["source_view_count"], 2)

    def test_confirmed_projection_activates_class_agnostic_region(self) -> None:
        region_id, region_confidence = self.region_arrays()
        output, metadata, report = select_propagated_regions(
            np.zeros((0, 8, 8), dtype=bool),
            [],
            region_id,
            region_confidence,
            {"window": self.projection()},
            {
                "window": {
                    "source_view_count": 3,
                    "seed_gaussian_count": 800,
                }
            },
            propagation_thresholds(),
        )

        self.assertEqual(output.shape, (1, 8, 8))
        np.testing.assert_array_equal(output[0], region_id == 3)
        self.assertEqual(metadata[0]["class"], "window")
        self.assertFalse(
            metadata[0]["hybrid_refinement"][
                "mask2former_semantic_class_used"
            ]
        )
        self.assertEqual(report["accepted_count"], 1)

    def test_existing_same_class_mask_prevents_duplicate_region(self) -> None:
        region_id, region_confidence = self.region_arrays()
        existing = (region_id == 3)[None, ...]
        output, metadata, report = select_propagated_regions(
            existing,
            [{"class": "window"}],
            region_id,
            region_confidence,
            {"window": self.projection()},
            {
                "window": {
                    "source_view_count": 2,
                    "seed_gaussian_count": 500,
                }
            },
            propagation_thresholds(),
        )

        np.testing.assert_array_equal(output, existing)
        self.assertEqual(metadata, [])
        self.assertEqual(report["skipped_existing_same_class"], 1)

    def test_competing_3d_identities_leave_region_unassigned(self) -> None:
        region_id, region_confidence = self.region_arrays()
        seed_reports = {
            class_name: {
                "source_view_count": 3,
                "seed_gaussian_count": 900,
            }
            for class_name in ("window", "door")
        }
        output, metadata, report = select_propagated_regions(
            np.zeros((0, 8, 8), dtype=bool),
            [],
            region_id,
            region_confidence,
            {
                "window": self.projection(),
                "door": self.projection(),
            },
            seed_reports,
            propagation_thresholds(),
        )

        self.assertEqual(output.shape[0], 0)
        self.assertEqual(metadata, [])
        self.assertEqual(report["identity_conflict_count"], 1)

    def test_mask2former_region_without_3d_seed_cannot_create_mask(self) -> None:
        region_id, region_confidence = self.region_arrays()
        output, metadata, report = select_propagated_regions(
            np.zeros((0, 8, 8), dtype=bool),
            [],
            region_id,
            region_confidence,
            {},
            {},
            propagation_thresholds(),
        )

        self.assertEqual(output.shape[0], 0)
        self.assertEqual(metadata, [])
        self.assertEqual(report["accepted_count"], 0)


class OnePassSchedulerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scheduler = SCHEDULER_PATH.read_text(encoding="utf-8")
        cls.fuser = FUSER_PATH.read_text(encoding="utf-8")
        cls.proposal_lifter = PROPOSAL_LIFTER_PATH.read_text(encoding="utf-8")

    def test_all_hybrid_stages_are_ordered_in_one_scheduler(self) -> None:
        markers = [
            "run_stage 01_mask2former_regions",
            "run_stage 02_grounded_sam",
            "run_stage 03_hybrid_mask_refinement",
            "run_stage 04_seed_flashsplat_proposals",
            "run_stage 05_multiview_seed_propagation",
            "run_stage 06_final_flashsplat_proposals",
            "run_stage 07_base_overlap_audit",
            "run_stage 08_multiview_3d_audit",
        ]
        positions = [self.scheduler.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))

    def test_report_only_runs_3d_audit_without_labels_or_ply(self) -> None:
        self.assertIn('REPORT_ONLY="${REPORT_ONLY:-1}"', self.scheduler)
        self.assertIn("FUSION_OUTPUT_ARGS+=(--report-only)", self.scheduler)
        self.assertIn('echo "semantic_labels_written=0"', self.scheduler)
        self.assertIn('echo "semantic_ply_written=0"', self.scheduler)
        self.assertIn('echo "base_modified=0"', self.scheduler)

    def test_mask2former_semantics_are_not_used_by_refinement(self) -> None:
        self.assertIn("--save-regions", self.scheduler)
        self.assertIn(
            "refine_sam_with_mask2former_regions",
            self.scheduler,
        )

    def test_one_pass_scheduler_uses_exactly_one_propagation_round(self) -> None:
        self.assertIn(
            "propagate_regions_from_3d_seeds",
            self.scheduler,
        )
        self.assertIn('echo "propagation_rounds=1"', self.scheduler)
        self.assertEqual(
            self.scheduler.count(
                "python -m scripts.task1.hybrid_refinement."
                "propagate_regions_from_3d_seeds"
            ),
            1,
        )

    def test_fuser_report_only_skips_label_and_label_map_writes(self) -> None:
        self.assertIn('parser.add_argument(\n        "--report-only"', self.fuser)
        self.assertIn("if not args.report_only:\n        np.save(labels_path, labels)", self.fuser)
        self.assertIn(
            '"semantic_labels_written": not args.report_only',
            self.fuser,
        )

    def test_one_pass_audit_applies_components_to_every_target_type(self) -> None:
        self.assertIn("--spatial-prune-all-classes", self.scheduler)
        self.assertIn(
            'parser.add_argument(\n        "--spatial-prune-all-classes"',
            self.fuser,
        )

    def test_candidate_label_mode_fails_closed_on_3d_identity_conflicts(self) -> None:
        self.assertIn("BASE_AUDIT_ARGS+=(--fail-on-conflict)", self.scheduler)

    def test_flashsplat_proposals_retain_hybrid_match_provenance(self) -> None:
        self.assertIn('"hybrid_refinement",', self.proposal_lifter)


if __name__ == "__main__":
    unittest.main()
