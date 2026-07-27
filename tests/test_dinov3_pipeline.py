from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.task1.dinov3.dinov3_segment_views import (
    CONFIDENCE_METRIC,
    EXPECTED_BACKBONE_SHA256,
    EXPECTED_SEGMENTOR_SHA256,
    PINNED_DINOV3_COMMIT,
    mask2former_confidence_maps,
    numeric_version,
    repository_provenance,
    verify_expected_sha256,
)

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_scene.a100.sbatch"
)
SEGMENTER_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "task1"
    / "dinov3"
    / "dinov3_segment_views.py"
)
LIFT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "task1"
    / "dinov2"
    / "lift_dinov2_view_votes.py"
)
FUSION_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "task1"
    / "dinov2"
    / "fuse_dinov2_multiview_votes.py"
)
SETUP_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "setup"
    / "install_dinov3_semantic.sh"
)


class Dinov3ProvenanceTest(unittest.TestCase):
    def test_numeric_version_ignores_cuda_suffix(self) -> None:
        self.assertEqual(numeric_version("2.7.1+cu128"), (2, 7, 1))

    def test_checkpoint_hash_can_be_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pth"
            path.write_bytes(b"test checkpoint")
            digest = verify_expected_sha256(path, "")
            self.assertEqual(verify_expected_sha256(path, digest), digest)
            with self.assertRaises(ValueError):
                verify_expected_sha256(path, "0" * 64)

    def test_mask2former_relative_margin_confidence(self) -> None:
        probabilities = np.asarray(
            [
                [[0.70, 0.40]],
                [[0.20, 0.40]],
                [[0.10, 0.20]],
            ],
            dtype=np.float32,
        )
        maps = mask2former_confidence_maps(probabilities)
        self.assertEqual(CONFIDENCE_METRIC, "relative_top1_top2_margin")
        np.testing.assert_allclose(
            maps["max_softmax_probability"],
            [[0.70, 0.40]],
        )
        np.testing.assert_allclose(
            maps["top1_top2_margin"],
            [[0.50, 0.0]],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            maps["confidence"],
            [[5.0 / 7.0, 0.0]],
            atol=1e-7,
        )
        self.assertTrue(
            np.logical_and(
                maps["normalized_entropy_confidence"] >= 0.0,
                maps["normalized_entropy_confidence"] <= 1.0,
            ).all()
        )

    def test_repository_commit_and_clean_state_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            (repository / ".git").mkdir()

            def fake_git_output(_repository: Path, *arguments: str) -> str:
                if arguments == ("rev-parse", "HEAD"):
                    return PINNED_DINOV3_COMMIT
                if arguments[:2] == ("status", "--porcelain"):
                    return ""
                if arguments == ("remote", "get-url", "origin"):
                    return "https://github.com/facebookresearch/dinov3.git"
                raise AssertionError(arguments)

            with patch(
                "scripts.task1.dinov3.dinov3_segment_views.git_output",
                side_effect=fake_git_output,
            ):
                provenance = repository_provenance(
                    repository,
                    PINNED_DINOV3_COMMIT,
                )
            self.assertEqual(provenance["commit"], PINNED_DINOV3_COMMIT)
            self.assertTrue(provenance["tracked_worktree_clean"])


class Dinov3AdapterContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.segmenter = SEGMENTER_PATH.read_text(encoding="utf-8")
        cls.lift = LIFT_PATH.read_text(encoding="utf-8")
        cls.fusion = FUSION_PATH.read_text(encoding="utf-8")

    def test_adapter_writes_raw_ade20k_contract(self) -> None:
        self.assertIn("class_id=raw_class", self.segmenter.replace(" ", ""))
        self.assertIn("confidence=confidence", self.segmenter)
        self.assertIn('"raw_class_storage": "uint8_ade20k_zero_based"', self.segmenter)
        self.assertIn(
            '"confidence_storage": "float16_relative_top1_top2_margin"',
            self.segmenter,
        )

    def test_adapter_uses_identity_preserving_ontology(self) -> None:
        self.assertIn('configs" / "ade20k_to_project.json"', self.segmenter)
        self.assertNotIn("ade20k_to_project.dense_backends.json", self.segmenter)

    def test_adapter_records_model_and_repository_provenance(self) -> None:
        for expected in (
            '"repository": repo_provenance',
            '"backbone_sha256": backbone_sha256',
            '"segmentor_sha256": segmentor_sha256',
            '"precision": args.precision',
        ):
            self.assertIn(expected, self.segmenter)

    def test_adapter_uses_relative_margin_and_records_calibration_maps(self) -> None:
        self.assertIn(
            '"raw_ade20k_class_and_relative_margin_confidence_v2"',
            self.segmenter,
        )
        self.assertIn('"confidence_metric": CONFIDENCE_METRIC', self.segmenter)
        self.assertIn("max_softmax_probability", self.segmenter)
        self.assertIn("top1_top2_margin", self.segmenter)
        self.assertIn("normalized_entropy_confidence", self.segmenter)

    def test_lift_accepts_an_explicit_segmentation_manifest(self) -> None:
        self.assertIn('"--segmentation-manifest"', self.lift)
        self.assertIn('"segmentation_source"', self.lift)

    def test_fusion_propagates_the_segmentation_source(self) -> None:
        self.assertIn("fused_source = f", self.fusion)
        self.assertIn('"source": fused_source', self.fusion)


class Dinov3SchedulerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scheduler = SCHEDULER_PATH.read_text(encoding="utf-8")

    def test_a100_and_separate_environments_are_selected(self) -> None:
        self.assertIn("#SBATCH --partition=a100", self.scheduler)
        self.assertIn('GAUSSIAN_ENV="${GAUSSIAN_ENV:-gaussian_grouping_true}"', self.scheduler)
        self.assertIn('DINOV3_ENV="${DINOV3_ENV:-dinov3_semantic}"', self.scheduler)

    def test_report_only_is_the_default_and_exits_before_lifting(self) -> None:
        self.assertIn('REPORT_ONLY="${REPORT_ONLY:-1}"', self.scheduler)
        self.assertIn('MIN_PIXEL_CONFIDENCE="${MIN_PIXEL_CONFIDENCE:-0.0}"', self.scheduler)
        report_only = self.scheduler.index('if [[ "${REPORT_ONLY}" = "1" ]]; then')
        vote_lift = self.scheduler.index("run_stage 05_lift_per_view_votes")
        self.assertLess(report_only, vote_lift)
        self.assertIn('echo "semantic_labels_written=0"', self.scheduler)
        self.assertIn('echo "semantic_ply_written=0"', self.scheduler)

    def test_destructive_reset_and_ply_are_opt_in(self) -> None:
        self.assertIn('RESET_OUTPUT="${RESET_OUTPUT:-0}"', self.scheduler)
        self.assertIn('WRITE_SEMANTIC_PLY="${WRITE_SEMANTIC_PLY:-0}"', self.scheduler)

    def test_exact_pinned_repository_revision_is_checked(self) -> None:
        self.assertIn(PINNED_DINOV3_COMMIT, self.scheduler)
        self.assertIn('git -C "${DINOV3_ROOT}" rev-parse HEAD', self.scheduler)

    def test_full_official_checkpoint_hashes_are_required_by_default(self) -> None:
        self.assertIn(EXPECTED_BACKBONE_SHA256, self.scheduler)
        self.assertIn(EXPECTED_SEGMENTOR_SHA256, self.scheduler)

    def test_scheduler_uses_maintained_ontology_and_lift(self) -> None:
        self.assertIn("configs/ade20k_to_project.json", self.scheduler)
        self.assertNotIn("ade20k_to_project.dense_backends.json", self.scheduler)
        self.assertIn(
            "--segmentation-manifest \"${VIEW_DIR}/dinov3_manifest.json\"",
            self.scheduler,
        )


class Dinov3SetupContractTest(unittest.TestCase):
    def test_setup_installs_code_and_dependencies_but_not_model_weights(self) -> None:
        setup = SETUP_PATH.read_text(encoding="utf-8")
        self.assertIn('TORCH_VERSION="2.7.1"', setup)
        self.assertIn('TORCHVISION_VERSION="0.22.1"', setup)
        self.assertIn(PINNED_DINOV3_COMMIT, setup)
        self.assertNotIn("huggingface-cli download", setup)
        self.assertNotIn("hf download", setup)
        self.assertNotIn("curl ", setup)
        self.assertNotIn("wget ", setup)


if __name__ == "__main__":
    unittest.main()
