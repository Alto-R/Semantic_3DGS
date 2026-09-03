from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.task1.dinov3.dinov3_segment_views import (
    CONFIDENCE_METRIC,
    EXPECTED_BACKBONE_SHA256,
    EXPECTED_SEGMENTOR_SHA256,
    GIB_BYTES,
    PINNED_DINOV3_COMMIT,
    checkpoint_state_dict_loader,
    configure_cuda_memory_limit,
    cuda_memory_usage,
    mask2former_confidence_maps,
    numeric_version,
    repository_provenance,
    short_side_resize_dimensions,
    validate_sliding_inference_geometry,
    verify_expected_sha256,
)

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "slurm"
    / "slurm_task1_dinov3_end_to_end_recovery_scene.sbatch"
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
    / "dinov3"
    / "lift_dense_view_votes.py"
)
SETUP_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "setup"
    / "install_dinov3_semantic.sh"
)


class Dinov3ProvenanceTest(unittest.TestCase):
    def test_official_short_side_resize_geometry(self) -> None:
        self.assertEqual(
            short_side_resize_dimensions(632, 960, 896),
            (896, 1361),
        )
        self.assertEqual(
            short_side_resize_dimensions(960, 632, 896),
            (1361, 896),
        )
        self.assertEqual(
            short_side_resize_dimensions(896, 896, 896),
            (896, 896),
        )

    def test_short_side_resize_rejects_invalid_dimensions(self) -> None:
        for dimensions in ((0, 960, 896), (632, 0, 896), (632, 960, 0)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    short_side_resize_dimensions(*dimensions)

    def test_sliding_geometry_requires_adapter_aligned_crop(self) -> None:
        validate_sliding_inference_geometry(512, 384)
        validate_sliding_inference_geometry(896, 596)
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            validate_sliding_inference_geometry(900, 596)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            validate_sliding_inference_geometry(512, 513)

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

    def test_cuda_memory_limit_uses_total_visible_device_memory(self) -> None:
        class FakeCuda:
            def __init__(self) -> None:
                self.calls: list[tuple[float, int]] = []

            def current_device(self) -> int:
                return 0

            def get_device_properties(self, device: int) -> SimpleNamespace:
                self.assert_device_index(device)
                return SimpleNamespace(total_memory=48 * GIB_BYTES)

            def set_per_process_memory_fraction(
                self,
                fraction: float,
                *,
                device: int,
            ) -> None:
                self.assert_device_index(device)
                self.calls.append((fraction, device))

            def mem_get_info(self, device: int) -> tuple[int, int]:
                self.assert_device_index(device)
                return 47 * GIB_BYTES, 48 * GIB_BYTES

            @staticmethod
            def assert_device_index(device: int) -> None:
                if not isinstance(device, int):
                    raise TypeError("CUDA API requires an integer device index")

        cuda = FakeCuda()
        torch = SimpleNamespace(
            cuda=cuda,
            device=lambda value: SimpleNamespace(
                type=value.partition(":")[0],
                index=(
                    int(value.partition(":")[2])
                    if value.partition(":")[2]
                    else None
                ),
            ),
        )
        record = configure_cuda_memory_limit(torch, "cuda", 35.0)

        self.assertTrue(record["enabled"])
        self.assertEqual(record["logical_device_index"], 0)
        self.assertEqual(record["requested_bytes"], 35 * GIB_BYTES)
        self.assertEqual(record["total_device_memory_bytes"], 48 * GIB_BYTES)
        self.assertFalse(record["hard_hardware_partition"])
        self.assertEqual(cuda.calls, [(35.0 / 48.0, 0)])

    def test_cuda_memory_limit_is_optional_and_cuda_only(self) -> None:
        self.assertFalse(
            configure_cuda_memory_limit(
                SimpleNamespace(),
                "cpu",
                None,
            )["enabled"]
        )
        with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
            configure_cuda_memory_limit(SimpleNamespace(), "cpu", 35.0)
        with self.assertRaisesRegex(ValueError, "finite positive"):
            configure_cuda_memory_limit(SimpleNamespace(), "cuda", 0.0)

    def test_cuda_memory_limit_must_be_below_total_device_memory(self) -> None:
        cuda = SimpleNamespace(
            current_device=lambda: 0,
            get_device_properties=lambda _device: SimpleNamespace(
                total_memory=48 * GIB_BYTES
            )
        )
        with self.assertRaisesRegex(ValueError, "smaller than total"):
            configure_cuda_memory_limit(
                SimpleNamespace(
                    cuda=cuda,
                    device=lambda _value: SimpleNamespace(
                        type="cuda",
                        index=None,
                    ),
                ),
                "cuda",
                48.0,
            )

    def test_cuda_memory_usage_records_allocated_and_reserved_peaks(self) -> None:
        cuda = SimpleNamespace(
            current_device=lambda: 0,
            synchronize=lambda _device: None,
            memory_allocated=lambda _device: 28 * GIB_BYTES,
            max_memory_allocated=lambda _device: 34 * GIB_BYTES,
            memory_reserved=lambda _device: 29 * GIB_BYTES,
            max_memory_reserved=lambda _device: 35 * GIB_BYTES,
        )
        record = cuda_memory_usage(
            SimpleNamespace(
                cuda=cuda,
                device=lambda _value: SimpleNamespace(
                    type="cuda",
                    index=None,
                ),
            ),
            "cuda",
        )
        self.assertEqual(record["logical_device_index"], 0)
        self.assertEqual(record["peak_allocated_bytes"], 34 * GIB_BYTES)
        self.assertEqual(record["peak_reserved_bytes"], 35 * GIB_BYTES)
        self.assertEqual(record["peak_allocated_gib"], 34.0)
        self.assertEqual(record["peak_reserved_gib"], 35.0)

    def test_local_mmap_intercepts_only_verified_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "weights.pth"
            checkpoint.write_bytes(b"synthetic")
            original_loader = Mock(return_value="standard")
            mmap_load = Mock(return_value={"weight": "mapped"})
            torch = SimpleNamespace(
                hub=SimpleNamespace(
                    load_state_dict_from_url=original_loader
                ),
                load=mmap_load,
            )

            with checkpoint_state_dict_loader(
                torch,
                "local_mmap",
                (checkpoint,),
                integrity_preverified=True,
            ) as record:
                loaded = torch.hub.load_state_dict_from_url(
                    checkpoint.resolve().as_uri(),
                    map_location="cpu",
                    check_hash=True,
                    weights_only=True,
                )

            self.assertEqual(loaded, {"weight": "mapped"})
            mmap_load.assert_called_once_with(
                checkpoint.resolve(),
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            original_loader.assert_not_called()
            self.assertEqual(
                record["intercepted_checkpoint_paths"],
                [str(checkpoint.resolve())],
            )
            self.assertIs(
                torch.hub.load_state_dict_from_url,
                original_loader,
            )

    def test_local_mmap_delegates_nonlocal_and_unlisted_urls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            allowed = Path(temporary) / "allowed.pth"
            unlisted = Path(temporary) / "unlisted.pth"
            original_loader = Mock(return_value="standard")
            torch = SimpleNamespace(
                hub=SimpleNamespace(
                    load_state_dict_from_url=original_loader
                ),
                load=Mock(),
            )

            with checkpoint_state_dict_loader(
                torch,
                "local_mmap",
                (allowed,),
                integrity_preverified=True,
            ):
                self.assertEqual(
                    torch.hub.load_state_dict_from_url(
                        "https://example.invalid/weights.pth"
                    ),
                    "standard",
                )
                self.assertEqual(
                    torch.hub.load_state_dict_from_url(
                        unlisted.resolve().as_uri()
                    ),
                    "standard",
                )

            self.assertEqual(original_loader.call_count, 2)
            torch.load.assert_not_called()

    def test_local_mmap_restores_loader_after_exception(self) -> None:
        original_loader = Mock()
        torch = SimpleNamespace(
            hub=SimpleNamespace(load_state_dict_from_url=original_loader),
            load=Mock(),
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with checkpoint_state_dict_loader(
                torch,
                "local_mmap",
                (Path("weights.pth"),),
                integrity_preverified=True,
            ):
                raise RuntimeError("synthetic failure")
        self.assertIs(torch.hub.load_state_dict_from_url, original_loader)

    def test_local_mmap_requires_preverified_integrity(self) -> None:
        torch = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "preverified"):
            with checkpoint_state_dict_loader(
                torch,
                "local_mmap",
                (Path("weights.pth"),),
                integrity_preverified=False,
            ):
                pass


class Dinov3AdapterContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.segmenter = SEGMENTER_PATH.read_text(encoding="utf-8")
        cls.lift = LIFT_PATH.read_text(encoding="utf-8")

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

    def test_adapter_resizes_short_side_before_sliding_inference(self) -> None:
        self.assertIn("short_side=crop_size", self.segmenter)
        self.assertIn(
            '"mode": "resize_short_side_to_crop_size_before_sliding"',
            self.segmenter,
        )
        self.assertIn('"dinov3_preprocessing": preprocessing', self.segmenter)

    def test_adapter_records_allocator_limit_and_observed_peak(self) -> None:
        self.assertIn('"--max-cuda-memory-gib"', self.segmenter)
        self.assertIn('"cuda_memory_limit": cuda_memory_limit', self.segmenter)
        self.assertIn('"cuda_memory_usage": memory_usage', self.segmenter)
        self.assertIn("torch.cuda.max_memory_allocated", self.segmenter)
        self.assertIn("torch.cuda.max_memory_reserved", self.segmenter)
        self.assertIn("return_memory_limit: bool = False", self.segmenter)
        self.assertIn("if return_memory_limit:", self.segmenter)

    def test_adapter_records_opt_in_local_mmap_loading(self) -> None:
        self.assertIn('"--checkpoint-load-mode"', self.segmenter)
        self.assertIn('choices=("standard", "local_mmap")', self.segmenter)
        self.assertIn('"checkpoint_loading": checkpoint_loading', self.segmenter)
        self.assertIn("weights_only=True", self.segmenter)
        self.assertIn("mmap=True", self.segmenter)
        self.assertIn("checkpoints_preverified=True", self.segmenter)
        self.assertIn(
            "integrity_preverified=checkpoints_preverified",
            self.segmenter,
        )

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

    def test_lift_propagates_the_segmentation_source(self) -> None:
        self.assertIn('"segmentation_source": manifest["source"]', self.lift)


class Dinov3SchedulerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scheduler = SCHEDULER_PATH.read_text(encoding="utf-8")

    def test_separate_runtime_environments_are_selected(self) -> None:
        self.assertIn(
            'GAUSSIAN_ENV="${GAUSSIAN_ENV:-semantic_3dgs_renderer}"',
            self.scheduler,
        )
        self.assertIn('DINOV3_ENV="${DINOV3_ENV:-dinov3_semantic}"', self.scheduler)

    def test_complete_pipeline_stages_run_in_order(self) -> None:
        stages = [
            "run_stage 01_render_real_camera_views",
            "run_stage 02_dinov3_ade20k_mask2former",
            "run_stage 03_lift_per_view_votes",
            "run_stage 04_recover_and_materialize",
            "run_stage 05_validate_semantic_ply",
        ]
        positions = [self.scheduler.index(stage) for stage in stages]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('VIEW_COUNT="${VIEW_COUNT:-0}"', self.scheduler)
        self.assertIn("--min-pixel-confidence 0.0", self.scheduler)

    def test_existing_output_requires_an_explicit_reset(self) -> None:
        self.assertIn('RESET_OUTPUT="${RESET_OUTPUT:-0}"', self.scheduler)
        self.assertIn("Output exists; choose a new OUTPUT_NAME", self.scheduler)

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
        self.assertIn(
            'DINOV3_CHECKPOINT_LOAD_MODE="${'
            'DINOV3_CHECKPOINT_LOAD_MODE:-local_mmap}"',
            self.scheduler,
        )
        self.assertIn(
            '--checkpoint-load-mode "${DINOV3_CHECKPOINT_LOAD_MODE}"',
            self.scheduler,
        )
        self.assertIn(
            "dinov3_checkpoint_load_mode=",
            self.scheduler,
        )
        self.assertIn(
            'DINOV3_MAX_CUDA_MEMORY_GIB="${DINOV3_MAX_CUDA_MEMORY_GIB:-42}"',
            self.scheduler,
        )
        self.assertIn("--max-cuda-memory-gib", self.scheduler)
        self.assertIn("DINOV3_CROP_SIZE must be a positive multiple of 32", self.scheduler)
        self.assertIn("DINOV3_STRIDE > DINOV3_CROP_SIZE", self.scheduler)
        self.assertIn("semantic_point_cloud.ply", self.scheduler)
        self.assertIn("semantic_point_cloud_supersplat_debug.ply", self.scheduler)


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
