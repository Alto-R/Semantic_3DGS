from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.task1.dinov3.dinov2_second_source import (
    SOURCE,
    aggregate_component_votes_agreement_gated,
    align_dinov2_evidence,
    collapse_dinov2_camera,
    exclude_dinov2_camera,
    load_dinov2_evidence,
)
from scripts.task1.dinov3.observed_black_component_graph_audit import (
    aggregate_component_votes,
    semantic_cache,
)


ROOT = Path(__file__).resolve().parents[1]


def write_dinov2_manifest(
    root: Path,
    frames: list[dict],
    *,
    gaussian_count: int,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": SOURCE,
        "gaussian_count": gaussian_count,
        "camera_count": len(frames),
        "frames": frames,
    }
    path = root / "vote_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class CollapseDinov2CameraTest(unittest.TestCase):
    def test_unique_winner_and_mass_without_normalization(self) -> None:
        indices = np.asarray([0, 0, 1, 1, 2], dtype=np.uint32)
        classes = np.asarray([15, 20, 15, 15, 0], dtype=np.uint16)
        weights = np.asarray([0.4, 0.3, 0.5, 0.2, 0.9], dtype=np.float32)
        winner, mass = collapse_dinov2_camera(
            indices,
            classes,
            weights,
            gaussian_count=4,
            class_count=25,
        )
        # Gaussian 0: 15 wins with 0.4; Gaussian 1: duplicate 15 rows
        # aggregate to 0.7; Gaussian 2: class 0 is unlabeled; Gaussian 3:
        # unseen.
        np.testing.assert_array_equal(winner, [15, 15, 0, 0])
        np.testing.assert_allclose(mass, [0.4, 0.7, 0.0, 0.0])

    def test_exact_tie_abstains(self) -> None:
        winner, mass = collapse_dinov2_camera(
            np.asarray([0, 0], dtype=np.uint32),
            np.asarray([15, 20], dtype=np.uint16),
            np.asarray([0.5, 0.5], dtype=np.float32),
            gaussian_count=1,
            class_count=25,
        )
        self.assertEqual(int(winner[0]), 0)
        self.assertEqual(float(mass[0]), 0.0)

    def test_rejects_invalid_arrays(self) -> None:
        with self.assertRaisesRegex(ValueError, "aligned"):
            collapse_dinov2_camera(
                np.asarray([0], dtype=np.uint32),
                np.asarray([15, 20], dtype=np.uint16),
                np.asarray([0.5, 0.5], dtype=np.float32),
                gaussian_count=1,
                class_count=25,
            )
        with self.assertRaisesRegex(ValueError, "outside the model"):
            collapse_dinov2_camera(
                np.asarray([4], dtype=np.uint32),
                np.asarray([15], dtype=np.uint16),
                np.asarray([0.5], dtype=np.float32),
                gaussian_count=4,
                class_count=25,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            collapse_dinov2_camera(
                np.asarray([0], dtype=np.uint32),
                np.asarray([15], dtype=np.uint16),
                np.asarray([-0.5], dtype=np.float32),
                gaussian_count=1,
                class_count=25,
            )


class LoadAndAlignTest(unittest.TestCase):
    def test_load_and_align_by_camera_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez_compressed(
                root / "camera_5.npz",
                indices=np.asarray([0], dtype=np.uint32),
                class_ids=np.asarray([15], dtype=np.uint16),
                weights=np.asarray([0.5], dtype=np.float32),
            )
            manifest_path = write_dinov2_manifest(
                root,
                [
                    {
                        "camera_index": 5,
                        "camera_id": 5,
                        "file": "camera_5.png",
                        "vote_file": "camera_5.npz",
                    }
                ],
                gaussian_count=2,
            )
            dinov2 = load_dinov2_evidence(
                manifest_path, gaussian_count=2, class_count=25
            )
            self.assertEqual(int(dinov2[0]["camera_id"]), 5)
            aligned = align_dinov2_evidence(
                [
                    {
                        "camera_index": 5,
                        "camera_id": 5,
                        "winners": np.zeros((2,), dtype=np.uint16),
                    }
                ],
                dinov2,
            )
            self.assertEqual(int(aligned[0]["camera_id"]), 5)

    def test_alignment_rejects_missing_counterpart(self) -> None:
        with self.assertRaisesRegex(ValueError, "no DINOv2 counterpart"):
            align_dinov2_evidence(
                [{"camera_index": 7, "camera_id": 7}],
                [{"camera_index": 5, "camera_id": 5}],
            )

    def test_exclude_dinov2_camera_removes_heldout_row(self) -> None:
        rows = [
            {"camera_index": 1, "camera_id": 1},
            {"camera_index": 2, "camera_id": 2},
            {"camera_index": 3, "camera_id": 3},
        ]
        remaining = exclude_dinov2_camera(rows, 2)
        self.assertEqual(
            [int(row["camera_id"]) for row in remaining], [1, 3]
        )


class AgreementGateTest(unittest.TestCase):
    @staticmethod
    def evidence(
        camera_id: int,
        winners: np.ndarray,
        masses: np.ndarray,
    ) -> dict:
        return {
            "camera_index": camera_id,
            "camera_id": camera_id,
            "file": f"camera_{camera_id}.png",
            "winners": winners,
            "mass": masses,
            "source": SOURCE,
        }

    def test_camera_contributes_only_when_component_winners_agree(self) -> None:
        gaussian_count = 6
        class_count = 25
        components = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
        # DINOv3: component 0 is class 15, component 1 is class 20.
        d3 = self.evidence(
            0,
            np.asarray([15, 15, 15, 20, 20, 20], dtype=np.uint16),
            np.asarray([0.4, 0.4, 0.4, 0.5, 0.5, 0.5], dtype=np.float32),
        )
        # DINOv2 agrees on component 0 but disagrees on component 1.
        d2 = self.evidence(
            0,
            np.asarray([15, 15, 15, 21, 21, 21], dtype=np.uint16),
            np.asarray([0.3, 0.3, 0.3, 0.5, 0.5, 0.5], dtype=np.float32),
        )
        reliability = {0: 0.8}
        d3_cache = semantic_cache(
            [d3], reliability, np.arange(gaussian_count), class_count=class_count
        )
        d2_cache = semantic_cache(
            [d2], reliability, np.arange(gaussian_count), class_count=class_count
        )
        votes = aggregate_component_votes_agreement_gated(
            components,
            d3_cache,
            d2_cache,
            [d3],
            [d2],
            reliability,
            class_count=class_count,
        )
        self.assertEqual(int(votes["camera_count"][0]), 1)
        self.assertEqual(int(votes["camera_count"][1]), 0)
        self.assertEqual(int(votes["raw"][15, 0]), 1)
        self.assertEqual(int(votes["raw"][20, 1]), 0)

    def test_matching_dinov2_matches_plain_aggregation(self) -> None:
        gaussian_count = 4
        class_count = 25
        components = np.asarray([0, 0, 1, 1], dtype=np.int32)
        item = self.evidence(
            1,
            np.asarray([15, 15, 20, 20], dtype=np.uint16),
            np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        )
        reliability = {1: 0.9}
        cache = semantic_cache(
            [item], reliability, np.arange(gaussian_count), class_count=class_count
        )
        plain = aggregate_component_votes(
            components, cache, [item], reliability, class_count=class_count
        )
        gated = aggregate_component_votes_agreement_gated(
            components,
            cache,
            cache,
            [item],
            [item],
            reliability,
            class_count=class_count,
        )
        np.testing.assert_array_equal(gated["raw"], plain["raw"])
        np.testing.assert_array_equal(gated["weighted"], plain["weighted"])
        np.testing.assert_array_equal(
            gated["camera_count"], plain["camera_count"]
        )

    def test_dinov2_abstention_abstains_even_when_dinov3_voted(self) -> None:
        gaussian_count = 2
        class_count = 25
        components = np.asarray([0, 0], dtype=np.int32)
        d3 = self.evidence(
            0,
            np.asarray([15, 15], dtype=np.uint16),
            np.asarray([0.5, 0.5], dtype=np.float32),
        )
        d2 = self.evidence(
            0,
            np.asarray([0, 0], dtype=np.uint16),
            np.asarray([0.0, 0.0], dtype=np.float32),
        )
        reliability = {0: 0.8}
        d3_cache = semantic_cache(
            [d3], reliability, np.arange(gaussian_count), class_count=class_count
        )
        d2_cache = semantic_cache(
            [d2], reliability, np.arange(gaussian_count), class_count=class_count
        )
        votes = aggregate_component_votes_agreement_gated(
            components,
            d3_cache,
            d2_cache,
            [d3],
            [d2],
            reliability,
            class_count=class_count,
        )
        self.assertEqual(int(votes["camera_count"][0]), 0)
        self.assertEqual(int(votes["raw"][15, 0]), 0)


class SchedulerContractTest(unittest.TestCase):
    def test_no_v5_references_in_second_source_module(self) -> None:
        module = ROOT / "scripts" / "task1" / "dinov3" / "dinov2_second_source.py"
        self.assertNotIn("v" + "5", module.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
