from __future__ import annotations

import unittest

import numpy as np

from scripts.task1.dinov2.recover_dinov2_abstentions_pipeline import (
    camera_reliabilities,
    recover_weighted,
)
from scripts.task1.dinov3.round_trip_fidelity_audit import (
    STATUS_ACCEPTED,
    consensus_statistics,
)


class Dinov2RecoveryPipelineTest(unittest.TestCase):
    def test_recovery_only_fills_multicamera_weighted_majority(self) -> None:
        gaussian_count = 6
        class_count = 5
        # camera winners per Gaussian (rows = cameras, 0 = abstain).
        winners = np.asarray(
            [
                [1, 1, 1, 1, 1, 0],   # camera 0
                [1, 2, 0, 1, 1, 0],   # camera 1
                [1, 2, 0, 2, 1, 0],   # camera 2
            ],
            dtype=np.uint16,
        )
        norm_mass = np.ones_like(winners, dtype=np.float32)
        evidence = [
            {
                "camera_index": camera,
                "camera_id": camera,
                "winners": winners[camera],
                "norm_mass": norm_mass[camera],
            }
            for camera in range(3)
        ]
        counts = np.zeros((class_count + 1, gaussian_count), dtype=np.uint16)
        for item in evidence:
            supported = np.flatnonzero(item["winners"])
            np.add.at(
                counts,
                (item["winners"][supported].astype(np.int64), supported),
                1,
            )
        statistics = consensus_statistics(counts, chunk_size=10)
        status = statistics["status"]
        # g0: 3x class1 -> accepted; g1: 1x class1 + 2x class2 -> no majority;
        # g2: single camera -> single; g3: 2x class1 + 1x class2 -> accepted;
        # g4: 3x class1 -> accepted; g5: unobserved.
        self.assertEqual(int(status[0]), STATUS_ACCEPTED)
        reliabilities = camera_reliabilities(statistics, evidence, z=1.96)
        for value in reliabilities.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        recovered, source_codes = recover_weighted(
            evidence,
            reliabilities,
            status,
            gaussian_count=gaussian_count,
            class_count=class_count,
        )
        # g0/g3/g4 are locked anchors.
        np.testing.assert_array_equal(source_codes[[0, 3, 4]], [1, 1, 1])
        # g5 stays unresolved (unobserved).
        self.assertEqual(int(source_codes[5]), 0)
        # g1 has 2 cameras -> may recover to the weighted winner; g2 has 1
        # camera -> never recovers.
        self.assertEqual(int(source_codes[2]), 0)
        self.assertEqual(int(recovered[2]), 0)
        self.assertEqual(int(recovered[5]), 0)


if __name__ == "__main__":
    unittest.main()
