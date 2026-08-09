from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.task1.common.wait_for_resources import (
    gpu_free_gib,
    host_available_gib,
    visible_gpu_id,
    wait_for_resources,
)


class ResourceWaitTest(unittest.TestCase):
    def test_reads_linux_memavailable_as_gib(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            meminfo = Path(temporary) / "meminfo"
            meminfo.write_text(
                "MemTotal:       131072000 kB\n"
                "MemAvailable:    83886080 kB\n",
                encoding="utf-8",
            )
            self.assertEqual(host_available_gib(meminfo), 80.0)

    def test_queries_one_explicit_gpu_without_creating_cuda_context(
        self,
    ) -> None:
        completed = SimpleNamespace(stdout="46080\n")
        with patch(
            "scripts.task1.common.wait_for_resources.subprocess.run",
            return_value=completed,
        ) as run:
            self.assertEqual(gpu_free_gib("1"), 45.0)
        run.assert_called_once_with(
            [
                "nvidia-smi",
                "--id",
                "1",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_uses_first_cuda_visible_device_when_gpu_id_is_omitted(
        self,
    ) -> None:
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,0"}):
            self.assertEqual(visible_gpu_id(None), "1")

    def test_waits_until_both_thresholds_are_satisfied(self) -> None:
        host_values = iter((70.0, 81.0))
        gpu_values = iter((44.0, 46.0))
        clock_values = iter((0.0, 0.0, 60.0))
        sleeps: list[float] = []
        output = io.StringIO()

        with redirect_stdout(output):
            record = wait_for_resources(
                min_host_available_gib=80.0,
                min_gpu_free_gib=45.0,
                gpu_id="1",
                interval_seconds=60.0,
                timeout_seconds=0.0,
                label="test",
                host_reader=lambda: next(host_values),
                gpu_reader=lambda _gpu_id: next(gpu_values),
                sleeper=sleeps.append,
                clock=lambda: next(clock_values),
            )

        self.assertEqual(record["state"], "ready")
        self.assertEqual(record["host_available_gib"], 81.0)
        self.assertEqual(record["gpu_free_gib"], 46.0)
        self.assertTrue(record["snapshot_only_not_a_reservation"])
        self.assertEqual(sleeps, [60.0])
        self.assertIn('"state": "waiting"', output.getvalue())
        self.assertIn('"state": "ready"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
