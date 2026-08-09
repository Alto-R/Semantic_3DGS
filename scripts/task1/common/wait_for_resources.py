#!/usr/bin/env python3
"""Wait for explicit host-RAM and GPU-free-memory thresholds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable


KIB_PER_GIB = 1024**2
MIB_PER_GIB = 1024


def host_available_gib(
    meminfo_path: Path = Path("/proc/meminfo"),
) -> float:
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition(":")
        if name == "MemAvailable" and separator:
            fields = value.split()
            if len(fields) != 2 or fields[1] != "kB":
                raise ValueError(f"Unexpected MemAvailable line: {line}")
            return int(fields[0]) / KIB_PER_GIB
    raise ValueError(f"MemAvailable is missing from {meminfo_path}")


def visible_gpu_id(explicit_gpu_id: str | None) -> str:
    if explicit_gpu_id:
        return explicit_gpu_id
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_id = visible_devices.partition(",")[0].strip()
    if not gpu_id or gpu_id in {"-1", "NoDevFiles"}:
        raise ValueError(
            "A GPU threshold requires --gpu-id or CUDA_VISIBLE_DEVICES"
        )
    return gpu_id


def gpu_free_gib(gpu_id: str) -> float:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            gpu_id,
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise ValueError(
            f"Expected one nvidia-smi memory value for {gpu_id}; got {lines}"
        )
    return float(lines[0]) / MIB_PER_GIB


def wait_for_resources(
    *,
    min_host_available_gib: float | None,
    min_gpu_free_gib: float | None,
    gpu_id: str | None,
    interval_seconds: float,
    timeout_seconds: float,
    label: str,
    host_reader: Callable[[], float] = host_available_gib,
    gpu_reader: Callable[[str], float] = gpu_free_gib,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    if min_host_available_gib is None and min_gpu_free_gib is None:
        raise ValueError("At least one resource threshold is required")
    for name, threshold in (
        ("min-host-available-gib", min_host_available_gib),
        ("min-gpu-free-gib", min_gpu_free_gib),
    ):
        if threshold is not None and threshold <= 0.0:
            raise ValueError(f"{name} must be positive")
    if interval_seconds <= 0.0:
        raise ValueError("interval-seconds must be positive")
    if timeout_seconds < 0.0:
        raise ValueError("timeout-seconds must be nonnegative")

    selected_gpu_id = (
        visible_gpu_id(gpu_id)
        if min_gpu_free_gib is not None
        else None
    )
    started = clock()
    while True:
        host_gib = (
            host_reader()
            if min_host_available_gib is not None
            else None
        )
        gpu_gib = (
            gpu_reader(selected_gpu_id)
            if selected_gpu_id is not None
            else None
        )
        host_ready = (
            min_host_available_gib is None
            or (
                host_gib is not None
                and host_gib >= min_host_available_gib
            )
        )
        gpu_ready = (
            min_gpu_free_gib is None
            or (
                gpu_gib is not None
                and gpu_gib >= min_gpu_free_gib
            )
        )
        elapsed_seconds = clock() - started
        record: dict[str, object] = {
            "source": "resource_availability_snapshot",
            "label": label,
            "state": "ready" if host_ready and gpu_ready else "waiting",
            "elapsed_seconds": round(elapsed_seconds, 3),
            "host_available_gib": host_gib,
            "min_host_available_gib": min_host_available_gib,
            "gpu_id": selected_gpu_id,
            "gpu_free_gib": gpu_gib,
            "min_gpu_free_gib": min_gpu_free_gib,
            "snapshot_only_not_a_reservation": True,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        if host_ready and gpu_ready:
            return record
        if timeout_seconds > 0.0 and elapsed_seconds >= timeout_seconds:
            raise TimeoutError(
                f"Resource wait timed out after {elapsed_seconds:.1f} seconds"
            )
        sleeper(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-host-available-gib", type=float)
    parser.add_argument("--min-gpu-free-gib", type=float)
    parser.add_argument("--gpu-id")
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help="Zero waits indefinitely.",
    )
    parser.add_argument("--label", default="dinov3")
    args = parser.parse_args()

    wait_for_resources(
        min_host_available_gib=args.min_host_available_gib,
        min_gpu_free_gib=args.min_gpu_free_gib,
        gpu_id=args.gpu_id,
        interval_seconds=args.interval_seconds,
        timeout_seconds=args.timeout_seconds,
        label=args.label,
    )


if __name__ == "__main__":
    main()
