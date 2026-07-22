#!/usr/bin/env python3
"""Measure projected semantic coverage from RGB and overlay render pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.int16)


def frame_coverage(
    rgb_path: Path,
    overlay_path: Path,
    difference_threshold: int,
    exclude_border: int,
) -> dict[str, Any]:
    rgb = load_rgb(rgb_path)
    overlay = load_rgb(overlay_path)
    if rgb.shape != overlay.shape:
        raise ValueError(f"Image shape mismatch: {rgb_path} {rgb.shape} != {overlay_path} {overlay.shape}")

    border = max(0, exclude_border)
    if border > 0:
        if min(rgb.shape[:2]) <= border * 2:
            raise ValueError(f"Border {border} is too large for image shape {rgb.shape}")
        rgb = rgb[border:-border, border:-border]
        overlay = overlay[border:-border, border:-border]

    absolute_difference = np.abs(overlay - rgb)
    changed = absolute_difference.max(axis=2) > max(0, difference_threshold)
    changed_count = int(changed.sum())
    pixel_count = int(changed.size)
    return {
        "file": overlay_path.name,
        "width": int(changed.shape[1]),
        "height": int(changed.shape[0]),
        "pixel_count": pixel_count,
        "overlay_changed_pixel_count": changed_count,
        "overlay_changed_pixel_ratio": changed_count / float(max(pixel_count, 1)),
        "mean_absolute_channel_difference": float(absolute_difference.mean()),
    }


def measure_directory(
    rgb_dir: Path,
    overlay_dir: Path,
    difference_threshold: int,
    exclude_border: int,
) -> dict[str, Any]:
    overlay_paths = sorted(overlay_dir.glob("*.png"))
    if not overlay_paths:
        raise ValueError(f"No PNG overlays found in {overlay_dir}")

    frames: list[dict[str, Any]] = []
    for overlay_path in overlay_paths:
        rgb_path = rgb_dir / overlay_path.name
        if not rgb_path.exists():
            raise FileNotFoundError(rgb_path)
        frames.append(
            frame_coverage(
                rgb_path,
                overlay_path,
                difference_threshold,
                exclude_border,
            )
        )

    ratios = np.asarray([frame["overlay_changed_pixel_ratio"] for frame in frames])
    changed_total = sum(int(frame["overlay_changed_pixel_count"]) for frame in frames)
    pixel_total = sum(int(frame["pixel_count"]) for frame in frames)
    return {
        "method": "rgb_vs_semantic_overlay_pixel_difference",
        "interpretation": (
            "Image-space proxy: a changed pixel received a visible semantic overlay. "
            "This is not semantic ground truth and can undercount labels whose palette color "
            "closely matches the original RGB value."
        ),
        "rgb_dir": str(rgb_dir),
        "overlay_dir": str(overlay_dir),
        "difference_threshold": max(0, difference_threshold),
        "excluded_border_pixels": max(0, exclude_border),
        "frame_count": len(frames),
        "pixel_count": pixel_total,
        "overlay_changed_pixel_count": changed_total,
        "overlay_changed_pixel_ratio": changed_total / float(max(pixel_total, 1)),
        "frame_ratio_min": float(ratios.min()),
        "frame_ratio_p10": float(np.quantile(ratios, 0.10)),
        "frame_ratio_median": float(np.median(ratios)),
        "frame_ratio_p90": float(np.quantile(ratios, 0.90)),
        "frame_ratio_max": float(ratios.max()),
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", required=True, type=Path)
    parser.add_argument("--overlay-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--difference-threshold", default=2, type=int)
    parser.add_argument("--exclude-border", default=1, type=int)
    args = parser.parse_args()

    report = measure_directory(
        args.rgb_dir,
        args.overlay_dir,
        args.difference_threshold,
        args.exclude_border,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
