"""QA rendering for SAM3 instance masks: overlays and contact sheets."""

from __future__ import annotations

import argparse
import colorsys
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


_GOLDEN_RATIO_CONJUGATE = 0.618033988749895


def instance_palette(count: int) -> list[tuple[int, int, int]]:
    """Deterministic, well-separated RGB colors via golden-ratio hues."""

    colors: list[tuple[int, int, int]] = []
    hue = 0.0
    for _ in range(count):
        red, green, blue = colorsys.hsv_to_rgb(hue % 1.0, 0.75, 0.95)
        colors.append(
            (int(round(red * 255)), int(round(green * 255)), int(round(blue * 255)))
        )
        hue += _GOLDEN_RATIO_CONJUGATE
    return colors


def overlay_instances(
    rgb: np.ndarray, mask_stack: np.ndarray, alpha: float = 0.55
) -> np.ndarray:
    """Blend one color per mask over the image; later masks draw on top."""

    image = np.asarray(rgb)
    stack = np.asarray(mask_stack)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape height x width x 3")
    result = image.astype(np.float64)
    if stack.size:
        if stack.shape[1:] != image.shape[:2]:
            raise ValueError("mask_stack must match the image size")
        for row, color in enumerate(instance_palette(stack.shape[0])):
            covered = stack[row] > 0
            tint = np.array(color, dtype=np.float64)
            result[covered] = (1.0 - alpha) * result[covered] + alpha * tint
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def contact_sheet(tiles: list[Image.Image], columns: int) -> Image.Image:
    """Arrange equally sized tiles into a grid image."""

    if not tiles:
        raise ValueError("contact sheet needs at least one tile")
    if columns < 1:
        raise ValueError("columns must be positive")
    width, height = tiles[0].size
    for tile in tiles:
        if tile.size != (width, height):
            raise ValueError("all contact sheet tiles must share one size")
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * width, rows * height), (255, 255, 255))
    for position, tile in enumerate(tiles):
        row, column = divmod(position, columns)
        sheet.paste(tile, (column * width, row * height))
    return sheet


def main(argv: list[str] | None = None) -> None:
    from scripts.task1.sam3.segment_views_core import validate_masks_manifest

    parser = argparse.ArgumentParser()
    parser.add_argument("--masks-manifest", required=True, type=Path)
    parser.add_argument("--masks-dir", required=True, type=Path)
    parser.add_argument("--rgb-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--columns", default=8, type=int)
    parser.add_argument("--alpha", default=0.55, type=float)
    args = parser.parse_args(argv)

    manifest = json.loads(args.masks_manifest.read_text(encoding="utf-8"))
    validate_masks_manifest(manifest)
    overlays_dir = args.output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    tiles: list[Image.Image] = []
    for frame in manifest["frames"]:
        rgb_path = args.rgb_dir / str(frame["file"])
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        with np.load(args.masks_dir / str(frame["mask_file"])) as data:
            mask_stack = data["mask_stack"]
        overlay = Image.fromarray(overlay_instances(rgb, mask_stack, args.alpha))
        overlay.save(overlays_dir / f"{Path(str(frame['file'])).stem}.png")
        tiles.append(overlay)
    if tiles:
        sheet = contact_sheet(tiles, min(args.columns, len(tiles)))
        sheet.save(args.output_dir / "instance_overlays_contact.jpg", quality=90)
    print(f"wrote {len(tiles)} overlays to {overlays_dir}")


if __name__ == "__main__":
    main()
