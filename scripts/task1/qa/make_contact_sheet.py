#!/usr/bin/env python3
"""Create a labeled contact sheet from a directory of rendered images."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw, ImageFont


def image_paths(image_dir: Path, patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(image_dir.glob(pattern))
    return sorted({path for path in paths if path.is_file()})


def fit_size(width: int, height: int, target_width: int) -> tuple[int, int]:
    if target_width <= 0 or width <= target_width:
        return width, height
    scale = target_width / float(width)
    return target_width, max(1, int(round(height * scale)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pattern", action="append", default=["*.png"])
    parser.add_argument("--columns", default=5, type=int)
    parser.add_argument("--thumb-width", default=240, type=int)
    parser.add_argument("--label-height", default=24, type=int)
    parser.add_argument("--background", default="#202124")
    parser.add_argument("--text", default="#f1f3f4")
    args = parser.parse_args()

    paths = image_paths(args.image_dir, args.pattern)
    if not paths:
        raise FileNotFoundError(f"No images matched in {args.image_dir}")
    if args.columns <= 0:
        raise ValueError("--columns must be positive")

    thumbs: List[tuple[Path, Image.Image]] = []
    cell_width = 0
    cell_image_height = 0
    for path in paths:
        with Image.open(path) as source:
            source = source.convert("RGB")
            size = fit_size(source.width, source.height, args.thumb_width)
            thumb = source.resize(size, Image.Resampling.LANCZOS)
        thumbs.append((path, thumb))
        cell_width = max(cell_width, thumb.width)
        cell_image_height = max(cell_image_height, thumb.height)

    rows = int(math.ceil(len(thumbs) / args.columns))
    cell_height = cell_image_height + args.label_height
    sheet = Image.new("RGB", (cell_width * args.columns, cell_height * rows), args.background)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, (path, thumb) in enumerate(thumbs):
        row = index // args.columns
        col = index % args.columns
        x = col * cell_width
        y = row * cell_height
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + cell_image_height + 4), path.stem, fill=args.text, font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"wrote {args.output} ({len(thumbs)} images)")


if __name__ == "__main__":
    main()
