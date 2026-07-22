#!/usr/bin/env python3
"""Write a versioned semantic color legend as JSON and PNG."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.task1.common.semantic_palette import (
    PALETTE_VERSION,
    label_color_key,
    normalize_class_name,
    rgb8_for_label,
    validate_color_mode,
)


def load_label_map(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("labels"), list):
        raise ValueError(f"{path} must contain a labels list")
    return data


def build_legend_records(
    label_map: dict[str, Any],
    labels: np.ndarray | None,
    color_mode: str,
) -> list[dict[str, Any]]:
    mode = validate_color_mode(color_mode)
    histogram: dict[int, int] = {}
    if labels is not None:
        if labels.ndim != 1:
            raise ValueError("labels array must be one-dimensional")
        ids, counts = np.unique(labels, return_counts=True)
        histogram = {int(label_id): int(count) for label_id, count in zip(ids, counts)}

    grouped: dict[str, dict[str, Any]] = {}
    for raw_item in label_map["labels"]:
        item = dict(raw_item)
        label_id = int(item["id"])
        class_name = normalize_class_name(
            item.get("class", "unlabeled" if label_id == 0 else "unknown")
        )
        color_key = label_color_key(label_id, item, mode)
        if color_key not in grouped:
            grouped[color_key] = {
                "color_key": color_key,
                "class": class_name,
                "rgb": rgb8_for_label(label_id, {label_id: item}, color_mode=mode),
                "label_ids": [],
                "label_names": [],
                "gaussian_count": 0,
            }
        record = grouped[color_key]
        record["label_ids"].append(label_id)
        record["label_names"].append(str(item.get("name", f"label_{label_id}")))
        record["gaussian_count"] += histogram.get(
            label_id,
            int(item.get("gaussian_count", 0)),
        )

    return sorted(
        grouped.values(),
        key=lambda item: (item["class"] == "unlabeled", item["class"], item["color_key"]),
    )


def render_legend(records: list[dict[str, Any]], output: Path, columns: int) -> None:
    if not records:
        raise ValueError("Cannot render an empty semantic legend")
    columns = max(1, min(columns, len(records)))
    rows = int(math.ceil(len(records) / columns))
    column_width = 390
    row_height = 30
    header_height = 48
    image = Image.new(
        "RGB",
        (columns * column_width, header_height + rows * row_height + 12),
        (248, 248, 248),
    )
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw.text((12, 10), f"Semantic palette: {PALETTE_VERSION}", fill=(20, 20, 20), font=font)
    draw.text((12, 26), "class | RGB | Gaussian count", fill=(70, 70, 70), font=font)
    for index, record in enumerate(records):
        column = index // rows
        row = index % rows
        x = column * column_width + 12
        y = header_height + row * row_height
        rgb = tuple(int(value) for value in record["rgb"])
        draw.rectangle((x, y + 4, x + 21, y + 25), fill=rgb, outline=(30, 30, 30))
        text = (
            f"{record['color_key']} | {rgb[0]:03d},{rgb[1]:03d},{rgb[2]:03d} | "
            f"{int(record['gaussian_count']):,}"
        )
        draw.text((x + 30, y + 8), text, fill=(20, 20, 20), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-map", required=True, type=Path)
    parser.add_argument("--labels-npy", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-png", required=True, type=Path)
    parser.add_argument("--color-mode", choices=("class", "instance"), default="class")
    parser.add_argument("--columns", default=3, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for output in (args.output_json, args.output_png):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    label_map = load_label_map(args.label_map)
    labels = np.load(args.labels_npy) if args.labels_npy is not None else None
    records = build_legend_records(label_map, labels, args.color_mode)
    payload = {
        "scene": label_map.get("scene", ""),
        "palette_version": PALETTE_VERSION,
        "color_mode": args.color_mode,
        "record_count": len(records),
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    render_legend(records, args.output_png, args.columns)
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
