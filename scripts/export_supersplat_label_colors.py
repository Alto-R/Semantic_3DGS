#!/usr/bin/env python3
"""Bake per-label debug colors into 3DGS SH coefficients for SuperSplat."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ply_utils import element_stride, read_ply_header, scalar_property_size
from semantic_palette import (
    load_label_items,
    normalize_classes,
    palette_records,
    rgb_for_label,
)


SH_C0 = 0.28209479177387814

SCALAR_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def dtype_for_property(data_type: str, endian: str) -> np.dtype:
    if data_type not in SCALAR_DTYPES:
        raise ValueError(f"Unsupported scalar PLY type: {data_type}")
    code = SCALAR_DTYPES[data_type]
    if code.endswith("1"):
        return np.dtype(code)
    return np.dtype(f"{endian}{code}")


def vertex_property_offsets(input_ply: Path) -> tuple[dict[str, tuple[int, np.dtype]], int]:
    header = read_ply_header(input_ply)
    vertex = header.element("vertex")
    if vertex is None:
        raise ValueError(f"{input_ply} has no vertex element")
    endian = "<" if header.fmt == "binary_little_endian" else ">"
    offsets: dict[str, tuple[int, np.dtype]] = {}
    offset = 0
    for prop in vertex.properties:
        if prop.is_list:
            raise ValueError(f"Vertex list properties are not supported: {prop.name}")
        offsets[prop.name] = (offset, dtype_for_property(prop.data_type, endian))
        offset += scalar_property_size(prop.data_type)
    return offsets, offset


def sh_dc_for_rgb(rgb: tuple[float, float, float]) -> np.ndarray:
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / SH_C0


def collect_label_colors(
    labels: np.ndarray,
    label_items: dict[int, dict[str, Any]],
    focus_classes: set[str],
) -> dict[int, np.ndarray]:
    return {
        int(label_id): sh_dc_for_rgb(rgb_for_label(int(label_id), label_items, focus_classes))
        for label_id in np.unique(labels)
    }


def mutable_column(
    blob: bytearray,
    count: int,
    stride: int,
    offset: int,
    dtype: np.dtype,
) -> np.ndarray:
    return np.ndarray(shape=(count,), dtype=dtype, buffer=blob, offset=offset, strides=(stride,))


def export_supersplat_debug_ply(
    input_ply: Path,
    output_ply: Path,
    label_map: Path | None,
    label_property: str,
    batch_size: int,
    overwrite: bool,
    focus_classes: set[str],
) -> dict[str, Any]:
    header = read_ply_header(input_ply)
    vertex = header.element("vertex")
    if vertex is None:
        raise ValueError(f"{input_ply} has no vertex element")
    if header.fmt not in {"binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Unsupported PLY format: {header.fmt}")
    if not header.elements or header.elements[0].name != "vertex":
        raise ValueError("Only PLY files with vertex as the first element are supported")
    if output_ply.exists() and not overwrite:
        raise FileExistsError(f"{output_ply} exists; pass --overwrite to replace it")

    offsets, stride = vertex_property_offsets(input_ply)
    required = [label_property, "f_dc_0", "f_dc_1", "f_dc_2"]
    missing = [name for name in required if name not in offsets]
    if missing:
        raise ValueError(f"{input_ply} is missing required SuperSplat color properties: {missing}")

    label_offset, label_dtype = offsets[label_property]
    label_items = load_label_items(label_map)
    dc_offsets = [offsets[f"f_dc_{index}"] for index in range(3)]
    for name, (_, dtype) in zip(["f_dc_0", "f_dc_1", "f_dc_2"], dc_offsets):
        if dtype.kind != "f":
            raise ValueError(f"{name} must be a floating-point PLY property")
    rest_offsets = [
        item
        for name, item in sorted(offsets.items(), key=lambda pair: pair[0])
        if name.startswith("f_rest_")
    ]
    for name, (_, dtype) in [(name, offsets[name]) for name in offsets if name.startswith("f_rest_")]:
        if dtype.kind != "f":
            raise ValueError(f"{name} must be a floating-point PLY property")

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    seen_labels: set[int] = set()
    with input_ply.open("rb") as source, output_ply.open("wb") as target:
        header_bytes = source.read(header.header_bytes)
        target.write(header_bytes)
        for start in range(0, vertex.count, batch_size):
            count = min(batch_size, vertex.count - start)
            blob = bytearray(source.read(stride * count))
            if len(blob) != stride * count:
                raise ValueError("Binary PLY body is shorter than declared vertex data")

            labels = mutable_column(blob, count, stride, label_offset, label_dtype).astype(np.int64)
            seen_labels.update(int(label_id) for label_id in np.unique(labels))
            dc_by_label = collect_label_colors(labels, label_items, focus_classes)

            for label_id, dc_color in dc_by_label.items():
                selected = labels == label_id
                for channel, (offset, dtype) in enumerate(dc_offsets):
                    column = mutable_column(blob, count, stride, offset, dtype)
                    column[selected] = dc_color[channel]

            for offset, dtype in rest_offsets:
                mutable_column(blob, count, stride, offset, dtype)[:] = 0.0

            target.write(blob)
        shutil.copyfileobj(source, target)

    return {
        "input_ply": str(input_ply),
        "output_ply": str(output_ply),
        "label_map": str(label_map) if label_map is not None else "",
        "vertex_count": vertex.count,
        "label_property": label_property,
        "color_mode": "supersplat_sh_dc",
        "focus_classes": sorted(focus_classes),
        "zeroed_f_rest_property_count": len(rest_offsets),
        "label_count": len(seen_labels),
        "palette": palette_records(seen_labels, label_items, focus_classes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ply", required=True, type=Path)
    parser.add_argument("--output-ply", required=True, type=Path)
    parser.add_argument("--label-map", type=Path)
    parser.add_argument("--label-property", default="label")
    parser.add_argument("--batch-size", default=100_000, type=int)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--focus-classes", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata = export_supersplat_debug_ply(
        input_ply=args.input_ply,
        output_ply=args.output_ply,
        label_map=args.label_map,
        label_property=args.label_property,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        focus_classes=normalize_classes(args.focus_classes),
    )
    metadata_path = args.metadata_json or args.output_ply.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
