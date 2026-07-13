#!/usr/bin/env python3
"""Append deterministic RGB debug colors to a labeled binary PLY."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "task1"))

from ply_utils import element_stride, read_ply_header, scalar_property_size
from semantic_palette import load_label_items, palette_records, rgb8_for_label


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


def insert_rgb_properties(lines: list[str]) -> bytes:
    new_lines: list[str] = []
    in_vertex = False
    inserted = False

    for line in lines:
        if line == "end_header" and in_vertex and not inserted:
            new_lines.extend(["property uchar red", "property uchar green", "property uchar blue"])
            inserted = True

        if line.startswith("element "):
            if in_vertex and not inserted:
                new_lines.extend(["property uchar red", "property uchar green", "property uchar blue"])
                inserted = True
            parts = line.split()
            in_vertex = len(parts) >= 2 and parts[1] == "vertex"

        new_lines.append(line)

    if not inserted:
        raise ValueError("Could not find vertex element in PLY header")
    return ("\n".join(new_lines) + "\n").encode("ascii")


def colors_for_labels(labels: np.ndarray, label_items: dict[int, dict[str, Any]]) -> np.ndarray:
    colors = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    for label_id in np.unique(labels.astype(np.int64)):
        colors[labels == label_id] = np.asarray(
            rgb8_for_label(int(label_id), label_items),
            dtype=np.uint8,
        )
    return colors


def label_property_offset(input_ply: Path, property_name: str) -> tuple[int, np.dtype]:
    header = read_ply_header(input_ply)
    vertex = header.element("vertex")
    if vertex is None:
        raise ValueError(f"{input_ply} has no vertex element")
    endian = "<" if header.fmt == "binary_little_endian" else ">"
    offset = 0
    for prop in vertex.properties:
        if prop.is_list:
            raise ValueError(f"Vertex list properties are not supported: {prop.name}")
        if prop.name == property_name:
            return offset, dtype_for_property(prop.data_type, endian)
        offset += scalar_property_size(prop.data_type)
    raise ValueError(f"{input_ply} has no vertex property named {property_name}")


def export_debug_ply(
    input_ply: Path,
    output_ply: Path,
    label_map: Path | None,
    property_name: str,
    batch_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    header = read_ply_header(input_ply)
    vertex = header.element("vertex")
    if vertex is None:
        raise ValueError(f"{input_ply} has no vertex element")
    if header.fmt not in {"binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Unsupported PLY format: {header.fmt}")
    if not header.elements or header.elements[0].name != "vertex":
        raise ValueError("Only PLY files with vertex as the first element are supported")
    if any(prop.name in {"red", "green", "blue"} for prop in vertex.properties):
        raise ValueError(f"{input_ply} already has red/green/blue vertex properties")
    if output_ply.exists() and not overwrite:
        raise FileExistsError(f"{output_ply} exists; pass --overwrite to replace it")

    label_offset, label_dtype = label_property_offset(input_ply, property_name)
    label_items = load_label_items(label_map)
    stride = element_stride(vertex)
    new_stride = stride + 3
    new_header = insert_rgb_properties(header.lines)

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    seen_labels: set[int] = set()
    with input_ply.open("rb") as source, output_ply.open("wb") as target:
        source.seek(header.header_bytes)
        target.write(new_header)
        for start in range(0, vertex.count, batch_size):
            count = min(batch_size, vertex.count - start)
            blob = source.read(stride * count)
            if len(blob) != stride * count:
                raise ValueError("Binary PLY body is shorter than declared vertex data")
            vertex_bytes = np.frombuffer(blob, dtype=np.uint8).reshape(count, stride)
            labels = np.ndarray(
                shape=(count,),
                dtype=label_dtype,
                buffer=blob,
                offset=label_offset,
                strides=(stride,),
            ).astype(np.int64)
            seen_labels.update(int(label_id) for label_id in np.unique(labels))
            colors = colors_for_labels(labels, label_items)
            combined = np.empty((count, new_stride), dtype=np.uint8)
            combined[:, :stride] = vertex_bytes
            combined[:, stride:] = colors
            target.write(combined.tobytes())
        shutil.copyfileobj(source, target)

    return {
        "input_ply": str(input_ply),
        "output_ply": str(output_ply),
        "label_map": str(label_map) if label_map is not None else "",
        "vertex_count": vertex.count,
        "label_property": property_name,
        "label_count": len(seen_labels),
        "palette": palette_records(seen_labels, label_items),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ply", required=True, type=Path)
    parser.add_argument("--output-ply", required=True, type=Path)
    parser.add_argument("--label-map", type=Path)
    parser.add_argument("--label-property", default="label")
    parser.add_argument("--batch-size", default=100_000, type=int)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata = export_debug_ply(
        input_ply=args.input_ply,
        output_ply=args.output_ply,
        label_map=args.label_map,
        property_name=args.label_property,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    metadata_path = args.metadata_json or args.output_ply.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
