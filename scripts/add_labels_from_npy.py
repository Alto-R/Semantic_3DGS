#!/usr/bin/env python3
"""Add per-vertex integer labels from a NumPy array to a scalar-vertex PLY."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from ply_utils import element_stride, read_ply_header


def insert_vertex_property(lines: list[str], property_name: str) -> bytes:
    new_lines: list[str] = []
    in_vertex = False
    inserted = False

    for line in lines:
        if line == "end_header" and in_vertex and not inserted:
            new_lines.append(f"property int {property_name}")
            inserted = True

        if line.startswith("element "):
            if in_vertex and not inserted:
                new_lines.append(f"property int {property_name}")
                inserted = True
            parts = line.split()
            in_vertex = len(parts) >= 2 and parts[1] == "vertex"

        new_lines.append(line)

    if not inserted:
        raise ValueError("Could not find vertex element in PLY header")

    return ("\n".join(new_lines) + "\n").encode("ascii")


def write_binary_with_labels(
    src: Path,
    dst: Path,
    labels: np.ndarray,
    property_name: str,
    batch_size: int = 100_000,
) -> None:
    header = read_ply_header(src)
    vertex = header.element("vertex")
    if vertex is None:
        raise ValueError("PLY has no vertex element")
    if any(prop.name == property_name for prop in vertex.properties):
        raise ValueError(f"PLY already has vertex property {property_name}")
    if not header.elements or header.elements[0].name != "vertex":
        raise ValueError("Only PLY files with vertex as the first element are supported")
    if labels.shape[0] != vertex.count:
        raise ValueError(f"Label count {labels.shape[0]} does not match vertex count {vertex.count}")

    if header.fmt == "binary_little_endian":
        label_dtype = np.dtype("<i4")
    elif header.fmt == "binary_big_endian":
        label_dtype = np.dtype(">i4")
    else:
        raise ValueError(f"Unsupported binary format: {header.fmt}")

    labels = np.asarray(labels, dtype=label_dtype)
    stride = element_stride(vertex)
    new_stride = stride + label_dtype.itemsize
    new_header = insert_vertex_property(header.lines, property_name)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as source, dst.open("wb") as target:
        source.seek(header.header_bytes)
        target.write(new_header)

        for start in range(0, vertex.count, batch_size):
            count = min(batch_size, vertex.count - start)
            blob = source.read(stride * count)
            if len(blob) != stride * count:
                raise ValueError("Binary PLY body is shorter than declared vertex data")

            vertex_bytes = np.frombuffer(blob, dtype=np.uint8).reshape(count, stride)
            label_bytes = labels[start : start + count].view(np.uint8).reshape(count, label_dtype.itemsize)
            combined = np.empty((count, new_stride), dtype=np.uint8)
            combined[:, :stride] = vertex_bytes
            combined[:, stride:] = label_bytes
            target.write(combined.tobytes())

        shutil.copyfileobj(source, target)


def write_ply_with_labels(
    src: Path,
    dst: Path,
    labels: np.ndarray,
    property_name: str = "label",
    batch_size: int = 100_000,
) -> None:
    header = read_ply_header(src)
    if header.fmt == "ascii":
        raise ValueError("ASCII PLY label writing is not implemented for per-vertex arrays")
    if header.fmt not in {"binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Unsupported PLY format: {header.fmt}")
    write_binary_with_labels(src, dst, labels, property_name, batch_size=batch_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ply", type=Path)
    parser.add_argument("labels_npy", type=Path)
    parser.add_argument("output_ply", type=Path)
    parser.add_argument("--property-name", default="label")
    parser.add_argument("--batch-size", default=100_000, type=int)
    args = parser.parse_args()

    labels = np.load(args.labels_npy)
    write_ply_with_labels(
        args.input_ply,
        args.output_ply,
        labels,
        property_name=args.property_name,
        batch_size=args.batch_size,
    )
    print(f"wrote {args.output_ply}")


if __name__ == "__main__":
    main()
