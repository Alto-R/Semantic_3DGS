#!/usr/bin/env python3
"""Add a constant integer label property to a scalar-vertex PLY file.

This is a bootstrap round-trip utility. It is intentionally conservative and
refuses to overwrite an existing label property.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "task1"))

from ply_utils import element_stride, read_ply_header


def insert_vertex_property(lines: Iterable[str], property_name: str) -> bytes:
    new_lines: List[str] = []
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


def line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    return ""


def write_ascii_with_label(src: Path, dst: Path, label: int, property_name: str) -> None:
    header = read_ply_header(src)
    vertex = header.element("vertex")
    if vertex is None:
        raise ValueError("PLY has no vertex element")
    if any(prop.name == property_name for prop in vertex.properties):
        raise ValueError(f"PLY already has vertex property {property_name}")
    if not header.elements or header.elements[0].name != "vertex":
        raise ValueError("Only PLY files with vertex as the first element are supported")

    new_header = insert_vertex_property(header.lines, property_name).decode("ascii")
    text = src.read_text(encoding="ascii")
    body = text[header.header_bytes:]
    body_lines = body.splitlines(keepends=True)
    if len(body_lines) < vertex.count:
        raise ValueError("ASCII PLY body has fewer vertex lines than declared")

    output_lines = [new_header]
    for index, line in enumerate(body_lines):
        if index < vertex.count:
            ending = line_ending(line)
            content = line[: -len(ending)] if ending else line
            output_lines.append(f"{content} {label}{ending}")
        else:
            output_lines.append(line)

    dst.write_text("".join(output_lines), encoding="ascii")


def write_binary_with_label(src: Path, dst: Path, label: int, property_name: str) -> None:
    header = read_ply_header(src)
    vertex = header.element("vertex")
    if vertex is None:
        raise ValueError("PLY has no vertex element")
    if any(prop.name == property_name for prop in vertex.properties):
        raise ValueError(f"PLY already has vertex property {property_name}")
    if not header.elements or header.elements[0].name != "vertex":
        raise ValueError("Only PLY files with vertex as the first element are supported")

    stride = element_stride(vertex)
    if header.fmt == "binary_little_endian":
        packed_label = struct.pack("<i", label)
    elif header.fmt == "binary_big_endian":
        packed_label = struct.pack(">i", label)
    else:
        raise ValueError(f"Unsupported binary format: {header.fmt}")

    data = src.read_bytes()
    body = data[header.header_bytes:]
    vertex_bytes = stride * vertex.count
    if len(body) < vertex_bytes:
        raise ValueError("Binary PLY body is shorter than declared vertex data")

    new_header = insert_vertex_property(header.lines, property_name)
    out = bytearray()
    out.extend(new_header)

    vertex_blob = body[:vertex_bytes]
    for offset in range(0, vertex_bytes, stride):
        out.extend(vertex_blob[offset : offset + stride])
        out.extend(packed_label)

    out.extend(body[vertex_bytes:])
    dst.write_bytes(bytes(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ply", type=Path)
    parser.add_argument("output_ply", type=Path)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--property-name", default="label")
    args = parser.parse_args()

    header = read_ply_header(args.input_ply)
    args.output_ply.parent.mkdir(parents=True, exist_ok=True)

    if header.fmt == "ascii":
        write_ascii_with_label(args.input_ply, args.output_ply, args.label, args.property_name)
    elif header.fmt in {"binary_little_endian", "binary_big_endian"}:
        write_binary_with_label(args.input_ply, args.output_ply, args.label, args.property_name)
    else:
        raise ValueError(f"Unsupported PLY format: {header.fmt}")

    print(f"wrote {args.output_ply}")


if __name__ == "__main__":
    main()
