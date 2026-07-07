"""Small PLY helpers for EyeNavGS semantic annotation scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


PLY_SCALAR_SIZES: Dict[str, int] = {
    "char": 1,
    "uchar": 1,
    "int8": 1,
    "uint8": 1,
    "short": 2,
    "ushort": 2,
    "int16": 2,
    "uint16": 2,
    "int": 4,
    "uint": 4,
    "int32": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}


@dataclass
class PlyProperty:
    name: str
    data_type: str
    is_list: bool = False
    count_type: Optional[str] = None


@dataclass
class PlyElement:
    name: str
    count: int
    properties: List[PlyProperty]


@dataclass
class PlyHeader:
    path: Path
    lines: List[str]
    header_bytes: int
    fmt: str
    version: str
    elements: List[PlyElement]

    def element(self, name: str) -> Optional[PlyElement]:
        for element in self.elements:
            if element.name == name:
                return element
        return None


def read_ply_header(path: Path) -> PlyHeader:
    path = Path(path)
    lines: List[str] = []
    header_bytes = 0

    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            if raw == b"":
                raise ValueError(f"{path} ended before end_header")
            header_bytes += len(raw)
            line = raw.decode("ascii", errors="strict").rstrip("\r\n")
            lines.append(line)
            if line == "end_header":
                break

    if not lines or lines[0] != "ply":
        raise ValueError(f"{path} is not a PLY file")

    fmt = ""
    version = ""
    elements: List[PlyElement] = []
    current: Optional[PlyElement] = None

    for line in lines[1:]:
        if not line or line.startswith("comment ") or line.startswith("obj_info "):
            continue
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            if len(parts) != 3:
                raise ValueError(f"Malformed format line: {line}")
            fmt = parts[1]
            version = parts[2]
        elif parts[0] == "element":
            if len(parts) != 3:
                raise ValueError(f"Malformed element line: {line}")
            current = PlyElement(name=parts[1], count=int(parts[2]), properties=[])
            elements.append(current)
        elif parts[0] == "property":
            if current is None:
                raise ValueError(f"Property before element: {line}")
            if len(parts) == 3:
                current.properties.append(PlyProperty(name=parts[2], data_type=parts[1]))
            elif len(parts) == 5 and parts[1] == "list":
                current.properties.append(
                    PlyProperty(
                        name=parts[4],
                        data_type=parts[3],
                        is_list=True,
                        count_type=parts[2],
                    )
                )
            else:
                raise ValueError(f"Malformed property line: {line}")

    if not fmt:
        raise ValueError(f"{path} has no format line")

    return PlyHeader(
        path=path,
        lines=lines,
        header_bytes=header_bytes,
        fmt=fmt,
        version=version,
        elements=elements,
    )


def scalar_property_size(data_type: str) -> int:
    try:
        return PLY_SCALAR_SIZES[data_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported PLY scalar type: {data_type}") from exc


def element_stride(element: PlyElement) -> int:
    stride = 0
    for prop in element.properties:
        if prop.is_list:
            raise ValueError(f"Element {element.name} contains list property {prop.name}")
        stride += scalar_property_size(prop.data_type)
    return stride

