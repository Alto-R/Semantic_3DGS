"""Tests for reading vertex positions from binary PLY files."""

import struct

import numpy as np
import pytest

from scripts.task1.common.ply_utils import read_vertex_xyz


def write_tiny_ply(tmp_path, *, with_xyz=True):
    properties = (
        ["property float x", "property float y", "property float z"]
        if with_xyz
        else ["property float nx", "property float ny", "property float nz"]
    )
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex 3",
            *properties,
            "property float opacity",
            "end_header",
            "",
        ]
    ).encode("ascii")
    rows = [
        (1.0, 2.0, 3.0, 0.5),
        (4.0, 5.0, 6.0, 0.5),
        (7.0, 8.0, 9.0, 0.5),
    ]
    body = b"".join(struct.pack("<4f", *row) for row in rows)
    path = tmp_path / "tiny.ply"
    path.write_bytes(header + body)
    return path


def test_read_vertex_xyz_roundtrip(tmp_path):
    xyz = read_vertex_xyz(write_tiny_ply(tmp_path))
    assert xyz.shape == (3, 3)
    assert xyz.dtype == np.float32
    np.testing.assert_allclose(xyz[1], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(xyz[:, 2], [3.0, 6.0, 9.0])


def test_read_vertex_xyz_requires_position_properties(tmp_path):
    with pytest.raises(ValueError, match="x"):
        read_vertex_xyz(write_tiny_ply(tmp_path, with_xyz=False))
