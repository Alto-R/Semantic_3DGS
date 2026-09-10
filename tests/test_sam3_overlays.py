"""Tests for instance overlay QA rendering."""

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from scripts.task1.sam3.render_instance_overlays import (  # noqa: E402
    contact_sheet,
    instance_palette,
    overlay_instances,
)


def test_instance_palette_is_deterministic_and_distinct():
    first = instance_palette(8)
    second = instance_palette(8)
    assert first == second
    assert len(first) == 8
    assert len(set(first)) == 8
    for color in first:
        assert len(color) == 3
        assert all(0 <= channel <= 255 for channel in color)


def test_overlay_changes_exactly_masked_pixels():
    rgb = np.full((6, 8, 3), 100, np.uint8)
    mask = np.zeros((6, 8), np.uint8)
    mask[2:4, 3:5] = 1
    stack = mask[None, ...]
    result = overlay_instances(rgb, stack, alpha=0.5)
    assert result.shape == rgb.shape
    assert result.dtype == np.uint8
    changed = np.any(result != rgb, axis=2)
    np.testing.assert_array_equal(changed, mask.astype(bool))


def test_overlay_empty_stack_returns_copy():
    rgb = np.full((4, 4, 3), 7, np.uint8)
    result = overlay_instances(rgb, np.zeros((0, 4, 4), np.uint8), alpha=0.5)
    np.testing.assert_array_equal(result, rgb)
    assert result is not rgb


def test_contact_sheet_grid_size():
    tiles = [Image.new("RGB", (40, 30), (index * 10, 0, 0)) for index in range(5)]
    sheet = contact_sheet(tiles, columns=2)
    assert sheet.size == (80, 90)  # 2 columns x 3 rows of 40x30 tiles
