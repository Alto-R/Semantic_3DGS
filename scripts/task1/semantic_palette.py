"""Shared class-aware colors for semantic overlays and debug PLY exports."""

from __future__ import annotations

import colorsys
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


CLASS_COLORS: dict[str, tuple[float, float, float]] = {
    "unlabeled": (0.08, 0.08, 0.08),
    "bicycle": (0.98, 0.10, 0.06),
    "bench": (0.05, 0.32, 0.98),
    "train": (0.88, 0.12, 0.08),
    "railroad_track": (0.42, 0.28, 0.16),
    "railway_platform": (0.76, 0.58, 0.36),
    "car": (0.98, 0.48, 0.04),
    "person": (0.90, 0.08, 0.58),
    "tree": (0.05, 0.62, 0.20),
    "vegetation": (0.48, 0.82, 0.08),
    "building": (0.72, 0.18, 0.82),
    "ground": (0.92, 0.72, 0.08),
    "road": (0.42, 0.44, 0.48),
    "sidewalk": (0.72, 0.62, 0.50),
    "sky": (0.08, 0.72, 0.98),
    "pole": (0.96, 0.88, 0.12),
    "sign": (0.94, 0.22, 0.16),
    "fence": (0.08, 0.78, 0.72),
    "guitar": (0.12, 0.76, 0.72),
    "chair": (0.56, 0.20, 0.86),
    "sofa": (0.10, 0.38, 0.92),
    "table": (0.76, 0.34, 0.10),
    "lamp": (0.98, 0.82, 0.10),
    "indoor_plant": (0.12, 0.68, 0.28),
    "cabinet": (0.72, 0.52, 0.18),
    "bookshelf": (0.04, 0.58, 0.62),
    "picture_frame": (0.90, 0.18, 0.64),
    "rug": (0.76, 0.16, 0.24),
    "door": (0.48, 0.26, 0.12),
    "window": (0.16, 0.70, 0.94),
    "wall": (0.68, 0.70, 0.74),
    "floor": (0.66, 0.46, 0.28),
    "ceiling": (0.88, 0.86, 0.72),
    "piano": (0.08, 0.56, 0.78),
    "television": (0.18, 0.20, 0.24),
    "speaker": (0.86, 0.38, 0.08),
    "media_console": (0.46, 0.34, 0.24),
    "curtain": (0.52, 0.24, 0.68),
}

STUFF_CLASSES = {
    "building",
    "ceiling",
    "curtain",
    "door",
    "floor",
    "ground",
    "railroad_track",
    "railway_platform",
    "road",
    "rug",
    "sidewalk",
    "sky",
    "terrain",
    "vegetation",
    "wall",
    "window",
}

MIN_LABEL_DELTA_E = 30.0


def load_label_items(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["id"]): item for item in data.get("labels", []) if "id" in item}


def normalize_classes(values: str | Iterable[str]) -> set[str]:
    items = values.split(",") if isinstance(values, str) else values
    return {str(item).strip().lower().replace(" ", "_") for item in items if str(item).strip()}


def fallback_color(label_id: int) -> tuple[float, float, float]:
    hue = (label_id * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.78, 0.96)


def _linear_srgb(channel: float) -> float:
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def lab_for_rgb(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    red, green, blue = (_linear_srgb(channel) for channel in rgb)
    x = (0.4124564 * red + 0.3575761 * green + 0.1804375 * blue) / 0.95047
    y = 0.2126729 * red + 0.7151522 * green + 0.0721750 * blue
    z = (0.0193339 * red + 0.1191920 * green + 0.9503041 * blue) / 1.08883

    delta = 6.0 / 29.0

    def transform(value: float) -> float:
        if value > delta**3:
            return value ** (1.0 / 3.0)
        return value / (3.0 * delta**2) + 4.0 / 29.0

    fx, fy, fz = transform(x), transform(y), transform(z)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def color_distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    first_lab = lab_for_rgb(first)
    second_lab = lab_for_rgb(second)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first_lab, second_lab)))


def _candidate_colors() -> tuple[tuple[float, float, float], ...]:
    colors: list[tuple[float, float, float]] = []
    for hue_index in range(48):
        hue = hue_index / 48.0
        for saturation, value in ((0.95, 0.95), (0.72, 0.98), (0.88, 0.72)):
            colors.append(colorsys.hsv_to_rgb(hue, saturation, value))
    return tuple(colors)


DISPLAY_COLOR_CANDIDATES = _candidate_colors()


def _label_signature(
    label_items: dict[int, dict[str, Any]],
) -> tuple[tuple[int, str, str], ...]:
    return tuple(
        sorted(
            (
                int(label_id),
                str(item.get("class", "unlabeled" if int(label_id) == 0 else "unknown")).lower(),
                str(item.get("name", "")),
            )
            for label_id, item in label_items.items()
        )
    )


@lru_cache(maxsize=128)
def _palette_from_signature(
    signature: tuple[tuple[int, str, str], ...],
) -> dict[int, tuple[float, float, float]]:
    palette: dict[int, tuple[float, float, float]] = {}
    used_colors: list[tuple[float, float, float]] = []

    if any(label_id == 0 for label_id, _, _ in signature):
        palette[0] = CLASS_COLORS["unlabeled"]
        used_colors.append(palette[0])

    for label_id, class_name, _ in signature:
        if label_id == 0:
            continue

        preferred = CLASS_COLORS.get(class_name, fallback_color(label_id))
        preferred_distance = min(
            (color_distance(preferred, used) for used in used_colors),
            default=float("inf"),
        )
        if preferred_distance >= MIN_LABEL_DELTA_E:
            selected = preferred
        else:
            selected = max(
                DISPLAY_COLOR_CANDIDATES,
                key=lambda candidate: min(color_distance(candidate, used) for used in used_colors),
            )
        palette[label_id] = selected
        used_colors.append(selected)

    return palette


def label_palette(
    label_items: dict[int, dict[str, Any]],
) -> dict[int, tuple[float, float, float]]:
    """Return deterministic colors separated across every label in one output."""
    return dict(_palette_from_signature(_label_signature(label_items)))


def rgb_for_label(
    label_id: int,
    label_items: dict[int, dict[str, Any]] | None = None,
    focus_classes: set[str] | None = None,
) -> tuple[float, float, float]:
    item = (label_items or {}).get(label_id, {})
    class_name = str(item.get("class", "unlabeled" if label_id == 0 else "unknown")).lower()
    if focus_classes and class_name not in focus_classes:
        return CLASS_COLORS["unlabeled"]

    if label_items:
        palette = _palette_from_signature(_label_signature(label_items))
        if label_id in palette:
            return palette[label_id]
    return CLASS_COLORS.get(class_name, fallback_color(label_id))


def rgb8_for_label(
    label_id: int,
    label_items: dict[int, dict[str, Any]] | None = None,
    focus_classes: set[str] | None = None,
) -> list[int]:
    return [int(round(channel * 255.0)) for channel in rgb_for_label(label_id, label_items, focus_classes)]


def palette_records(
    label_ids: Iterable[int],
    label_items: dict[int, dict[str, Any]] | None = None,
    focus_classes: set[str] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for label_id in sorted({int(value) for value in label_ids}):
        item = (label_items or {}).get(label_id, {})
        records.append(
            {
                "id": label_id,
                "name": item.get("name", ""),
                "class": item.get("class", ""),
                "rgb": rgb8_for_label(label_id, label_items, focus_classes),
            }
        )
    return records
