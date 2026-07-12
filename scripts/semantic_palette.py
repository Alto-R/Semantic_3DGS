"""Shared class-aware colors for semantic overlays and debug PLY exports."""

from __future__ import annotations

import colorsys
import json
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
}

STUFF_CLASSES = {
    "building",
    "ceiling",
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


def rgb_for_label(
    label_id: int,
    label_items: dict[int, dict[str, Any]] | None = None,
    focus_classes: set[str] | None = None,
) -> tuple[float, float, float]:
    item = (label_items or {}).get(label_id, {})
    class_name = str(item.get("class", "unlabeled" if label_id == 0 else "unknown")).lower()
    if focus_classes and class_name not in focus_classes:
        return CLASS_COLORS["unlabeled"]

    base = CLASS_COLORS.get(class_name, fallback_color(label_id))
    if label_id == 0 or class_name in STUFF_CLASSES:
        return base

    # Keep instances in the same color family while making adjacent instances distinguishable.
    name = str(item.get("name", ""))
    try:
        instance = int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        instance = label_id
    factor = (1.0, 0.78, 0.90, 0.68)[(max(instance, 1) - 1) % 4]
    return tuple(min(1.0, max(0.0, channel * factor)) for channel in base)


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
