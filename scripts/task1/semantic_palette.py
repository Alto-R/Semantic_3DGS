"""Stable semantic-class colors for overlays, legends, and debug PLY exports."""

from __future__ import annotations

import colorsys
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PALETTE_VERSION = "semantic-class-v1"
COLOR_MODES = {"class", "instance"}

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
    "fireplace": (0.82, 0.42, 0.12),
    "toy": (0.96, 0.18, 0.48),
    "cushion": (0.98, 0.58, 0.12),
    "book": (0.28, 0.48, 0.96),
    "tree_stump": (0.48, 0.25, 0.10),
    "log": (0.66, 0.36, 0.14),
    "path": (0.52, 0.50, 0.46),
    "radiator": (0.76, 0.76, 0.80),
    "monitor": (0.20, 0.26, 0.34),
    "stroller": (0.62, 0.20, 0.78),
    "staircase": (0.78, 0.48, 0.20),
}


def normalize_class_name(value: Any) -> str:
    normalized = "_".join(str(value or "").strip().lower().replace("-", " ").split())
    return normalized or "unknown"


def validate_color_mode(color_mode: str) -> str:
    mode = str(color_mode).strip().lower()
    if mode not in COLOR_MODES:
        raise ValueError(f"color mode must be one of {sorted(COLOR_MODES)}; got {color_mode!r}")
    return mode


def load_label_items(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["id"]): item for item in data.get("labels", []) if "id" in item}


def normalize_classes(values: str | Iterable[str]) -> set[str]:
    items = values.split(",") if isinstance(values, str) else values
    return {normalize_class_name(item) for item in items if str(item).strip()}


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def fallback_class_color(class_name: str) -> tuple[float, float, float]:
    """Return a run-independent color keyed only by normalized class name."""

    key = normalize_class_name(class_name)
    digest = _digest(key)
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    saturation = 0.68 + (digest[2] / 255.0) * 0.22
    value = 0.78 + (digest[3] / 255.0) * 0.18
    return colorsys.hsv_to_rgb(hue, saturation, value)


def rgb_for_class(class_name: str) -> tuple[float, float, float]:
    key = normalize_class_name(class_name)
    return CLASS_COLORS.get(key, fallback_class_color(key))


def instance_color(class_name: str, instance_name: str) -> tuple[float, float, float]:
    """Return a stable class-related shade keyed by semantic and instance names."""

    key = normalize_class_name(class_name)
    if key == "unlabeled":
        return rgb_for_class(key)
    base = rgb_for_class(key)
    hue, saturation, value = colorsys.rgb_to_hsv(*base)
    digest = _digest(f"{key}:{normalize_class_name(instance_name)}")
    hue_offset = ((digest[0] / 255.0) - 0.5) * 0.08
    saturation_scale = 0.88 + (digest[1] / 255.0) * 0.20
    value_scale = 0.86 + (digest[2] / 255.0) * 0.22
    return colorsys.hsv_to_rgb(
        (hue + hue_offset) % 1.0,
        min(1.0, max(0.35, saturation * saturation_scale)),
        min(1.0, max(0.35, value * value_scale)),
    )


def label_color_key(label_id: int, item: dict[str, Any], color_mode: str) -> str:
    mode = validate_color_mode(color_mode)
    class_name = normalize_class_name(
        item.get("class", "unlabeled" if int(label_id) == 0 else "unknown")
    )
    if mode == "class":
        return class_name
    instance_name = normalize_class_name(item.get("name", f"label_{int(label_id)}"))
    return f"{class_name}:{instance_name}"


def label_palette(
    label_items: dict[int, dict[str, Any]],
    color_mode: str = "class",
) -> dict[int, tuple[float, float, float]]:
    """Return colors independent of label IDs, map ordering, and other labels."""

    mode = validate_color_mode(color_mode)
    palette: dict[int, tuple[float, float, float]] = {}
    for label_id, item in label_items.items():
        class_name = normalize_class_name(
            item.get("class", "unlabeled" if int(label_id) == 0 else "unknown")
        )
        if mode == "class":
            palette[int(label_id)] = rgb_for_class(class_name)
        else:
            palette[int(label_id)] = instance_color(
                class_name,
                str(item.get("name", f"label_{int(label_id)}")),
            )
    return palette


def rgb_for_label(
    label_id: int,
    label_items: dict[int, dict[str, Any]] | None = None,
    focus_classes: set[str] | None = None,
    color_mode: str = "class",
) -> tuple[float, float, float]:
    mode = validate_color_mode(color_mode)
    item = (label_items or {}).get(label_id, {})
    class_name = normalize_class_name(
        item.get("class", "unlabeled" if label_id == 0 else "unknown")
    )
    if focus_classes and class_name not in focus_classes:
        return rgb_for_class("unlabeled")
    if mode == "instance":
        return instance_color(class_name, str(item.get("name", f"label_{label_id}")))
    return rgb_for_class(class_name)


def rgb8_for_class(class_name: str) -> list[int]:
    return [int(round(channel * 255.0)) for channel in rgb_for_class(class_name)]


def rgb8_for_label(
    label_id: int,
    label_items: dict[int, dict[str, Any]] | None = None,
    focus_classes: set[str] | None = None,
    color_mode: str = "class",
) -> list[int]:
    return [
        int(round(channel * 255.0))
        for channel in rgb_for_label(label_id, label_items, focus_classes, color_mode)
    ]


def palette_records(
    label_ids: Iterable[int],
    label_items: dict[int, dict[str, Any]] | None = None,
    focus_classes: set[str] | None = None,
    color_mode: str = "class",
) -> list[dict[str, Any]]:
    mode = validate_color_mode(color_mode)
    records = []
    for label_id in sorted({int(value) for value in label_ids}):
        item = (label_items or {}).get(label_id, {})
        records.append(
            {
                "id": label_id,
                "name": item.get("name", ""),
                "class": item.get("class", ""),
                "palette_version": PALETTE_VERSION,
                "color_mode": mode,
                "color_key": label_color_key(label_id, item, mode),
                "rgb": rgb8_for_label(label_id, label_items, focus_classes, mode),
            }
        )
    return records
