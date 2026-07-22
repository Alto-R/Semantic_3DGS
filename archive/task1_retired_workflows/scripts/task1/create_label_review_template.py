#!/usr/bin/env python3
"""Create a CSV review sheet for automatic 3DGS object-group labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "source_id",
    "source_name",
    "source_class",
    "gaussian_count",
    "proposal_count",
    "source_view_count",
    "final_id",
    "final_name",
    "final_class",
    "review_status",
    "notes",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def build_rows(label_map: dict[str, Any], summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    labels_by_id: dict[int, dict[str, Any]] = {}
    for item in label_map.get("labels", []):
        labels_by_id[int(item["id"])] = item

    summary_by_id: dict[int, dict[str, Any]] = {}
    histogram: dict[int, int] = {}
    if summary:
        for item in summary.get("groups", []):
            summary_by_id[int(item["id"])] = item
        histogram = {int(label): int(count) for label, count in summary.get("label_histogram", {}).items()}

    source_ids = sorted(set(labels_by_id) | set(summary_by_id) | {0})
    rows: list[dict[str, Any]] = []
    for source_id in source_ids:
        label_item = labels_by_id.get(source_id, {})
        summary_item = summary_by_id.get(source_id, {})
        source_name = str(label_item.get("name", f"object_group_{source_id:03d}" if source_id else "unlabeled"))
        source_class = str(label_item.get("class", "object_candidate" if source_id else "unlabeled"))
        gaussian_count = as_int(
            summary_item.get("gaussian_count", label_item.get("gaussian_count", histogram.get(source_id, 0)))
        )
        proposal_count = as_int(summary_item.get("proposal_count", label_item.get("proposal_count", 0)))
        source_view_count = as_int(summary_item.get("source_view_count", label_item.get("source_view_count", 0)))

        rows.append(
            {
                "source_id": source_id,
                "source_name": source_name,
                "source_class": source_class,
                "gaussian_count": gaussian_count,
                "proposal_count": proposal_count,
                "source_view_count": source_view_count,
                "final_id": source_id,
                "final_name": "unlabeled" if source_id == 0 else source_name,
                "final_class": "unlabeled" if source_id == 0 else source_class,
                "review_status": "locked" if source_id == 0 else "todo",
                "notes": "",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-label-map", required=True, type=Path)
    parser.add_argument("--auto-summary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    label_map = load_json(args.auto_label_map)
    summary = load_json(args.auto_summary) if args.auto_summary else None
    rows = build_rows(label_map, summary)
    write_csv(args.output, rows, overwrite=args.overwrite)
    print(f"wrote {args.output} with {len(rows)} review rows")
    print("Edit final_id/final_name/final_class before applying the review.")


if __name__ == "__main__":
    main()
