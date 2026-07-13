#!/usr/bin/env python3
"""Summarize EyeNavGS trace CSVs for Rutgers and NTHU dataset repos."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_RUTGERS = Path("/lab/haoq_lab/cse12312032/data/EyeNavGS/Rutgers")
DEFAULT_NTHU = Path("/lab/haoq_lab/cse12312032/data/EyeNavGS/NTHU")


def count_data_rows(path: Path) -> int:
    with path.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    return max(0, line_count - 1)


def read_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def read_scene_settings(root: Path) -> List[Dict[str, str]]:
    scene_setting = root / "scene_setting.csv"
    if not scene_setting.exists():
        return []
    with scene_setting.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_trace_files(site: str, root: Path) -> Iterable[Path]:
    if site == "Rutgers":
        yield from sorted((root / "dataset").glob("*/*.csv"))
    elif site == "NTHU":
        yield from sorted(root.glob("*/*.csv"))
    else:
        raise ValueError(f"Unknown site: {site}")


def scene_from_trace_path(site: str, path: Path) -> str:
    if site == "Rutgers":
        return path.parent.name
    if site == "NTHU":
        return path.parent.name
    raise ValueError(f"Unknown site: {site}")


def summarize_site(site: str, root: Path) -> Dict[str, Any]:
    root = root.resolve()
    files = list(find_trace_files(site, root))
    by_scene: Dict[str, List[Path]] = defaultdict(list)
    for path in files:
        by_scene[scene_from_trace_path(site, path)].append(path)

    settings = read_scene_settings(root)
    scene_settings = {
        row.get("Scene_Name") or row.get("Dataset_Name"): row
        for row in settings
        if row.get("Scene_Name") or row.get("Dataset_Name")
    }

    scenes: List[Dict[str, Any]] = []
    for scene, scene_files in sorted(by_scene.items()):
        header_counts: Counter[str] = Counter()
        row_count = 0
        sample_file: Optional[str] = None
        for path in scene_files:
            header = read_header(path)
            header_counts[",".join(header)] += 1
            row_count += count_data_rows(path)
            if sample_file is None:
                sample_file = str(path)

        scenes.append(
            {
                "scene": scene,
                "trace_files": len(scene_files),
                "rows": row_count,
                "sample_file": sample_file,
                "header_variants": [
                    {"count": count, "columns": header.split(",")}
                    for header, count in header_counts.most_common()
                ],
                "scene_setting": scene_settings.get(scene),
            }
        )

    return {
        "site": site,
        "root": str(root),
        "scene_count": len(scenes),
        "trace_file_count": len(files),
        "row_count": sum(scene["rows"] for scene in scenes),
        "scenes": scenes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rutgers", type=Path, default=DEFAULT_RUTGERS)
    parser.add_argument("--nthu", type=Path, default=DEFAULT_NTHU)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summaries = []
    if args.rutgers.exists():
        summaries.append(summarize_site("Rutgers", args.rutgers))
    if args.nthu.exists():
        summaries.append(summarize_site("NTHU", args.nthu))

    if args.json:
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return

    for site in summaries:
        print(
            f"{site['site']}: {site['scene_count']} scenes, "
            f"{site['trace_file_count']} CSVs, {site['row_count']} rows"
        )
        print("scene\tcsvs\trows\theaders\tsetting")
        for scene in site["scenes"]:
            print(
                f"{scene['scene']}\t{scene['trace_files']}\t{scene['rows']}\t"
                f"{len(scene['header_variants'])}\t"
                f"{'yes' if scene['scene_setting'] else 'missing'}"
            )
        print()


if __name__ == "__main__":
    main()
