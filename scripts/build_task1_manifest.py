#!/usr/bin/env python3
"""Build a Task 1 scene manifest from EyeNavGS scene settings and 3DGS PLYs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from find_point_clouds import summarize_point_cloud


DEFAULT_SETTINGS = Path("/lab/haoq_lab/cse12312032/data/EyeNavGS/Rutgers/scene_setting.csv")
DEFAULT_MODEL_ROOT = Path("/lab/haoq_lab/cse12312032/data/3dgs_models")


def read_scene_settings(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def iteration_score(path: str) -> int:
    parts = Path(path).parts
    for part in parts:
        if part.startswith("iteration_"):
            suffix = part.removeprefix("iteration_")
            if suffix.isdigit():
                return int(suffix)
    return -1


def choose_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (iteration_score(item["path"]), item["vertex_count"] or 0),
        reverse=True,
    )[0]


def build_manifest(scene_settings: Path, model_root: Path) -> Dict[str, Any]:
    settings = read_scene_settings(scene_settings)
    model_root = model_root.resolve()

    point_clouds = [
        summarize_point_cloud(path, model_root)
        for path in sorted(model_root.rglob("point_cloud.ply"))
    ]

    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for item in point_clouds:
        by_scene.setdefault(item["scene"].lower(), []).append(item)

    scenes: List[Dict[str, Any]] = []
    for row in settings:
        scene = row["Scene_Name"]
        candidates = by_scene.get(scene.lower(), [])
        chosen = choose_candidate(candidates)
        scenes.append(
            {
                "scene": scene,
                "scene_setting": row,
                "candidate_count": len(candidates),
                "chosen_point_cloud": chosen["path"] if chosen else None,
                "chosen_vertex_count": chosen["vertex_count"] if chosen else None,
                "all_candidates": candidates,
            }
        )

    return {
        "scene_setting_csv": str(scene_settings),
        "model_root": str(model_root),
        "point_cloud_count": len(point_clouds),
        "scene_count": len(scenes),
        "matched_scene_count": sum(1 for scene in scenes if scene["chosen_point_cloud"]),
        "scenes": scenes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(args.scene_settings, args.model_root)

    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    print(
        f"{manifest['matched_scene_count']}/{manifest['scene_count']} scenes matched; "
        f"{manifest['point_cloud_count']} point_cloud.ply files found"
    )
    print("scene\tcandidates\tvertices\tchosen_point_cloud")
    for scene in manifest["scenes"]:
        print(
            f"{scene['scene']}\t{scene['candidate_count']}\t"
            f"{scene['chosen_vertex_count'] or ''}\t"
            f"{scene['chosen_point_cloud'] or 'MISSING'}"
        )


if __name__ == "__main__":
    main()

