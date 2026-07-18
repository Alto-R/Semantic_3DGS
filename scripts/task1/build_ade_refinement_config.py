#!/usr/bin/env python3
"""Build a scene-neutral Grounding config from supported DINO ADE classes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from dinov2_ontology import load_ontology, normalize_class_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONTOLOGY = PROJECT_ROOT / "configs" / "ade20k_to_project.json"
DEFAULT_ALIASES = PROJECT_ROOT / "configs" / "ade20k_grounding_prompts.json"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def normalize_prompt(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def load_aliases(path: Path) -> dict[str, list[str]]:
    payload = load_object(path, "ADE prompt aliases")
    raw_aliases = payload.get("aliases", {})
    if not isinstance(raw_aliases, dict):
        raise ValueError("ADE prompt aliases must contain an aliases object")
    result: dict[str, list[str]] = {}
    for raw_class, raw_prompts in raw_aliases.items():
        class_name = normalize_class_name(raw_class)
        if not isinstance(raw_prompts, list):
            raise ValueError(f"Prompt aliases for {class_name} must be a list")
        prompts = [normalize_prompt(value) for value in raw_prompts]
        if any(not value for value in prompts):
            raise ValueError(f"Prompt aliases for {class_name} contain an empty prompt")
        if len(prompts) != len(set(prompts)):
            raise ValueError(f"Prompt aliases for {class_name} contain duplicates")
        result[class_name] = prompts
    return result


def dino_class_records(label_map: dict[str, Any]) -> list[dict[str, Any]]:
    raw_labels = label_map.get("labels")
    if not isinstance(raw_labels, list):
        raise ValueError("DINO label map must contain a labels list")
    grouped: dict[str, dict[str, Any]] = {}
    for raw in raw_labels:
        if not isinstance(raw, dict):
            raise ValueError("Every DINO label-map record must be an object")
        label_id = int(raw["id"])
        class_name = normalize_class_name(raw.get("class", ""))
        if label_id == 0 or class_name in {"", "unlabeled", "unknown"}:
            continue
        count = int(raw.get("gaussian_count", 0))
        source_views = int(raw.get("source_view_count", 0))
        record = grouped.setdefault(
            class_name,
            {
                "class": class_name,
                "gaussian_count": 0,
                "source_view_count": 0,
                "label_ids": [],
            },
        )
        record["gaussian_count"] += count
        record["source_view_count"] = max(record["source_view_count"], source_views)
        record["label_ids"].append(label_id)
    return list(grouped.values())


def prompt_word_count(classes: list[dict[str, Any]]) -> int:
    unique_prompts = {
        normalize_prompt(prompt)
        for item in classes
        for prompt in item.get("prompts", [])
        if normalize_prompt(prompt)
    }
    return sum(len(prompt.split()) for prompt in unique_prompts)


def ensure_unique_prompts(classes: list[dict[str, Any]]) -> None:
    owners: dict[str, str] = {}
    for item in classes:
        class_name = str(item["class"])
        for raw_prompt in item["prompts"]:
            prompt = normalize_prompt(raw_prompt)
            owner = owners.get(prompt)
            if owner is not None and owner != class_name:
                raise ValueError(
                    f"Grounding prompt {prompt!r} belongs to both {owner} and {class_name}"
                )
            owners[prompt] = class_name


def augment_with_extensions(
    refinement: dict[str, Any],
    scene_config: dict[str, Any],
    extension_config: dict[str, Any],
    include_classes: str,
    ontology_classes: set[str],
    max_prompt_words: int,
) -> dict[str, Any]:
    """Add selected non-ADE open-vocabulary classes to one Grounding run."""
    scene = str(refinement.get("scene", "")).strip()
    extension_scene = str(extension_config.get("scene", "")).strip()
    if extension_scene != scene:
        raise ValueError(
            f"Extension config scene {extension_scene!r} does not match {scene!r}"
        )
    raw_specs = scene_config.get("classes")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("Scene class config must contain a non-empty classes list")
    scene_specs: dict[str, dict[str, Any]] = {}
    for raw in raw_specs:
        if not isinstance(raw, dict):
            raise ValueError("Every scene class specification must be an object")
        class_name = normalize_class_name(raw.get("class", raw.get("name", "")))
        prompts = [normalize_prompt(value) for value in raw.get("prompts", [])]
        prompts = [value for value in prompts if value]
        if not prompts:
            prompts = [normalize_prompt(class_name)]
        kind = str(raw.get("type", "thing")).strip().lower()
        if kind not in {"thing", "stuff"}:
            raise ValueError(f"Invalid scene class type for {class_name}: {kind}")
        scene_specs[class_name] = {
            "class": class_name,
            "type": kind,
            "prompts": list(dict.fromkeys(prompts)),
        }

    candidates = [
        normalize_class_name(value)
        for value in extension_config.get("candidate_classes", [])
    ]
    defaults = [
        normalize_class_name(value)
        for value in extension_config.get("default_enabled_classes", [])
    ]
    selected = (
        [normalize_class_name(value) for value in include_classes.split(",") if value.strip()]
        if include_classes.strip()
        else defaults
    )
    if not selected:
        raise ValueError("No missing-vocabulary extension classes were selected")
    if len(selected) != len(set(selected)):
        raise ValueError("Selected extension classes contain duplicates")
    unknown = sorted(set(selected) - set(candidates))
    if unknown:
        raise ValueError(f"Selected extension classes are not candidates: {unknown}")
    missing_specs = sorted(set(selected) - set(scene_specs))
    if missing_specs:
        raise ValueError(f"Extension classes have no scene prompt specification: {missing_specs}")
    collisions = sorted(set(selected) & ontology_classes)
    if collisions:
        raise ValueError(f"Extension classes collide with the ADE ontology: {collisions}")

    extension_specs = [
        {
            **scene_specs[class_name],
            "role": "extension",
            "source": "selected_missing_vocabulary_extension",
        }
        for class_name in selected
    ]
    combined = [*refinement["classes"], *extension_specs]
    ensure_unique_prompts(combined)
    words = prompt_word_count(combined)
    if words > max_prompt_words:
        raise ValueError(
            f"Unified refinement vocabulary uses {words} words, exceeding "
            f"the global budget {max_prompt_words}"
        )
    return {
        **refinement,
        "source": "automatic_supported_dinov2_ade_plus_missing_vocabulary_extensions",
        "missing_vocabulary_enabled": True,
        "selected_extension_classes": selected,
        "extension_class_count": len(selected),
        "classes": combined,
        "assignment_priority": [
            *refinement["selected_refinement_classes"],
            *selected,
        ],
        "prompt_word_count": words,
    }


def build_refinement_config(
    dino_label_map: dict[str, Any],
    ontology_path: Path,
    aliases: dict[str, list[str]],
    min_anchor_gaussians: int,
    min_source_views: int,
    max_prompt_words: int,
) -> dict[str, Any]:
    if min_anchor_gaussians < 1:
        raise ValueError("min_anchor_gaussians must be positive")
    if min_source_views < 1:
        raise ValueError("min_source_views must be positive")
    if max_prompt_words < 1:
        raise ValueError("max_prompt_words must be positive")
    scene = str(dino_label_map.get("scene", "")).strip()
    if not scene:
        raise ValueError("DINO label map has no scene")

    ontology = load_ontology(ontology_path)
    ontology_by_name = {item.project_class: item for item in ontology.classes}
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in dino_class_records(dino_label_map):
        class_name = str(record["class"])
        ontology_class = ontology_by_name.get(class_name)
        reasons: list[str] = []
        if ontology_class is None:
            reasons.append("not_in_ade_ontology")
        if int(record["gaussian_count"]) < min_anchor_gaussians:
            reasons.append(f"gaussian_count<{min_anchor_gaussians}")
        if int(record["source_view_count"]) < min_source_views:
            reasons.append(f"source_view_count<{min_source_views}")
        if reasons:
            rejected.append({**record, "reasons": reasons})
            continue

        canonical = normalize_prompt(class_name)
        prompts: list[str] = []
        for prompt in [*aliases.get(class_name, []), canonical]:
            normalized = normalize_prompt(prompt)
            if normalized and normalized not in prompts:
                prompts.append(normalized)
        eligible.append(
            {
                **record,
                "type": ontology_class.kind,
                "ade_id": ontology_class.ade_id,
                "project_id": ontology_class.project_id,
                "prompts": prompts,
                "role": "ade_refinement",
                "source": "supported_dinov2_ade_class",
            }
        )

    eligible.sort(
        key=lambda item: (
            str(item["type"]) == "stuff",
            -int(item["gaussian_count"]),
            str(item["class"]),
        )
    )
    ensure_unique_prompts(eligible)
    words = prompt_word_count(eligible)
    if words > max_prompt_words:
        raise ValueError(
            f"Automatic ADE prompt vocabulary uses {words} words, exceeding "
            f"the global budget {max_prompt_words}"
        )
    class_names = [str(item["class"]) for item in eligible]
    return {
        "version": 1,
        "scene": scene,
        "source": "automatic_supported_dinov2_ade_refinement",
        "missing_vocabulary_enabled": False,
        "eligibility": {
            "min_anchor_gaussians": min_anchor_gaussians,
            "min_source_views": min_source_views,
            "definition": "surviving DINO ADE class with global support gates",
        },
        "selected_refinement_classes": class_names,
        "selected_class_count": len(class_names),
        "rejected_classes": rejected,
        "assignment_priority": class_names,
        "prompt_word_count": words,
        "max_prompt_words": max_prompt_words,
        "classes": eligible,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov2-label-map", required=True, type=Path)
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY, type=Path)
    parser.add_argument("--prompt-aliases", default=DEFAULT_ALIASES, type=Path)
    parser.add_argument("--scene-class-config", type=Path)
    parser.add_argument("--extension-config", type=Path)
    parser.add_argument("--include-extension-classes", default="")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-anchor-gaussians", default=500, type=int)
    parser.add_argument("--min-source-views", default=2, type=int)
    parser.add_argument("--max-prompt-words", default=180, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")

    dino_label_map = load_object(args.dinov2_label_map, "DINO label map")
    aliases = load_aliases(args.prompt_aliases)
    payload = build_refinement_config(
        dino_label_map,
        args.ontology,
        aliases,
        args.min_anchor_gaussians,
        args.min_source_views,
        args.max_prompt_words,
    )
    if (args.scene_class_config is None) != (args.extension_config is None):
        raise ValueError(
            "--scene-class-config and --extension-config must be supplied together"
        )
    if args.extension_config is not None and args.scene_class_config is not None:
        ontology_classes = {
            item.project_class for item in load_ontology(args.ontology).classes
        }
        payload = augment_with_extensions(
            payload,
            load_object(args.scene_class_config, "scene class config"),
            load_object(args.extension_config, "extension config"),
            args.include_extension_classes,
            ontology_classes,
            args.max_prompt_words,
        )
    payload["sources"] = {
        "dinov2_label_map": {
            "path": str(args.dinov2_label_map),
            "sha256": sha256_file(args.dinov2_label_map),
        },
        "ontology": {"path": str(args.ontology), "sha256": sha256_file(args.ontology)},
        "prompt_aliases": {
            "path": str(args.prompt_aliases),
            "sha256": sha256_file(args.prompt_aliases),
        },
    }
    if args.scene_class_config is not None and args.extension_config is not None:
        payload["sources"].update(
            {
                "scene_class_config": {
                    "path": str(args.scene_class_config),
                    "sha256": sha256_file(args.scene_class_config),
                },
                "extension_config": {
                    "path": str(args.extension_config),
                    "sha256": sha256_file(args.extension_config),
                },
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key not in {"classes", "sources"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
