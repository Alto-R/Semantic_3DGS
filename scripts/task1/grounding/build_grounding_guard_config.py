#!/usr/bin/env python3
"""Build a deterministic GroundingDINO guard-plus-extension class config."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def normalize_name(value: Any) -> str:
    return "_".join(
        re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split()
    )


def normalized_text(value: Any) -> str:
    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split()
    )


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return data


def normalized_unique(values: list[Any], label: str) -> list[str]:
    result = [normalize_name(value) for value in values]
    if any(not value for value in result):
        raise ValueError(f"{label} contains an empty class name")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate class names: {result}")
    return result


def parse_scene_specs(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    raw_specs = config.get("classes")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("Scene class config must contain a non-empty classes list")
    specs: list[dict[str, Any]] = []
    for raw in raw_specs:
        if not isinstance(raw, dict):
            raise ValueError("Every scene class specification must be an object")
        class_name = normalize_name(raw.get("class", raw.get("name", "")))
        if not class_name:
            raise ValueError(f"Scene class specification has no class: {raw}")
        kind = str(raw.get("type", "thing")).strip().lower()
        if kind not in {"thing", "stuff"}:
            raise ValueError(f"Invalid type for {class_name}: {kind}")
        raw_prompts = raw.get("prompts", [class_name.replace("_", " ")])
        if isinstance(raw_prompts, str):
            raw_prompts = [raw_prompts]
        prompts: list[str] = []
        seen_prompts: set[str] = set()
        for value in raw_prompts:
            prompt = normalized_text(value)
            if prompt and prompt not in seen_prompts:
                prompts.append(prompt)
                seen_prompts.add(prompt)
        if not prompts:
            prompts = [class_name.replace("_", " ")]
        specs.append({"class": class_name, "type": kind, "prompts": prompts})

    names = [str(item["class"]) for item in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"Scene class config contains duplicate classes: {names}")
    raw_priority = config.get("assignment_priority", names)
    if not isinstance(raw_priority, list):
        raise ValueError("assignment_priority must be a list")
    priority = normalized_unique(raw_priority, "assignment_priority")
    unknown_priority = [value for value in priority if value not in set(names)]
    if unknown_priority:
        raise ValueError(
            f"assignment_priority references classes absent from the config: {unknown_priority}"
        )
    for name in names:
        if name not in priority:
            priority.append(name)
    return specs, priority


def resolve_extension_classes(config: dict[str, Any], include_classes: str) -> list[str]:
    candidates = normalized_unique(
        list(config.get("candidate_classes", [])),
        "candidate_classes",
    )
    if include_classes.strip():
        selected = normalized_unique(include_classes.split(","), "include_classes")
    else:
        selected = normalized_unique(
            list(config.get("default_enabled_classes", [])),
            "default_enabled_classes",
        )
    unknown = [value for value in selected if value not in candidates]
    if unknown:
        raise ValueError(f"Selected extension classes are not candidates: {unknown}")
    if not selected:
        raise ValueError("No extension classes were selected")
    return selected


def dino_class_records(label_map: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = label_map.get("labels")
    if not isinstance(raw_items, list):
        raise ValueError("DINO label map must contain a labels list")
    grouped: dict[str, dict[str, Any]] = {}
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ValueError("Every DINO label-map item must be an object")
        label_id = int(raw["id"])
        class_name = normalize_name(raw.get("class", ""))
        if label_id == 0 or class_name in {"", "unlabeled", "unknown"}:
            continue
        kind = str(raw.get("type", "thing")).strip().lower()
        if kind not in {"thing", "stuff"}:
            raise ValueError(f"Invalid DINO label type for {class_name}: {kind}")
        count = int(raw.get("gaussian_count", 0))
        record = grouped.setdefault(
            class_name,
            {
                "class": class_name,
                "type": kind,
                "gaussian_count": 0,
                "label_ids": [],
                "label_names": [],
            },
        )
        if record["type"] != kind:
            raise ValueError(f"DINO class {class_name} has inconsistent thing/stuff types")
        record["gaussian_count"] += count
        record["label_ids"].append(label_id)
        record["label_names"].append(str(raw.get("name", f"label_{label_id}")))
    return sorted(grouped.values(), key=lambda item: str(item["class"]))


def _token_relation(left: str, right: str) -> float:
    left_tokens = frozenset(normalized_text(left).split())
    right_tokens = frozenset(normalized_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    if left_tokens == right_tokens:
        return 1.0
    if left_tokens < right_tokens or right_tokens < left_tokens:
        return min(len(left_tokens), len(right_tokens)) / float(
            max(len(left_tokens), len(right_tokens))
        )
    return 0.0


def guard_match_score(dino_class: str, scene_spec: dict[str, Any]) -> float:
    dino_text = normalized_text(dino_class)
    scene_class = normalized_text(scene_spec["class"])
    if dino_text == scene_class:
        return 100.0
    prompt_texts = [normalized_text(value) for value in scene_spec["prompts"]]
    if dino_text in prompt_texts:
        return 95.0
    class_relation = _token_relation(dino_text, scene_class)
    if class_relation:
        return 85.0 + class_relation
    prompt_relations = [_token_relation(dino_text, prompt) for prompt in prompt_texts]
    best_prompt_relation = max(prompt_relations, default=0.0)
    if best_prompt_relation:
        return 75.0 + best_prompt_relation
    candidates = [scene_class, *prompt_texts]
    if any(
        min(len(dino_text), len(candidate)) >= 5
        and (dino_text in candidate or candidate in dino_text)
        for candidate in candidates
    ):
        return 65.0
    return 0.0


def match_scene_guard(
    dino_class: str,
    scene_specs: list[dict[str, Any]],
    extension_classes: set[str],
) -> dict[str, Any] | None:
    scored = [
        (guard_match_score(dino_class, spec), spec)
        for spec in scene_specs
        if str(spec["class"]) not in extension_classes
    ]
    best_score = max((score for score, _spec in scored), default=0.0)
    if best_score <= 0.0:
        return None
    best = [spec for score, spec in scored if score == best_score]
    return best[0] if len(best) == 1 else None


def prompt_word_count(specs: list[dict[str, Any]]) -> int:
    prompts = {
        normalized_text(prompt)
        for spec in specs
        for prompt in spec.get("prompts", [])
        if normalized_text(prompt)
    }
    return sum(len(prompt.split()) for prompt in prompts)


def ensure_unique_prompts(specs: list[dict[str, Any]]) -> None:
    owners: dict[str, str] = {}
    for spec in specs:
        class_name = str(spec["class"])
        for raw_prompt in spec["prompts"]:
            prompt = normalized_text(raw_prompt)
            owner = owners.get(prompt)
            if owner is not None and owner != class_name:
                raise ValueError(
                    f"Grounding prompt {prompt!r} belongs to both {owner} and {class_name}"
                )
            owners[prompt] = class_name


def apply_prompt_budget(
    extensions: list[dict[str, Any]],
    guards: list[dict[str, Any]],
    max_prompt_words: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if max_prompt_words <= 0:
        raise ValueError("max_prompt_words must be positive")
    if prompt_word_count(extensions) > max_prompt_words:
        raise ValueError("Extension prompts alone exceed the Grounding prompt budget")
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    ranked = sorted(
        guards,
        key=lambda item: (
            str(item["type"]) == "stuff",
            -int(item.get("gaussian_count", 0)),
            str(item["class"]),
        ),
    )
    for guard in ranked:
        if prompt_word_count([*extensions, *kept, guard]) <= max_prompt_words:
            kept.append(guard)
        else:
            dropped.append(guard)
    return kept, dropped


def build_guard_config(
    dino_label_map: dict[str, Any],
    scene_config: dict[str, Any],
    extension_config: dict[str, Any],
    include_classes: str,
    max_prompt_words: int,
) -> dict[str, Any]:
    scene = str(dino_label_map.get("scene", "")).strip()
    if not scene:
        raise ValueError("DINO label map has no scene")
    extension_scene = str(extension_config.get("scene", scene)).strip()
    if extension_scene != scene:
        raise ValueError(
            f"Extension config scene {extension_scene!r} does not match DINO scene {scene!r}"
        )
    scene_specs, scene_priority = parse_scene_specs(scene_config)
    scene_by_name = {str(item["class"]): item for item in scene_specs}
    selected_extensions = resolve_extension_classes(extension_config, include_classes)
    missing_extensions = [value for value in selected_extensions if value not in scene_by_name]
    if missing_extensions:
        raise ValueError(
            f"Extension classes have no scene prompt specification: {missing_extensions}"
        )

    extension_set = set(selected_extensions)
    extension_specs = [
        {
            **scene_by_name[class_name],
            "role": "extension",
            "source": "extension_config",
            "dinov2_classes": [],
            "gaussian_count": 0,
        }
        for class_name in selected_extensions
    ]

    matched_guards: dict[str, dict[str, Any]] = {}
    raw_guards: list[dict[str, Any]] = []
    dino_records = dino_class_records(dino_label_map)
    for record in dino_records:
        dino_class = str(record["class"])
        scene_spec = match_scene_guard(dino_class, scene_specs, extension_set)
        if scene_spec is None:
            raw_guards.append(
                {
                    "class": dino_class,
                    "type": str(record["type"]),
                    "prompts": [dino_class.replace("_", " ")],
                    "role": "guard",
                    "source": "dinov2_label_map",
                    "dinov2_classes": [dino_class],
                    "gaussian_count": int(record["gaussian_count"]),
                }
            )
            continue
        guard_name = str(scene_spec["class"])
        guard = matched_guards.setdefault(
            guard_name,
            {
                **scene_spec,
                "role": "guard",
                "source": "dinov2_label_map+scene_prompt_config",
                "dinov2_classes": [],
                "gaussian_count": 0,
            },
        )
        guard["dinov2_classes"].append(dino_class)
        guard["gaussian_count"] += int(record["gaussian_count"])

    guard_specs = [*matched_guards.values(), *raw_guards]
    kept_guards, dropped_guards = apply_prompt_budget(
        extension_specs,
        guard_specs,
        max_prompt_words,
    )
    kept_names = {str(item["class"]) for item in kept_guards}
    all_by_name = {
        str(item["class"]): item for item in [*extension_specs, *kept_guards]
    }
    if len(all_by_name) != len(extension_specs) + len(kept_guards):
        raise ValueError("Generated guard and extension class names collide")

    # Existing DINO classes are conservative ownership guards. Put them ahead
    # of extensions so the fusion tie-break preserves the existing semantic
    # owner when multi-view evidence is exactly equal. Evidence, not priority,
    # still decides every non-tied comparison.
    ordered_names: list[str] = []
    for name in scene_priority:
        if name in kept_names and name not in ordered_names:
            ordered_names.append(name)
    remaining_guards = sorted(
        (item for item in kept_guards if str(item["class"]) not in ordered_names),
        key=lambda item: (
            str(item["type"]) == "stuff",
            -int(item.get("gaussian_count", 0)),
            str(item["class"]),
        ),
    )
    ordered_names.extend(str(item["class"]) for item in remaining_guards)
    for name in scene_priority:
        if name in extension_set and name not in ordered_names:
            ordered_names.append(name)
    for name in selected_extensions:
        if name not in ordered_names:
            ordered_names.append(name)
    ordered_specs = [all_by_name[name] for name in ordered_names]
    ensure_unique_prompts(ordered_specs)

    return {
        "version": 1,
        "scene": scene,
        "source": "dinov2_scene_guards_plus_grounding_extensions",
        "selected_extension_classes": selected_extensions,
        "guard_classes": [
            str(item["class"]) for item in ordered_specs if item["role"] == "guard"
        ],
        "dropped_guard_classes": [str(item["class"]) for item in dropped_guards],
        "dino_source_class_count": len(dino_records),
        "guard_class_count": len(kept_names),
        "extension_class_count": len(extension_specs),
        "prompt_word_count": prompt_word_count(ordered_specs),
        "max_prompt_words": max_prompt_words,
        "assignment_priority": ordered_names,
        "classes": ordered_specs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov2-label-map", required=True, type=Path)
    parser.add_argument("--scene-class-config", required=True, type=Path)
    parser.add_argument("--extension-config", required=True, type=Path)
    parser.add_argument("--include-classes", default="")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-prompt-words", default=180, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")

    dino_label_map = load_object(args.dinov2_label_map, "DINO label map")
    scene_config = load_object(args.scene_class_config, "scene class config")
    extension_config = load_object(args.extension_config, "extension config")
    payload = build_guard_config(
        dino_label_map,
        scene_config,
        extension_config,
        args.include_classes,
        args.max_prompt_words,
    )
    payload["sources"] = {
        "dinov2_label_map": {
            "path": str(args.dinov2_label_map),
            "sha256": sha256_file(args.dinov2_label_map),
        },
        "scene_class_config": {
            "path": str(args.scene_class_config),
            "sha256": sha256_file(args.scene_class_config),
        },
        "extension_config": {
            "path": str(args.extension_config),
            "sha256": sha256_file(args.extension_config),
        },
    }
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
