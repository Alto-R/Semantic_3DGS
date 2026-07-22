"""Load the ADE20K(150) -> project ontology used by the dense-semantic route.

The compact project class space reserves id 0 for ignore/no-vote. Project
classes receive stable compact ids 1..C in first-appearance order of the
ontology file, so the same config always yields the same id assignment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ADE20K_NUM_CLASSES = 150
IGNORE_NAME = "ignore"


@dataclass(frozen=True)
class ProjectOntology:
    """Compact project class table plus the ADE20K index remap."""

    # class_names[0] == "ignore"; project classes are class_names[1:].
    class_names: tuple[str, ...]
    # class_types[i] is "ignore" | "thing" | "stuff" for class_names[i].
    class_types: tuple[str, ...]
    # ade_to_project[k] maps raw ADE20K index k -> compact project id (0..C).
    ade_to_project: np.ndarray
    source_path: str

    @property
    def num_project_classes(self) -> int:
        return len(self.class_names) - 1

    def is_stuff(self, compact_id: int) -> bool:
        return self.class_types[compact_id] == "stuff"

    def is_thing(self, compact_id: int) -> bool:
        return self.class_types[compact_id] == "thing"

    def remap(self, ade_class_map: np.ndarray) -> np.ndarray:
        """Remap an (H, W) raw ADE20K class map into compact project ids."""
        if ade_class_map.min() < 0 or ade_class_map.max() >= ADE20K_NUM_CLASSES:
            raise ValueError(
                f"ADE20K class map has out-of-range values "
                f"[{ade_class_map.min()}, {ade_class_map.max()}]"
            )
        return self.ade_to_project[ade_class_map]

    def describe(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "num_project_classes": self.num_project_classes,
            "class_names": list(self.class_names),
            "class_types": list(self.class_types),
        }


def load_ontology(path: Path) -> ProjectOntology:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data.get("classes")
    if not isinstance(entries, list):
        raise ValueError(f"{path} must contain a classes list")
    if len(entries) != ADE20K_NUM_CLASSES:
        raise ValueError(
            f"{path} must map all {ADE20K_NUM_CLASSES} ADE20K classes, "
            f"found {len(entries)}"
        )

    seen_indices: set[int] = set()
    class_names: list[str] = [IGNORE_NAME]
    class_types: list[str] = [IGNORE_NAME]
    compact_by_name: dict[str, int] = {}
    ade_to_project = np.zeros((ADE20K_NUM_CLASSES,), dtype=np.int16)

    for entry in entries:
        index = int(entry["index"])
        if index < 0 or index >= ADE20K_NUM_CLASSES:
            raise ValueError(f"ADE20K index out of range: {index}")
        if index in seen_indices:
            raise ValueError(f"Duplicate ADE20K index: {index}")
        seen_indices.add(index)

        project = str(entry["project"]).strip().lower().replace(" ", "_")
        kind = str(entry["type"]).strip().lower()
        if kind not in {IGNORE_NAME, "thing", "stuff"}:
            raise ValueError(f"Invalid type {kind!r} for ADE20K index {index}")
        if (kind == IGNORE_NAME) != (project == IGNORE_NAME):
            raise ValueError(
                f"ADE20K index {index}: project={project!r} and type={kind!r} "
                "must both be ignore or neither"
            )

        if project == IGNORE_NAME:
            ade_to_project[index] = 0
            continue

        compact_id = compact_by_name.get(project)
        if compact_id is None:
            compact_id = len(class_names)
            compact_by_name[project] = compact_id
            class_names.append(project)
            class_types.append(kind)
        elif class_types[compact_id] != kind:
            raise ValueError(
                f"Project class {project!r} is declared both "
                f"{class_types[compact_id]!r} and {kind!r}"
            )
        ade_to_project[index] = compact_id

    return ProjectOntology(
        class_names=tuple(class_names),
        class_types=tuple(class_types),
        ade_to_project=ade_to_project,
        source_path=str(path),
    )
