"""Load and validate the global ADE20K-to-project ontology."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OntologyClass:
    ade_id: int
    ade_name: str
    project_id: int
    project_class: str
    kind: str


@dataclass(frozen=True)
class Ontology:
    version: int
    source: str
    ignore_index: int
    classes: tuple[OntologyClass, ...]

    @property
    def class_count(self) -> int:
        return len(self.classes)

    @property
    def by_project_id(self) -> dict[int, OntologyClass]:
        return {item.project_id: item for item in self.classes}

    @property
    def ade_to_project(self) -> np.ndarray:
        lookup = np.zeros((256,), dtype=np.uint16)
        for item in self.classes:
            lookup[item.ade_id] = np.uint16(item.project_id)
        lookup[self.ignore_index] = np.uint16(0)
        return lookup


def normalize_class_name(value: Any) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", " ").split())


def load_ontology(path: Path) -> Ontology:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_classes = data.get("classes")
    if not isinstance(raw_classes, list):
        raise ValueError("ontology classes must be a list")

    classes: list[OntologyClass] = []
    for raw in raw_classes:
        if not isinstance(raw, dict):
            raise ValueError("every ontology class must be an object")
        item = OntologyClass(
            ade_id=int(raw["ade_id"]),
            ade_name=normalize_class_name(raw["ade_name"]),
            project_id=int(raw["project_id"]),
            project_class=normalize_class_name(raw["project_class"]),
            kind=str(raw["type"]).strip().lower(),
        )
        if item.kind not in {"thing", "stuff"}:
            raise ValueError(f"invalid ontology type for {item.project_class}: {item.kind}")
        if not item.ade_name or not item.project_class:
            raise ValueError("ontology class names must be non-empty")
        classes.append(item)

    expected_ade_ids = list(range(150))
    ade_ids = sorted(item.ade_id for item in classes)
    if ade_ids != expected_ade_ids:
        raise ValueError("ontology must contain every ADE20K id from 0 through 149 exactly once")

    expected_project_ids = list(range(1, len(classes) + 1))
    project_ids = sorted(item.project_id for item in classes)
    if project_ids != expected_project_ids:
        raise ValueError("project ids must be contiguous from 1 through the class count")

    project_classes = [item.project_class for item in classes]
    if len(project_classes) != len(set(project_classes)):
        raise ValueError("project class names must be unique")

    ignore_index = int(data.get("ignore_index", 255))
    if not 0 <= ignore_index <= 255:
        raise ValueError("ignore_index must fit in uint8")
    return Ontology(
        version=int(data.get("version", 1)),
        source=str(data.get("source", "ADE20K-150")),
        ignore_index=ignore_index,
        classes=tuple(sorted(classes, key=lambda item: item.ade_id)),
    )
