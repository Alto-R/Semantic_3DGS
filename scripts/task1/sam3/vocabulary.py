"""Per-scene SAM3 prompt vocabulary configuration.

The vocabulary lists the noun phrases prompted to SAM3 for one scene. Each
entry owns a canonical concept (its phrase); synonyms are prompted separately
but fold back into the same concept. Roles gate acceptance criteria in QA,
never model behavior.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


ALLOWED_ROLES = ("gnn_node", "context_probe", "oov_probe")


@dataclass(frozen=True)
class VocabularyEntry:
    phrase: str
    role: str
    synonyms: tuple[str, ...]


@dataclass(frozen=True)
class Vocabulary:
    scene: str
    entries: tuple[VocabularyEntry, ...]
    expected_part_of: tuple[tuple[str, str], ...]

    def phrase_to_concept(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for entry in self.entries:
            mapping[entry.phrase] = entry.phrase
            for synonym in entry.synonyms:
                mapping[synonym] = entry.phrase
        return mapping

    def prompt_phrases(self) -> tuple[str, ...]:
        phrases: list[str] = []
        for entry in self.entries:
            phrases.append(entry.phrase)
            phrases.extend(entry.synonyms)
        return tuple(phrases)

    def concepts(self) -> tuple[str, ...]:
        return tuple(entry.phrase for entry in self.entries)

    def sha256(self) -> str:
        payload = {
            "scene": self.scene,
            "phrases": [
                {
                    "phrase": entry.phrase,
                    "role": entry.role,
                    "synonyms": list(entry.synonyms),
                }
                for entry in self.entries
            ],
            "expected_part_of": [list(pair) for pair in self.expected_part_of],
        }
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_vocabulary(path: Path) -> Vocabulary:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scene = str(payload.get("scene", ""))
    if not scene:
        raise ValueError("vocabulary must name its scene")

    raw_phrases = payload.get("phrases")
    if not isinstance(raw_phrases, list) or not raw_phrases:
        raise ValueError("vocabulary must list at least one phrase")

    entries: list[VocabularyEntry] = []
    seen: set[str] = set()
    for raw in raw_phrases:
        phrase = str(raw.get("phrase", "")).strip()
        if not phrase:
            raise ValueError("every phrase entry needs a non-empty phrase")
        role = str(raw.get("role", ""))
        if role not in ALLOWED_ROLES:
            raise ValueError(
                f"phrase {phrase!r} has unknown role {role!r}; "
                f"allowed roles are {ALLOWED_ROLES}"
            )
        synonyms = tuple(str(value).strip() for value in raw.get("synonyms", []))
        if any(not synonym for synonym in synonyms):
            raise ValueError(f"phrase {phrase!r} has an empty synonym")
        for token in (phrase, *synonyms):
            if token in seen:
                raise ValueError(f"duplicate vocabulary phrase {token!r}")
            seen.add(token)
        entries.append(VocabularyEntry(phrase=phrase, role=role, synonyms=synonyms))

    concepts = {entry.phrase for entry in entries}
    raw_pairs = payload.get("expected_part_of", [])
    if not isinstance(raw_pairs, list):
        raise ValueError("expected_part_of must be a list of [child, parent] pairs")
    pairs: list[tuple[str, str]] = []
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, list) or len(raw_pair) != 2:
            raise ValueError("expected_part_of entries must be [child, parent] pairs")
        child, parent = (str(raw_pair[0]), str(raw_pair[1]))
        for concept in (child, parent):
            if concept not in concepts:
                raise ValueError(
                    f"expected_part_of references unknown concept {concept!r}"
                )
        pairs.append((child, parent))

    return Vocabulary(
        scene=scene,
        entries=tuple(entries),
        expected_part_of=tuple(pairs),
    )
