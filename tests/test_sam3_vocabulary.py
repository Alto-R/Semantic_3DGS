"""Tests for the SAM3 vocabulary configuration."""

import json
from pathlib import Path

import pytest

from scripts.task1.sam3.vocabulary import load_vocabulary

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_config(tmp_path, payload):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


BASE = {
    "scene": "old_street",
    "phrases": [
        {"phrase": "car", "role": "gnn_node", "synonyms": ["vehicle"]},
        {"phrase": "tree", "role": "gnn_node"},
        {"phrase": "building", "role": "context_probe"},
        {"phrase": "window", "role": "gnn_node"},
        {"phrase": "manhole cover", "role": "oov_probe"},
    ],
    "expected_part_of": [["window", "building"]],
}


def test_load_vocabulary_roundtrip(tmp_path):
    vocab = load_vocabulary(write_config(tmp_path, BASE))
    assert vocab.scene == "old_street"
    assert [entry.phrase for entry in vocab.entries] == [
        "car",
        "tree",
        "building",
        "window",
        "manhole cover",
    ]
    assert vocab.phrase_to_concept()["vehicle"] == "car"
    assert vocab.phrase_to_concept()["car"] == "car"
    assert vocab.prompt_phrases() == (
        "car",
        "vehicle",
        "tree",
        "building",
        "window",
        "manhole cover",
    )


def test_vocabulary_sha256_is_stable(tmp_path):
    first = load_vocabulary(write_config(tmp_path, BASE)).sha256()
    second = load_vocabulary(write_config(tmp_path, BASE)).sha256()
    assert first == second
    assert len(first) == 64


def test_duplicate_phrase_rejected(tmp_path):
    bad = dict(BASE)
    bad["phrases"] = BASE["phrases"] + [{"phrase": "car", "role": "gnn_node"}]
    with pytest.raises(ValueError, match="duplicate"):
        load_vocabulary(write_config(tmp_path, bad))


def test_duplicate_synonym_rejected(tmp_path):
    bad = dict(BASE)
    bad["phrases"] = BASE["phrases"] + [
        {"phrase": "sedan", "role": "gnn_node", "synonyms": ["vehicle"]}
    ]
    with pytest.raises(ValueError, match="duplicate"):
        load_vocabulary(write_config(tmp_path, bad))


def test_unknown_role_rejected(tmp_path):
    bad = dict(BASE)
    bad["phrases"] = [{"phrase": "car", "role": "hero"}]
    with pytest.raises(ValueError, match="role"):
        load_vocabulary(write_config(tmp_path, bad))


def test_empty_phrase_list_rejected(tmp_path):
    bad = dict(BASE)
    bad["phrases"] = []
    bad["expected_part_of"] = []
    with pytest.raises(ValueError, match="phrase"):
        load_vocabulary(write_config(tmp_path, bad))


def test_expected_part_of_must_reference_concepts(tmp_path):
    bad = dict(BASE)
    bad["expected_part_of"] = [["window", "castle"]]
    with pytest.raises(ValueError, match="castle"):
        load_vocabulary(write_config(tmp_path, bad))


def test_old_street_config_loads():
    vocab = load_vocabulary(
        PROJECT_ROOT / "configs" / "task1_sam3_vocabulary.old_street.json"
    )
    assert vocab.scene == "old_street"
    roles = {entry.role for entry in vocab.entries}
    assert roles == {"gnn_node", "context_probe", "oov_probe"}
    assert ("window", "building") in vocab.expected_part_of
