"""Tests for the SAM3 backend interface and view segmentation core."""

import json

import numpy as np
import pytest

from scripts.task1.sam3.sam3_backend import InstanceMask, MockSam3Backend
from scripts.task1.sam3.segment_views_core import (
    MASKS_CONTRACT,
    MASKS_SOURCE,
    build_masks_manifest,
    segment_view,
    write_view_masks,
)
from scripts.task1.sam3.vocabulary import load_vocabulary


@pytest.fixture()
def vocab(tmp_path):
    payload = {
        "scene": "old_street",
        "phrases": [
            {"phrase": "car", "role": "gnn_node", "synonyms": ["vehicle"]},
            {"phrase": "building", "role": "context_probe"},
        ],
        "expected_part_of": [],
    }
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_vocabulary(path)


def make_backend():
    # phrase -> list of (y0, y1, x0, x1, score) rectangles
    return MockSam3Backend(
        {
            "car": [(2, 6, 1, 5, 0.9), (2, 6, 8, 12, 0.8)],
            "vehicle": [(2, 6, 1, 5, 0.7)],
            "building": [(0, 10, 0, 16, 0.95)],
        }
    )


def test_mock_backend_returns_rect_masks():
    masks = make_backend().segment(np.zeros((10, 16, 3), np.uint8), "car")
    assert len(masks) == 2
    assert isinstance(masks[0], InstanceMask)
    assert masks[0].mask.shape == (10, 16)
    assert masks[0].mask.dtype == np.bool_
    assert masks[0].mask[2:6, 1:5].all()
    assert masks[0].mask.sum() == 16
    assert masks[0].score == 0.9


def test_mock_backend_clips_rectangles_to_image():
    backend = MockSam3Backend({"car": [(5, 20, 10, 30, 0.9)]})
    masks = backend.segment(np.zeros((10, 16, 3), np.uint8), "car")
    assert len(masks) == 1
    assert masks[0].mask.sum() == (10 - 5) * (16 - 10)


def test_mock_backend_unknown_phrase_is_empty():
    assert make_backend().segment(np.zeros((10, 16, 3), np.uint8), "sky") == []


def test_segment_view_prompts_all_phrases_and_filters_score(vocab):
    view = segment_view(
        np.zeros((10, 16, 3), np.uint8), vocab, make_backend(), min_score=0.75
    )
    assert [mask.phrase for mask in view] == ["car", "car", "building"]
    assert [mask.concept for mask in view] == ["car", "car", "building"]


def test_segment_view_keeps_synonyms_under_canonical_concept(vocab):
    view = segment_view(
        np.zeros((10, 16, 3), np.uint8), vocab, make_backend(), min_score=0.0
    )
    assert [(mask.phrase, mask.concept) for mask in view] == [
        ("car", "car"),
        ("car", "car"),
        ("vehicle", "car"),
        ("building", "building"),
    ]


def test_write_view_masks_roundtrip(tmp_path, vocab):
    view = segment_view(
        np.zeros((10, 16, 3), np.uint8), vocab, make_backend(), min_score=0.0
    )
    fragment = write_view_masks(tmp_path / "00000_cam0192.npz", view)
    with np.load(tmp_path / "00000_cam0192.npz") as data:
        stack = data["mask_stack"]
    assert stack.shape == (4, 10, 16)
    assert stack.dtype == np.uint8
    assert fragment["mask_count"] == 4
    assert [mask["phrase"] for mask in fragment["masks"]] == [
        "car",
        "car",
        "vehicle",
        "building",
    ]
    assert fragment["masks"][0]["mask_index"] == 0
    assert fragment["masks"][0]["concept"] == "car"
    assert fragment["masks"][0]["pixel_count"] == 16


def test_masks_manifest_contract(vocab):
    manifest = build_masks_manifest(
        scene="old_street",
        vocabulary=vocab,
        model_id="facebook/sam3",
        model_revision="mock",
        min_score=0.5,
        frames=[],
    )
    assert manifest["source"] == MASKS_SOURCE
    assert manifest["contract"] == MASKS_CONTRACT
    assert manifest["scene"] == "old_street"
    assert manifest["model_id"] == "facebook/sam3"
    assert manifest["model_revision"] == "mock"
    assert manifest["min_score"] == 0.5
    assert manifest["vocabulary_sha256"] == vocab.sha256()
    assert manifest["frames"] == []
