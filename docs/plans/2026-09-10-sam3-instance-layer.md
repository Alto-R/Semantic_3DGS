# SAM3 Instance Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the SAM3 instance route (segment -> lift -> associate -> vote -> hierarchy -> QA) as locally testable code with a mocked SAM3 backend; cluster execution is wired but deferred.

**Architecture:** New subpackage `scripts/task1/sam3/` mirroring the dense DINOv3 route's structure: pure functions with strict contracts tested by pytest, thin CUDA/cluster drivers that reuse `scripts/task1/common/flashsplat_cameras.py`. Design authority: `docs/plans/2026-09-10-sam3-instance-layer-design.md`.

**Tech Stack:** Python 3.10+, numpy, pytest (8.3.4 local). PIL only for QA overlays. torch/FlashSplat/transformers imports stay inside cluster-only `main()` functions, exactly like `lift_dense_view_votes.py` does.

**Ground rules (from design):**
- No sum-to-one constraint across concepts; per-concept independent votes.
- Instance ids are uint16, 1-based; 0 always means background/unlabeled.
- Every artifact gets `source`/`contract` strings and a manifest, matching repo style.
- Local machine never downloads a checkpoint and never imports torch in tests.
- Commit style: short imperative subject, no type prefix (match `git log`), plus the Claude Co-Authored-By trailer.
- Run tests from repo root: `python -m pytest tests/<file> -v`.

---

### Task 1: Vocabulary config module

**Files:**
- Create: `scripts/task1/sam3/__init__.py` (empty)
- Create: `scripts/task1/sam3/vocabulary.py`
- Create: `configs/task1_sam3_vocabulary.old_street.json`
- Test: `tests/test_sam3_vocabulary.py`

**Step 1: Write the failing tests**

```python
"""Tests for the SAM3 vocabulary configuration."""
import json
import pytest
from scripts.task1.sam3.vocabulary import load_vocabulary

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
    assert [e.phrase for e in vocab.entries] == [
        "car", "tree", "building", "window", "manhole cover"]
    # synonyms fold to the canonical concept of their entry
    assert vocab.phrase_to_concept()["vehicle"] == "car"
    assert vocab.phrase_to_concept()["car"] == "car"
    # every prompted phrase appears exactly once in the prompt list
    assert vocab.prompt_phrases() == (
        "car", "vehicle", "tree", "building", "window", "manhole cover")

def test_vocabulary_sha256_is_stable(tmp_path):
    a = load_vocabulary(write_config(tmp_path, BASE)).sha256()
    b = load_vocabulary(write_config(tmp_path, BASE)).sha256()
    assert a == b and len(a) == 64

def test_duplicate_phrase_rejected(tmp_path):
    bad = dict(BASE)
    bad["phrases"] = BASE["phrases"] + [{"phrase": "car", "role": "gnn_node"}]
    with pytest.raises(ValueError, match="duplicate"):
        load_vocabulary(write_config(tmp_path, bad))

def test_unknown_role_rejected(tmp_path):
    bad = dict(BASE)
    bad["phrases"] = [{"phrase": "car", "role": "hero"}]
    with pytest.raises(ValueError, match="role"):
        load_vocabulary(write_config(tmp_path, bad))

def test_expected_part_of_must_reference_concepts(tmp_path):
    bad = dict(BASE)
    bad["expected_part_of"] = [["window", "castle"]]
    with pytest.raises(ValueError, match="castle"):
        load_vocabulary(write_config(tmp_path, bad))
```

**Step 2:** `python -m pytest tests/test_sam3_vocabulary.py -v` — expect FAIL (module not found).

**Step 3: Implement `vocabulary.py`**

Frozen dataclasses `VocabularyEntry(phrase, role, synonyms)` and
`Vocabulary(scene, entries, expected_part_of)`. Roles allowed:
`{"gnn_node", "context_probe", "oov_probe"}`. Methods:
- `phrase_to_concept()` — canonical concept = entry phrase; synonyms map to it;
- `prompt_phrases()` — entry phrase then its synonyms, order preserved, tuple;
- `sha256()` — sha256 of `json.dumps(payload, sort_keys=True)` of the parsed
  canonical structure.
Validation: unique phrases across phrases+synonyms, known roles, part_of pairs
reference canonical concepts, non-empty phrase list. Raise `ValueError` with
the offending token in the message.

**Step 4:** rerun — expect 5 PASS.

**Step 5: Write the real config** `configs/task1_sam3_vocabulary.old_street.json`
with roles per design: gnn_node (car, tree, person, streetlight, bench,
signboard, storefront, window, door, bicycle, motorbike, trash can, potted
plant), context_probe (building, road, sidewalk, wall, sky), oov_probe
(manhole cover, air conditioning unit, electric scooter, awning), and
`expected_part_of` at least `[["window","building"],["storefront","building"],
["signboard","storefront"]]`. Load it in a test:

```python
def test_old_street_config_loads():
    vocab = load_vocabulary(Path("configs/task1_sam3_vocabulary.old_street.json"))
    roles = {e.role for e in vocab.entries}
    assert roles == {"gnn_node", "context_probe", "oov_probe"}
```

**Step 6: Commit** `Add SAM3 vocabulary configuration module`

---

### Task 2: SAM3 backend interface, mock backend, S1 segmentation core

**Files:**
- Create: `scripts/task1/sam3/sam3_backend.py`
- Create: `scripts/task1/sam3/segment_views_core.py`
- Test: `tests/test_sam3_segment_views.py`

**Step 1: Failing tests**

```python
import numpy as np
from scripts.task1.sam3.sam3_backend import InstanceMask, MockSam3Backend
from scripts.task1.sam3.segment_views_core import (
    MASKS_CONTRACT, MASKS_SOURCE, build_masks_manifest, segment_view,
    write_view_masks)

def make_backend():
    # phrase -> list of (y0, y1, x0, x1, score) rectangles
    return MockSam3Backend({
        "car": [(2, 6, 1, 5, 0.9), (2, 6, 8, 12, 0.8)],
        "vehicle": [(2, 6, 1, 5, 0.7)],
        "building": [(0, 10, 0, 16, 0.95)],
    })

def test_mock_backend_returns_rect_masks():
    masks = make_backend().segment(np.zeros((10, 16, 3), np.uint8), "car")
    assert len(masks) == 2
    assert masks[0].mask.shape == (10, 16) and masks[0].mask.dtype == np.bool_
    assert masks[0].mask[2:6, 1:5].all() and masks[0].mask.sum() == 16
    assert masks[0].score == 0.9

def test_segment_view_prompts_all_phrases_and_filters_score(vocab_fixture):
    view = segment_view(np.zeros((10, 16, 3), np.uint8), vocab_fixture,
                        make_backend(), min_score=0.75)
    phrases = [m.phrase for m in view]
    assert phrases == ["car", "car", "building"]  # vehicle 0.7 filtered
    assert all(m.concept in {"car", "building"} for m in view)

def test_write_view_masks_roundtrip(tmp_path, vocab_fixture):
    view = segment_view(np.zeros((10, 16, 3), np.uint8), vocab_fixture,
                        make_backend(), min_score=0.0)
    fragment = write_view_masks(tmp_path / "00000_cam0192.npz", view)
    with np.load(tmp_path / "00000_cam0192.npz") as data:
        stack = data["mask_stack"]
    assert stack.shape == (4, 10, 16) and stack.dtype == np.uint8
    assert [m["phrase"] for m in fragment["masks"]] == [
        "car", "car", "vehicle", "building"]
    assert fragment["masks"][0]["mask_index"] == 0

def test_masks_manifest_contract(tmp_path, vocab_fixture):
    manifest = build_masks_manifest(
        scene="old_street", vocabulary=vocab_fixture,
        model_id="facebook/sam3", model_revision="mock",
        min_score=0.5, frames=[])
    assert manifest["source"] == MASKS_SOURCE
    assert manifest["contract"] == MASKS_CONTRACT
    assert manifest["vocabulary_sha256"] == vocab_fixture.sha256()
```

Add a `vocab_fixture` pytest fixture building the Task 1 BASE config in
`tmp_path` (module-level helper, reuse via conftest is unnecessary — keep it
in this file).

**Step 2:** run — FAIL. **Step 3: Implement.**

- `InstanceMask` frozen dataclass `(mask: np.ndarray, score: float)`;
  `Sam3Backend` Protocol with `segment(image, phrase) -> list[InstanceMask]`;
  `MockSam3Backend` produces clipped rectangle masks, deterministic order.
- `segment_views_core.ViewMask(phrase, concept, score, mask)`;
  `segment_view` iterates `vocabulary.prompt_phrases()`, maps phrase->concept,
  drops masks below `min_score` or empty after clipping, preserves backend
  order per phrase.
- `write_view_masks` saves `mask_stack` (M,H,W uint8, savez_compressed) and
  returns the frame manifest fragment with per-mask
  `{mask_index, phrase, concept, score, pixel_count}`.
- `MASKS_SOURCE = "sam3_promptable_concept_segmentation"`,
  `MASKS_CONTRACT = "per_view_binary_instance_masks_v1"`;
  `build_masks_manifest` records scene, model id/revision, min_score,
  vocabulary sha, frames.

**Step 4:** run — PASS. **Step 5: Commit** `Add SAM3 mock backend and view segmentation core`

---

### Task 3: S2 pure lift — concept index maps and membership votes

**Files:**
- Create: `scripts/task1/sam3/lift_mask_votes_core.py`
- Test: `tests/test_sam3_mask_lift.py`

**Step 1: Failing tests**

```python
import numpy as np
import pytest
from scripts.task1.sam3.lift_mask_votes_core import (
    concept_index_map, mask_membership_votes, observed_gaussians)

def test_concept_index_map_scores_break_overlaps():
    a = np.zeros((4, 4), np.uint8); a[0:2, 0:2] = 1
    b = np.zeros((4, 4), np.uint8); b[1:3, 1:3] = 1
    index_map = concept_index_map(np.stack([a, b]), np.array([0.6, 0.9]))
    assert index_map.dtype == np.float32
    assert index_map[0, 0] == 1.0        # only a
    assert index_map[1, 1] == 2.0        # overlap -> higher score wins
    assert index_map[3, 3] == 0.0        # background

def test_membership_votes_are_per_mask_fractions():
    # rows: background + 2 masks, 5 gaussians
    used = np.array([
        [0.0, 1.0, 0.5, 0.0, 0.0],
        [2.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0]], np.float32)
    visibility = used.sum(axis=0)
    indices, mask_ids, weights = mask_membership_votes(
        used, visibility, view_mask_indices=np.array([4, 7], np.uint16))
    assert indices.tolist() == [0, 1, 2]
    assert mask_ids.tolist() == [4, 4, 7]
    np.testing.assert_allclose(weights, [1.0, 0.5, 0.5])
    # no sum-to-one requirement anywhere

def test_membership_votes_rejects_bad_shapes():
    with pytest.raises(ValueError):
        mask_membership_votes(np.zeros((1, 4), np.float32),
                              np.zeros(4, np.float32),
                              np.array([1], np.uint16))  # needs bg + 1 row

def test_observed_gaussians():
    observed = observed_gaussians(np.array([0.0, 0.4, 0.0, 2.0], np.float32))
    assert observed.dtype == np.uint32 and observed.tolist() == [1, 3]
```

**Step 2:** FAIL. **Step 3: Implement.**

- `concept_index_map(mask_stack, scores)` — argmax by score over overlapping
  same-concept masks, float32 map, 0 background, row k -> k+1; validate
  shapes and score length.
- `mask_membership_votes(used_count, visibility, view_mask_indices)` —
  `used_count` has 1 background row + len(view_mask_indices) mask rows
  (validate), weights `used[row]/visibility` where positive; emitted
  `mask_ids` are the per-view mask indices from S1 (uint16); finite and
  non-negative checks copied from `dense_sparse_view_votes`, but NO
  sum-to-one check. `view_mask_indices` maps the pass rows back to S1
  mask_index values.
- `observed_gaussians(visibility)` — uint32 indices of visibility > 0.

**Step 4:** PASS. **Step 5: Commit** `Add per-concept mask lift core`

---

### Task 4: S3 cross-view instance association

**Files:**
- Create: `scripts/task1/sam3/associate_instances.py`
- Test: `tests/test_sam3_association.py`

**Step 1: Failing tests**

```python
import numpy as np
from scripts.task1.sam3.associate_instances import (
    MaskSupport, associate_masks, build_instance_registry, weighted_jaccard)

def sup(view, mask_index, concept, pairs, score=0.9):
    idx = np.array([p[0] for p in pairs], np.uint32)
    wts = np.array([p[1] for p in pairs], np.float32)
    return MaskSupport(view=view, mask_index=mask_index, concept=concept,
                       score=score, indices=idx, weights=wts)

def test_weighted_jaccard():
    a = sup("v0", 0, "car", [(1, 1.0), (2, 0.5)])
    b = sup("v1", 0, "car", [(2, 0.5), (3, 1.0)])
    assert weighted_jaccard(a, b) == 0.5 / 2.5

def test_same_object_across_views_merges():
    car_v0 = sup("v0", 0, "car", [(i, 1.0) for i in range(10)])
    car_v1 = sup("v1", 0, "car", [(i, 1.0) for i in range(2, 12)])
    other  = sup("v1", 1, "car", [(i, 1.0) for i in range(100, 110)])
    groups = associate_masks([car_v0, car_v1, other], threshold=0.3)
    assert sorted(len(g) for g in groups) == [1, 2]

def test_concepts_never_merge():
    car = sup("v0", 0, "car", [(i, 1.0) for i in range(10)])
    tree = sup("v1", 0, "tree", [(i, 1.0) for i in range(10)])
    assert len(associate_masks([car, tree], threshold=0.1)) == 2

def test_same_view_pairs_do_not_union_directly():
    a = sup("v0", 0, "car", [(i, 1.0) for i in range(10)])
    b = sup("v0", 1, "car", [(i, 1.0) for i in range(10)])
    assert len(associate_masks([a, b], threshold=0.1)) == 2

def test_registry_shape_and_conflict_count():
    a = sup("v0", 0, "car", [(i, 1.0) for i in range(10)])
    b = sup("v1", 0, "car", [(i, 1.0) for i in range(10)])
    c = sup("v0", 1, "car", [(i, 1.0) for i in range(5, 15)])
    registry = build_instance_registry(
        associate_masks([a, b, c], threshold=0.3), threshold=0.3)
    inst = registry["instances"]
    assert inst[0]["instance_id"] == 1  # ids are 1-based uint16 range
    merged = max(inst, key=lambda i: len(i["members"]))
    assert merged["supporting_camera_count"] == 2
    # a and c share view v0 inside one group -> counted as a conflict
    assert registry["same_view_conflict_groups"] == 1
```

**Step 2:** FAIL. **Step 3: Implement.**

- `MaskSupport` frozen dataclass as in the test.
- `weighted_jaccard` on sorted sparse arrays: sum(min)/sum(max) via
  `np.intersect1d(..., return_indices=True)` plus the disjoint remainders.
- `associate_masks`: per concept, build an inverted index gaussian->mask list
  to collect candidate pairs (skip pairs from the same view), union-find over
  pairs with J >= threshold, return groups as lists of input positions.
- `build_instance_registry(groups, threshold)`: instances sorted by
  (concept, first view, mask_index) for determinism; per instance:
  `instance_id` (1-based, fail if > 65535), concept, members
  `[{view, mask_index, score}]`, `supporting_camera_count` (distinct views),
  `same_view_conflict_groups` count where a group holds >= 2 masks from one
  view (report only, no splitting — pilot metric), plus
  `{"source": "sam3_support_overlap_association",
    "contract": "per_concept_weighted_jaccard_union_v1", "threshold": ...}`.

**Step 4:** PASS. **Step 5: Commit** `Add cross-view instance association`

---

### Task 5: S4 per-concept consensus and CSR membership

**Files:**
- Create: `scripts/task1/sam3/instance_membership.py`
- Test: `tests/test_sam3_membership.py`

**Step 1: Failing tests**

```python
import numpy as np
from scripts.task1.sam3.instance_membership import (
    STATUS_ACCEPTED, STATUS_SINGLE_CAMERA, STATUS_WEAK_MAJORITY,
    accumulate_membership, view_concept_winners)

def test_view_concept_winners_require_dominant_unique_mass():
    indices = np.array([0, 0, 1, 2], np.uint32)
    mask_ids = np.array([4, 7, 4, 7], np.uint16)
    weights = np.array([0.8, 0.2, 0.4, 0.5], np.float32)
    g, m = view_concept_winners(indices, mask_ids, weights, min_weight=0.5)
    # gaussian 0: mask 4 wins with 0.8; gaussian 1: below min_weight;
    # gaussian 2: exactly 0.5 counts
    assert g.tolist() == [0, 2] and m.tolist() == [4, 7]

def test_view_concept_winners_tie_abstains():
    indices = np.array([0, 0], np.uint32)
    mask_ids = np.array([4, 7], np.uint16)
    weights = np.array([0.5, 0.5], np.float32)
    g, _ = view_concept_winners(indices, mask_ids, weights, min_weight=0.5)
    assert g.size == 0

def test_accumulate_membership_statuses():
    # gaussian 0 observed by 3 cameras, supported by 2 -> accepted
    # gaussian 1 observed by 3, supported by 1 -> single camera
    # gaussian 2 observed by 4, supported by 2 -> weak majority
    events = [
        (np.array([0, 1, 2], np.uint32), np.array([1, 1, 1], np.uint16)),
        (np.array([0, 2], np.uint32), np.array([1, 1], np.uint16)),
    ]
    observe_counts = np.array([3, 3, 4, 9], np.uint16)
    csr = accumulate_membership(events, observe_counts, gaussian_count=4)
    assert csr.indptr.tolist() == [0, 1, 2, 3, 3]
    assert csr.instance_ids.tolist() == [1, 1, 1]
    assert csr.support_counts.tolist() == [2, 1, 2]
    np.testing.assert_allclose(csr.scores, [2/3, 1/3, 2/4])
    assert csr.status.tolist() == [
        STATUS_ACCEPTED, STATUS_SINGLE_CAMERA, STATUS_WEAK_MAJORITY]
```

**Step 2:** FAIL. **Step 3: Implement.**

- `view_concept_winners`: group by gaussian (lexsort), unique strict argmax
  with weight >= min_weight, exact ties abstain — mirrors the audited
  "one unique max else abstain" policy.
- `MembershipCSR` frozen dataclass `(indptr int64, instance_ids uint16,
  support_counts uint16, observe_counts uint16, scores float32,
  status uint8)`; entries sorted by (gaussian, instance).
- `accumulate_membership(events, observe_counts, gaussian_count)`: events are
  per-view `(gaussian_indices, global_instance_ids)` winner arrays (the
  driver maps per-view mask winners to global ids via the registry); combine
  with a `g * 2**16 + i` key, `np.unique(..., return_counts=True)`; statuses:
  support >= 2 and support > observe/2 -> ACCEPTED (1); support == 1 ->
  SINGLE_CAMERA (2); else WEAK_MAJORITY (3). observe_count 0 with support is
  a `RuntimeError` (vote without visibility is a driver bug).
- `save_membership(path, csr)` / `load_membership(path)` NPZ round-trip
  (add a small round-trip test in the same file).

**Step 4:** PASS. **Step 5: Commit** `Add per-concept membership consensus`

---

### Task 6: PLY vertex xyz reader

**Files:**
- Modify: `scripts/task1/common/ply_utils.py` (add `read_vertex_xyz`)
- Test: `tests/test_sam3_ply_xyz.py`

Read `scripts/task1/common/ply_utils.py` and `scripts/task1/qa/add_labels_from_npy.py`
first; reuse the existing header parsing. Test writes a tiny
binary_little_endian PLY (3 vertices, properties x,y,z float + one extra
float property to prove offset handling) with plain struct packing, then:

```python
def test_read_vertex_xyz_roundtrip(tmp_path):
    path = write_tiny_ply(tmp_path)          # helper in the test file
    xyz = read_vertex_xyz(path)
    assert xyz.shape == (3, 3) and xyz.dtype == np.float32
    np.testing.assert_allclose(xyz[1], [4.0, 5.0, 6.0])
```

Implement with a numpy structured dtype built from the parsed header
(fail on ascii PLYs and on missing x/y/z). Commit:
`Add PLY vertex position reader`

---

### Task 7: S5 hierarchy — overlap classification and scene graph

**Files:**
- Create: `scripts/task1/sam3/instance_hierarchy.py`
- Test: `tests/test_sam3_hierarchy.py`

**Step 1: Failing tests**

```python
import numpy as np
from scripts.task1.sam3.instance_hierarchy import (
    OverlapThresholds, build_scene_graph, classify_overlaps,
    containment, instance_geometry)

def test_containment():
    a = np.array([1, 2, 3], np.uint32)
    b = np.array([2, 3, 4, 5], np.uint32)
    assert containment(a, b) == 2 / 3

def test_classify_part_of_and_noise():
    building = {"instance_id": 1, "concept": "building",
                "support": np.arange(0, 100, dtype=np.uint32)}
    window = {"instance_id": 2, "concept": "window",
              "support": np.arange(10, 20, dtype=np.uint32)}
    car = {"instance_id": 3, "concept": "car",
           "support": np.arange(98, 130, dtype=np.uint32)}
    result = classify_overlaps(
        [building, window, car],
        OverlapThresholds(part_of_child=0.6, part_of_parent=0.3,
                          duplicate_mutual=0.8))
    assert result.part_of_edges == [
        {"child": 2, "parent": 1, "containment_child_in_parent": 1.0,
         "containment_parent_in_child": 0.1}]
    assert result.merges == []
    assert result.noise_pairs == [(1, 3)]

def test_classify_duplicates_merge_same_concept():
    a = {"instance_id": 1, "concept": "car",
         "support": np.arange(0, 10, dtype=np.uint32)}
    b = {"instance_id": 2, "concept": "car",
         "support": np.arange(0, 9, dtype=np.uint32)}
    result = classify_overlaps([a, b], OverlapThresholds())
    assert result.merges == [(2, 1)]  # smaller merges into larger

def test_instance_geometry_weighted():
    xyz = np.array([[0, 0, 0], [2, 0, 0], [9, 9, 9]], np.float32)
    geometry = instance_geometry(
        np.array([0, 1], np.uint32), np.array([1.0, 3.0], np.float32), xyz)
    np.testing.assert_allclose(geometry["centroid"], [1.5, 0.0, 0.0])
    np.testing.assert_allclose(geometry["bbox_min"], [0, 0, 0])
    np.testing.assert_allclose(geometry["bbox_max"], [2, 0, 0])

def test_scene_graph_nodes_edges_and_no_cycles():
    # build from the part_of fixture above; assert node fields
    # {instance_id, concept, gaussian_count, centroid, bbox_min, bbox_max}
    # and edges list; assert a cycle raises ValueError
    ...
```

Write the cycle test concretely: feeding edges `1->2` and `2->1` into
`build_scene_graph` must raise `ValueError("part_of cycle")`.

**Step 2:** FAIL. **Step 3: Implement.**

- `containment(a, b)` = |A∩B| / |A| via `np.intersect1d`.
- `OverlapThresholds` dataclass with the design defaults
  (part_of_child=0.6, part_of_parent=0.3, duplicate_mutual=0.8).
- `classify_overlaps(instances, thresholds)`: pairwise over instances with
  any support intersection; same concept + mutual containment >=
  duplicate_mutual -> merge (smaller into larger); different concepts with
  asymmetric containment -> part_of edge (child = contained); everything
  else with nonzero overlap -> noise pair. Deterministic ordering by ids.
- `instance_geometry(indices, scores, xyz)`: score-weighted centroid,
  axis-aligned bbox of the support.
- `build_scene_graph(instances, edges, merges, xyz, vocabulary)`: apply
  merges, drop merged ids, attach geometry per node, validate acyclicity by
  topological sort, return the JSON-ready dict with
  `source="sam3_instance_scene_graph"`,
  `contract="membership_derived_nodes_part_of_edges_v1"`, and an
  `expected_part_of_found` QA block comparing against
  `vocabulary.expected_part_of`.

**Step 4:** PASS. **Step 5: Commit** `Add overlap classification and scene graph`

---

### Task 8: QA overlays and contact sheet

**Files:**
- Create: `scripts/task1/sam3/render_instance_overlays.py`
- Test: `tests/test_sam3_overlays.py`

Pure PIL/numpy: `instance_palette(n)` (golden-ratio HSV, deterministic),
`overlay_instances(rgb, mask_stack, alpha=0.55)`, `contact_sheet(images,
columns)` returning a PIL image. Tests assert output sizes, that overlay
changes exactly the masked pixels, and palette determinism. Guard the module
import with `pytest.importorskip("PIL")` in the test if Pillow is absent
locally; if it is absent, `pip install pillow` into the local env first.
Commit: `Add instance overlay QA rendering`

---

### Task 9: Drivers and cluster glue (code only — cluster-verify later)

**Files:**
- Create: `scripts/task1/sam3/sam3_segment_views.py` (S1 driver)
- Create: `scripts/task1/sam3/lift_mask_view_votes.py` (S2 driver)
- Create: `scripts/task1/sam3/run_sam3_instance_scene.sh`
- Create: `scripts/slurm/slurm_task1_sam3_instance_scene.sbatch`
- Test: `tests/test_sam3_drivers.py` (argument wiring + manifest contracts
  only, using the mock backend; never imports torch)

Rules, mirroring the dense route exactly:
- S1 driver: `--rgb-dir --vocabulary --output-dir --model-id --revision
  --min-score --backend {transformers,mock}`; the transformers backend class
  `Sam3TransformersBackend` lives behind a lazy import inside `main()`;
  loading it locally without the package must raise a clear RuntimeError.
  Write `sam3_masks_manifest.json` via Task 2 helpers. Refuse to overwrite
  without `--overwrite` (FileExistsError, repo convention).
- S2 driver: clone the `main()` skeleton of
  `scripts/task1/dinov3/lift_dense_view_votes.py` (camera loading, FlashSplat
  modules, per-view loop) but per concept pass: build the index map with
  `concept_index_map`, call `render_flashsplat` once per (view, concept),
  emit one NPZ per view (`indices`, `mask_ids`, `weights`, `observed`) plus
  `sam3_vote_manifest.json` with
  `source="sam3_mask_flashsplat_votes"`,
  `contract="per_concept_pass_membership_votes_v1"` and per-frame concept
  pass bookkeeping. Validate the S1 manifest source/contract before lifting
  (test this validation locally with a wrong manifest).
- S3–S5 drivers are pure CPU: extend `associate_instances.py`,
  `instance_membership.py`, `instance_hierarchy.py` each with a `main()`
  reading the upstream manifests, sha256-pinning inputs like
  `materialize_hard_vote_consensus.py` does, writing
  `instance_registry.json`, `membership.npz` + `membership_summary.json`,
  `hierarchy.json`, `scene_graph.json`, and a derived flat
  `gaussian_instances.npy` (top accepted score per Gaussian, ties -> 0) for
  the existing PLY tooling.
- Shell + sbatch: chain S1..S5 + overlays in the existing scheduler style
  (read one existing sbatch for the header conventions before writing).
- Driver tests: run S1 end-to-end in tmp_path with the mock backend on
  synthetic PNGs (8x8), then assert manifest contracts chain; assert the S2
  manifest validator rejects a manifest whose contract string is wrong.

Commit: `Add SAM3 route drivers and scheduler entry points`

---

### Task 10: Documentation

**Files:**
- Modify: `README.md` (new "SAM3 instance route (experimental)" section:
  purpose, entry points, output contract; mark cluster-verification pending)
- Modify: `scripts/README.md` (script inventory rows)
- Modify: `docs/EXTERNAL_REPOS.md` (SAM3 model pin: `facebook/sam3`,
  revision recorded at cluster setup, transformers version note)

Commit: `Document the SAM3 instance route`

---

### Task 11: Full verification

Run: `python -m pytest tests/ -v` — the whole suite (old + new) must pass.
Then `git log --oneline` to confirm one commit per task. Update
`docs/plans/2026-09-10-sam3-instance-layer.md` checkboxes if used. Report:
what is locally verified (pure route on mocks) versus deferred
(SAM3 checkpoint, FlashSplat passes, Slurm run — cluster).
