"""Driver tests for the SAM3 route: S1 with the mock backend, manifest
validation, and a synthetic S3->S4->S5 mini pipeline. Nothing here imports
torch or downloads a model."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from scripts.task1.sam3 import (  # noqa: E402
    associate_instances,
    instance_hierarchy,
    instance_membership,
    render_instance_overlays,
    sam3_segment_views,
)
from scripts.task1.sam3.lift_mask_view_votes import (  # noqa: E402
    VOTES_CONTRACT,
    VOTES_SOURCE,
    validate_votes_manifest,
)
from scripts.task1.sam3.segment_views_core import (  # noqa: E402
    MASKS_CONTRACT,
    MASKS_SOURCE,
    validate_masks_manifest,
)


VOCAB = {
    "scene": "old_street",
    "phrases": [
        {"phrase": "window", "role": "gnn_node"},
        {"phrase": "building", "role": "context_probe"},
    ],
    "expected_part_of": [["window", "building"]],
}

# Rectangles (y0, y1, x0, x1, score) reused by every synthetic view.
BOXES = {
    "window": [(0, 2, 0, 2, 0.9), (0, 2, 4, 6, 0.85)],
    "building": [(0, 8, 0, 8, 0.8)],
}

VIEW_STEMS = ("00000_cam0001", "00001_cam0002")


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def make_s1_inputs(tmp_path):
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir(exist_ok=True)
    for stem in VIEW_STEMS:
        Image.new("RGB", (8, 8), (90, 90, 90)).save(rgb_dir / f"{stem}.png")
    vocab_path = write_json(tmp_path / "vocab.json", VOCAB)
    boxes_path = write_json(tmp_path / "boxes.json", BOXES)
    return rgb_dir, vocab_path, boxes_path


def run_s1(tmp_path, extra=()):
    rgb_dir, vocab_path, boxes_path = make_s1_inputs(tmp_path)
    output_dir = tmp_path / "s1"
    argv = [
        "--rgb-dir", str(rgb_dir),
        "--vocabulary", str(vocab_path),
        "--output-dir", str(output_dir),
        "--backend", "mock",
        "--mock-boxes", str(boxes_path),
        "--min-score", "0.5",
        *extra,
    ]
    sam3_segment_views.main(argv)
    return output_dir


def test_s1_driver_writes_masks_and_manifest(tmp_path):
    output_dir = run_s1(tmp_path)
    manifest = json.loads(
        (output_dir / "sam3_masks_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source"] == MASKS_SOURCE
    assert manifest["contract"] == MASKS_CONTRACT
    assert manifest["frame_count"] == 2
    assert manifest["model_id"] == "mock"
    frames = manifest["frames"]
    assert [frame["file"] for frame in frames] == [
        "00000_cam0001.png",
        "00001_cam0002.png",
    ]
    for frame in frames:
        assert frame["mask_count"] == 3
        assert [mask["concept"] for mask in frame["masks"]] == [
            "window", "window", "building"]
        mask_path = output_dir / "masks" / frame["mask_file"]
        assert mask_path.is_file()
        with np.load(mask_path) as data:
            assert data["mask_stack"].shape == (3, 8, 8)


def test_s1_driver_refuses_overwrite(tmp_path):
    run_s1(tmp_path)
    with pytest.raises(FileExistsError):
        run_s1(tmp_path)


def test_s1_driver_overwrite_flag(tmp_path):
    run_s1(tmp_path)
    run_s1(tmp_path, extra=("--overwrite",))


def test_validate_masks_manifest_rejects_wrong_contract():
    with pytest.raises(ValueError):
        validate_masks_manifest(
            {"source": MASKS_SOURCE, "contract": "something_else", "frames": []}
        )


def test_validate_votes_manifest_rejects_wrong_source():
    with pytest.raises(ValueError):
        validate_votes_manifest(
            {
                "source": "dense_votes",
                "contract": VOTES_CONTRACT,
                "gaussian_count": 10,
                "camera_count": 1,
                "frames": [{}],
            }
        )


def test_validate_votes_manifest_rejects_wrong_contract():
    with pytest.raises(ValueError):
        validate_votes_manifest(
            {
                "source": VOTES_SOURCE,
                "contract": "something_else",
                "gaussian_count": 10,
                "camera_count": 1,
                "frames": [{}],
            }
        )


def test_mask_support_join_rejects_duplicate_stems():
    masks_manifest = {
        "frames": [
            {"file": "a.png", "masks": []},
            {"file": "a.png", "masks": []},
        ]
    }
    votes_manifest = {"frames": []}
    with pytest.raises(ValueError, match="duplicate"):
        associate_instances.load_mask_supports(
            masks_manifest, votes_manifest, Path(".")
        )


@pytest.mark.parametrize(
    "module",
    [
        "scripts.task1.sam3.sam3_segment_views",
        "scripts.task1.sam3.lift_mask_view_votes",
        "scripts.task1.sam3.associate_instances",
        "scripts.task1.sam3.instance_membership",
        "scripts.task1.sam3.instance_hierarchy",
        "scripts.task1.sam3.render_instance_overlays",
    ],
)
def test_driver_modules_are_executable(module):
    # The sbatch invokes every stage as `python -m <module>`; a missing
    # __main__ guard makes the stage a silent no-op with exit status 0.
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_overlay_driver_writes_views_and_contact_sheet(tmp_path):
    output_dir = run_s1(tmp_path)
    overlays_dir = tmp_path / "qa"
    render_instance_overlays.main(
        [
            "--masks-manifest", str(output_dir / "sam3_masks_manifest.json"),
            "--masks-dir", str(output_dir / "masks"),
            "--rgb-dir", str(tmp_path / "rgb"),
            "--output-dir", str(overlays_dir),
            "--columns", "2",
        ]
    )
    assert (overlays_dir / "overlays" / "00000_cam0001.png").is_file()
    assert (overlays_dir / "overlays" / "00001_cam0002.png").is_file()
    sheet = Image.open(overlays_dir / "instance_overlays_contact.jpg")
    assert sheet.size == (16, 8)  # two 8x8 tiles in two columns


# --- synthetic mini pipeline: S1 (mock) -> synthetic S2 -> S3 -> S4 -> S5 ---

GAUSSIAN_COUNT = 20


def synth_votes(tmp_path, masks_manifest_path):
    """Hand-built S2 output consistent with the S1 mock masks.

    window mask 0 -> gaussians 0..4, window mask 1 -> 8..12,
    building mask 2 -> 0..19; every view observes every gaussian.
    """

    indices = np.concatenate(
        [
            np.arange(0, 5, dtype=np.uint32),
            np.arange(8, 13, dtype=np.uint32),
            np.arange(0, 20, dtype=np.uint32),
        ]
    )
    mask_ids = np.concatenate(
        [
            np.full(5, 0, np.uint16),
            np.full(5, 1, np.uint16),
            np.full(20, 2, np.uint16),
        ]
    )
    weights = np.concatenate(
        [
            np.full(5, 0.9, np.float32),
            np.full(5, 0.9, np.float32),
            np.full(20, 0.8, np.float32),
        ]
    )
    observed = np.arange(GAUSSIAN_COUNT, dtype=np.uint32)

    vote_dir = tmp_path / "s2" / "view_votes"
    vote_dir.mkdir(parents=True)
    for stem in VIEW_STEMS:
        np.savez_compressed(
            vote_dir / f"{stem}.npz",
            indices=indices,
            mask_ids=mask_ids,
            weights=weights,
            observed=observed,
        )
    frames = [
        {"file": f"{stem}.png", "vote_file": f"view_votes/{stem}.npz"}
        for stem in VIEW_STEMS
    ]
    manifest = {
        "source": VOTES_SOURCE,
        "contract": VOTES_CONTRACT,
        "masks_manifest": str(masks_manifest_path),
        "gaussian_count": GAUSSIAN_COUNT,
        "camera_count": len(frames),
        "frames": frames,
    }
    return write_json(tmp_path / "s2" / "sam3_vote_manifest.json", manifest)


def write_tiny_ply(tmp_path, count):
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {count}",
            "property float x",
            "property float y",
            "property float z",
            "end_header",
            "",
        ]
    ).encode("ascii")
    body = b"".join(
        struct.pack("<3f", float(i), 0.0, 0.0) for i in range(count)
    )
    path = tmp_path / "scene.ply"
    path.write_bytes(header + body)
    return path


def run_mini_pipeline(tmp_path):
    s1_dir = run_s1(tmp_path)
    masks_manifest = s1_dir / "sam3_masks_manifest.json"
    votes_manifest = synth_votes(tmp_path, masks_manifest)

    registry_path = tmp_path / "instance_registry.json"
    associate_instances.main(
        [
            "--masks-manifest", str(masks_manifest),
            "--votes-manifest", str(votes_manifest),
            "--output", str(registry_path),
            "--threshold", "0.3",
        ]
    )

    membership_dir = tmp_path / "s4"
    instance_membership.main(
        [
            "--masks-manifest", str(masks_manifest),
            "--votes-manifest", str(votes_manifest),
            "--registry", str(registry_path),
            "--output-dir", str(membership_dir),
            "--min-weight", "0.5",
        ]
    )

    graph_dir = tmp_path / "s5"
    instance_hierarchy.main(
        [
            "--membership", str(membership_dir / "membership.npz"),
            "--registry", str(registry_path),
            "--source-ply", str(write_tiny_ply(tmp_path, GAUSSIAN_COUNT)),
            "--vocabulary", str(tmp_path / "vocab.json"),
            "--output-dir", str(graph_dir),
        ]
    )
    return registry_path, membership_dir, graph_dir


def test_stage_reruns_require_overwrite(tmp_path):
    registry_path, membership_dir, graph_dir = run_mini_pipeline(tmp_path)
    masks_manifest = tmp_path / "s1" / "sam3_masks_manifest.json"
    votes_manifest = tmp_path / "s2" / "sam3_vote_manifest.json"

    s4 = [
        "--masks-manifest", str(masks_manifest),
        "--votes-manifest", str(votes_manifest),
        "--registry", str(registry_path),
        "--output-dir", str(membership_dir),
        "--min-weight", "0.5",
    ]
    with pytest.raises(FileExistsError):
        instance_membership.main(s4)
    instance_membership.main([*s4, "--overwrite"])

    s5 = [
        "--membership", str(membership_dir / "membership.npz"),
        "--registry", str(registry_path),
        "--source-ply", str(tmp_path / "scene.ply"),
        "--vocabulary", str(tmp_path / "vocab.json"),
        "--output-dir", str(graph_dir),
    ]
    with pytest.raises(FileExistsError):
        instance_hierarchy.main(s5)
    instance_hierarchy.main([*s5, "--overwrite"])

    qa = [
        "--masks-manifest", str(masks_manifest),
        "--masks-dir", str(tmp_path / "s1" / "masks"),
        "--rgb-dir", str(tmp_path / "rgb"),
        "--output-dir", str(tmp_path / "qa"),
    ]
    render_instance_overlays.main(qa)
    with pytest.raises(FileExistsError):
        render_instance_overlays.main(qa)
    render_instance_overlays.main([*qa, "--overwrite"])


def test_provenance_hashes_recorded(tmp_path):
    from scripts.task1.sam3.provenance import sha256_file

    registry_path, membership_dir, graph_dir = run_mini_pipeline(tmp_path)
    masks_manifest = tmp_path / "s1" / "sam3_masks_manifest.json"
    votes_manifest = tmp_path / "s2" / "sam3_vote_manifest.json"

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["masks_manifest_sha256"] == sha256_file(masks_manifest)
    assert registry["votes_manifest_sha256"] == sha256_file(votes_manifest)

    summary = json.loads(
        (membership_dir / "membership_summary.json").read_text(encoding="utf-8")
    )
    assert summary["masks_manifest_sha256"] == sha256_file(masks_manifest)
    assert summary["votes_manifest_sha256"] == sha256_file(votes_manifest)
    assert summary["registry_sha256"] == sha256_file(registry_path)

    hierarchy = json.loads(
        (graph_dir / "hierarchy.json").read_text(encoding="utf-8")
    )
    assert hierarchy["membership_sha256"] == sha256_file(
        membership_dir / "membership.npz"
    )
    assert hierarchy["registry_sha256"] == sha256_file(registry_path)


def test_membership_rejects_tampered_upstream(tmp_path):
    registry_path, membership_dir, _ = run_mini_pipeline(tmp_path)
    masks_manifest = tmp_path / "s1" / "sam3_masks_manifest.json"
    votes_manifest = tmp_path / "s2" / "sam3_vote_manifest.json"
    # regenerating an upstream artifact after association must be detected
    votes_manifest.write_text(
        votes_manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="changed"):
        instance_membership.main(
            [
                "--masks-manifest", str(masks_manifest),
                "--votes-manifest", str(votes_manifest),
                "--registry", str(registry_path),
                "--output-dir", str(membership_dir),
                "--overwrite",
            ]
        )


def test_mini_pipeline_registry(tmp_path):
    registry_path, _, _ = run_mini_pipeline(tmp_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["instance_count"] == 3
    by_concept = {}
    for instance in registry["instances"]:
        by_concept.setdefault(instance["concept"], []).append(instance)
    assert len(by_concept["window"]) == 2
    assert len(by_concept["building"]) == 1
    assert all(
        instance["supporting_camera_count"] == 2
        for instance in registry["instances"]
    )


def test_mini_pipeline_membership(tmp_path):
    _, membership_dir, _ = run_mini_pipeline(tmp_path)
    membership = instance_membership.load_membership(
        membership_dir / "membership.npz"
    )
    # gaussian 0 belongs to a window instance and the building concurrently
    row = slice(membership.indptr[0], membership.indptr[1])
    assert membership.instance_ids[row].size == 2
    assert (membership.status[row] == instance_membership.STATUS_ACCEPTED).all()
    summary = json.loads(
        (membership_dir / "membership_summary.json").read_text(encoding="utf-8")
    )
    assert summary["gaussian_count"] == GAUSSIAN_COUNT
    assert summary["status_counts"]["accepted"] > 0


def test_mini_pipeline_scene_graph_and_flat_labels(tmp_path):
    _, _, graph_dir = run_mini_pipeline(tmp_path)
    graph = json.loads(
        (graph_dir / "scene_graph.json").read_text(encoding="utf-8")
    )
    concepts = sorted(node["concept"] for node in graph["nodes"])
    assert concepts == ["building", "window", "window"]
    assert graph["edge_count"] == 2
    assert all(edge["type"] == "part_of" for edge in graph["edges"])
    assert graph["expected_part_of_found"] == [
        {"child": "window", "parent": "building", "found": True}
    ]
    flat = np.load(graph_dir / "gaussian_instances.npy")
    assert flat.shape == (GAUSSIAN_COUNT,)
    window_ids = {
        node["instance_id"] for node in graph["nodes"]
        if node["concept"] == "window"
    }
    building_id = next(
        node["instance_id"] for node in graph["nodes"]
        if node["concept"] == "building"
    )
    # multi-label gaussians resolve to the most specific accepted instance
    assert flat[0] in window_ids
    assert flat[9] in window_ids
    assert flat[6] == building_id
    assert flat[19] == building_id
