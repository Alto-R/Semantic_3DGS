#!/usr/bin/env python3
"""Report-only leave-one-camera-out DINOv3 2D-to-3D fidelity audit.

The audit consumes the automatically selected dense DINOv3 cache and its
FlashSplat class distributions.  Each camera/Gaussian distribution is reduced
to one unique hard class winner (or abstention), so no camera can dominate the
3D identity consensus through vote scale.  Every camera is then excluded in
turn, a strict majority is computed from the other cameras, and that temporary
3D prediction is rendered back into the held-out camera.

No accepted Gaussian semantic labels, label map, or semantic PLY are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from scripts.task1.common.flashsplat_cameras import (
    background_tensor,
    default_pipeline,
    load_cameras,
    load_flashsplat,
    load_gaussians,
    make_camera,
    point_cloud_path,
    render_flashsplat,
)
from scripts.task1.common.semantic_palette import rgb8_for_class
from scripts.task1.dinov2.dinov2_ontology import Ontology, load_ontology
from scripts.task1.dinov3.selected_view_cache import CACHE_CONTRACT


SOURCE = "dinov3_cached_2d_3d_round_trip_fidelity_audit"
CONTRACT = "report_only_leave_one_camera_out_round_trip_fidelity_v1"
VOTE_SOURCE = "dinov3_dense_pixel_flashsplat_votes"
VOTE_CONTRACT = "complete_dense_argmax_pixels_normalized_per_camera_v1"

STATUS_ACCEPTED = 0
STATUS_UNOBSERVED = 1
STATUS_SINGLE_CAMERA = 2
STATUS_EXACT_TIE = 3
STATUS_NO_STRICT_MAJORITY = 4
STATUS_NAMES = {
    STATUS_ACCEPTED: "accepted_strict_majority",
    STATUS_UNOBSERVED: "unobserved",
    STATUS_SINGLE_CAMERA: "single_camera",
    STATUS_EXACT_TIE: "exact_tie",
    STATUS_NO_STRICT_MAJORITY: "no_strict_majority",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": 0}
    names = ("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max")
    quantiles = np.quantile(finite, (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0))
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        **{name: float(value) for name, value in zip(names, quantiles)},
    }


def boundary_mask(class_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(class_ids)
    if values.ndim != 2:
        raise ValueError("class map must be two-dimensional")
    boundary = np.zeros(values.shape, dtype=bool)
    horizontal = values[:, 1:] != values[:, :-1]
    vertical = values[1:, :] != values[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def collapse_camera_distribution(
    indices: np.ndarray,
    class_ids: np.ndarray,
    weights: np.ndarray,
    *,
    gaussian_count: int,
    class_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return one unique hard identity per camera/Gaussian, else zero."""

    idx = np.asarray(indices)
    classes = np.asarray(class_ids)
    values = np.asarray(weights, dtype=np.float32)
    if not (idx.shape == classes.shape == values.shape) or idx.ndim != 1:
        raise ValueError("camera vote arrays must be aligned one-dimensional arrays")
    if idx.size and (np.any(idx < 0) or int(idx.max()) >= gaussian_count):
        raise ValueError("camera vote references a Gaussian outside the model")
    if idx.size and (np.any(classes <= 0) or int(classes.max()) > class_count):
        raise ValueError("camera vote references an invalid project class")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("camera vote weights must be finite and positive")

    totals = np.zeros((gaussian_count,), dtype=np.float32)
    np.add.at(totals, idx.astype(np.int64, copy=False), values)
    observed = totals > 0.0
    if observed.any() and not np.allclose(
        totals[observed], 1.0, rtol=1e-5, atol=1e-5
    ):
        raise ValueError("each camera/Gaussian distribution must sum to one")

    winner = np.zeros((gaussian_count,), dtype=np.uint16)
    best = np.zeros((gaussian_count,), dtype=np.float32)
    tied = np.zeros((gaussian_count,), dtype=bool)
    for project_id in np.unique(classes):
        selected = classes == project_id
        positions = idx[selected].astype(np.int64, copy=False)
        scores = values[selected]
        if np.unique(positions).size != positions.size:
            raise ValueError("camera has duplicate class mass for one Gaussian")
        current = best[positions]
        greater = scores > current
        equal_positive = (scores == current) & (scores > 0.0)
        if np.any(greater):
            changed = positions[greater]
            best[changed] = scores[greater]
            winner[changed] = np.uint16(project_id)
            tied[changed] = False
        if np.any(equal_positive):
            tied[positions[equal_positive]] = True
    winner[tied] = 0
    winner[~observed] = 0
    accepted = winner > 0
    return winner, {
        "visible_gaussian_count": int(np.count_nonzero(observed)),
        "unique_camera_winner_count": int(np.count_nonzero(accepted)),
        "camera_abstain_count": int(np.count_nonzero(observed & ~accepted)),
        "exact_tie_count": int(np.count_nonzero(tied)),
        "winning_mass": quantile_summary(best[accepted]),
    }


def consensus_statistics(
    camera_counts: np.ndarray,
    *,
    chunk_size: int = 100_000,
) -> dict[str, np.ndarray]:
    """Summarize hard camera identities without storing a Gaussian label output."""

    counts = np.asarray(camera_counts)
    if counts.ndim != 2 or counts.shape[0] < 2:
        raise ValueError("camera_counts must have shape classes-plus-zero x gaussians")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    gaussian_count = counts.shape[1]
    total = counts[1:].sum(axis=0, dtype=np.uint16)
    winner = np.zeros((gaussian_count,), dtype=np.uint16)
    maximum = np.zeros((gaussian_count,), dtype=np.uint16)
    second = np.zeros((gaussian_count,), dtype=np.uint16)
    tie_count = np.zeros((gaussian_count,), dtype=np.uint16)
    tie_first = np.zeros((gaussian_count,), dtype=np.uint16)
    tie_second = np.zeros((gaussian_count,), dtype=np.uint16)

    for start in range(0, gaussian_count, chunk_size):
        end = min(start + chunk_size, gaussian_count)
        local = np.asarray(counts[1:, start:end], dtype=np.uint16)
        local_max = local.max(axis=0)
        local_winner = local.argmax(axis=0).astype(np.uint16) + np.uint16(1)
        local_ties = (local == local_max[None, :]) & (local_max[None, :] > 0)
        local_tie_count = local_ties.sum(axis=0, dtype=np.uint16)
        local_second = np.partition(local, -2, axis=0)[-2]
        first = np.zeros((end - start,), dtype=np.uint16)
        second_tied = np.zeros((end - start,), dtype=np.uint16)
        for offset in range(local.shape[0]):
            matched = local_ties[offset]
            put_first = matched & (first == 0)
            first[put_first] = np.uint16(offset + 1)
            put_second = matched & (first != 0) & (first != offset + 1) & (second_tied == 0)
            second_tied[put_second] = np.uint16(offset + 1)
        winner[start:end] = np.where(local_max > 0, local_winner, 0)
        maximum[start:end] = local_max
        second[start:end] = local_second
        tie_count[start:end] = local_tie_count
        tie_first[start:end] = first
        tie_second[start:end] = second_tied

    status = np.full((gaussian_count,), STATUS_NO_STRICT_MAJORITY, dtype=np.uint8)
    status[total == 0] = STATUS_UNOBSERVED
    status[total == 1] = STATUS_SINGLE_CAMERA
    enough = total >= 2
    status[enough & (tie_count > 1)] = STATUS_EXACT_TIE
    accepted = enough & (tie_count == 1) & (maximum.astype(np.uint16) * 2 > total)
    status[accepted] = STATUS_ACCEPTED
    share = np.divide(
        maximum,
        total,
        out=np.zeros((gaussian_count,), dtype=np.float32),
        where=total > 0,
    )
    return {
        "total": total,
        "winner": winner,
        "maximum": maximum,
        "second": second,
        "tie_count": tie_count,
        "tie_first": tie_first,
        "tie_second": tie_second,
        "status": status,
        "winner_share": share,
    }


def leave_one_out_consensus(
    statistics: dict[str, np.ndarray],
    heldout_winner: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return strict-majority consensus after removing one camera's winner."""

    heldout = np.asarray(heldout_winner, dtype=np.uint16)
    total = statistics["total"].astype(np.int32) - (heldout > 0).astype(np.int32)
    maximum = statistics["maximum"].astype(np.int32)
    second = statistics["second"].astype(np.int32)
    global_winner = statistics["winner"]
    ties = statistics["tie_count"]
    candidate = np.zeros(heldout.shape, dtype=np.uint16)
    candidate_count = np.zeros(heldout.shape, dtype=np.int32)

    unique = ties == 1
    candidate[unique] = global_winner[unique]
    candidate_count[unique] = maximum[unique]
    removed_unique_winner = unique & (heldout == global_winner) & (heldout > 0)
    candidate_count[removed_unique_winner] -= 1
    became_tied = removed_unique_winner & (candidate_count == second)
    candidate[became_tied] = 0

    exactly_two = ties == 2
    first = statistics["tie_first"]
    second_tied = statistics["tie_second"]
    removed_first = exactly_two & (heldout == first) & (heldout > 0)
    removed_second = exactly_two & (heldout == second_tied) & (heldout > 0)
    candidate[removed_first] = second_tied[removed_first]
    candidate[removed_second] = first[removed_second]
    candidate_count[removed_first | removed_second] = maximum[removed_first | removed_second]

    accepted = (total >= 2) & (candidate > 0) & (candidate_count * 2 > total)
    labels = np.where(accepted, candidate, 0).astype(np.uint16)
    return labels, total.astype(np.uint16)


def render_binary_project_ids(
    labels: np.ndarray,
    camera: Any,
    gaussians: Any,
    modules: dict[str, Any],
    pipeline: Any,
    background: Any,
    *,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render eight binary class-ID features and decode their pixel majority."""

    import torch

    labels_cuda = torch.from_numpy(np.asarray(labels, dtype=np.int32)).to(
        device="cuda", dtype=torch.int32
    )
    labeled = labels_cuda > 0
    occupancy_color = labeled.to(dtype=torch.float32)[:, None].repeat(1, 3)
    occupancy_pkg = render_flashsplat(
        camera, gaussians, modules, pipeline, background, override_color=occupancy_color
    )
    occupancy = occupancy_pkg["render"][0].detach().float().cpu().numpy()
    decoded = np.zeros(occupancy.shape, dtype=np.uint16)
    minimum_margin = np.ones(occupancy.shape, dtype=np.float32)

    for first_bit in (0, 3, 6):
        channels = []
        bit_numbers = []
        for channel in range(3):
            bit = first_bit + channel
            bit_numbers.append(bit)
            if bit < 8:
                channels.append(((labels_cuda >> bit) & 1).to(dtype=torch.float32))
            else:
                channels.append(torch.zeros_like(labels_cuda, dtype=torch.float32))
        colors = torch.stack(channels, dim=1)
        render_pkg = render_flashsplat(
            camera, gaussians, modules, pipeline, background, override_color=colors
        )
        rendered = render_pkg["render"].detach().float().cpu().numpy()
        for channel, bit in enumerate(bit_numbers):
            if bit >= 8:
                continue
            values = rendered[channel]
            one = values > (occupancy * np.float32(0.5))
            decoded |= one.astype(np.uint16) << np.uint16(bit)
            normalized_margin = np.divide(
                np.abs(values - occupancy * np.float32(0.5)) * np.float32(2.0),
                occupancy,
                out=np.zeros_like(occupancy),
                where=occupancy > 0.0,
            )
            minimum_margin = np.minimum(minimum_margin, normalized_margin)
        del render_pkg, rendered, colors

    valid = (occupancy > 0.0) & (decoded > 0) & (decoded <= class_count)
    decoded[~valid] = 0
    minimum_margin[occupancy <= 0.0] = 0.0
    del labels_cuda, occupancy_color, occupancy_pkg
    torch.cuda.empty_cache()
    return decoded, valid, minimum_margin


def palette_rgb(project_ids: np.ndarray, ontology: Ontology) -> np.ndarray:
    output = np.zeros((*project_ids.shape, 3), dtype=np.uint8)
    for project_id in np.unique(project_ids):
        if int(project_id) == 0:
            continue
        item = ontology.by_project_id[int(project_id)]
        output[project_ids == project_id] = np.asarray(
            rgb8_for_class(item.project_class), dtype=np.uint8
        )
    return output


def save_visuals(
    base_rgb: np.ndarray,
    predicted: np.ndarray,
    valid: np.ndarray,
    source: np.ndarray,
    ontology: Ontology,
    overlay_path: Path,
    disagreement_path: Path,
) -> None:
    from PIL import Image

    colored = palette_rgb(predicted, ontology)
    overlay = base_rgb.copy()
    overlay[valid] = (
        0.45 * overlay[valid].astype(np.float32)
        + 0.55 * colored[valid].astype(np.float32)
    ).astype(np.uint8)
    disagreement = np.zeros_like(base_rgb)
    disagreement[~valid] = (48, 48, 48)
    disagreement[valid & (predicted == source)] = (30, 180, 70)
    disagreement[valid & (predicted != source)] = (230, 45, 45)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    disagreement_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, mode="RGB").save(overlay_path)
    Image.fromarray(disagreement, mode="RGB").save(disagreement_path)


def validate_provenance(
    cache_report: dict[str, Any],
    view_manifest: dict[str, Any],
    dino_manifest: dict[str, Any],
    vote_manifest: dict[str, Any],
) -> list[int]:
    if cache_report.get("contract") != CACHE_CONTRACT:
        raise ValueError("source is not a validated selected-view DINOv3 cache")
    for field, expected in (
        ("manual_camera_selection_used", False),
        ("semantic_vote_lifting_run", False),
        ("semantic_fusion_run", False),
        ("semantic_labels_written", False),
        ("label_map_written", False),
        ("semantic_ply_written", False),
    ):
        if cache_report.get(field) is not expected:
            raise ValueError(f"selected cache violates {field}={expected}")
    if vote_manifest.get("source") != VOTE_SOURCE or vote_manifest.get("contract") != VOTE_CONTRACT:
        raise ValueError("vote manifest is not complete dense DINOv3 evidence")
    for field, expected in (
        ("query_region_filtering_used", False),
        ("confidence_threshold_used", False),
        ("one_normalized_vote_per_camera", True),
        ("v5_used", False),
        ("dinov2_used", False),
    ):
        if vote_manifest.get(field) is not expected:
            raise ValueError(f"vote manifest violates {field}={expected}")

    expected = [int(value) for value in cache_report.get("camera_indices", [])]
    expected_ids = [int(value) for value in cache_report.get("camera_ids", [])]
    if not expected or len(expected) != len(expected_ids):
        raise ValueError("selected cache has invalid camera provenance")
    for label, manifest in (
        ("view", view_manifest),
        ("DINOv3", dino_manifest),
        ("vote", vote_manifest),
    ):
        frames = manifest.get("frames")
        if not isinstance(frames, list) or len(frames) != len(expected):
            raise ValueError(f"{label} manifest does not contain the exact selected cameras")
        if [int(frame.get("camera_index", -1)) for frame in frames] != expected:
            raise ValueError(f"{label} camera order differs from selected cache")
        if [int(frame.get("camera_id", -1)) for frame in frames] != expected_ids:
            raise ValueError(f"{label} camera IDs differ from selected cache")
    expected_files = [str(frame.get("file", "")) for frame in view_manifest["frames"]]
    if any(not value for value in expected_files) or len(set(expected_files)) != len(expected_files):
        raise ValueError("view manifest filenames must be nonempty and unique")
    for label, manifest in (("DINOv3", dino_manifest), ("vote", vote_manifest)):
        if [str(frame.get("file", "")) for frame in manifest["frames"]] != expected_files:
            raise ValueError(f"{label} filenames differ from rendered views")
    if int(view_manifest.get("max_width", -1)) != int(cache_report["render"]["max_width"]):
        raise ValueError("source render width differs from selected cache report")
    if int(vote_manifest.get("render_max_width", -1)) != int(cache_report["render"]["max_width"]):
        raise ValueError("vote render width differs from selected cache report")
    return expected


def _status_counts(status: np.ndarray) -> dict[str, int]:
    return {
        STATUS_NAMES[code]: int(np.count_nonzero(status == code))
        for code in sorted(STATUS_NAMES)
    }


def main() -> None:
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source-view-dir", required=True, type=Path)
    parser.add_argument("--selected-cache-report", required=True, type=Path)
    parser.add_argument("--vote-manifest", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flashsplat-root", required=True, type=Path)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--chunk-size", default=100_000, type=int)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    required_paths = (
        args.selected_cache_report,
        args.vote_manifest,
        args.ontology,
        args.source_view_dir / "view_manifest.json",
        args.source_view_dir / "dinov3_manifest.json",
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes_before = {str(path): sha256_file(path) for path in required_paths}
    cache_report = json.loads(args.selected_cache_report.read_text(encoding="utf-8"))
    view_manifest = json.loads((args.source_view_dir / "view_manifest.json").read_text(encoding="utf-8"))
    dino_manifest = json.loads((args.source_view_dir / "dinov3_manifest.json").read_text(encoding="utf-8"))
    vote_manifest = json.loads(args.vote_manifest.read_text(encoding="utf-8"))
    camera_indices = validate_provenance(cache_report, view_manifest, dino_manifest, vote_manifest)
    source_root = args.source_view_dir.parents[1]
    if Path(str(cache_report.get("output_dir", ""))).resolve() != source_root.resolve():
        raise ValueError("selected-cache report belongs to a different output root")
    if Path(str(vote_manifest.get("model_path", ""))).resolve() != args.model_path.resolve():
        raise ValueError("vote manifest belongs to a different Gaussian model")
    if Path(str(vote_manifest.get("segmentation_manifest", ""))).resolve() != (
        args.source_view_dir / "dinov3_manifest.json"
    ).resolve():
        raise ValueError("vote manifest belongs to a different DINOv3 cache")
    expected_manifest_hashes = cache_report.get("output_manifest_sha256", {})
    if not isinstance(expected_manifest_hashes, dict):
        raise ValueError("selected-cache report is missing manifest hashes")
    if expected_manifest_hashes.get("view_manifest") != hashes_before[
        str(args.source_view_dir / "view_manifest.json")
    ]:
        raise ValueError("view manifest hash differs from selected-cache provenance")
    if expected_manifest_hashes.get("dinov3_manifest") != hashes_before[
        str(args.source_view_dir / "dinov3_manifest.json")
    ]:
        raise ValueError("DINOv3 manifest hash differs from selected-cache provenance")
    ontology = load_ontology(args.ontology)
    class_count = ontology.class_count
    gaussian_count = int(vote_manifest.get("gaussian_count", -1))
    if gaussian_count < 1:
        raise ValueError("vote manifest has an invalid Gaussian count")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    mask_dir = args.output_dir / "heldout_masks"
    overlay_dir = args.output_dir / "heldout_overlays"
    disagreement_dir = args.output_dir / "heldout_disagreement"
    for directory in (mask_dir, overlay_dir, disagreement_dir):
        directory.mkdir()

    camera_summaries: list[dict[str, Any]] = []
    camera_counts: np.memmap | None = None
    with tempfile.TemporaryDirectory(prefix="round_trip_counts_", dir=args.output_dir) as temporary:
        camera_counts = np.memmap(
            Path(temporary) / "camera_counts.uint8",
            mode="w+",
            dtype=np.uint8,
            shape=(class_count + 1, gaussian_count),
        )
        camera_counts[:] = 0
        for frame in vote_manifest["frames"]:
            vote_path = args.vote_manifest.parent / str(frame["vote_file"])
            if not vote_path.is_file():
                raise FileNotFoundError(vote_path)
            with np.load(vote_path, allow_pickle=False) as data:
                winners, summary = collapse_camera_distribution(
                    data["indices"], data["class_ids"], data["weights"],
                    gaussian_count=gaussian_count, class_count=class_count,
                )
            supported = np.flatnonzero(winners).astype(np.int64)
            camera_counts[winners[supported], supported] += np.uint8(1)
            camera_summaries.append({
                "camera_index": int(frame["camera_index"]),
                "camera_id": int(frame["camera_id"]),
                "file": str(frame["file"]),
                **summary,
            })
            print(f"collapsed camera {frame['camera_index']}: {supported.size} unique winners")
        camera_counts.flush()
        statistics = consensus_statistics(camera_counts, chunk_size=args.chunk_size)

        np.savez_compressed(
            args.output_dir / "gaussian_camera_agreement_diagnostics.npz",
            semantic_camera_count=statistics["total"],
            winner_camera_count=statistics["maximum"],
            winner_share=statistics["winner_share"].astype(np.float16),
            consensus_status=statistics["status"],
        )

        cameras = load_cameras(args.model_path)
        if any(index < 0 or index >= len(cameras) for index in camera_indices):
            raise ValueError("selected cache references a camera outside cameras.json")
        expected_ids = [int(value) for value in cache_report["camera_ids"]]
        if [int(cameras[index]["id"]) for index in camera_indices] != expected_ids:
            raise ValueError("selected-cache camera IDs differ from current cameras.json")
        modules = load_flashsplat(args.flashsplat_root)
        ply_path = point_cloud_path(args.model_path, args.iteration)
        if Path(str(vote_manifest.get("ply_path", ""))).resolve() != ply_path.resolve():
            raise ValueError("vote manifest belongs to a different Gaussian PLY")
        gaussians = load_gaussians(modules, ply_path, args.sh_degree)
        if int(gaussians.get_xyz.shape[0]) != gaussian_count:
            raise ValueError("model Gaussian count differs from vote manifest")
        pipeline = default_pipeline()
        background = background_tensor(False)
        render_max_width = int(cache_report["render"]["max_width"])
        lookup = ontology.ade_to_project
        rgb_dir = args.source_view_dir / "rgb_renders"
        confusion = np.zeros((class_count + 1, class_count + 1), dtype=np.uint64)
        aggregate = {
            "pixels": 0,
            "projected": 0,
            "agreed": 0,
            "boundary_pixels": 0,
            "boundary_projected": 0,
            "boundary_agreed": 0,
            "interior_pixels": 0,
            "interior_projected": 0,
            "interior_agreed": 0,
        }

        for ordinal, (vote_frame, dino_frame) in enumerate(
            zip(vote_manifest["frames"], dino_manifest["frames"])
        ):
            vote_path = args.vote_manifest.parent / str(vote_frame["vote_file"])
            with np.load(vote_path, allow_pickle=False) as data:
                heldout, _summary = collapse_camera_distribution(
                    data["indices"], data["class_ids"], data["weights"],
                    gaussian_count=gaussian_count, class_count=class_count,
                )
            loo_labels, remaining_views = leave_one_out_consensus(statistics, heldout)
            camera_index = int(vote_frame["camera_index"])
            camera = make_camera(cameras[camera_index], modules, render_max_width)
            predicted, valid, binary_margin = render_binary_project_ids(
                loo_labels, camera, gaussians, modules, pipeline, background,
                class_count=class_count,
            )
            segment_path = args.source_view_dir / str(dino_frame["segment_file"])
            with np.load(segment_path, allow_pickle=False) as segment:
                raw = np.asarray(segment["class_id"], dtype=np.uint8)
            source_project = lookup[raw]
            if source_project.shape != predicted.shape:
                raise ValueError("held-out DINO map and 3D projection shapes differ")
            boundary = boundary_mask(source_project)
            interior = ~boundary
            agreement = valid & (predicted == source_project)
            flat = source_project[valid].astype(np.int64) * (class_count + 1) + predicted[valid].astype(np.int64)
            confusion += np.bincount(flat, minlength=(class_count + 1) ** 2).reshape(
                class_count + 1, class_count + 1
            ).astype(np.uint64)

            pixels = int(source_project.size)
            projected = int(np.count_nonzero(valid))
            agreed = int(np.count_nonzero(agreement))
            boundary_projected = int(np.count_nonzero(valid & boundary))
            interior_projected = int(np.count_nonzero(valid & interior))
            frame_summary = {
                "file": str(vote_frame["file"]),
                "camera_index": camera_index,
                "camera_id": int(vote_frame["camera_id"]),
                "source_pixel_count": pixels,
                "projected_pixel_count": projected,
                "projected_pixel_ratio": projected / pixels,
                "heldout_agreement_count": agreed,
                "heldout_agreement_of_projected": agreed / projected if projected else 0.0,
                "boundary_pixel_count": int(np.count_nonzero(boundary)),
                "boundary_projected_count": boundary_projected,
                "boundary_agreement_count": int(np.count_nonzero(agreement & boundary)),
                "interior_pixel_count": int(np.count_nonzero(interior)),
                "interior_projected_count": interior_projected,
                "interior_agreement_count": int(np.count_nonzero(agreement & interior)),
                "binary_projection_margin": quantile_summary(binary_margin[valid]),
                "remaining_camera_evidence": quantile_summary(remaining_views[loo_labels > 0]),
                **camera_summaries[ordinal],
            }
            camera_summaries[ordinal] = frame_summary
            for key, value in (
                ("pixels", pixels), ("projected", projected), ("agreed", agreed),
                ("boundary_pixels", frame_summary["boundary_pixel_count"]),
                ("boundary_projected", boundary_projected),
                ("boundary_agreed", frame_summary["boundary_agreement_count"]),
                ("interior_pixels", frame_summary["interior_pixel_count"]),
                ("interior_projected", interior_projected),
                ("interior_agreed", frame_summary["interior_agreement_count"]),
            ):
                aggregate[key] += int(value)

            stem = Path(str(vote_frame["file"])).stem
            np.savez_compressed(
                mask_dir / f"{stem}.npz",
                projected_project_id=predicted.astype(np.uint8),
                projection_valid=valid,
                source_boundary=boundary,
                agreement=agreement,
                minimum_binary_margin=binary_margin.astype(np.float16),
            )
            base_rgb = np.asarray(
                Image.open(rgb_dir / str(vote_frame["file"])).convert("RGB"),
                dtype=np.uint8,
            )
            save_visuals(
                base_rgb, predicted, valid, source_project, ontology,
                overlay_dir / str(vote_frame["file"]),
                disagreement_dir / str(vote_frame["file"]),
            )
            print(f"held out camera {camera_index}: coverage={projected / pixels:.4f} agreement={agreed / projected if projected else 0.0:.4f}")

        del camera_counts
        camera_counts = None

    status = statistics["status"]
    class_rows: list[dict[str, Any]] = []
    for item in ontology.classes:
        row = confusion[item.project_id]
        total = int(row.sum())
        correct = int(row[item.project_id])
        if total:
            class_rows.append({
                "project_id": item.project_id,
                "class": item.project_class,
                "projected_source_pixel_count": total,
                "correct_pixel_count": correct,
                "agreement": correct / total,
            })
    confusions: list[dict[str, Any]] = []
    off_diagonal = confusion.copy()
    np.fill_diagonal(off_diagonal, 0)
    for flat_index in np.argsort(off_diagonal.ravel())[::-1][:50]:
        count = int(off_diagonal.ravel()[flat_index])
        if count == 0:
            break
        source_id, predicted_id = np.unravel_index(flat_index, off_diagonal.shape)
        confusions.append({
            "source_project_id": int(source_id),
            "source_class": ontology.by_project_id[int(source_id)].project_class,
            "predicted_project_id": int(predicted_id),
            "predicted_class": ontology.by_project_id[int(predicted_id)].project_class,
            "pixel_count": count,
        })
    np.savez_compressed(
        args.output_dir / "heldout_confusion_matrix.npz",
        counts=confusion,
        project_ids=np.arange(class_count + 1, dtype=np.uint16),
    )

    hashes_after = {str(path): sha256_file(path) for path in required_paths}
    if hashes_before != hashes_after:
        raise RuntimeError("audit provenance inputs changed during execution")
    report = {
        "source": SOURCE,
        "contract": CONTRACT,
        "report_only": True,
        "model_path": str(args.model_path),
        "source_view_dir": str(args.source_view_dir),
        "selected_cache_report": str(args.selected_cache_report),
        "vote_manifest": str(args.vote_manifest),
        "ontology": str(args.ontology),
        "camera_count": len(camera_indices),
        "gaussian_count": gaussian_count,
        "camera_indices": camera_indices,
        "camera_vote_policy": "one_unique_max_class_per_camera_and_gaussian_else_abstain",
        "camera_vote_scale": "one_equal_identity_vote_per_camera",
        "consensus_policy": "at_least_two_cameras_unique_strict_majority_else_abstain",
        "heldout_policy": "exclude_the_target_camera_before_every_3d_consensus",
        "projection_policy": "eight_binary_project_id_feature_renders_decoded_at_each_pixel",
        "boundary_definition": "four_connected_change_in_heldout_dinov3_project_class",
        "gaussian_agreement": {
            "semantic_camera_count": quantile_summary(statistics["total"]),
            "winner_camera_count": quantile_summary(statistics["maximum"]),
            "winner_share": quantile_summary(statistics["winner_share"][statistics["total"] > 0]),
            "status_counts": _status_counts(status),
            "zero_camera_gaussian_count": int(np.count_nonzero(statistics["total"] == 0)),
            "single_camera_gaussian_count": int(np.count_nonzero(statistics["total"] == 1)),
            "multicamera_gaussian_count": int(np.count_nonzero(statistics["total"] >= 2)),
        },
        "heldout_pixel_metrics": {
            **aggregate,
            "projected_ratio": aggregate["projected"] / aggregate["pixels"],
            "agreement_of_projected": aggregate["agreed"] / aggregate["projected"] if aggregate["projected"] else 0.0,
            "boundary_projected_ratio": aggregate["boundary_projected"] / aggregate["boundary_pixels"] if aggregate["boundary_pixels"] else 0.0,
            "boundary_agreement_of_projected": aggregate["boundary_agreed"] / aggregate["boundary_projected"] if aggregate["boundary_projected"] else 0.0,
            "interior_projected_ratio": aggregate["interior_projected"] / aggregate["interior_pixels"] if aggregate["interior_pixels"] else 0.0,
            "interior_agreement_of_projected": aggregate["interior_agreed"] / aggregate["interior_projected"] if aggregate["interior_projected"] else 0.0,
        },
        "per_camera": camera_summaries,
        "per_source_class": class_rows,
        "largest_off_diagonal_confusions": confusions,
        "input_sha256": hashes_before,
        "dinov3_inference_rerun": False,
        "flashsplat_lifting_rerun": True,
        "manual_camera_selection_used": False,
        "manual_gaussian_selection_used": False,
        "scene_specific_rules": False,
        "class_specific_thresholds": False,
        "accepted_gaussian_labels_written": False,
        "gaussian_project_class_array_written": False,
        "label_map_written": False,
        "semantic_ply_written": False,
    }
    report_path = args.output_dir / "round_trip_fidelity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "contract": CONTRACT,
        "camera_count": len(camera_indices),
        "gaussian_agreement": report["gaussian_agreement"],
        "heldout_pixel_metrics": report["heldout_pixel_metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
