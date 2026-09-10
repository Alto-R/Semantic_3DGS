"""S4: per-concept equal-camera consensus into a sparse membership matrix.

Each global instance is voted on independently: a camera that observes a
Gaussian supports the instance when the Gaussian's within-concept winner in
that view maps to it. Statuses mirror the audited hard-vote policy, applied
per (Gaussian, instance) pair instead of per Gaussian.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MEMBERSHIP_SOURCE = "sam3_instance_membership"
MEMBERSHIP_CONTRACT = "per_instance_equal_camera_strict_majority_v1"

STATUS_ACCEPTED = 1
STATUS_SINGLE_CAMERA = 2
STATUS_WEAK_MAJORITY = 3

STATUS_NAMES = {
    STATUS_ACCEPTED: "accepted",
    STATUS_SINGLE_CAMERA: "single_camera",
    STATUS_WEAK_MAJORITY: "weak_majority",
}

_ID_SPACE = 65536  # uint16 instance id space, id 0 reserved


@dataclass(frozen=True)
class MembershipCSR:
    """Sparse (gaussians x instances) membership with vote statistics."""

    indptr: np.ndarray
    instance_ids: np.ndarray
    support_counts: np.ndarray
    observe_counts: np.ndarray
    scores: np.ndarray
    status: np.ndarray


def view_concept_winners(
    indices: np.ndarray,
    mask_ids: np.ndarray,
    weights: np.ndarray,
    min_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-Gaussian unique dominant mask within one concept pass.

    A Gaussian yields a winner only when its highest mask membership reaches
    ``min_weight`` and is strictly unique; exact ties abstain, mirroring the
    audited one-unique-max-per-camera policy.
    """

    gaussians = np.asarray(indices, dtype=np.int64)
    masks = np.asarray(mask_ids, dtype=np.uint16)
    mass = np.asarray(weights, dtype=np.float32)
    if not (gaussians.shape == masks.shape == mass.shape):
        raise ValueError("indices, mask_ids, and weights must align")
    if gaussians.size == 0:
        return np.zeros(0, np.uint32), np.zeros(0, np.uint16)

    order = np.lexsort((masks.astype(np.int64), -mass.astype(np.float64), gaussians))
    sorted_g = gaussians[order]
    sorted_m = masks[order]
    sorted_w = mass[order]

    first = np.ones(sorted_g.size, dtype=bool)
    first[1:] = sorted_g[1:] != sorted_g[:-1]
    first_pos = np.flatnonzero(first)
    winner_g = sorted_g[first_pos]
    winner_m = sorted_m[first_pos]
    winner_w = sorted_w[first_pos]

    # A winner ties when the row right after it holds the same Gaussian at the
    # same weight. Clamping keeps the final winner's lookup in range; the
    # has_runner_up mask discards that borrowed comparison.
    next_pos = first_pos + 1
    has_runner_up = next_pos < sorted_g.size
    runner_up = np.minimum(next_pos, sorted_g.size - 1)
    tie = (
        has_runner_up
        & (sorted_g[runner_up] == winner_g)
        & (sorted_w[runner_up] == winner_w)
    )

    keep = (winner_w >= np.float32(min_weight)) & ~tie
    return winner_g[keep].astype(np.uint32), winner_m[keep].astype(np.uint16)


def accumulate_membership(
    events: list[tuple[np.ndarray, np.ndarray]],
    observe_counts: np.ndarray,
    gaussian_count: int,
) -> MembershipCSR:
    """Combine per-view (gaussian, global instance) winner events.

    ``observe_counts`` holds, per Gaussian, the number of cameras with any
    rendered visibility of it. A vote on an unobserved Gaussian is a driver
    bug and raises.
    """

    observed = np.asarray(observe_counts, dtype=np.uint16)
    if observed.shape != (gaussian_count,):
        raise ValueError("observe_counts must have one entry per Gaussian")

    keys_parts: list[np.ndarray] = []
    for gaussians, instances in events:
        g = np.asarray(gaussians, dtype=np.int64)
        i = np.asarray(instances, dtype=np.int64)
        if g.shape != i.shape:
            raise ValueError("event gaussians and instances must align")
        if g.size and (g.min() < 0 or g.max() >= gaussian_count):
            raise ValueError("event references a Gaussian out of range")
        if i.size and (i.min() < 1 or i.max() >= _ID_SPACE):
            raise ValueError("instance ids must be 1-based uint16 values")
        keys_parts.append(g * _ID_SPACE + i)

    if keys_parts:
        keys = np.concatenate(keys_parts)
    else:
        keys = np.zeros(0, dtype=np.int64)
    unique_keys, support = np.unique(keys, return_counts=True)
    entry_gaussians = unique_keys // _ID_SPACE
    entry_instances = (unique_keys % _ID_SPACE).astype(np.uint16)
    support = support.astype(np.uint16)

    entry_observed = observed[entry_gaussians]
    if np.any(entry_observed == 0):
        raise RuntimeError("a camera voted on a Gaussian it never observed")
    if np.any(support.astype(np.int64) > entry_observed.astype(np.int64)):
        raise RuntimeError(
            "an instance gathered more votes than observing cameras"
        )
    scores = (support / entry_observed).astype(np.float32)

    status = np.full(support.shape, STATUS_WEAK_MAJORITY, dtype=np.uint8)
    status[support.astype(np.int64) * 2 > entry_observed] = STATUS_ACCEPTED
    status[support == 1] = STATUS_SINGLE_CAMERA

    counts_per_gaussian = np.bincount(entry_gaussians, minlength=gaussian_count)
    indptr = np.zeros(gaussian_count + 1, dtype=np.int64)
    np.cumsum(counts_per_gaussian, out=indptr[1:])

    return MembershipCSR(
        indptr=indptr,
        instance_ids=entry_instances,
        support_counts=support,
        observe_counts=entry_observed.astype(np.uint16),
        scores=scores,
        status=status,
    )


def save_membership(path: Path, membership: MembershipCSR) -> None:
    np.savez_compressed(
        Path(path),
        indptr=membership.indptr,
        instance_ids=membership.instance_ids,
        support_counts=membership.support_counts,
        observe_counts=membership.observe_counts,
        scores=membership.scores,
        status=membership.status,
    )


def load_membership(path: Path) -> MembershipCSR:
    with np.load(Path(path), allow_pickle=False) as data:
        return MembershipCSR(
            indptr=data["indptr"],
            instance_ids=data["instance_ids"],
            support_counts=data["support_counts"],
            observe_counts=data["observe_counts"],
            scores=data["scores"],
            status=data["status"],
        )


def main(argv: list[str] | None = None) -> None:
    from scripts.task1.sam3.associate_instances import validate_instance_registry
    from scripts.task1.sam3.lift_mask_view_votes import validate_votes_manifest
    from scripts.task1.sam3.segment_views_core import validate_masks_manifest

    parser = argparse.ArgumentParser()
    parser.add_argument("--masks-manifest", required=True, type=Path)
    parser.add_argument("--votes-manifest", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-weight", default=0.5, type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(args.output_dir)
    masks_manifest = json.loads(args.masks_manifest.read_text(encoding="utf-8"))
    validate_masks_manifest(masks_manifest)
    votes_manifest = json.loads(args.votes_manifest.read_text(encoding="utf-8"))
    validate_votes_manifest(votes_manifest)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    validate_instance_registry(registry)

    from scripts.task1.sam3.provenance import sha256_file

    masks_sha256 = sha256_file(args.masks_manifest)
    votes_sha256 = sha256_file(args.votes_manifest)
    for recorded, actual, name in (
        (registry.get("masks_manifest_sha256"), masks_sha256, "masks manifest"),
        (registry.get("votes_manifest_sha256"), votes_sha256, "votes manifest"),
    ):
        if recorded != actual:
            raise RuntimeError(
                f"the {name} changed since instance association"
            )

    gaussian_count = int(votes_manifest["gaussian_count"])
    global_id: dict[tuple[str, int], int] = {}
    for instance in registry["instances"]:
        for member in instance["members"]:
            key = (str(member["view"]), int(member["mask_index"]))
            global_id[key] = int(instance["instance_id"])

    from scripts.task1.sam3.segment_views_core import stem_index

    concept_of: dict[str, dict[int, str]] = {}
    for stem, frame in stem_index(
        masks_manifest["frames"], "masks manifest"
    ).items():
        concept_of[stem] = {
            int(mask["mask_index"]): str(mask["concept"]) for mask in frame["masks"]
        }

    observe_totals = np.zeros(gaussian_count, dtype=np.int64)
    events: list[tuple[np.ndarray, np.ndarray]] = []
    votes_dir = args.votes_manifest.parent
    for frame in votes_manifest["frames"]:
        stem = Path(str(frame["file"])).stem
        with np.load(votes_dir / str(frame["vote_file"]), allow_pickle=False) as data:
            indices = data["indices"]
            mask_ids = data["mask_ids"]
            weights = data["weights"]
            observed = data["observed"]
        observe_totals[observed.astype(np.int64)] += 1

        frame_concepts = concept_of.get(stem)
        if frame_concepts is None:
            raise ValueError(f"masks manifest does not know view {stem}")
        masks_by_concept: dict[str, list[int]] = {}
        for mask_index, concept in frame_concepts.items():
            masks_by_concept.setdefault(concept, []).append(mask_index)
        for concept_masks in masks_by_concept.values():
            rows = np.isin(mask_ids, np.array(concept_masks, dtype=np.uint16))
            winners_g, winners_m = view_concept_winners(
                indices[rows], mask_ids[rows], weights[rows], args.min_weight
            )
            if winners_g.size == 0:
                continue
            instance_ids = np.empty(winners_m.shape, dtype=np.uint16)
            for position, mask_index in enumerate(winners_m.tolist()):
                key = (stem, int(mask_index))
                if key not in global_id:
                    raise RuntimeError(
                        f"registry does not map view {stem} mask {mask_index}"
                    )
                instance_ids[position] = global_id[key]
            events.append((winners_g, instance_ids))

    if int(observe_totals.max(initial=0)) > 65535:
        raise ValueError("camera count exceeds the uint16 observation space")
    membership = accumulate_membership(
        events, observe_totals.astype(np.uint16), gaussian_count
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_membership(args.output_dir / "membership.npz", membership)

    status_counts = {
        name: int(np.count_nonzero(membership.status == code))
        for code, name in sorted(STATUS_NAMES.items())
    }
    summary: dict[str, Any] = {
        "source": MEMBERSHIP_SOURCE,
        "contract": MEMBERSHIP_CONTRACT,
        "masks_manifest": str(args.masks_manifest),
        "votes_manifest": str(args.votes_manifest),
        "registry": str(args.registry),
        "masks_manifest_sha256": masks_sha256,
        "votes_manifest_sha256": votes_sha256,
        "registry_sha256": sha256_file(args.registry),
        "gaussian_count": gaussian_count,
        "camera_count": int(votes_manifest["camera_count"]),
        "min_weight": args.min_weight,
        "entry_count": int(membership.instance_ids.shape[0]),
        "status_counts": status_counts,
        "camera_vote_policy": "one_unique_dominant_mask_per_camera_else_abstain",
        "consensus_policy": "per_instance_at_least_two_cameras_strict_majority",
    }
    (args.output_dir / "membership_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
