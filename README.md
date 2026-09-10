# EyeNavGS Semantic 3D Gaussian Splatting

This repository builds semantic labels for pretrained EyeNavGS/3D Gaussian
Splatting scenes. It includes complete DINOv3/dino.txt and SAM3 routes.
The DINOv3 route renders the original scene
cameras, predicts ADE20K semantics with DINOv3, lifts the evidence onto
Gaussians, adds out-of-vocabulary classes with dino.txt and SAM, and renders the
final labels back into every camera.

The current deliverable is semantic annotation (D1). Downstream navigation and
VR integration work are not implemented in this repository yet.

## Current state

- The production base uses DINOv3 ViT-7B ADE20K predictions from every real
  reconstruction camera and abstains when the multiview evidence is weak.
- Reliability-calibrated recovery fills detected ties and weak majorities. It
  does not fill Gaussians that no camera observes.
- Classes missing from ADE20K use competitive dino.txt classification, SAM
  masks, and multiview OOV fusion against the DINOv3 base.
- Retired DINOv3 3D-first, hybrid-refinement, and singleton-recovery
  experiments are preserved under `archive/`.
- Every accepted result remains versioned. No stage overwrites a base output.

See [Project status](docs/PROJECT_STATUS.md) for the verified reference result
and known limitations.

## Complete DINOv3 and dino.txt pipeline

The complete route starts from a Graphdeco reconstruction and creates fresh
semantic outputs. It does not require an accepted semantic result from an
earlier run.

```text
DINOv3 route
  render real camera views
        |
        v
  DINOv3 ViT-7B + ADE20K Mask2Former
        |
        v
  per-view FlashSplat lift
        |
        v
  equal-camera strict-majority hard vote  ---> immutable hard-vote base
        |
        +-- reliability-calibrated abstention recovery
        |       (single camera, exact tie, no strict majority)
        |
        v
  labels + label map + semantic PLY + SuperSplat PLY + summary
```

The direct commands are in the
[complete quickstart](docs/DINOV3_DINOTXT_SAM_QUICKSTART.md). The longer
[pipeline guide](docs/DINOV3_DINOTXT_SAM_END_TO_END_PIPELINE.md) explains the
inputs, intermediate artifacts, and fusion rules.

## Main entry points

| Purpose | Entry point |
|---|---|
| Run the general DINOv3 abstention-recovery pipeline end to end | `scripts/slurm/slurm_task1_dinov3_end_to_end_recovery_scene.sbatch` |
| Fuse dino.txt and SAM masks into a DINOv3 base | `scripts/task1/dinov3/run_oov_multiclass_scene.sh` |
| Run the complete SAM3 semantic and instance pipeline | `scripts/slurm/slurm_task1_sam3_instance_scene.sbatch` |

The repository still contains the earlier DINOv2 base, ADE v5 refinement,
GroundingDINO source, reviewed-extension, and recoloring entry points. They are
legacy alternatives, not stages of the maintained DINOv3 and dino.txt route.
See the [script inventory](scripts/README.md) for their locations.

## Complete SAM3 semantic and instance pipeline

The SAM3 route starts from a pretrained reconstruction, renders its cameras
(or reuses existing RGB views), and produces semantic classes, multi-label
instance memberships, a scene graph, and final 3D render-back images.
SAM3 is its recognition model; no DINO labels are required for this route.

```text
3DGS model + vocabulary
  -> RGB camera renders (or existing RGB + manifest)
  -> SAM3 masks -> per-concept FlashSplat lift
  -> visibility reliability + constrained cross-view instance association
  -> per-instance consensus + pooled semantic class consensus
  -> hierarchy / scene graph / label arrays
  -> final semantic and instance 3D render-back + object layers
```

Run it with `scripts/slurm/slurm_task1_sam3_instance_scene.sbatch`.
The [SAM3 quickstart](docs/SAM3_QUICKSTART.md) includes the separate environment,
verified ModelScope checkpoint download, fresh/reused camera inputs and all
refinement flags. The original strict route remains the default; the
quickstart explicitly enables the verified refinement.

The 129-camera old_street refinement achieves **80.67% semantic coverage**
and **74.22% instance coverage**, with 1,395 nodes and 1,390 part_of edges.
Class agreement and global instance identity are kept separate; these are
coverage counts, not ground-truth accuracy. See
[refinement semantics and limitations](docs/SAM3_REFINEMENT.md).
Result arrays, images, logs and model weights are not committed.

## Quick start

Follow the [complete quickstart](docs/DINOV3_DINOTXT_SAM_QUICKSTART.md). Its
single shell block starts from a Graphdeco reconstruction and creates fresh
DINOv3 base labels, dino.txt/SAM evidence, final semantic and SuperSplat PLYs,
and class-colored PNGs for every reconstruction camera.

## Expected workspace

```text
<workspace>/
  projects/<this-repository>/
  data/3dgs_models/graphdeco/<scene>/
  external/
  outputs/eyenavgs_task1/
```

Absolute runtime paths are derived by the schedulers. Tracked configuration and
documentation use repository-relative paths.

## Repository layout

```text
configs/          declarative semantic and reviewed-extension configurations
docs/             maintained pipeline, method, status, and dependency guides
scripts/setup/    environment setup
scripts/cluster/  cluster helpers
scripts/slurm/    maintained Slurm entry points
scripts/task1/    semantic subpackages: common, DINOv2, DINOv3, grounding, merge, and QA
tests/            unit and configuration tests
archive/          retired workflows, rejected experiments, and project history
```

## Output contract

A completed maintained run is expected to contain:

- `gaussian_labels.npy`: one integer semantic label per Gaussian;
- `label_map.json`: label identifiers, names, and provenance;
- `semantic_point_cloud.ply` and
  `semantic_point_cloud_supersplat_debug.ply` for both the base and final OOV
  fusion;
- dino.txt probabilities, SAM masks, and their manifest;
- class-colored semantic PNGs and RGB PNGs for every camera;
- fusion summaries and vote manifests.

The PLY `label` property is an integer. Image-space overlay coverage is a useful
diagnostic, not a measurement of semantic accuracy.

## Documentation

- [Current project status](docs/PROJECT_STATUS.md)
- [Complete DINOv3, dino.txt, and SAM quickstart](docs/DINOV3_DINOTXT_SAM_QUICKSTART.md)
- [Detailed DINOv3, dino.txt, and SAM pipeline](docs/DINOV3_DINOTXT_SAM_END_TO_END_PIPELINE.md)
- [External repositories and setup](docs/EXTERNAL_REPOS.md)
- [Script and scheduler map](scripts/README.md)
- [Archive index](archive/README.md)
