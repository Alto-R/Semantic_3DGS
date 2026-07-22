# EyeNavGS Semantic 3D Gaussian Splatting

This repository builds semantic labels for pretrained EyeNavGS/3D Gaussian
Splatting scenes. The maintained pipeline renders the original scene cameras,
predicts 2D semantics with DINOv2, lifts the evidence onto Gaussians, fuses it
across views, and optionally applies conservative GroundingDINO+SAM refinements.

The current deliverable is semantic annotation (D1). Downstream navigation and
VR integration work are not implemented in this repository yet.

## Current state

- The production semantic base is prompt-free DINOv2 ADE20K fusion using all
  available real cameras and explicit abstention for weak evidence.
- The active automatic refinement is adaptive instance-guard v5. It may repair
  incomplete ADE20K instances, but never overwrites its immutable DINOv2 base.
- Classes missing from ADE20K are handled separately with reviewed
  GroundingDINO+SAM evidence. They are not merged automatically.
- The experimental v6-v10 singleton-recovery methods were rejected after an
  independent multiview audit and are retained under `archive/`.
- Every accepted result remains versioned. No stage overwrites a base output.

See [Project status](docs/PROJECT_STATUS.md) for scene-level decisions and known
limitations.

## Pipeline

```text
pretrained 3DGS + cameras.json
              |
              v
render real camera views
              |
              v
DINOv2 ViT-L/14 + ADE20K head
              |
              v
per-view FlashSplat lift
              |
              v
exact multiview fusion + abstention  ---> immutable DINOv2 base
              |
              +-- ADE v5 refinement ------> versioned refined output
              |
              +-- reviewed custom classes -> versioned extension output
              |
              v
labels + label map + overlays + validation + optional semantic PLY
```

The full stage contract, scheduler variables, review gates, and output layout
are documented in [Pipeline guide](docs/PIPELINE.md).

## Main entry points

| Purpose | Scheduler |
|---|---|
| Build the prompt-free DINOv2 base | `scripts/slurm/slurm_task1_dinov2_scene.sbatch` |
| Run adaptive instance-guard v5 | `scripts/slurm/slurm_task1_ade_refinement_scene.sbatch` |
| Replay a v5 merge from cached evidence | `scripts/slurm/slurm_task1_ade_refinement_replay_scene.sbatch` |
| Generate custom-class source evidence | `scripts/slurm/slurm_task1_semantic_scene.sbatch` |
| Merge explicitly reviewed custom classes | `scripts/slurm/slurm_task1_reviewed_extensions_scene.sbatch` |
| Recolor an existing semantic output | `scripts/slurm/slurm_task1_recolor_output.sbatch` |

The user owns Slurm submission and monitoring. Schedulers that expose
`CONFIG_ONLY` can be resolved without submission by setting `CONFIG_ONLY=1` and
running the scheduler through `bash`.

## Quick start

Run from the repository root on the cluster. The schedulers derive the workspace,
data, external-repository, and output roots from the repository location.

```bash
sbatch --chdir="$(pwd -P)" \
  --export=ALL,SCENE=<scene>,OUTPUT_NAME=<scene>_dinov2_separate_abstain_allviews_v1,VIEW_COUNT=0,FUSION_MODE=separate_abstain,MIN_SEMANTIC_EVIDENCE=0.50,ENABLE_GROUNDINGDINO=0,RESET_OUTPUT=0,COLOR_MODE=class \
  scripts/slurm/slurm_task1_dinov2_scene.sbatch
```

`VIEW_COUNT=0` selects every real camera. Do not treat a successful job as an
accepted annotation: inspect its overlays, fused counts, label map, and
validation report first.

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
scripts/task1/    semantic subpackages: common, DINOv2, grounding, merge, and QA
tests/            unit and configuration tests
archive/          retired workflows, rejected experiments, and project history
```

## Output contract

A reviewed semantic output is expected to contain:

- `gaussian_labels.npy`: one integer semantic label per Gaussian;
- `label_map.json`: label identifiers, names, and provenance;
- class and instance overlays plus contact sheets;
- validation and summary reports;
- `semantic_point_cloud.ply` only when that stage intentionally publishes one.

The PLY `label` property is an integer. Image-space overlay coverage is a useful
diagnostic, not a measurement of semantic accuracy.

## Documentation

- [Pipeline guide](docs/PIPELINE.md)
- [Current project status](docs/PROJECT_STATUS.md)
- [DINOv2 multiview fusion](docs/DINOV2_MULTIVIEW_VOTING.md)
- [External repositories and setup](docs/EXTERNAL_REPOS.md)
- [Script and scheduler map](scripts/README.md)
- [Archive index](archive/README.md)
