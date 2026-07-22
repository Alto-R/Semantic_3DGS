# External Repositories and Environment

The semantic pipeline uses external research repositories and model weights that
are intentionally not vendored here. Setup scripts place them under the shared
workspace's `external/` directory and keep the project checkout independent of
machine-specific paths.

## Required capabilities

- Graphdeco/3D Gaussian Splatting scene rendering
- DINOv2 ViT-L/14 and the ADE20K linear segmentation head
- GroundingDINO text-conditioned detection
- Segment Anything mask generation
- FlashSplat projection and Gaussian-mask lifting

Exact repository revisions and environment commands are maintained by the setup
scripts under `scripts/setup/`. Cluster helpers live under `scripts/cluster/`.

## Expected layout

```text
<workspace>/
  projects/<this-repository>/
  external/<dependency-checkouts>/
  data/3dgs_models/graphdeco/<scene>/
  outputs/eyenavgs_task1/
```

Slurm entry points derive `<workspace>` from the current repository checkout. If
a dependency or model cache must be overridden, provide it through the documented
job environment rather than editing a tracked script with an absolute path.

## Setup and verification

1. Run the applicable script in `scripts/setup/` on the target cluster.
2. Verify that the expected scene point cloud and `cameras.json` exist.
3. Run `scripts/slurm/slurm_gpu_check.sbatch` when validating a new environment.
4. Resolve the intended semantic scheduler with `CONFIG_ONLY=1` before
   submitting an expensive job when that entry point supports it.

External repositories are dependencies, not synchronization targets for this
project. Repository maintenance applies only to this repository's canonical
`origin`; the configured `team` remote is read-only.
