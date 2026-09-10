# Complete SAM3 pipeline

This pipeline starts from an existing Graphdeco 3DGS reconstruction and ends
with semantic labels, global instance memberships, a scene graph, and actual
3D render-back images. It can generate its own RGB inputs or reuse existing
ones. It does not train the original reconstruction, train a downstream GNN,
or implement VR navigation.

## Environment and checkpoint

Use the workspace layout from the root README: this checkout is under
`<workspace>/projects/<repository>`, FlashSplat is under `external/FlashSplat`,
and each scene model contains `cameras.json` and
`point_cloud/iteration_30000/point_cloud.ply`.

Prepare the separate `semantic_3dgs_renderer` environment and editable
FlashSplat rasterizer as described in [external dependencies](EXTERNAL_REPOS.md).
The verified renderer uses torch 2.0.0+cu117 and scipy 1.11.4. Keep this
environment separate from the modern SAM3 dependencies.

From the repository root:

```bash
bash scripts/setup/install_sam3_semantic.sh
export WORKSPACE_ROOT="$(cd ../.. && pwd -P)"
export SAM3_MODEL_REVISION=96f3e1b404ba14f2cfac60ee6ae87c269a7b7923
export SAM3_MODEL_ID="${WORKSPACE_ROOT}/data/models/sam3/modelscope-${SAM3_MODEL_REVISION}"
conda run --no-capture-output -n sam3_semantic python scripts/setup/download_sam3.py \
  --output-dir "${SAM3_MODEL_ID}"
```

The installer pins the verified torch 2.7.1+cu118, transformers 5.3.0,
Pillow 12.2.0 and ModelScope 1.40.0. The downloader uses ModelScope and verifies
all 12 Transformers snapshot files against checked-in hashes. It excludes
the alternative native `sam3.pt`; neither model weights nor credentials are
stored in Git. `SAM3_MODEL_REVISION` is the ModelScope commit, not an HF commit.
Use `--verify-only` to validate an already downloaded checkpoint.

## Start from a reconstruction

```bash
export PROJECT_ROOT="$(pwd -P)"
export SCENE=old_street
export MODEL_DIR="${WORKSPACE_ROOT}/data/3dgs_models/graphdeco/${SCENE}"
export OUTPUT_NAME=old_street_sam3_refined_run1
export SAM3_ENV=sam3_semantic
export GAUSSIAN_ENV=semantic_3dgs_renderer
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export VISIBILITY_WEIGHTING=soft
export VISIBILITY_ABSOLUTE_SCALE=0.01
export VISIBILITY_RELATIVE_SCALE=0.05
export CONSENSUS_THRESHOLD=0.4
export PREVENT_DISJOINT_INSTANCE_MERGES=1
export POOL_CONCEPT_VOTES=1
export SEMANTIC_THRESHOLD=0.4
unset RGB_DIR RENDER_MANIFEST
bash scripts/slurm/slurm_task1_sam3_instance_scene.sbatch
```

This works directly with `bash`, without Slurm. Adjust the scheduler header
before submitting with `sbatch` on another cluster. `CAMERA_COUNT=0` (default)
renders all reconstruction cameras; `CAMERA_COUNT=3` is a small execution
smoke test, not a quality evaluation. The RGB renderer lives in the shared
legacy package `dinov2/render_task1_views.py` but performs no DINO inference.

Set a new `OUTPUT_NAME` for each experiment. `CONFIG_ONLY=1 bash ...` prints
the resolved configuration without model execution. `OVERWRITE=1` is an
explicit replacement option for a failed run, not the normal workflow.

For another scene, provide `VOCABULARY=/path/to/vocabulary.json`; use the
old_street configuration as the schema reference. The default old_street
vocabulary has 22 concepts and 25 independently prompted English phrases.

## Reuse existing RGB inputs

Set both variables before running the same script:

```bash
export RGB_DIR="/path/to/existing/stages/01_real_camera_views/rgb_renders"
export RENDER_MANIFEST="/path/to/existing/stages/01_real_camera_views/view_manifest.json"
```

`MODEL_DIR` must be the exact reconstruction used in this manifest, including
camera indexing and Gaussian order. In the verified old_street run this was
the prepared `selected_129_model`. Setting only one input is rejected.
Existing RGB inputs are read without being overwritten.

## Stages and outputs

| Stage | Product |
|---|---|
| 00, optional | RGB camera renders and manifest |
| 01 | SAM3 per-view instance masks and prompt metadata |
| 02 | Per-concept FlashSplat Gaussian votes |
| 02b, optional | Per-camera rendered mass cache |
| 03 | Cross-view instance registry, optionally with cannot-link constraints |
| 04 | Per-instance consensus, raw counts and optional reliability weights |
| 04b, optional | Pooled semantic concept memberships and label arrays |
| 05 | Instance hierarchy, scene graph, Gaussian instance IDs |
| 06 | Two-dimensional mask inspection overlays |
| 07 | Final semantic/instance labels rendered through the original 3D Gaussians |

The new semantic output is under `stages/04_semantic_consensus`:
`gaussian_labels.npy`, `gaussian_labels_objects_first.npy`, `label_map.json`,
`concept_membership.npz`, `instance_unresolved.npy`, and `semantic_summary.json`.
Stage 05 instance IDs are a different namespace from semantic class IDs.

`renderback/` contains per-camera semantic, objects-first and instance label
images and overlays, standalone concept layers, contact sheets and a
hash-linked `render_summary.json`. `RENDER_LAYERS=window,signboard,storefront`
controls the independent concept layers. This renderer can also be invoked
on a completed run using `python -m scripts.task1.sam3.render_results --help`.

The verified old_street refinement achieved **80.67% semantic coverage** and
**74.22% instance coverage** across 129 cameras, with 1,395 nodes and 1,390
part_of edges. Coverage is not accuracy; 278,770 Gaussians have an accepted
class while their instance identity remains unresolved. See
[refinement semantics and limitations](SAM3_REFINEMENT.md).

The original strict instance route remains available by leaving refinement
options at their defaults. Actual run data, images, logs and weights remain
outside this repository.

## Integration validation

The complete entry point was executed on the RTX 6000 Ada server from the
old_street reconstruction with three freshly rendered cameras and all 25
SAM3 prompts. Every stage, including semantic pooling and the final actual
3D render-back, completed successfully. This smoke run verifies orchestration;
the separate 129-view run above supplies the quality/coverage reference.

Before integration into main, the repository suite passed **301 tests and
155 subtests**. The SAM3 subset passed 98 tests. Checkpoint verification
matched all 12 pinned files, and the setup/runner shell syntax checks passed.
