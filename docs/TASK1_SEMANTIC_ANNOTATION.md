# Task 1: Semantic Annotation of EyeNavGS 3DGS Scenes

Task 1 assigns one object/semantic label to every Gaussian in each EyeNavGS scene.
The deliverable for each scene is:

```text
semantic_point_cloud.ply
label_map.json
overlay_renders/
```

## Output Contract

The semantic PLY must preserve the original 3DGS Gaussian properties and add one
new scalar integer property:

```text
property int label
```

Do not modify the original model PLY in place. Write a separate semantic output
file.

The label map must use stable integer IDs:

```json
{
  "scene": "bicycle",
  "labels": [
    {"id": 0, "name": "unlabeled", "class": "unlabeled"},
    {"id": 1, "name": "ground", "class": "ground"}
  ]
}
```

## Pilot Scene

Default pilot scene: `bicycle`.

Fallback rule: use the first scene that has:

- a complete 3DGS `point_cloud.ply`
- usable camera/render metadata
- enough visual structure to validate labels from overlays

Current data/model status:

- EyeNavGS Rutgers and NTHU trace repos are cloned under the cluster `external/`
  directory.
- `/lab/haoq_lab/cse12312032/data/EyeNavGS/Rutgers` and `NTHU` are symlinks to
  those cloned trace repos.
- Trace inventory can be checked with:

  ```bash
  python scripts/inventory_eyenavgs_traces.py
  ```

  Current inventory notes:

  - Rutgers: 12 scenes, 264 CSVs, 2,481,596 rows.
  - NTHU: 12 expected scenes plus one anomalous `trian` folder, 289 CSVs,
    1,384,969 rows.
  - `trian/user21_trian.csv` has no matching `scene_setting.csv` entry and
    should not be merged into `train` without manual confirmation.

- GraphDeco official pretrained models are downloaded with
  `scripts/download_graphdeco_models.sh`.
- After extraction, discover available PLY files with:

  ```bash
  python scripts/find_point_clouds.py /lab/haoq_lab/cse12312032/data/3dgs_models/graphdeco
  ```

  Build the initial scene/model manifest with:

  ```bash
  python scripts/build_task1_manifest.py
  ```

  Current GraphDeco model manifest:

  - 26 `point_cloud.ply` files extracted under
    `/lab/haoq_lab/cse12312032/data/3dgs_models/graphdeco`.
  - 8/12 EyeNavGS scenes match official GraphDeco pretrained models:
    `truck`, `treehill`, `train`, `stump`, `room`, `playroom`, `drjohnson`,
    `bicycle`.
  - The four missing scenes are `nyc`, `london`, `berlin`, and `alameda`;
    these are ZipNeRF-derived scenes and require a separate 3DGS model source.
  - Manifest JSON was saved to:
    `/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/manifests/task1_scene_manifest_graphdeco.json`.
  - Pilot scene candidate: `bicycle`, iteration 30000, 6,131,954 Gaussians.

## Pipeline

1. Locate the scene model:

   ```text
   <scene>/point_cloud/iteration_30000/point_cloud.ply
   ```

2. Inspect the PLY:

   ```bash
   python scripts/inspect_ply.py /path/to/point_cloud.ply --json
   ```

3. Render 100-200 representative RGB views from the original 3DGS model.

4. Generate 2D masks using the mask generator expected by the chosen tool:

   - FlashSplat baseline path first.
   - SAGA mask/feature path for refinement.
   - SAM2 only if the chosen toolchain requires or benefits from it.

5. Lift 2D masks/groups to Gaussian labels:

   - Run FlashSplat for a fast full-scene baseline.
   - Export per-Gaussian group IDs.
   - Convert groups into the project label schema.

6. Name and validate labels:

   - Create semantic overlay renders.
   - Use contact sheets to assign human-readable `name` and `class`.
   - Keep `0` as `unlabeled`.

7. Refine important objects with SAGA:

   High-priority object types:

   - people
   - animals
   - vehicles
   - doors
   - signs

   Medium-priority object types:

   - buildings
   - trees
   - furniture
   - screens

   Low-priority object types:

   - ground
   - sky
   - walls
   - road

8. Export final scene artifacts:

   ```text
   /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/<scene>/semantic_point_cloud.ply
   /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/<scene>/label_map.json
   /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/<scene>/overlay_renders/
   ```

## Validation Criteria

For a scene to count as Task 1 complete:

- every Gaussian has a `label` value
- `label_map.json` covers every nonzero label ID
- overlay renders show plausible object boundaries from several viewpoints
- obvious high-priority objects are not merged into background labels
- the scene has a short notes file describing known weak labels

## Baseline PLY Round Trip

Before running semantic tools, verify that the repo can safely add a label column:

```bash
python scripts/add_dummy_labels.py /path/to/point_cloud.ply /tmp/semantic_test.ply --label 0
python scripts/inspect_ply.py /tmp/semantic_test.ply --json
```

The output PLY should have the same vertex count and all original properties,
plus `label`.

## Bicycle Pilot Bootstrap

Run the first full bootstrap on the cluster from the project checkout:

```bash
cd /lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
bash scripts/build_graphdeco_rasterizer.sh
sbatch scripts/slurm_task1_bicycle_pilot.sbatch
```

Default resources target the L40 node:

```text
partition=a100
qos=a100
nodelist=l40gpu002
```

If that node is busy, submit the same job on the RTX8000 node:

```bash
sbatch --partition=titan --qos=titan --nodelist=rtx8000 scripts/slurm_task1_bicycle_pilot.sbatch
```

Expected first-stage outputs:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/point_cloud_inspection.json
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/semantic_point_cloud.ply
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/semantic_point_cloud_inspection.json
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/label_map.json
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/rgb_renders/
```

The rendered PNGs are RGB sanity frames from `cameras.json`. They are generated
without original ground-truth images, because the pretrained GraphDeco model
folder already contains camera intrinsics/extrinsics but its saved `cfg_args`
points to the original author's local image path. The renderer prefers the
official GraphDeco rasterizer submodule, which must be built once with
`scripts/build_graphdeco_rasterizer.sh`; this avoids using the
Gaussian-Grouping-modified rasterizer for plain RGB sanity renders.

The default pilot render uses a conservative camera set selected by
`scripts/check_graphdeco_projection.py`:

```text
41,43,148,70,115,68,64,87,69,39,85,120,42,153,123,88,37,40,7,111
```

These views avoid the worst near-plane/radius cases in the bicycle model. To
try a higher-resolution run after the 320px sanity pass:

```bash
sbatch --export=ALL,RENDER_MAX_WIDTH=640 scripts/slurm_task1_bicycle_pilot.sbatch
```

To return to evenly spaced camera sampling, submit with an empty camera list:

```bash
sbatch --export=ALL,RENDER_CAMERA_INDICES=,RENDER_COUNT=20 scripts/slurm_task1_bicycle_pilot.sbatch
```

## Bicycle FlashSplat/SAM Pilot

After the RGB sanity renders look valid, run the first binary semantic lift for
the bicycle object:

```bash
cd /lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
sbatch scripts/slurm_task1_bicycle_flashsplat.sbatch
```

This job runs two project-owned adapters instead of FlashSplat's stock
COLMAP-source loader:

1. `scripts/generate_flashsplat_prompt_masks.py`
   - loads the pretrained GraphDeco model from `cameras.json`
   - uses seed-view prompt points on camera index `70`
   - projects the nearest prompted Gaussians into the selected views
   - runs SAM on each rendered view
   - writes masks and mask overlays under:

   ```text
   /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/prompt_masks_bicycle/
   ```

2. `scripts/run_flashsplat_cameras.py`
   - loads those binary masks
   - calls FlashSplat's `flashsplat_render` directly
   - accumulates per-Gaussian `used_count`
   - converts the binary FlashSplat decision into project labels
   - writes the semantic PLY and validation overlays

Default seed prompts in the 320x213 `cam0070` render:

```text
positive:
93,110
218,111
160,80
130,60
205,48

negative:
45,80
285,80
286,132
35,132
160,30
```

Expected outputs:

```text
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/semantic_point_cloud.ply
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/label_map.json
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/flashsplat/flashsplat_counts.pt
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/flashsplat/gaussian_labels.npy
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/flashsplat/flashsplat_manifest.json
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/overlay_renders/
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/overlay_contact_sheet.png
```

This is a one-object pilot. It should not be counted as a fully labeled scene
until additional labels are added and visually validated.

Current pilot result:

- Slurm job `91862` completed on `rtx8000`.
- The semantic PLY has 6,131,954 vertices and the expected final `label`
  property.
- Binary label histogram:

  ```json
  {"0": 5841032, "1": 290922}
  ```

- Visual overlays confirm that the FlashSplat/SAM adapter runs end to end, but
  the bicycle label still leaks into the bench in many views. Treat this output
  as a technical baseline, not an accepted final semantic annotation.
