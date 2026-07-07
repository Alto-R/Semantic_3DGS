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

- GraphDeco official pretrained models are downloaded with
  `scripts/download_graphdeco_models.sh`.
- After extraction, discover available PLY files with:

  ```bash
  python scripts/find_point_clouds.py /lab/haoq_lab/cse12312032/data/3dgs_models/graphdeco
  ```

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
