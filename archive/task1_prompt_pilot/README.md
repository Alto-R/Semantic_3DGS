# Archived Task 1 Prompt Pilot

This folder keeps the first prompt-based bicycle pilot for provenance only.
It should not be used as the active Task 1 pipeline.

Why it was archived:

- It used manually chosen positive/negative SAM prompt points.
- Task 1 needs an automatic object-level segmentation workflow.
- The prompt pilot technically validated the render -> mask -> FlashSplat ->
  semantic PLY chain, but the bicycle mask leaked into the bench.

Recorded result for the `bicycle` scene:

- Label histogram:

  ```json
  {"0": 5841032, "1": 290922}
  ```

Active replacement:

```text
scripts/task1/generate_grounded_sam_masks.py
scripts/task1/run_flashsplat_mask_proposals.py
scripts/task1/cluster_semantic_flashsplat_proposals.py
scripts/slurm/slurm_task1_semantic_scene.sbatch
```
