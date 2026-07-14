# Script Layout

Run project commands from the repository root so the relative paths used by
the Slurm entry points resolve consistently.

- `task1/`: maintained Task 1 Python pipeline, validation, visualization, and
  inspection tools. These files stay together because several import shared
  local modules.
- `slurm/`: maintained Slurm entry points. Use
  `slurm/slurm_task1_semantic_scene.sbatch` for scene-configurable runs; the
  bicycle entry point is a compatibility wrapper.
- `cluster/`: cluster bootstrap, health-check, dependency-clone, and dependency
  inventory helpers.
- `setup/`: renderer and semantic-extension installation helpers.

Superseded experiments and utilities are retained under
`archive/task1_legacy/scripts/` and `archive/task1_prompt_pilot/scripts/` for
provenance. They are not active pipeline entry points.

Scene vocabularies live in `configs/task1_semantic_classes.<scene>.json`. If a
scene-specific file is absent, the generic scheduler falls back to
`configs/task1_semantic_classes.example.json`.
