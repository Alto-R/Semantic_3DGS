# Retired Task 1 Workflows

These files were removed from active directories because no maintained scheduler,
module, or test depends on them.

```text
scripts/task1/   manual review templates, inventory helpers, and projection checks
scripts/slurm/   obsolete compatibility wrapper
configs/         stale example path configuration
docs/            superseded implementation and method histories
```

`docs/DINOV2_MULTIVIEW_VOTING_SUMMARY.md` is the concise final description of
the retired DINOv2 route. The older document beside it preserves the fuller
experimental history.

They are preserved verbatim for provenance. Do not add new runtime dependencies
on this directory; restore and modernize a file explicitly if it becomes useful
again.
