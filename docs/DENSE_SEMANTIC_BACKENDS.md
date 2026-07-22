# Dense-Semantic Route with Pluggable 2D Backends

Status: code complete, not yet run on the cluster.

This route is a backend-upgrade candidate for the maintained DINOv2 multiview
fusion base (`docs/DINOV2_MULTIVIEW_VOTING.md`). The maintained base runs the
DINOv2 ViT-L/14 ADE20K **linear head**, whose patch-level (14 px) boundaries
and ~47 ADE20K mIoU are the primary quality ceiling of the 2D evidence. This
route keeps the same render -> segment -> lift -> fuse shape but makes the 2D
segmenter pluggable:

- `mask2former` (default): HuggingFace
  `facebook/mask2former-swin-large-ade-semantic`, ADE20K mIoU ~56,
  pixel-level boundaries, `pip install transformers`, runs on any cluster GPU
  (lighter than SAM ViT-H).
- `dinov3`: official facebookresearch/dinov3 ViT-7B/16 plus the released
  ADE20K M2F segmentor via `torch.hub` (`dinov3_vit7b16_ms`), ADE20K mIoU
  ~63. Needs a 40 GB-class GPU, a local repo clone, and both checkpoints
  (`weights=` segmentor head, `backbone_weights=` backbone).

## Relationship to the maintained DINOv2 base

This is a parallel implementation, not a replacement (yet). Differences:

| Aspect | Maintained base (`scripts/task1/dinov2/`) | This route (`scripts/task1/dense_seg/`) |
| --- | --- | --- |
| 2D model | DINOv2 ViT-L/14 linear head | Mask2Former-Swin-L or DINOv3-7B M2F |
| Ontology | `configs/ade20k_to_project.json` (identity, 150 classes) | `configs/ade20k_to_project.dense_backends.json` (merging, ~40 classes + ignore) |
| Lifting | per-view vote records on disk | one FlashSplat multi-object index-mask call per <=32-class batch |
| Fusion | exact disk-backed aggregation, separate_abstain | in-memory vote matrix, `--mode full` or `--mode fill` |
| Extra gate | evidence/agreement/views | adds `MIN_VISIBLE_RATIO` (views_supporting / visible_views) |

The long-term intent is to port the winning backend into
`scripts/task1/dinov2/dinov2_segment_views.py` behind the same per-view
contract, and keep only one fusion stack. Until a side-by-side comparison on
`room` justifies that, both routes coexist.

## Pipeline

```text
scripts/task1/dense_seg/segment_views_semantic.py   render + segment + remap
scripts/task1/dense_seg/lift_semantic_votes.py      index-mask FlashSplat lift
scripts/task1/dense_seg/fuse_semantic_votes.py      threshold votes, instances,
                                                    pruning, PLY/label-map export
scripts/slurm/slurm_task1_dense_semantic_scene.sbatch
```

Fusion modes:

- `--mode full`: every Gaussian labeled from votes alone (pure dense route).
- `--mode fill`: a baseline `gaussian_labels.npy` + `label_map.json` is kept
  atomic; only base-unlabeled Gaussians receive vote labels, stuff-only by
  default. Stuff votes merge into an existing base entry only when it follows
  the stuff naming convention (`name == class`), so accepted thing instances
  can never be extended or contradicted. The sbatch auto-detects the baseline
  layout (`03_semantic_fusion`, `03_exact_fusion`, or `deliverables/`).

Vote decision per Gaussian (column c wins):

```text
assign iff  sum votes > 0
        and views_supporting[c] >= MIN_VIEWS          (default 2)
        and votes[c] / sum votes >= MIN_AGREEMENT     (default 0.5)
        and views_supporting[c] / visible_views >= MIN_VISIBLE_RATIO (default 0)
```

Views where a Gaussian contributes only to ignore/low-confidence pixels still
count toward `visible_views`, so systematic single-view artifacts cannot pass
a nonzero `MIN_VISIBLE_RATIO` by hiding their visibility.

## Configuration

```text
SCENE, MODEL_DIR, OUTPUT_NAME
MODE                    fill | full           (default fill)
SEG_BACKEND             mask2former | dinov3  (default mask2former)
VIEW_COUNT              0 = all cameras.json cameras (default)
MIN_CONFIDENCE          pixel gate at lift time      (default 0.35)
MIN_VIEWS / MIN_AGREEMENT / MIN_VISIBLE_RATIO
FILL_SOURCE             baseline run dir for fill mode
BASE_LABELS / BASE_LABEL_MAP   explicit overrides for nonstandard layouts
MASK2FORMER_MODEL       HF model id
DINOV3_REPO / DINOV3_BACKBONE_WEIGHTS / DINOV3_SEGMENTOR_WEIGHTS
```

Confidence gating happens in the lift stage, so re-lifting with a different
`MIN_CONFIDENCE` never re-runs the segmentation model.

## Known limitations

- ADE20K closed set: no railroad-track class, no wheel class; unknown objects
  are force-classified into the nearest ADE20K class. The `MIN_CONFIDENCE`
  gate and `ignore` mappings are the mitigation.
- The merging ontology (`*.dense_backends.json`) is intentionally different
  from the maintained identity ontology; the two are not interchangeable and
  each stage validates that its manifest and config agree (names and
  thing/stuff types).

## Tests

`tests/test_dense_semantic_vote.py` (numpy-only, no GPU): ontology integrity,
vote gates, fill semantics, merge-target safety, pruning rules.
