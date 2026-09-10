# External repositories and environment

The semantic pipeline uses external research repositories and model weights
that are not vendored here. Keep repository checkouts under the shared
workspace's `external/` directory and model files under `data/models/`.

## Required capabilities

- Graphdeco/3D Gaussian Splatting scene rendering
- FlashSplat projection and Gaussian-mask lifting
- DINOv3 ViT-7B with the ADE20K Mask2Former head
- DINOv3 ViT-L/16 with the dino.txt vision and text head
- Segment Anything ViT-H mask generation

The current pipeline requires these external checkouts:

```text
<workspace>/external/dinov3/
<workspace>/external/FlashSplat/
<workspace>/external/segment-anything/
```

The DINOv3 setup pins the DINOv3 checkout to commit
`6876159a11b4df116f30f667f8c9888617df0751`. The complete run also needs these
user-provided files under `<workspace>/data/models/dinov3/`:

```text
dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth
dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth
bpe_simple_vocab_16e6.txt.gz
```

The SAM ViT-H checkpoint path is supplied through `SAM_CHECKPOINT`. Setup
scripts do not download model weights.

DINOv2 and GroundingDINO remain dependencies of legacy alternative workflows.
They are not required by the maintained DINOv3 and dino.txt route.

## Experimental SAM3 instance route

The SAM3 route loads `facebook/sam3` through Hugging Face transformers inside
a dedicated conda environment (default name `sam3_semantic`). It does not use
the external checkouts above except FlashSplat. The checkpoint revision is
not defaulted: every scheduler run must set `SAM3_MODEL_REVISION` to the
pinned revision, and the exact revision plus the transformers version are
recorded here once cluster verification completes. Local unit tests use a
mock backend and never download the checkpoint.

Exact repository revisions and environment commands are maintained by the setup
scripts under `scripts/setup/`. Cluster helpers live under `scripts/cluster/`.

## Expected layout

```text
<workspace>/
  projects/<this-repository>/
  external/dinov3/
  external/FlashSplat/
  external/segment-anything/
  data/3dgs_models/graphdeco/<scene>/
  data/models/dinov3/
  outputs/eyenavgs_task1/
```

Slurm entry points derive `<workspace>` from the current repository checkout. If
a dependency or model cache must be overridden, provide it through the documented
job environment rather than editing a tracked script with an absolute path.

## Setup and verification

1. Run `scripts/setup/install_dinov3_semantic.sh` to create or verify the
   `dinov3_semantic` environment and pinned DINOv3 checkout.
2. Create or verify the `semantic_3dgs_renderer` environment with FlashSplat
   and the Graphdeco renderer dependencies.
3. Put the five DINOv3 and dino.txt files listed above in the model directory.
4. Verify that the scene has `cameras.json` and
   `point_cloud/iteration_30000/point_cloud.ply`.
5. Follow `docs/DINOV3_DINOTXT_SAM_QUICKSTART.md` for the complete run.

External repositories are dependencies, not synchronization targets for this
project. Repository maintenance applies only to this repository's canonical
`origin`; the configured `team` remote is read-only.

## SAM3 cluster smoke verification (2026-09-10)

On `aiseon` (RTX 6000 Ada), the Transformers backend and all six stages passed
an old_street smoke test with 3 existing rendered cameras and all 25 prompt
phrases (22 concepts, including configured synonyms). The run produced 419
2D masks, 4 accepted scene-graph nodes, and 2 part_of edges in 95 seconds.
At that smoke-test stage this verified execution only; the completed
129-camera pilot and later refinement are documented below and in SAM3_REFINEMENT.md.

- Environment: Python 3.11, torch 2.7.1+cu118, transformers 5.3.0, Pillow 12.2.0.
- Renderer: existing semantic_3dgs_renderer, torch 2.0.0+cu117, scipy 1.11.4.
- Download source: ModelScope `facebook/sam3`, commit
  `96f3e1b404ba14f2cfac60ee6ae87c269a7b7923`.
- Transformers weight: `model.safetensors`, 3,439,938,512 bytes, SHA-256
  `6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a`.
  All 12 downloaded files were verified against the pinned Git/LFS metadata.
  The alternative native `sam3.pt` was not needed or downloaded.
- `SAM3_MODEL_ID` points to the local ModelScope snapshot;
  `SAM3_MODEL_REVISION` records its ModelScope commit, not the HF commit.
- Camera indices must use the `selected_129_model` referenced by the existing
  view manifest, rather than the default scene path in the runner.

Real high-footprint Gaussians exposed FP32 atomic-sum roundoff between concept
passes (maximum observed relative deviation about 0.10%). The lift validates
identical observed sets, finite totals and a 0.2% relative mass tolerance;
each concept normalizes with its own summed mass. The tolerance is solely a
numerical guard. The association and acceptance thresholds are unchanged.
The SAM3 suite, including this regression, passes 77 tests.

Machine-local launch configuration and provenance are under
`<workspace>/config/sam3_old_street.env`, `sam3_modelscope_provenance.json`,
and `sam3_readiness.json`. `run_sam3_old_street.sh` is the prepared full-pilot
launcher; `run_sam3_smoke3.sh` was the smoke launcher.

## Full old_street pilot (129 views, 2026-09-10)

The six-stage full pilot completed successfully in 24m12s, using the same
checkpoint and thresholds as the smoke test. It produced 13,820 masks;
12,900 had nonzero lifted support and associated into 6,397 registry instances.
Strict consensus retained 915 scene-graph nodes and 777 part_of edges.
Accepted membership covers 2,665,785 / 4,327,550 Gaussians (61.6%).

These are review outputs, not accepted object ground truth. 4,903 registry
instances have only one supporting camera; 407 accepted nodes have fewer
than 100 Gaussians, and 63 accepted nodes contain same-phrase/same-view
multi-mask association conflicts. Review products and all-camera contact
sheets are in `<workspace>/outputs/eyenavgs_task1/old_street_sam3_instance_pilot_v1/quality_review`.

Two execution optimizations preserve the decision rules:

- SAM3 caches the current immutable image's vision features and prefilters
  scores before resizing masks. The BF16 threshold is stepped downward one
  representable value and the original float-score filter remains decisive.
  All masks and scores matched bitwise on three views with all 25 prompts;
  measured inference changed from ~13s to ~1.2s per view.
- Hierarchy support intersections use an integer sparse Gram product.
  All three real-scene outputs (`scene_graph.json`, `hierarchy.json`, and
  `gaussian_instances.npy`) matched the original implementation byte-for-byte.
  The complete SAM3 test suite passes 78 tests.
