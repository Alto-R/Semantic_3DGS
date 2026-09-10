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
