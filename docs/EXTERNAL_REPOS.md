# External Repositories

Third-party repositories are cloned outside the project repository under:

```text
external/
```

Do not commit these repositories into this project.

## Planned Dependencies

| Name | Purpose | URL |
|---|---|---|
| EyeNavGS Software | Replay, visualization, coordinate utilities | https://github.com/symmru/EyeNavGS_Software |
| EyeNavGS Rutgers Dataset | Dataset metadata and traces | https://github.com/symmru/EyeNavGS_Rutgers_Dataset |
| EyeNavGS NTHU Dataset | Dataset metadata and traces | https://github.com/sawalee0811/EyeNavGS_NTHU_Dataset |
| Gaussian Splatting | Base 3DGS model format and renderer | https://github.com/graphdeco-inria/gaussian-splatting |
| FlashSplat | Mask-to-Gaussian proposal lifting | https://github.com/florinshen/FlashSplat |
| SegAnyGAussians | Semantic rasterizer extensions and Segment Anything dependency | https://github.com/Jumpat/SegAnyGAussians |
| Grounded Segment Anything | GroundingDINO and SAM semantic mask generation | https://github.com/IDEA-Research/Grounded-Segment-Anything |
| DINOv2 | ViT-L/14 backbone and ADE20K linear semantic head | https://github.com/facebookresearch/dinov2 |

## Commit Recording Command

From the cluster:

```bash
bash scripts/cluster/record_external_repos.sh
```

## Clone Location

Current clone root:

```text
external/
```

The repositories are cloned with shallow history and `GIT_LFS_SKIP_SMUDGE=1`
to avoid pulling large assets into project storage.

Required submodules for GraphDeco, FlashSplat, and SAGA were initialized. SAGA's
GitHub SSH submodule URLs are rewritten to HTTPS in environments without GitHub
SSH credentials.

The DINOv2 semantic route uses its own `dinov2_segmentation` Conda environment.
Create it and download the official ViT-L/14 ADE20K linear checkpoints with:

```bash
bash scripts/setup/install_dinov2_segmentation.sh
```

The environment uses `mmcv-full==1.7.2` and `mmsegmentation==0.30.0`. This is
the compatible MMSeg 0.x pair for which OpenMMLab publishes a Python 3.10,
PyTorch 2.0, CUDA 11.7 wheel; forcing a binary wheel prevents an accidental
MMCV source build on a login node. NumPy is pinned to `1.26.4` and OpenCV to
`4.11.0.86`, because the official Torch 2.0 binaries cannot interoperate with
NumPy 2.x.

Current DINOv2 source requires Python 3.10 syntax. If an older dedicated
environment already exists, rebuild it explicitly:

```bash
RECREATE_INCOMPATIBLE_ENV=1 bash scripts/setup/install_dinov2_segmentation.sh
```
