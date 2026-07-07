# External Repositories

Third-party repositories are cloned on the cluster under:

```text
/lab/haoq_lab/cse12312032/external
```

Do not commit these repos into this project. After cloning, record the exact
commit hash used for every dependency.

## Planned Dependencies

| Name | Purpose | URL | Commit |
|---|---|---|---|
| EyeNavGS Software | Replay, visualization, coordinate utilities | https://github.com/symmru/EyeNavGS_Software | `2121d03548e126d04e824f09764df1d4fce9fcea` |
| EyeNavGS Rutgers Dataset | Dataset metadata and traces | https://github.com/symmru/EyeNavGS_Rutgers_Dataset | `7f264fda0223cbc74e191f115ef9fd3b580d8cf4` |
| EyeNavGS NTHU Dataset | Dataset metadata and traces | https://github.com/sawalee0811/EyeNavGS_NTHU_Dataset | `608790e7d932534f18a2a124887605d1f6c85384` |
| Gaussian Splatting | Base 3DGS model format and renderer | https://github.com/graphdeco-inria/gaussian-splatting | `54c035f7834b564019656c3e3fcc3646292f727d` |
| FlashSplat | Fast baseline semantic grouping | https://github.com/florinshen/FlashSplat | `3e3b14786333bf0163ba1b8541e86a3765112d7d` |
| SegAnyGAussians | SAGA refinement | https://github.com/Jumpat/SegAnyGAussians | `4acdaa6ba1a8801bd6ac479cb0b9ece653d44f0e` |
| SAM2 | Optional 2D mask generation | https://github.com/facebookresearch/sam2 | TODO |

## Commit Recording Command

From the cluster:

```bash
bash scripts/record_external_repos.sh
```

## Cluster Clone Location

Current clone root:

```text
/lab/haoq_lab/cse12312032/external
```

The repos were cloned with shallow history and `GIT_LFS_SKIP_SMUDGE=1` to avoid
pulling large assets into `/lab`, which is already near capacity.

Required submodules for GraphDeco, FlashSplat, and SAGA were initialized. SAGA's
GitHub SSH submodule URLs were rewritten to HTTPS in the cluster clone because
the cluster account does not have a GitHub SSH key.
