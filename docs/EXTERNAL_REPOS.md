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
| EyeNavGS Software | Replay, visualization, coordinate utilities | https://github.com/symmru/EyeNavGS_Software | TODO |
| EyeNavGS Rutgers Dataset | Dataset metadata and traces | https://github.com/symmru/EyeNavGS_Rutgers_Dataset | TODO |
| EyeNavGS NTHU Dataset | Dataset metadata and traces | https://github.com/sawalee0811/EyeNavGS_NTHU_Dataset | TODO |
| Gaussian Splatting | Base 3DGS model format and renderer | https://github.com/graphdeco-inria/gaussian-splatting | TODO |
| FlashSplat | Fast baseline semantic grouping | https://github.com/florinshen/FlashSplat | TODO |
| SegAnyGAussians | SAGA refinement | https://github.com/Jumpat/SegAnyGAussians | TODO |
| SAM2 | Optional 2D mask generation | https://github.com/facebookresearch/sam2 | TODO |

## Commit Recording Command

From the cluster:

```bash
for repo in /lab/haoq_lab/cse12312032/external/*/.git; do
  worktree="$(dirname "$repo")"
  printf "%s %s\n" "$(basename "$worktree")" "$(git -C "$worktree" rev-parse HEAD)"
done
```

