# Project Workflow Lessons

This note condenses reusable setup lessons from the EyeNavGS / 3DGS semantic
annotation bootstrap. It is meant for future projects so we can skip repeated
setup confusion and go directly to implementation.

## Canonical Workspace

- The canonical local project directory is:

  ```text
  C:\Users\dhana\PROJECTS\PKU 3DGS VR
  ```

- Older files may still exist under:

  ```text
  C:\Users\dhana\Documents\PKU 3DGS VR
  ```

- Treat `PROJECTS` as the real repo. The `Documents` path is only a historical
  workspace/sandbox artifact.
- For future Codex sessions, open the project directly from `PROJECTS` when
  possible. Otherwise, Codex may still be sandboxed to `Documents`, which causes
  extra path-copying and approval friction.
- The user has explicitly approved treating repo-item moves/copies from
  `Documents` to `PROJECTS` as normal project maintenance. The Codex app may
  still require a filesystem approval because of sandbox rules, but conceptually
  this is not a user-level decision point.

## Cluster Access

- Cluster login command:

  ```bash
  ssh -p 10022 cse12312032@172.18.34.25
  ```

- SSH alias:

  ```sshconfig
  Host haoqi
    HostName 172.18.34.25
    Port 10022
    User cse12312032
    IdentityFile C:\Users\dhana\.ssh\id_ed25519
    IdentitiesOnly yes
  ```

- Cluster root:

  ```text
  /lab/haoq_lab/cse12312032
  ```

- Do not put passwords, passphrases, or private keys into scripts, commands,
  docs, logs, or commits.
- Key-based SSH was already working after the user entered the local key
  passphrase. Installing a public key into `authorized_keys` is only needed when
  the cluster does not already accept that key.
- The cluster IP is private/internal. Off the required network, direct SSH will
  not work unless there is a VPN, bastion/jump host, remote lab workstation, or
  approved overlay network such as WireGuard/Tailscale/ZeroTier.
- If SSH suddenly reports connection-level `Permission denied`, retry later
  rather than repeatedly hammering it. We saw transient denial during file-copy
  attempts even after jobs had completed successfully.
- `scp` may fail even when normal SSH recently worked. For small artifacts,
  fallback options are:
  - retry after a delay
  - use `ssh` plus `base64` transfer
  - inspect metadata on the cluster and pull artifacts later

## Git Pattern

- Local canonical repo:

  ```text
  C:\Users\dhana\PROJECTS\PKU 3DGS VR
  ```

- Original cluster bare remote, now deprecated:

  ```text
  /lab/haoq_lab/cse12312032/git/pku-3dgs-vr.git
  ```

- Current GitHub remote:

  ```text
  https://github.com/DhanaKresnawijaya237/pku-3dgs-vr
  ```

- Cluster checkout:

  ```text
  /lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
  ```

- Current remote pattern:

  ```text
  origin -> https://github.com/DhanaKresnawijaya237/pku-3dgs-vr.git
  ```

- Push local code first, then pull on the cluster:

  ```powershell
  git -C "C:\Users\dhana\PROJECTS\PKU 3DGS VR" push origin main
  ```

  ```bash
  cd /lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
  git pull --ff-only
  ```

- Because the GitHub repo is private, the cluster checkout needs GitHub
  authentication before it can pull directly from GitHub.

- Use Git for code and docs only. Do not commit datasets, 3DGS models,
  generated PLYs, checkpoints, render outputs, third-party repos, or SAM/feature
  caches.
- Keep third-party repositories under:

  ```text
  /lab/haoq_lab/cse12312032/external
  ```

- Record third-party repo URLs and commit hashes in tracked docs instead of
  vendoring those repos.
- Configure Git author explicitly at project start. For this project:

  ```text
  Dhana Kresnawijaya <dhanakresnawijaya@gmail.com>
  ```

## Cluster Directory Layout

Use this layout consistently:

```text
/lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
/lab/haoq_lab/cse12312032/data/EyeNavGS
/lab/haoq_lab/cse12312032/data/3dgs_models
/lab/haoq_lab/cse12312032/external
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1
```

Rules:

- Download heavy data directly on the cluster.
- Keep large files off Windows unless they are tiny samples for inspection.
- Keep outputs under `/outputs`, not inside the Git repo.
- Build small project-owned utility scripts under `scripts/`.
- Keep environment setup scripts reproducible and rerunnable.

## Slurm And GPU Notes

- The login node has no usable GPU. GPU checks and CUDA imports must run inside
  Slurm jobs.
- Known Slurm account:

  ```text
  gpulab02
  ```

- Known working resource targets:

  ```bash
  # RTX 8000
  --partition=titan --qos=titan --nodelist=rtx8000

  # L40
  --partition=a100 --qos=a100 --nodelist=l40gpu002
  ```

- RTX8000 and L40 both worked for semantic environment smoke tests.
- Prefer high-VRAM nodes for large Gaussian scenes. `bicycle` has about 6.1M
  Gaussians and needs careful rendering choices.
- Use Slurm logs under:

  ```text
  /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/logs
  ```

- Always inspect both `.out` and `.err`; successful partial outputs can exist
  even when a later stage fails.

## Python / CUDA Environment

- Reused Conda env:

  ```text
  gaussian_grouping_true
  ```

- Verified useful imports:
  - `torch`
  - `cv2`
  - `numpy`
  - `PIL`
  - `tqdm`
  - `segment_anything`
  - `diff_gaussian_rasterization`
  - `simple_knn`
  - FlashSplat/SAGA rasterizer extensions

- Do not assume one rasterizer package is compatible with every repo. We hit
  GraphDeco renderer/API mismatches because different repos install different
  `diff_gaussian_rasterization` variants under the same Python package name.
- For plain GraphDeco RGB rendering, prefer the official GraphDeco rasterizer
  submodule built in-place:

  ```bash
  bash scripts/build_graphdeco_rasterizer.sh
  ```

- For semantic methods, keep the semantic extensions available in the Conda env,
  but avoid replacing third-party source code unless absolutely necessary.

## What Worked

- Key-based SSH worked once the local key passphrase was entered.
- Git remote to the cluster bare repo worked well for syncing project code.
- Cluster checkout with `git pull --ff-only` kept local and cluster code aligned.
- The fixed directory layout made it clear what belongs in Git versus outputs.
- Slurm smoke tests on RTX8000 and L40 confirmed GPU availability and core CUDA
  imports.
- Binary PLY round-trip worked: preserving all original Gaussian properties and
  adding a final integer `label` property.
- The dummy semantic PLY for `bicycle` was successfully written:

  ```text
  /lab/haoq_lab/cse12312032/outputs/eyenavgs_task1/bicycle/semantic_point_cloud.ply
  ```

- RGB rendering worked after:
  - using the official GraphDeco rasterizer
  - selecting safer camera IDs
  - lowering pilot render width
  - running on RTX8000

## What Did Not Work Immediately

- Treating `Documents` as the real workspace caused repeated confusion. The real
  repo should be `PROJECTS`.
- Trying to use the stock GraphDeco renderer directly was not ideal because
  pretrained model `cfg_args` pointed to the original author's local image path.
  The fix was to render from `cameras.json` without ground-truth image loading.
- The Gaussian-Grouping-modified rasterizer was not compatible with plain
  GraphDeco RGB rendering in this setup. It caused API mismatch and then huge
  CUDA allocation failures.
- Even the official rasterizer can fail with unsafe views. Camera 0 triggered
  enormous projected splat extents and requested over 100 TiB of CUDA memory.
- Evenly spaced camera sampling is not safe by default for large 3DGS scenes.
  Use projection diagnostics or known-safe camera IDs first.
- `scp` was unreliable during this session. It sometimes failed with
  connection-level `Permission denied` after jobs had already completed.

## Reusable Debugging Pattern

When cluster work fails:

1. Check Slurm state:

   ```bash
   squeue -u cse12312032
   sacct -j <job_id> --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS,NodeList
   ```

2. Read both logs:

   ```bash
   sed -n '1,220p' /path/to/job.out
   sed -n '1,220p' /path/to/job.err
   ```

3. Check whether partial outputs were still created.
4. Reduce the job to one scene, one camera, one output.
5. Lower image resolution before assuming the method is broken.
6. Verify package signatures when CUDA extension names overlap:

   ```bash
   python -c "import inspect, diff_gaussian_rasterization as r; print(r.__file__); print(inspect.signature(r.GaussianRasterizationSettings))"
   ```

7. Prefer a diagnostic script over guessing camera conventions.

## User Preferences And Decisions

- Use the cluster for heavy work: downloads, 3DGS model handling, rendering,
  segmentation, CUDA builds, and large outputs.
- Use Windows local workspace for docs, code edits, Git control, and small
  sample inspections.
- Keep large data off Windows unless specifically needed.
- Use private GitHub as the primary remote. The old cluster bare remote was
  useful for bootstrapping, but should not be required for future syncs.
- Do not ask again whether repo items can be copied/moved from `Documents` to
  `PROJECTS`; treat `PROJECTS` as canonical.
- Do not store any cluster password or key material in the repo.
- Prefer direct implementation once a plan is accepted; do not stop at proposal
  unless there is a real blocker.

## Future Project Bootstrap Checklist

1. Pick the canonical local repo path first.
2. Open Codex directly in that path if possible.
3. Configure SSH alias and test `ssh <alias> "pwd; hostname"`.
4. Create a bare cluster Git remote or private GitHub remote.
5. Configure Git author.
6. Create `.gitignore` before downloads or generated outputs.
7. Establish cluster directory layout before cloning external repos.
8. Clone third-party repos under `/external`, not inside the project repo.
9. Record third-party commits immediately.
10. Build and test CUDA extensions with a tiny import job before real jobs.
11. Start with one scene and one camera.
12. Save logs, manifests, and inspection JSONs under `/outputs`.
13. Only scale to more scenes after the single-scene pilot has real visual QA.
