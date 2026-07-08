# Local and Cluster Workflow

This project uses Windows for coordination and the Linux GPU cluster for heavy
processing.

## Machines

Local project folder:

```text
C:\Users\dhana\PROJECTS\PKU 3DGS VR
```

Cluster login:

```text
ssh -p 10022 cse12312032@172.18.34.25
```

Preferred SSH alias:

```text
haoqi
```

Cluster root:

```text
/lab/haoq_lab/cse12312032
```

## SSH Setup

The existing key is:

```text
C:\Users\dhana\.ssh\id_ed25519
```

If `ssh haoqi` prints this prompt, the public key is already accepted by the
cluster:

```text
Enter passphrase for key 'C:\Users\dhana\.ssh\id_ed25519':
```

That prompt asks for the local private-key passphrase, not the cluster account
password. Codex and scripts cannot answer that interactive prompt reliably, so
load the key into `ssh-agent` once per login session:

```powershell
Get-Service ssh-agent | Set-Service -StartupType Manual
Start-Service ssh-agent
ssh-add $env:USERPROFILE\.ssh\id_ed25519
```

Enter the key passphrase when `ssh-add` asks. After that, this should work
without a prompt:

```powershell
ssh haoqi "pwd; hostname"
```

Recommended `C:\Users\dhana\.ssh\config` entry:

```sshconfig
Host haoqi
  HostName 172.18.34.25
  Port 10022
  User cse12312032
  IdentityFile C:\Users\dhana\.ssh\id_ed25519
  IdentitiesOnly yes
```

## Git Workflow

Use Git for source code, docs, configs, and small reproducible utilities.

Current remotes:

- `origin`: private GitHub remote at
  `https://github.com/DhanaKresnawijaya237/pku-3dgs-vr.git`
- The old `cluster` bare remote under
  `/lab/haoq_lab/cse12312032/git/pku-3dgs-vr.git` is deprecated and should not
  be used for new syncs.

Local setup used for GitHub:

```powershell
git remote add origin https://github.com/DhanaKresnawijaya237/pku-3dgs-vr.git
git push -u origin main
```

Daily local-to-cluster sync:

```powershell
cd "C:\Users\dhana\PROJECTS\PKU 3DGS VR"
git status
git add <changed-files>
git commit -m "<short message>"
git push
```

On the cluster:

```bash
cd /lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
git pull --ff-only
```

The cluster checkout needs GitHub authentication before it can pull a private
GitHub repo. Use a GitHub SSH key, deploy key, or `gh auth login` on the
cluster. Do not store GitHub tokens in project files.

Large data and generated files must not be committed.

## Cluster Layout

Use this fixed layout:

```text
/lab/haoq_lab/cse12312032/projects/pku-3dgs-vr      # this Git repo
/lab/haoq_lab/cse12312032/data/EyeNavGS             # EyeNavGS traces/scenes
/lab/haoq_lab/cse12312032/data/3dgs_models          # pretrained 3DGS models
/lab/haoq_lab/cse12312032/external                  # third-party repos
/lab/haoq_lab/cse12312032/outputs/eyenavgs_task1    # semantic labels/renders
```

Create the directories on the cluster:

```bash
bash scripts/bootstrap_cluster.sh
```

Check cluster capabilities:

```bash
bash scripts/cluster_check.sh
```

Current cluster facts:

- Login node: `login02`
- GPU jobs must run through Slurm; the login node does not expose `nvidia-smi`
- Slurm account: `gpulab02`
- Allowed GPU QoS/partitions for this user: `rtx2080ti`, `titan`, `a100`
- Default smoke-test target: `rtx2080ti`
- Verified GPU smoke test: job `91780` ran on `gpu022` with an RTX 2080 Ti
- Verified higher-VRAM smoke tests:
  - job `91793` ran on `rtx8000` with a Quadro RTX 8000, 49 GB VRAM
  - job `91794` ran on `l40gpu002` with an NVIDIA L40, 46 GB VRAM
- `/lab` is nearly full: 32T total, about 624G free during setup on 2026-07-07

Submit a GPU smoke test:

```bash
cd /lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
sbatch scripts/slurm_gpu_check.sbatch
```

Target a specific higher-VRAM node:

```bash
sbatch --partition=titan --qos=titan --nodelist=rtx8000 scripts/slurm_gpu_check.sbatch
sbatch --partition=a100 --qos=a100 --nodelist=l40gpu002 scripts/slurm_gpu_check.sbatch
```

Install missing local CUDA extensions into the selected Conda environment:

```bash
cd /lab/haoq_lab/cse12312032/projects/pku-3dgs-vr
bash scripts/install_semantic_extensions.sh gaussian_grouping_true
```

Current semantic environment status:

- Reused Conda env: `gaussian_grouping_true`
- PyTorch: `2.0.0+cu117`
- Verified imports on the login node:
  - `torch`
  - `cv2`
  - `segment_anything`
  - `diff_gaussian_rasterization`
  - `simple_knn`
  - `flashsplat_rasterization`
  - `diff_gaussian_rasterization_contrastive_f`
  - `diff_gaussian_rasterization_depth`

## Third-Party Code

Clone third-party repos under:

```text
/lab/haoq_lab/cse12312032/external
```

Do not vendor them into this repository. Record URLs and commit hashes in
`docs/EXTERNAL_REPOS.md`.

Required or likely repos:

- EyeNavGS software
- EyeNavGS Rutgers dataset repo
- EyeNavGS NTHU dataset repo
- Original 3D Gaussian Splatting
- FlashSplat
- SAGA / SegAnyGAussians
- SAM2, only if needed for mask generation

## Artifact Movement

Move only small artifacts back to Windows:

- summary CSVs
- label maps
- contact sheets
- selected validation images
- logs
- reports

Keep full datasets, model folders, raw rendered views, masks, and checkpoints on
the cluster.
