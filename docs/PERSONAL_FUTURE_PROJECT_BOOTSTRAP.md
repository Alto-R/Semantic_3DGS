# Personal Future Project Bootstrap

Use this file to reduce setup time and token cost for future projects. It is
personal to this machine/user/cluster setup, but intentionally avoids details
from any one research project.

## Canonical Local Setup

- Put future project repos under:

  ```text
  C:\Users\dhana\PROJECTS\<PROJECT_NAME>
  ```

- Treat `C:\Users\dhana\PROJECTS` as the canonical local root.
- If Codex opens in another folder, copy only tracked repo items into the
  canonical `PROJECTS` repo and continue there.
- The user has approved routine copying/moving of repo files from temporary or
  sandbox folders into `C:\Users\dhana\PROJECTS\<PROJECT_NAME>`.
- Do not keep two active repo copies.

## Git Identity

Use this Git author identity:

```bash
git config user.name "Dhana Kresnawijaya"
git config user.email "dhanakresnawijaya@gmail.com"
```

Default Git strategy:

- initialize Git immediately
- add `.gitignore` before generating outputs
- commit code/docs/configs/scripts
- do not commit datasets, models, outputs, checkpoints, caches, or external repos
- prefer private GitHub for long-term backup/sharing when a remote is provided
- if GitHub remote is not provided yet, use a local/cluster remote first

## SSH And Cluster

Primary cluster SSH:

```bash
ssh -p 10022 cse12312032@172.18.34.25
```

Preferred SSH alias:

```sshconfig
Host haoqi
  HostName 172.18.34.25
  Port 10022
  User cse12312032
  IdentityFile C:\Users\dhana\.ssh\id_ed25519
  IdentitiesOnly yes
```

Cluster root:

```text
/lab/haoq_lab/cse12312032
```

Reusable cluster layout:

```text
/lab/haoq_lab/cse12312032/projects/<PROJECT_NAME>   # Git checkout
/lab/haoq_lab/cse12312032/data/<PROJECT_NAME>       # datasets and large inputs
/lab/haoq_lab/cse12312032/outputs/<PROJECT_NAME>    # generated outputs and logs
/lab/haoq_lab/cse12312032/external                  # third-party repos
/lab/haoq_lab/cse12312032/git/<PROJECT_NAME>.git    # optional bare remote
```

Rules:

- never put passwords/passphrases/private keys/tokens in files or commands
- key-based SSH already works when the local key is available/unlocked
- do not reinstall the public key unless SSH key auth stops working
- `172.18.34.25` is an internal/private address; off-network access needs VPN,
  a bastion/jump host, or a lab machine already inside the network
- if SSH/scp returns transient connection-level `Permission denied`, wait and
  retry instead of repeatedly hammering it

## Cluster Git Remote Pattern

If GitHub is not set up yet, use a bare remote on the cluster:

```bash
mkdir -p /lab/haoq_lab/cse12312032/git
git init --bare /lab/haoq_lab/cse12312032/git/<PROJECT_NAME>.git
```

Local remote:

```bash
git remote add cluster haoqi:/lab/haoq_lab/cse12312032/git/<PROJECT_NAME>.git
git push -u cluster main
```

Cluster checkout:

```bash
git clone /lab/haoq_lab/cse12312032/git/<PROJECT_NAME>.git \
  /lab/haoq_lab/cse12312032/projects/<PROJECT_NAME>
```

Daily flow:

```bash
# local
git status
git add <files>
git commit -m "<message>"
git push cluster main
```

```bash
# cluster
cd /lab/haoq_lab/cse12312032/projects/<PROJECT_NAME>
git pull --ff-only
```

If the cluster is unavailable, commit locally and leave the repo ahead of the
remote until network access returns.

## Slurm / GPU Notes

Known account:

```text
gpulab02
```

Known working GPU targets:

```bash
# RTX 8000, more VRAM
--partition=titan --qos=titan --nodelist=rtx8000

# L40, more VRAM
--partition=a100 --qos=a100 --nodelist=l40gpu002

# RTX 2080 Ti, smaller jobs
--partition=rtx2080ti --qos=rtx2080ti
```

Cluster checks:

```bash
squeue -u cse12312032
sacct -j <JOB_ID> --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS,NodeList
```

Smoke test every new project:

```bash
pwd
hostname
nvidia-smi
df -h /lab/haoq_lab/cse12312032
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Standard New Project Skeleton

Create this early:

```text
README.md
.gitignore
.gitattributes
configs/paths.example.yaml
docs/WORKFLOW.md
docs/EXTERNAL_REPOS.md
docs/ENVIRONMENT.md
scripts/
```

Minimum `.gitignore`:

```gitignore
data/
outputs/
external/
checkpoints/
.venv/
__pycache__/
*.log
```

## Third-Party Code

- Clone third-party repos under:

  ```text
  /lab/haoq_lab/cse12312032/external
  ```

- Do not vendor third-party repos into the project repo.
- Track only URLs, commit hashes, setup notes, and local paths in
  `docs/EXTERNAL_REPOS.md`.

## Default Work Split

Local Windows:

- edit code/docs
- manage Git commits
- inspect small samples
- write plans and notes

Cluster:

- download large files
- clone heavy third-party repos
- build CUDA extensions
- run GPU jobs
- store large outputs

## Reusable Debugging Rules

- Start with one tiny end-to-end run before scaling.
- Always read both `.out` and `.err` logs.
- Check partial outputs even when a job fails.
- If a Python/CUDA package name overlaps across repos, inspect the actual import
  path and function signature:

  ```bash
  python -c "import inspect, PACKAGE as p; print(p.__file__); print(inspect.signature(p.SYMBOL))"
  ```

- For visual outputs, verify file existence, size, dimensions, and at least one
  actual image/sample.
- Do not treat a completed job as success until output quality is inspected.

## Things To Avoid Repeating

- Do not start coding before choosing the canonical local and remote paths.
- Do not ask again whether `PROJECTS` is the real local root.
- Do not use the cluster login node as if it has GPUs.
- Do not commit generated outputs or heavy files.
- Do not repeatedly try SSH/scp during network denial; wait or switch network.
- Do not patch third-party code unless a project-owned wrapper cannot solve it.
- Do not run full-scale jobs before a single-input smoke test passes.
