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

Expected flow:

```powershell
cd "C:\Users\dhana\PROJECTS\PKU 3DGS VR"
git status
git add README.md .gitignore docs configs scripts
git commit -m "Bootstrap EyeNavGS semantic annotation workflow"
git remote add origin <private-git-remote-url>
git push -u origin main
```

On the cluster:

```bash
mkdir -p /lab/haoq_lab/cse12312032/projects
cd /lab/haoq_lab/cse12312032/projects
git clone <private-git-remote-url> pku-3dgs-vr
cd pku-3dgs-vr
```

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
