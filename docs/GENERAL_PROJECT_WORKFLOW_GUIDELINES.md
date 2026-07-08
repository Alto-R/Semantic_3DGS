# General Project Workflow Guidelines

This guide is for future projects that use a local development machine plus a
remote Linux server or GPU cluster. It avoids project-specific names and should
be reusable across unrelated work.

## Core Principle

Decide the workflow before doing technical work:

- one canonical local project directory
- one canonical remote project directory
- one Git synchronization path
- one place for large data
- one place for generated outputs
- one repeatable way to run remote jobs

Most avoidable mistakes come from leaving one of those implicit.

## Start With Canonical Paths

Pick and record these paths first:

```text
<LOCAL_PROJECT_ROOT>
<REMOTE_PROJECT_ROOT>
<REMOTE_DATA_ROOT>
<REMOTE_EXTERNAL_ROOT>
<REMOTE_OUTPUT_ROOT>
```

Example layout:

```text
<REMOTE_ROOT>/projects/<project-name>   # tracked code and docs
<REMOTE_ROOT>/data/<project-name>       # datasets, models, large inputs
<REMOTE_ROOT>/external                  # third-party repositories
<REMOTE_ROOT>/outputs/<project-name>    # generated results, logs, reports
```

Rules:

- Treat one local directory as canonical.
- Open the coding tool directly in that canonical directory when possible.
- Do not keep two active copies of the repo.
- If a sandboxed tool starts in the wrong folder, clearly declare which folder
  is canonical and copy only tracked project changes into it.
- Do not store large data inside the Git repo.

## Local Versus Remote Responsibilities

Use the local machine for:

- code editing
- documentation
- Git commits
- small test files
- small visual inspections
- project planning

Use the remote machine or cluster for:

- large downloads
- model training or inference
- GPU builds
- long-running jobs
- large generated outputs
- dataset preprocessing

This keeps the local workspace fast and keeps expensive artifacts near the
compute environment.

## SSH And Network Setup

Create a stable SSH alias early:

```sshconfig
Host <remote-alias>
  HostName <remote-host>
  Port <remote-port>
  User <remote-user>
  IdentityFile <path-to-private-key>
  IdentitiesOnly yes
```

Test it with:

```bash
ssh <remote-alias> "pwd; hostname"
```

Rules:

- Never store passwords, passphrases, private keys, or tokens in code, docs,
  commands, logs, or commits.
- If the remote host is on a private/internal network, direct SSH only works
  from that network or through a VPN, bastion host, remote workstation, or
  approved overlay network.
- If key-based SSH already works, do not reinstall the public key unless there
  is evidence it is missing.
- Use a bastion or jump host explicitly when needed:

  ```sshconfig
  Host <target>
    HostName <private-target-host>
    User <target-user>
    ProxyJump <bastion-alias>
  ```

- If SSH suddenly fails at the connection level, wait and retry. Do not assume
  the project setup is broken.

## Git Setup

Create the repo before adding generated files:

```bash
git init
git config user.name "<name>"
git config user.email "<email>"
```

Add `.gitignore` immediately. Ignore at least:

```text
data/
outputs/
external/
checkpoints/
.venv/
__pycache__/
*.log
```

Recommended sync patterns:

- private hosted remote, if internet access and policy allow it
- bare Git remote on the server, if hosted Git is unavailable
- direct `rsync` only for data or artifacts, not as the primary code history

Bare remote pattern:

```bash
# remote
mkdir -p <REMOTE_ROOT>/git
git init --bare <REMOTE_ROOT>/git/<project-name>.git
```

```bash
# local
git remote add cluster <remote-alias>:<REMOTE_ROOT>/git/<project-name>.git
git push -u cluster main
```

```bash
# remote checkout
git clone <REMOTE_ROOT>/git/<project-name>.git <REMOTE_PROJECT_ROOT>
```

Day-to-day flow:

```bash
# local
git status
git add <files>
git commit -m "<clear message>"
git push
```

```bash
# remote
cd <REMOTE_PROJECT_ROOT>
git pull --ff-only
```

Rules:

- Commit code, docs, configs, and small scripts.
- Do not commit datasets, models, generated outputs, checkpoints, caches, or
  third-party repositories.
- Record external repository URLs and commit hashes in a tracked notes file.
- Keep commits small enough that a future reader can understand what changed.

## Third-Party Code

Clone third-party repositories outside the project repo:

```text
<REMOTE_EXTERNAL_ROOT>/<repo-name>
```

Track only a lock/notes file with:

```text
name
url
commit hash
purpose
local path
setup notes
```

Rules:

- Prefer official repositories and documented install commands.
- Do not patch third-party code casually.
- If a third-party patch is unavoidable, record exactly why.
- Keep project-owned adapters and utilities in the project repo.

## Environment Setup

Create an environment setup note before installing many packages:

```text
environment name
Python version
CUDA version
PyTorch version
required system modules
install commands
known incompatible packages
```

Verify the environment with small import checks:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

For overlapping CUDA extensions:

```bash
python -c "import inspect, some_package as p; print(p.__file__); print(inspect.signature(p.SomeClass))"
```

Rules:

- Do not assume two repositories use compatible packages just because the Python
  package name is the same.
- Build CUDA extensions in the environment that will run the job.
- Test imports in a real GPU job when login nodes do not expose GPUs.
- Keep install scripts rerunnable.

## Remote Job Workflow

For clusters, create one small smoke job before running real workloads.

Smoke job should check:

```bash
pwd
hostname
nvidia-smi
df -h <REMOTE_ROOT>
python -c "import torch; print(torch.cuda.is_available())"
```

For Slurm-style systems, standardize:

```bash
sbatch <job-script>
squeue -u <user>
sacct -j <job-id> --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS,NodeList
```

Job scripts should:

- use absolute paths
- activate the environment explicitly
- print host and GPU information
- create output directories
- write logs under `<REMOTE_OUTPUT_ROOT>/logs`
- fail fast with `set -euo pipefail`
- support environment-variable overrides for resources and parameters

## Data And Output Discipline

Use this rule:

```text
inputs go under data/
third-party code goes under external/
generated files go under outputs/
tracked project code goes under projects/
```

For each generated result, save:

- command or job script used
- input paths
- output paths
- manifest JSON if useful
- log files
- small inspection summary

Do not overwrite original input data. Write transformed outputs separately.

## Visual And Artifact Validation

Do not treat a successful command as proof of a good result.

For image, video, 3D, or model outputs:

- check files exist
- check sizes are nonzero
- check dimensions and metadata
- inspect at least one sample visually
- create contact sheets for batches
- compare against expected qualitative behavior
- save validation notes

For structured files:

- inspect schema
- count rows/items before and after transformation
- verify expected properties were preserved
- verify only intended properties were added or changed

## Debugging Pattern

When something fails:

1. Read the exact error.
2. Check whether partial outputs were created.
3. Reduce to one input, one output, one job.
4. Lower resolution or batch size.
5. Verify environment/package paths.
6. Verify function signatures for compiled extensions.
7. Add a diagnostic script instead of guessing.
8. Save the diagnostic in the repo if it will be useful again.

Common useful checks:

```bash
which python
python --version
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi
df -h
du -sh <path>
```

## Permission And Approval Hygiene

At project start, agree on:

- canonical local repo path
- whether copying tracked files between sandbox and canonical repo is allowed
- whether commits should be made automatically
- whether pushes should happen automatically
- which remote is canonical
- what must never be uploaded or committed

Rules:

- Never run destructive commands unless explicitly requested.
- Never revert user changes unless explicitly requested.
- Ask before deleting, resetting, or overwriting unclear state.
- Prefer precise file copies over broad directory syncs.
- Keep secrets out of the repo even if that slows setup.

## Documentation To Create Early

Recommended initial docs:

```text
README.md
docs/WORKFLOW.md
docs/ENVIRONMENT.md
docs/EXTERNAL_REPOS.md
docs/RUNBOOK.md
configs/paths.example.yaml
```

`README.md` should say:

- goal
- current status
- main deliverables
- where to run code

`WORKFLOW.md` should say:

- local path
- remote path
- Git flow
- data/output policy

`ENVIRONMENT.md` should say:

- environment name
- install steps
- verification commands

`EXTERNAL_REPOS.md` should say:

- external repo URLs
- commit hashes
- why each repo is needed

`RUNBOOK.md` should say:

- common commands
- job submission commands
- troubleshooting steps

## Common Pitfalls To Avoid

- Starting work before choosing the canonical repo path.
- Keeping two repo copies and editing both.
- Downloading large data locally when the cluster should own it.
- Committing generated outputs.
- Assuming SSH works from every network.
- Assuming `scp` works just because `ssh` works.
- Assuming login nodes have GPUs.
- Running full jobs before one-scene or one-file smoke tests.
- Trusting successful exit codes without inspecting outputs.
- Mixing incompatible CUDA extension variants.
- Editing third-party code when a project-owned adapter would be safer.
- Letting undocumented manual cluster steps become required workflow.

## New Project Bootstrap Checklist

1. Choose `<LOCAL_PROJECT_ROOT>`.
2. Open the coding tool in that directory.
3. Initialize Git.
4. Add `.gitignore`.
5. Configure Git author.
6. Add `README.md`.
7. Add workflow docs.
8. Configure SSH alias.
9. Verify remote login.
10. Create or choose Git remote.
11. Create remote directory layout.
12. Clone/pull project on remote.
13. Clone third-party repos under `<REMOTE_EXTERNAL_ROOT>`.
14. Record third-party commits.
15. Create environment setup script or notes.
16. Run environment import smoke test.
17. Run GPU smoke job.
18. Run one tiny end-to-end job.
19. Inspect output manually.
20. Scale only after the tiny job is validated.
