#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${1:-$(cd -- "${PROJECT_ROOT}/../.." && pwd)}"

echo "user: $(whoami)"
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "date: $(date -Is)"
echo

echo "storage:"
df -h "$ROOT" || true
echo

echo "gpu:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi
echo

echo "tools:"
for tool in git python python3 conda mamba nvcc cmake gcc g++; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf "  %-8s %s\n" "$tool" "$(command -v "$tool")"
  else
    printf "  %-8s missing\n" "$tool"
  fi
done
echo

echo "scheduler:"
for tool in sinfo squeue sbatch qstat qsub; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf "  %-8s %s\n" "$tool" "$(command -v "$tool")"
  fi
done

if command -v sacctmgr >/dev/null 2>&1; then
  echo
  echo "slurm associations:"
  sacctmgr show assoc "user=$(whoami)" format=User,Account,Partition,QOS%40 || true
fi

if command -v sinfo >/dev/null 2>&1; then
  echo
  echo "slurm partitions:"
  sinfo
fi
