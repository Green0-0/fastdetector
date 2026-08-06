#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --job-name=fd_analysis
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/usr/project/xtmp/fastdetector_hf_cache}"

echo "Running analysis..."
python scripts/analysis.py \
    --globals-config config/globals.toml \
    --analysis-config config/analysis.toml

echo "Analysis complete."
