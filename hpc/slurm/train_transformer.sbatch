#!/bin/bash
#SBATCH --job-name=bindsite-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Train the cross-attention transformer.
#
#   sbatch hpc/slurm/train_transformer.sbatch configs/homology_split.yaml
#
# The controls run FIRST and the job fails if any of them fails. If the
# label-shuffle control scores above chance, the features leak and nothing
# trained afterwards would mean anything, so there is no point spending the
# GPU allocation.
#
# Memory: attention is O(L^2) in chain length. At max_length 400 and batch 8
# the attention matrices are small; the distance-bias RBF expansion is the
# larger term at n_bases x L^2 per head. 64G host RAM is generous and 16G of
# GPU memory is ample. Raise --max-length before raising --batch-size if
# chains are being truncated: truncation discards residues, which changes what
# the model can see, while a smaller batch only changes the optimisation.

set -euo pipefail

CONFIG="${1:?usage: sbatch train_transformer.sbatch <config.yaml>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs results

module load anaconda3 cuda 2>/dev/null || true
source activate bindsite 2>/dev/null || true

# scikit-learn already parallelises across the allocation; nested BLAS
# threading measurably slows the job.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "host:    $(hostname)"
echo "config:  $CONFIG"
echo "started: $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/validate_controls.py --json "results/controls_${SLURM_JOB_ID:-local}.json"

python -m bindsite.cli run -c "$CONFIG" -v \
  > "results/run_${SLURM_JOB_ID:-local}.json"

echo "finished: $(date -Iseconds)"
