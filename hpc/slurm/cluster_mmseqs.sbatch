#!/bin/bash
#SBATCH --job-name=bindsite-cluster
#SBATCH --output=logs/cluster_%j.out
#SBATCH --error=logs/cluster_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=short

# Cluster sequences with MMseqs2, for datasets beyond a few thousand proteins.
#
#   sbatch hpc/slurm/cluster_mmseqs.sbatch configs/homology_split.yaml
#
# CPU only, and heavily parallel. The built-in exact clustering is O(n^2)
# alignments — fine for the few hundred proteins in this repository, hopeless
# for 145000 — so MMseqs2 is what a full-scale run needs.
#
# IMPORTANT: MMseqs2's clustering, like any centroid or set-cover scheme, does
# not compute the transitive closure of the similarity relation. It can
# therefore place a homologous pair in different clusters, and a split built
# from it can leak even though no cluster is shared. Always keep
# split.verify_identity on, which checks the split against the sequences
# themselves. This repository found exactly that failure: greedy clustering
# put two 46.8%-identical proteins in different clusters and the verification
# caught it.

set -euo pipefail

CONFIG="${1:?usage: sbatch cluster_mmseqs.sbatch <config.yaml>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs results

module load anaconda3 mmseqs2 2>/dev/null || true
source activate bindsite 2>/dev/null || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if ! command -v mmseqs >/dev/null 2>&1; then
  echo "mmseqs is not on PATH; load the module or conda install -c bioconda mmseqs2" >&2
  exit 4
fi

mmseqs version
python -m bindsite.cli cluster -c "$CONFIG" \
  > "results/clusters_${SLURM_JOB_ID:-local}.json"
