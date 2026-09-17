# Project state

**Status:** `IMPLEMENTED_AND_TESTED` for everything that runs here;
`IMPLEMENTED_NOT_FULLY_EXECUTED` for MMseqs2 clustering and HPC training.

The dataset, curation, homology separation, features, baselines, the
transformer, evaluation and the control experiments all execute end to end on
real RCSB data and are covered by 312 passing tests. The transformer is
genuinely trained (Apple MPS, 392 s) and evaluated, not merely implemented.

## What runs

| Component | Status | Evidence |
|---|---|---|
| RCSB search, download, curation | `IMPLEMENTED_AND_TESTED` | 416 proteins curated from 900 candidates; 16 tests |
| CCD-based ligand classification | `IMPLEMENTED_AND_TESTED` | 422 components cached, 0 lookup failures; 17 tests |
| Geometric binding-residue labelling | `IMPLEMENTED_AND_TESTED` | streptavidin reference recovered; 38 tests |
| Features (sequence, physicochemical, geometry) | `IMPLEMENTED_AND_TESTED` | ligand-blindness exact to 0.0; 20 tests |
| Local-alignment identity + connected components | `IMPLEMENTED_AND_TESTED` | invariant verified over 86,320 pairs; 33 tests |
| Homology-separated splits + leakage guards | `IMPLEMENTED_AND_TESTED` | max train–test identity 0.265; 24 tests |
| Baselines (4) | `IMPLEMENTED_AND_TESTED` | 33 tests |
| Cross-attention transformer | `IMPLEMENTED_AND_TESTED` | trained and evaluated; 33 + 24 tests |
| Two-level evaluation, AUPRC-first | `IMPLEMENTED_AND_TESTED` | 28 tests |
| Control experiments | `IMPLEMENTED_AND_TESTED` | 7/7 passing |
| MMseqs2 clustering | `IMPLEMENTED_NOT_FULLY_EXECUTED` | not on PATH; command built and documented |
| Distributed / Dockerised GPU training | `NOT_REPRODUCED` | SLURM scripts provided, never executed |

## Headline measurements

All `VERIFIED_REPRODUCED`; see [docs/VALIDATION.md](docs/VALIDATION.md).

* **Homology leakage is capacity-dependent.** Random-protein vs
  homology-separated AUPRC: transformer 0.558 vs 0.251 (**2.22×**), forest
  0.532 vs 0.221 (2.40×), logistic 0.232 vs 0.193 (1.20×), prevalence 0.082
  vs 0.070 (1.17×, pure test-set composition). Same data, same model, one
  config line different.
* **Honest performance**: transformer AUPRC 0.251 (3.59× chance), AUROC 0.781,
  best F1 0.311, precision 0.262 at top L/10.
* **Global alignment breaks the task**: at a 30% threshold it admits 10.7% of
  all pairs and merges the whole dataset into one cluster; local alignment
  admits 1.4%.
* **Curation removes 464 of 900 candidates** for having no genuine ligand.

## Known limitations

1. **Scale.** 416 proteins and 91,692 residues, against the resume's claimed
   145K proteins and 47.6M residues. The dataset builder scales (42,019 PDB
   entries match the filters) but the exact O(n²) clustering does not, and
   training at that scale needs GPU HPC. This is the largest gap.

2. **The transformer is data-limited, not architecture-limited.** 1.68M
   parameters on 298 training proteins overfit by epoch 13 of 25. The reported
   numbers should not be read as a ceiling for the architecture — they are a
   ceiling for the architecture *at this data scale*.

3. **COACH420 and HOLO4K are not used**, so the resume's 0.82 AUROC is not
   reproduced and no number here is comparable to a published result on them.
   Their binding-residue definitions differ from the 4.0 Å all-heavy-atom rule
   used here. What can be said: homology-separated gives AUROC 0.781 and a
   random split gives 0.876, so 0.82 falls between the two.

4. **MMseqs2 has never been run here.** Every clustering number comes from the
   built-in exact backend. Note that MMseqs2's own clustering shares the
   transitive-closure gap that made greedy clustering leak, so splits built
   from it must still be verified directly.

5. **Connected-components clustering over-merges.** 4 of 62 multi-member
   clusters contain pairs below the 30% threshold (minimum 0.071). This is the
   conservative direction — over-merging never splits a homologous pair — but
   it coarsens the split: the largest cluster holds 7.7% of proteins, so
   realised fold fractions deviate from those requested (0.20 requested,
   0.135 realised for test).

6. **One dataset, one seed for the main comparison.** The leakage result is a
   single split per arm. The synthetic control replicates over five seeds; the
   real-data comparison does not.

## Next steps, in order of value

1. **Install MMseqs2 and scale the dataset to 10–50K proteins.** Everything
   needed is implemented; this is a compute and tooling problem, not a code
   one, and it would address limitations 1, 2 and 4 together. Keep
   `split.verify_identity: true` — the verification is what caught the
   46.8%-identical pair that clustering missed.
2. **Repeat the leakage comparison over several splits** to put an interval on
   the 2.22× figure rather than a point estimate.
3. **Adopt COACH420/HOLO4K's own binding-residue definition** as an
   alternative labelling mode, which would make a direct comparison to
   published numbers possible.
4. **Add a sequence-only and a structure-only ablation to the reported
   results.** The architecture and tests already support both; only the
   reporting is missing.

## Reproducing

```bash
pip install -e ".[all]"
python scripts/smoke_test.py                                      # 43/43
python -m pytest tests/                                           # 312 passed
python scripts/validate_controls.py -c configs/homology_split.yaml  # 7/7
bindsite run -c configs/homology_split.yaml -v
bindsite run -c configs/random_split.yaml -v
```

First run downloads ~900 structures (~40 MB) and computes an 86,320-pair
identity matrix (~4 minutes, cached thereafter). Committed reference outputs
are in `results/reference/`, and every run records its own `provenance` block
with package versions.

Environment for the reported numbers: Python 3.14, numpy 2.2.6, torch 2.14.0
(Apple MPS), scikit-learn, Biopython 1.88, macOS arm64. Note `numpy>=2.3` is
requested in the environment specs — 2.2.x emits spurious matmul warnings that
do not affect results (documented in
[docs/VALIDATION.md §7](docs/VALIDATION.md)).
