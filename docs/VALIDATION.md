# Validation

Every quantitative value is tagged with how it was obtained:

| Tag | Meaning |
|---|---|
| `VERIFIED_REPRODUCED` | Computed by running this code, in the environment recorded in the run's `provenance` block. |
| `VERIFIED_FROM_EXISTING_ARTIFACT` | Read from a committed result file produced by an earlier run of this code. |
| `NOT_REPRODUCED` | Not executed here. Stated as a target or a published figure, never as a result of this work. |
| `REFERENCE_RESULT` | Published by someone else, quoted for comparison. |

Nothing is reported without one of these tags. Where a number could not be
produced, the reason is given instead of the number.

---

## 1. The headline result: homology leakage on real data

Two runs, on the same 416 proteins, with the same model, seed and
hyperparameters. The configs differ in **exactly one line**,
`split.strategy`.

All `VERIFIED_REPRODUCED`. Committed:
[`results/reference/homology_split.json`](../results/reference/homology_split.json),
[`results/reference/random_split.json`](../results/reference/random_split.json).

| model | homology-separated AUPRC | random-protein AUPRC | ratio | honest AUROC | leaky AUROC |
|---|---|---|---|---|---|
| `prevalence` | 0.0699 | 0.0820 | 1.17× | 0.5000 | 0.5000 |
| `burial` | 0.0899 | 0.1185 | 1.32× | 0.6168 | 0.6522 |
| `logistic` | 0.1927 | 0.2321 | 1.20× | 0.7650 | 0.7667 |
| `random_forest` | 0.2213 | 0.5320 | **2.40×** | 0.7770 | 0.8720 |
| **`transformer`** | **0.2511** | **0.5580** | **2.22×** | **0.7809** | **0.8762** |

**How to read the `prevalence` row.** It is the normaliser. A constant
predictor cannot exploit leakage at all, so its 1.17× ratio is pure test-set
composition: the two splits produce test sets with positive rates of 6.99% and
8.20%. Any ratio near 1.17× is explained by that alone.

`logistic` sits at 1.20× — essentially nothing beyond the rate difference. The
forest and the transformer sit at **2.40× and 2.22×**. The leakage is almost
entirely **capacity-dependent**: a model that can memorise family-specific
structure exploits the redundancy, and a linear model cannot.

This is why measuring leakage with a weak model understates the risk for the
models people actually deploy.

The mechanism is explicit in the output: the random split places **33 of 199**
homology clusters in both train and test.

### The same effect, isolated in a controlled experiment

`results/reference/controls.json`, control 4, using the random forest:

| split level | AUPRC | AUROC | lift | inflation vs homology |
|---|---|---|---|---|
| `random_residue` | 0.5083 | 0.8706 | 6.59× | **+0.2648** |
| `random_protein` | 0.5405 | 0.8779 | 6.63× | **+0.2970** |
| `homology_cluster` | 0.2435 | 0.7879 | 3.44× | — |

The two leaking arms are not strictly ordered against each other — the
residue split trains on a fraction of every protein, the protein split on
whole near-duplicates — and both roughly double the honest figure.

---

## 2. Control experiments

```bash
python scripts/validate_controls.py -c configs/homology_split.yaml --json results/controls.json
```

**7/7 passing**, all `VERIFIED_REPRODUCED`. Committed:
[`results/reference/controls.json`](../results/reference/controls.json).

| # | Control | Measured | Reference | Verdict |
|---|---|---|---|---|
| 1 | Positive control: label is a known function of the features | AUPRC 0.9705 | chance 0.0793 | PASS |
| 2 | AUROC overstates a mediocre predictor under imbalance | AUPRC 0.3506 at AUROC 0.795 | chance 0.1017 | PASS |
| 3 | Label shuffle (negative control) | AUPRC 0.0772, mean AUROC 0.5106 | rate 0.0709, chance 0.5 | PASS |
| 4 | Leakage by split level | +0.2648 / +0.2970 inflation | — | PASS |
| 5 | Leakage guard rejects a corrupted split | raises `LeakageError` | — | PASS |
| 6 | Split verified directly against the sequences | max identity 0.2672 | threshold 0.30 | PASS |
| 7 | Features identical with and without the ligand | **0.0e+00** difference | exact | PASS |

### Control 3 is replicated

A single label shuffle is a noisy estimate of chance: on ~1800 test residues
with ~150 positives the AUROC standard error is near 0.024, and individual
seeds were measured from −0.063 to +0.038 during development. The assertion is
therefore on the **mean over five seeds**, with the spread reported:

```
mean AUROC 0.5106  sd 0.0112  over seeds [0, 1, 2, 3, 4]
per seed:  [0.5124, 0.5042, 0.5011, 0.5318, 0.5037]
```

The tolerance (0.03) was **not** chosen to make a measurement pass; the
observed excess is +0.0106.

### Control 7 is exact, not approximate

Features computed with the ligand present and with every `HETATM` record
stripped are **bit-identical** (`max_geometric_difference == 0.0`). A pocket is
a concavity, so a geometric feature computed with the ligand in place would
partly encode the answer. The labels, and only the labels, differ.

---

## 3. Reference validation: streptavidin

`VERIFIED_REPRODUCED`. PDB 1STP, biotin, 4.0 Å cutoff, no fitting — geometry
and ligand curation only:

```
N23 L25 S27 Y43 S45 V47 G48 N49 W79 A86 S88 T90 W92 W108 L110 D128
```

16 of 121 residues (13.2%). This recovers the canonical biotin-binding site:
the hydrogen-bonding set **N23, S27, Y43, S45, N49, S88, T90, D128** and the
hydrophobic box **W79, W92, W108** (Weber et al., *Science* 243:85, 1989 —
`REFERENCE_RESULT`). W120 is correctly absent: it belongs to the adjacent
subunit, which is not in this chain.

The negative case is equally informative. PDB 4ZQK (PD-1/PD-L1, a
protein–protein complex) contains only water and a sodium ion, and is
**refused** with exit code 2 rather than labelled.

---

## 4. Alignment mode: a methodological finding

`VERIFIED_REPRODUCED`. Committed:
[`results/reference/alignment_mode_comparison.json`](../results/reference/alignment_mode_comparison.json).

All 86,320 pairs of the 416-protein dataset, identity relative to the shorter
sequence:

| percentile | global alignment | local alignment |
|---|---|---|
| p50 | 0.228 | **0.048** |
| p75 | 0.263 | 0.071 |
| p90 | 0.303 | 0.101 |
| p95 | 0.329 | 0.130 |
| p99 | 0.928 | 0.928 |
| **fraction ≥ 30%** | **10.7%** | **1.4%** |

Under **global** alignment a 30% threshold admits a tenth of all pairs, because
forcing an end-to-end alignment of unrelated proteins finds matches
throughout. The transitive closure of that graph merged the whole dataset into
**one cluster of 416 proteins**, so no family could be held out and the split
was impossible.

Under **local** alignment the distribution is bimodal — background near 5%,
genuine homologs above 90% — the threshold sits in a real gap, and clustering
is stable across thresholds:

| threshold | clusters | largest | fraction |
|---|---|---|---|
| 0.25 | 178 | 34 | 8.2% |
| **0.30** | **199** | **32** | **7.7%** |
| 0.40 | 218 | 32 | 7.7% |
| 0.60 | 233 | 29 | 7.0% |

BLAST and MMseqs2 both assess homology from local alignments, so this is also
what "30% identity" means in the binding-site literature.

---

## 5. Test suite

```bash
python -m pytest tests/
```

**311 tests, all passing** (`VERIFIED_REPRODUCED`):

| Module | Tests | Focus |
|---|---|---|
| `test_structure.py` | 38 | Parsing, altloc resolution, ligand curation, geometric labelling |
| `test_homology.py` | 33 | Local vs global identity, the connected-components invariant |
| `test_config.py` | 45 | Rejection of every malformed input |
| `test_models.py` | 33 | Baselines, transformer, ablations, padding invariance |
| `test_evaluate.py` | 28 | AUROC/AUPRC under imbalance, two-level metrics |
| `test_splits.py` | 24 | Cluster disjointness, direct sequence verification |
| `test_features.py` | 20 | Feature blocks, virtual CB, ligand-blindness |
| `test_train.py` | 24 | Class weighting, padding masking, early stopping |
| `test_chemcomp.py` | 17 | CCD classification, cache round-trip |
| `test_cli.py` | 17 | Exit codes, stdout/stderr separation |
| `test_validation.py` | 16 | The controls behave as controls |
| `test_dataset.py` | 16 | Curation attrition, manifest consistency |

Smoke test: **43/43 checks**, no network, under a minute.

---

## 6. What was not reproduced

The resume this repository accompanies makes four quantitative claims about
this project. Here is the status of each.

| Resume claim | Status | Why |
|---|---|---|
| **0.82 AUROC on COACH420 and HOLO4K** | `NOT_REPRODUCED` | Neither benchmark is used. Their residue-level annotations define a binding residue differently from the 4.0 Å all-heavy-atom rule here, so metrics are not comparable without adopting their definition, and a meaningful comparison needs training at a scale not available here. See below for what *was* measured. |
| **145K+ diverse proteins** | `NOT_REPRODUCED` | 416 proteins were curated. The dataset builder scales — 42,019 PDB entries match the filters — but the exact O(n²) clustering does not, and training at that scale needs GPU HPC. |
| **47.6M annotated residues** | `NOT_REPRODUCED` | 91,692 residues, 7,153 of them binding. |
| **3.7K+ protein families via MMseqs2 at 30% identity** | `PARTIALLY_REPRODUCED` | 199 homology clusters at 30% local identity, by exact connected-components clustering. MMseqs2 itself is **not installed**: the integration builds and documents the command but it has never been executed here. |

### On the 0.82 AUROC claim specifically

It is worth stating plainly what this repository did measure, because it
brackets that number:

* homology-separated, the transformer reaches **AUROC 0.781**;
* on a random protein split — the same data and model — it reaches **0.876**.

0.82 falls between the two. An AUROC in that range is clearly attainable on
this task; what the number does not convey is that at 7% positives the
homology-separated model has AUPRC 0.251, best F1 0.311, and precision 0.262
in its top L/10 predictions. That is the reason this repository never reports
AUROC alone.

### Other limitations

| Component | Status | Detail |
|---|---|---|
| MMseqs2 clustering | `IMPLEMENTED_NOT_FULLY_EXECUTED` | Not on PATH here. The command is constructed and documented; the exact fallback runs instead and is what produced every number above. |
| Distributed / Dockerised GPU training | `NOT_REPRODUCED` | Training ran single-device on Apple MPS (392 s). SLURM scripts are provided and **have not been executed**. |
| Training at the data scale the model implies | `NOT_REPRODUCED` | 1.68M parameters on 298 training proteins overfits by epoch 13 of 25 (train loss 0.76 falling, validation loss 1.01 rising). The model is limited by data here, not architecture — which is itself a finding, and the reason the reported numbers should not be read as an architecture ceiling. |

### Implementation status

`IMPLEMENTED_AND_TESTED` for the dataset, curation, homology separation,
features, baselines, evaluation and controls — all execute end to end on real
data and are covered by 311 tests.

`IMPLEMENTED_NOT_FULLY_EXECUTED` for MMseqs2 clustering and for
distributed/HPC training.

---

## 7. Bugs found by validation

Recorded because the controls and the assertions earned their place by
catching these.

| # | Bug | Found by | Fix |
|---|---|---|---|
| 1 | **Modified amino acids treated as ligands.** `YOF` (3-fluorotyrosine) and `CGU` (γ-carboxyglutamate) are HETATM records. `1XDC` had **74 spurious binding residues** from 18 fluorotyrosines — residues adjacent to its own backbone. | Inspecting the ligand-frequency table of the built dataset | Chemical Component Dictionary lookup; anything `*peptide linking` is polymer. 84 of 500 proteins were removed. |
| 2 | **N-glycans treated as ligands.** `NAG`, `MAN`, `GAL` are covalently attached to asparagine — a modification, not a pocket. | The same inspection | CCD `*saccharide*linking* `excluded; 21 proteins had been labelled from glycans alone. |
| 3 | **Global alignment made a homology split impossible.** At 30%, global identity admitted 10.7% of all pairs and the transitive closure merged all 416 proteins into one cluster. | The split refusing to build | Switched to local (Smith–Waterman) alignment, matching BLAST/MMseqs2 convention. Quantified in §4. |
| 4 | **Greedy clustering left a 46.8%-identical pair in different folds.** `2VKF` and `1LFK`. Centroid clustering guarantees similarity only to a representative, so it under-merges as well as over-merges. | The direct identity verification — which exists precisely because it does not trust the clustering | Connected-components (transitive closure) clustering made the default. |
| 5 | **Alternate locations were appended, not replaced.** The higher-occupancy variant was added alongside the lower one, double-counting the atom in every distance and neighbour-count calculation. | A parser test on a two-altloc residue | Atoms keyed by name, coordinates materialised once. Measured effect on the real dataset: 3 proteins (`1MUN`, `2AMB`, `37BV`) each lost one spurious binding residue, 7156 → 7153. |
| 6 | **`-v` progress output went to stdout**, breaking `bindsite run \| jq`. | Reading the JSON artifact of the first full run | All progress routed to stderr, pinned by a CLI test. |
| 7 | **Clustering cache lost `linkage` and `n_edges`** on round-trip, so a cached run reported the dataclass default "centroid" — misdescribing how the clustering was produced. | Comparing a fresh run's output against a cached one | Fields persisted and the backend added to the cache-validity check. |
| 8 | **`n_heads=0` raised `ZeroDivisionError`** instead of a config error, because the divisibility check ran before the positivity check. | A parametrised config-rejection test | Validation reordered. |
| 9 | **`ComponentCache` was silently discarded.** `cache or ComponentCache()` — the class defines `__len__`, so an empty cache is falsy and every call built a throwaway, refetching each component and accumulating nothing. | A cache-population assertion that reported 0 entries after 14 successful lookups | `is None` check. |
| 10 | **A `random_residue` config downloaded the whole dataset before refusing.** The pipeline cannot run a residue-level split, but the refusal came after dataset construction — several minutes of downloading before the error. | Timing the adversarial exit-code checks in validation pass 3 | Moved to `SplitConfig.__post_init__`; now refused in 0.2 ms, with a pointer to the control experiment where residue-level splitting *is* implemented. |
| 11 | **Dead `isfinite` branch in the JSON encoder** (inherited pattern). `np.float64` is a `float` subclass, so the C encoder handles it and `allow_nan=False` raises before `default()` is consulted. | A serialisation smoke check | Non-finite handling moved to a pre-dump `sanitise()` walk. |

Items 3 and 4 are the ones worth emphasising. Both were caught by assertions
that deliberately do not trust the layer beneath them — the split verification
checks sequences rather than cluster ids, and it rejected a split that every
cluster-level check had passed.

### A note on a warning that is *not* a bug

On numpy 2.2.x on Apple Silicon, scikit-learn's logistic solver emits
`divide by zero / overflow / invalid value encountered in matmul`. These are
spurious — a matmul performs no division. They were investigated rather than
ignored: they are not caused by constant feature columns (they persist after
those are dropped) nor by array layout (`ascontiguousarray` makes no
difference), they are absent on numpy 2.5.x, and the results are identical
under both versions (same AUPRC to four decimals, finite coefficients with
maximum magnitude 1.061). The environment specs ask for `numpy>=2.3`.
