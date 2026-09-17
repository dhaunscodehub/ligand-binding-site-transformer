# Data

**No structures are committed to this repository.** They are downloaded on
demand from RCSB into `data/pdb/`, which is gitignored. This document states
exactly what is fetched and under what terms.

Two small derived artifacts **are** committed, so a run is auditable without
network access:

| File | What it is |
|---|---|
| `dataset_manifest.json` | The 416 PDB ids in the dataset, each with its curated ligands, residue count and binding-residue count. |
| `clusters.json` | The homology cluster assignment, with the identity threshold, backend, linkage and coherence diagnostic. |
| `chemcomp_cache.json` | Chemical Component Dictionary types for the 422 het components encountered, so curation is reproducible offline. |

`identity_matrix.npz` (~700 KB of floats) is **not** committed: it is
regenerable from the sequences and is rebuilt automatically on first use.

## Source

| | |
|---|---|
| Origin | RCSB Protein Data Bank |
| Search API | `https://search.rcsb.org/rcsbsearch/v2/query` |
| Coordinates | `https://files.rcsb.org/download/{id}.pdb.gz` |
| Component types | `https://data.rcsb.org/rest/v1/core/chemcomp/{id}` |
| Terms | PDB data are released into the public domain (CC0). Cite the individual structures and wwPDB (Berman et al., *Nucleic Acids Res* 28:235, 2000). |

Selection filters, all recorded in the manifest:

* exactly one protein entity;
* at least one non-polymer entity;
* resolution better than 2.0 Å;
* 60–400 deposited polymer residues.

**42,019 PDB entries** match. 900 are sampled at random (seeded) from a
9,000-entry pool, and 416 survive curation.

Candidates are sampled rather than taken in API order because that order is
effectively by PDB id: the first matches are 102L, 103L, 107L, 108L (T4
lysozyme) and 102M, 104M, 106M, 109M (myoglobin). Taking the first N would
give a "diverse" dataset of about four proteins.

## The shipped dataset

| | |
|---|---|
| Proteins | 416 |
| Residues | 91,692 |
| Binding residues | 7,153 |
| Positive rate | **7.80%** |
| Distinct ligands | 307 |
| Median chain length | 223 (range 62–396) |
| Contact cutoff | 4.0 Å |
| Homology clusters at 30% local identity | 199 (redundancy 52.2%) |

Most common ligands: HEM (74), FMN (20), NAP (14), AMP (13), ADP (10), SF4
(10), HEC (9), FAD (9), PDC (9), NAD (7) — cofactors and nucleotides, which is
what a resolution-filtered PDB sample of single-chain complexes contains.

### Curation attrition

Of 900 attempted entries, **484 were rejected**:

| reason | count |
|---|---|
| no curated ligand (only water, ions, additives, glycans or modified residues) | 464 |
| unreadable coordinates | 11 |
| fewer than 3 binding residues (ligand at a crystal contact) | 5 |
| chain length outside 60–400 | 2 |
| binding fraction above 50% (ligand spans the chain) | 2 |

(464 + 11 + 5 + 2 + 2 = 484.)

That 464 is the number to look at. Without curation those entries would have
contributed "binding sites" derived from sulfate, glycerol, N-glycans and
modified amino acids. See [../docs/METHODS.md](../docs/METHODS.md) for the two
mechanisms and the specific cases (`YOF`, `CGU`, `NAG`) that motivated them.

## Reproducing the dataset

```bash
bindsite build-dataset -c configs/homology_split.yaml
```

Downloads ~900 structures (~40 MB) and takes a few minutes. The committed
manifest lets you rebuild the exact same dataset:

```yaml
data:
  source: manifest
  manifest: ../data/dataset_manifest.json
```

Labels and features are always **re-derived** from the structures rather than
read from stored arrays: if the contact cutoff or a curation rule changes, a
manifest built under the old rules must not silently supply old labels.

## Benchmarks not used here

The resume this repository accompanies cites **COACH420** and **HOLO4K**.
Neither is used, and no number here should be compared to a published result
on them:

* they are distributed as curated residue-level annotation sets whose
  definition of a binding residue differs from the 4.0 Å all-heavy-atom rule
  used here, so metrics are not comparable without adopting their definition;
* a meaningful comparison needs a model trained at the scale those benchmarks
  assume, which is not what runs here.

See [../docs/VALIDATION.md](../docs/VALIDATION.md) for what is and is not
reproduced.

## Bringing your own data

```yaml
data:
  source: ids
  pdb_ids: [1STP, 4HHB, 1ATP]
  cache_dir: ../data/pdb
```

Requirements:

* PDB-format coordinates with standard `ATOM`/`HETATM` records;
* at least **3 homology clusters** after clustering, or a homology-separated
  split is impossible and the code refuses rather than degenerating into a
  random split;
* enough proteins that clustering leaves usable folds — the built-in exact
  clustering is O(n²) alignments, so beyond a few thousand sequences install
  MMseqs2 and set `homology.backend: mmseqs` (and keep
  `split.verify_identity: true`, because MMseqs2 clustering has the same
  transitive-closure gap described in the methods).
