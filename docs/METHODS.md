# Methods

## Labelling binding residues

A binding residue is one with any heavy atom within **4.0 Å** of any heavy
atom of a curated ligand. The cutoff is a parameter, recorded in every output,
because the positive rate — and therefore every metric — depends on it.

Distances use **all** heavy atoms of the residue, not CA or CB. A lysine side
chain reaches ~6 Å from its CA, so a CA-based cutoff would make a residue's
label depend on its size rather than on its contacts.

### What counts as a ligand

This is the scientifically load-bearing decision in the whole pipeline. Most
PDB entries contain water, ions and crystallisation additives, and a naive
"any HETATM is a ligand" rule labels sulfate and glycerol contacts as binding
sites. Two complementary mechanisms are needed, because neither suffices:

**The Chemical Component Dictionary** identifies modified amino acids and
linking saccharides. A hand-written list cannot keep up: in a 500-protein
sample drawn from the PDB, `YOF` (3-fluorotyrosine) and `CGU`
(γ-carboxyglutamate) appear as HETATM records. Treating them as ligands makes
a protein's "binding site" the residues adjacent to its own backbone —
`1XDC` had **74 spurious binding residues** from 18 fluorotyrosines before
this check existed. `NAG`, `MAN`, `BMA` and `GAL` are N-glycans, covalently
attached to asparagine: a glycosylation site is a post-translational
modification, not a pocket.

**Curated exclusion lists** identify solvent, ions and additives. The CCD
cannot: water, sulfate, glycerol and zinc are all legitimately typed
`non-polymer`, so the CCD has no basis to distinguish a cryoprotectant from a
substrate. The lists follow the exclusion principle used by BioLiP (Yang, Roy
& Zhang, *Nucleic Acids Res* 41:D1096, 2013) and P2Rank (Krivák & Hoksza,
*J Cheminform* 10:39, 2018).

A ligand must also have at least **6 heavy atoms**: below that it is a
fragment or ion cluster that cannot define a pocket.

The rejection tally is reported, not discarded. On the shipped dataset, **464
of 900 candidate entries were rejected for having no curated ligand** — if
that number were small, the curation would not be doing anything.

### Further dataset filters

* single protein entity (labels are per chain);
* resolution better than 2.0 Å (a mis-placed ligand mislabels a pocket);
* 60–400 residues;
* at least 3 binding residues (fewer usually means a ligand at a crystal
  contact, which no structural feature can explain);
* binding fraction at most 50% (above that the "ligand" spans the chain).

Candidates are **randomly sampled** from a large pool, not taken in the order
RCSB returns them. The API's default ordering is effectively by PDB id, and
the first entries matching these filters are 102L, 103L, 107L, 108L (all T4
lysozyme) and 102M, 104M, 106M, 109M (all myoglobin) — taking the first N
would build a "curated dataset of diverse proteins" containing about four
distinct proteins.

---

## Features

Three blocks, kept separate because the model treats sequence and structure as
distinct modalities and the ablation needs to switch them independently.

**Sequence** — one-hot residue identity, 20 amino acids plus an unknown column.

**Physicochemical** — published per-residue scales, each cited in
`features.py`: Kyte & Doolittle hydropathy (*J Mol Biol* 157:105, 1982),
maximum accessible surface area (Tien et al., *PLoS One* 8:e80635, 2013),
side-chain volume (Zamyatnin, *Prog Biophys Mol Biol* 24:107, 1974), formal
charge, hydrogen-bond donor and acceptor counts, aromaticity, and glycine and
proline flags. Normalisation is by fixed published constants, never by dataset
statistics: using dataset means would give the same residue different features
in train and test.

**Backbone geometry** — neighbour count and a size-normalised burial measure,
depth from the centroid, side-chain direction relative to the outward radial,
local backbone curvature, CA–CA distances to sequence neighbours, and
proximity to the nearer terminus. Glycine gets a reconstructed virtual CB from
ideal tetrahedral geometry, so it is comparable to every other residue instead
of being a special case of zeros.

On real data these separate binding from non-binding residues in the
biophysically expected direction. Measured on streptavidin (PDB 1STP), the
strongest is side-chain direction: binding residues average **−0.478** and
non-binding **+0.208** — binding residues point their side chains *inward*,
into the cavity, which is what pocket geometry predicts.

### Features never see the ligand

A pocket is a concavity, so any geometric feature computed with the ligand
still present would partly encode the answer. The ligand is used to derive
labels and for nothing else. This is asserted, not assumed: a control strips
every HETATM record and checks that the features are bit-identical
(`max_geometric_difference == 0.0`).

---

## Homology separation

Binding-site predictors are routinely reported on random splits. The PDB is
enormously redundant — many entries are the same protein with a different
ligand — so a random split puts near-identical sequences in train and test,
and the reported number measures memorisation of protein families.

### Local alignment, not global

Identity is computed from **Smith–Waterman local** alignment with BLOSUM62 and
BLAST's default gap costs, divided by the length of the shorter sequence.

The alignment mode matters more than it looks, and getting it wrong is what
made a first attempt here fail outright. Measured over all 86,320 pairs of the
416-protein dataset:

| | global | local |
|---|---|---|
| median identity | 0.228 | **0.048** |
| 90th percentile | 0.303 | 0.101 |
| 95th percentile | 0.329 | 0.130 |
| 99th percentile | 0.928 | 0.928 |
| fraction of pairs ≥ 30% | **10.7%** | **1.4%** |

Under global alignment a 30% threshold admits a tenth of all pairs, because
forcing an end-to-end alignment of unrelated proteins finds matches
throughout. The transitive closure of that similarity graph merged the entire
dataset into **one cluster of 416 proteins**, leaving no family to hold out.
Under local alignment the distribution is bimodal — background at ~5%,
genuine homologs above 90% — the 30% threshold sits in a real gap, and the
clustering is stable across thresholds from 0.25 to 0.60 (178 to 233
clusters). BLAST and MMseqs2 both assess homology from local alignments, so
this is also what "30% identity" means in this literature.

### Connected components, not greedy centroids

A split is valid only if no train–test pair exceeds the threshold. That
property is exactly *"clusters are the connected components of the graph whose
edges join pairs above the threshold"*, so that is what the default backend
computes.

Greedy centroid clustering — the scheme CD-HIT and MMseqs2's linclust use —
compares each sequence only to a cluster **representative**, and fails in both
directions:

* it **over-merges**: A and B both within threshold of representative R join
  one cluster even when A and B are unrelated. Conservative for leakage,
  costly for split granularity.
* it **under-merges**: A can be 47% identical to B while below threshold to
  B's representative. This is the dangerous direction, and it happened here —
  greedy clustering placed `2VKF` and `1LFK`, **46.8% identical**, in
  different clusters, and the run was rejected by the leakage assertion.

Over-merging is retained and reported rather than hidden. The coherence
diagnostic measures it: on the shipped dataset **4 of 62 multi-member clusters
contain pairs below the 30% threshold**, minimum observed 0.071.

Cost is all pairs, O(n²) alignments — about 86,000 for 416 sequences, a few
minutes, cached. For 145,000 sequences it is impossible and MMseqs2 is
required, with the caveat that MMseqs2 has the same transitive-closure gap, so
its splits must still be verified directly.

### Verification, not trust

`assert_no_homology_leakage` runs two checks:

1. **Cluster disjointness** — no cluster id in more than one fold. Cheap, and
   catches assembly bugs.
2. **Direct identity** — the maximum pairwise identity between train and test
   sequences, read from the cached identity matrix. This does not trust the
   clustering, which is the point: it is what caught the 46.8% pair.

The matrix is cached as a separate artifact from the clustering, because it is
the expensive part and does not depend on the threshold. Without the cache the
verification costs O(n_train × n_test) alignments on every run — enough that
it would end up switched off, which defeats it.

### Three split levels

| level | what it splits | validity |
|---|---|---|
| `random_residue` | residues within each protein | the most severe leak available: neighbouring residues share nearly all features |
| `random_protein` | whole proteins, homology ignored | the leak most published numbers contain |
| `homology` | whole clusters | the only level at which a test protein is from an unseen family |

The first two are implemented **only** so the control experiment can measure
them. The pipeline refuses to run `random_residue`, and `random_protein`
carries a caveat in its own output.

---

## Model

A multimodal cross-attention transformer, ~286k parameters at the shipped
settings.

**Two encoders.** Sequence features and structure features are embedded
separately. Concatenating them would let the first linear layer mix them
immediately, making it impossible to ask what each modality contributes.

**Cross-attention between modalities.** Each stream attends to the other, so
structural context is interpreted in light of chemistry and vice versa. This
is what makes the model structure-*aware* rather than a sequence model with
extra columns: a hydrophobic residue in a concave pocket and the same residue
on a convex surface get different representations.

**Self-attention within each stream.** A binding site is a set of residues
close in space but often far apart in sequence. Attention over the whole chain
lets a residue see those partners; a convolutional or windowed model cannot,
which is the main architectural argument for attention here.

**Structural attention bias.** Raw attention has no notion of distance, so it
would treat a residue 3 Å away and one 40 Å away identically. A learned bias
from the CA–CA distance matrix is added to the attention logits. Distances are
expanded into radial basis functions first: a single scalar fed to a linear
layer could only produce a monotonic bias, but the useful relationship is not
monotonic — residues lining a pocket sit at a characteristic separation, while
both sequence neighbours and distant residues are less informative. Note that
a *uniform* distance matrix produces no effect at all, because a constant
added to every attention logit cancels in the softmax.

Both streams read the **previous** layer's other stream, so neither gets a
within-layer advantage from evaluation order.

### Class imbalance

Binding residues are ~8% of residues. The positive class is weighted in
`BCEWithLogitsLoss` by `negatives / positives`, computed from the **training
fold only** — computing it over the whole dataset would use the test fold's
label distribution. The weight is capped at 20: on a protein with 3 binding
residues out of 300 the raw weight is 99, and a model trained at that weight
predicts nearly everything positive.

### Two details that matter more than they look

**Padding is masked out of the loss, not just the attention.** Proteins have
different lengths, so batches are padded, and a padded position carries label
0 — the majority class. Including it trains the model on positions that do not
exist and reweights the loss by batch composition.

**Early stopping is on validation AUPRC, not loss.** Under imbalance the loss
is dominated by the negative class, so it can improve while the ranking of
positives gets worse. The weights from the best-AUPRC epoch are restored, not
the last epoch's.

---

## Evaluation

### AUROC is not reported alone

Binding residues are ~8% of a protein. ROC curves use the false positive rate,
whose denominator is the large negative class, so a model can produce many
more false positives than true positives and still show a high AUROC. On a
simulated 10%-positive task with a modest real signal, AUROC reaches 0.79
while the best achievable F1 is 0.40 at precision 0.36.

AUROC is computed for comparability with the literature, and it is never
printed without AUPRC, the positive rate, the AUPRC lift over chance, and
precision in the top L/10. Model comparison ranks by **AUPRC**, because under
imbalance AUROC compresses exactly the differences that matter.

**Precision and recall at top L/10** is the practitioner's metric: given a
protein of length L, take the L/10 highest-scoring residues and ask how many
are truly binding. Unlike a fixed threshold it adapts to protein size, and
unlike AUROC it reflects what a short list actually contains.

### Two levels

**Residue level, pooled.** All residues from all test proteins in one pool.
This is what "0.82 AUROC" normally means. It is dominated by large proteins
and by proteins with many binding residues.

**Protein level.** The metric is computed per protein and averaged, so each
protein counts once. This is what matters in use, and it is usually lower and
more variable. Proteins with fewer than 20 residues, or with no positives, are
excluded and **listed** rather than averaged in. The gap between the two
levels is reported as `pooled_minus_per_protein_auprc`.

Undefined metrics are `None`, never a substituted value: one-vs-rest AUROC has
no meaning when a class has no positive example.

### Baselines

| baseline | what it establishes |
|---|---|
| `prevalence` | The AUPRC scale. AUROC is 0.5 by construction and AUPRC equals the positive rate. |
| `burial` | One hand-picked geometric feature. Any learned model must beat it. |
| `logistic` | Local chemistry only, no context. Isolates how much of the signal is per-residue. |
| `random_forest` | Non-linear on the same features. Separates "needs non-linearity" from "needs context". |

None can see other residues, which is the capability the transformer adds.
Comparing against them is how that capability gets measured instead of
asserted.

---

## Exit codes

| code | meaning |
|---|---|
| 0 | success |
| 2 | bad input or invalid configuration |
| 3 | a required optional dependency is missing |
| 4 | an external tool (MMseqs2) is unavailable or failed |
| 5 | **data leakage detected** |

Code 5 is distinct because a leaking split is not a crash and not a bad
config: it is a scientifically invalid run, and a caller in a batch script
should be able to tell that apart without parsing a message.

Status messages go to **stderr** and JSON to **stdout**, so
`bindsite run -c config.yaml | jq .comparison` works.
