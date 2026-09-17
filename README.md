# Ligand-binding site prediction with homology-separated evaluation

A structure-aware cross-attention transformer that predicts which residues of
a protein contact a ligand — built around the measurement that decides whether
any such number means anything: **is the reported accuracy a property of the
biology, or of protein families the model has already seen?**

The central result is a controlled comparison, not a leaderboard entry. Two
runs on the same 416 proteins, same model, same seed, configs differing in
**exactly one line**:

| | homology-separated | random protein split | ratio |
|---|---|---|---|
| **transformer AUPRC** | **0.251** | **0.558** | **2.22×** |
| transformer AUROC | 0.781 | 0.876 | |
| `logistic` AUPRC | 0.193 | 0.232 | 1.20× |
| `prevalence` AUPRC | 0.070 | 0.082 | 1.17× |

A constant predictor cannot exploit leakage, so its 1.17× ratio is pure
test-set composition — and the linear model, at 1.20×, gains essentially
nothing beyond that. The transformer gains **2.22×**. Leakage here is almost
entirely **capacity-dependent**: models able to memorise family structure
exploit the PDB's redundancy, and weak models cannot. Measuring leakage with a
weak model understates the risk for the models people deploy.

The mechanism is not inferred — the random split puts **33 of 199** homology
clusters in both train and test, and says so in its own output.

```bash
pip install -e ".[all]"
python scripts/smoke_test.py                  # 43 checks, no network
python scripts/validate_controls.py           # the control experiments
bindsite label 1STP                           # streptavidin's biotin pocket
bindsite run -c configs/homology_split.yaml | jq .comparison
```

---

## Status

| | |
|---|---|
| **Implementation** | `IMPLEMENTED_AND_TESTED` (dataset, curation, homology separation, features, models, evaluation, controls) |
| **Tests** | 311 passing |
| **Smoke test** | 43/43 |
| **Controls** | 7/7 |
| **Dataset** | 416 proteins, 91,692 residues, 7.80% positive rate, 307 distinct ligands — all real, curated from RCSB |
| **Not reproduced** | The resume's 0.82 AUROC on COACH420/HOLO4K, its 145K-protein / 47.6M-residue scale, and MMseqs2 clustering. See [docs/VALIDATION.md §6](docs/VALIDATION.md). |

Every number carries a provenance tag (`VERIFIED_REPRODUCED`,
`NOT_REPRODUCED`, `REFERENCE_RESULT`).

---

## Results

Homology-separated, 56 held-out test proteins from families the model never
saw. Test positive rate 6.99%, so AUPRC chance is 0.070.

| model | AUPRC | lift | AUROC | best F1 | P@L/10 | per-protein AUPRC |
|---|---|---|---|---|---|---|
| `prevalence` | 0.070 | 1.00× | 0.500 | 0.131 | 0.064 | 0.079 |
| `burial` | 0.090 | 1.29× | 0.617 | 0.171 | 0.075 | 0.114 |
| `logistic` | 0.193 | 2.76× | 0.765 | 0.276 | 0.218 | 0.315 |
| `random_forest` | 0.221 | 3.17× | 0.777 | 0.292 | 0.244 | 0.308 |
| **`transformer`** | **0.251** | **3.59×** | **0.781** | **0.311** | **0.262** | **0.312** |

The ordering is the point: the transformer beats a non-linear model on the
same per-residue features, which beats a linear one, which beats one
hand-picked geometric feature, which beats predicting the prevalence. None of
the baselines can see other residues — that is the capability attention adds,
and this is how it gets measured rather than asserted.

### AUROC is not reported alone

At 7% positives, ROC's false-positive-rate denominator is the large negative
class, so many false positives barely move it. The transformer's AUROC of
0.781 comes with best F1 0.311 and precision 0.262 in its top L/10
predictions. A control demonstrates the general case: a simulated predictor
reaches AUROC 0.795 while its best achievable F1 is 0.400 at precision 0.358.

Model comparison therefore ranks by **AUPRC**, always printed with its chance
level and lift.

### Reference validation

`bindsite label 1STP` recovers streptavidin's canonical biotin site with no
fitting at all — geometry plus ligand curation:

```
N23 L25 S27 Y43 S45 V47 G48 N49 W79 A86 S88 T90 W92 W108 L110 D128
```

That is the hydrogen-bonding set N23, S27, Y43, S45, N49, S88, T90, D128 plus
the hydrophobic box W79, W92, W108 (Weber et al., *Science* 243:85, 1989).
W120 is correctly absent — it belongs to the adjacent subunit.

And `bindsite label 4ZQK` **refuses**: PD-1/PD-L1 contains only water and a
sodium ion, which are not binding sites.

---

## How leakage is prevented

**Connected-components clustering, not greedy centroids.** A split is valid
only if no train–test pair exceeds the identity threshold — which is exactly
"clusters are the connected components of the above-threshold graph". Greedy
centroid clustering (what CD-HIT and MMseqs2's linclust do) compares each
sequence only to a representative, so it both over-merges *and* under-merges.
Here it placed `2VKF` and `1LFK` — **46.8% identical** — in different
clusters.

**Verification against the sequences, not the clustering.** That 46.8% pair
was caught because the assertion reads a cached pairwise-identity matrix
rather than trusting cluster ids. It runs on every split (max train–test
identity 0.265 against a 0.30 threshold, over 16,688 pairs).

**Local alignment, not global.** This one decided whether the project worked
at all. Measured over all 86,320 pairs, global-alignment identity has median
0.228 and admits **10.7%** of pairs at a 30% threshold — enough that the
transitive closure merged all 416 proteins into **one cluster**, leaving no
family to hold out. Local (Smith–Waterman) alignment gives median 0.048, only
1.4% above threshold, and a bimodal distribution with genuine homologs above
90%. BLAST and MMseqs2 both use local alignment, so this is also what "30%
identity" means in this literature.

**Curation that decides what a ligand is.** 464 of 900 candidate entries were
rejected for containing only water, ions, additives, glycans or modified
residues. Two mechanisms are needed: the PDB Chemical Component Dictionary
catches modified amino acids and glycans (`1XDC` had **74 spurious binding
residues** from its own 18 fluorotyrosines), while curated exclusion lists
catch solvent and cryoprotectants — which the CCD types as `non-polymer` and
cannot distinguish from substrates.

**A distinct exit code.** Detected leakage exits **5**, separate from bad
input (2) or a crash. A leaking split is not an error; it is a scientifically
invalid run.

**Features that never see the ligand.** A pocket is a concavity, so a
geometric feature computed with the ligand present would encode the answer. A
control strips every `HETATM` record and asserts the features are
bit-identical — measured difference exactly `0.0`.

---

## The model

~1.7M parameters. Two encoders (sequence+physicochemistry, backbone geometry),
cross-attention between them, self-attention within each, and a learned
attention bias from the CA–CA distance matrix.

Cross-attention is what makes it structure-*aware* rather than a sequence
model with extra columns: a hydrophobic residue in a concave pocket and the
same residue on a convex surface get different representations. Self-attention
matters because a binding site is a set of residues close in space but often
far apart in sequence.

The distance bias expands distances into radial basis functions before the
linear map — a scalar fed to a linear layer could only produce a monotonic
bias, but the useful relationship is not monotonic. (A *uniform* distance
matrix correctly has no effect: a constant added to every attention logit
cancels in the softmax.)

Binding residues are ~8% of residues, so the positive class is weighted in the
loss by `negatives/positives` from the **training fold only**, capped at 20.
Padding is masked out of the loss, not just the attention — a padded position
carries label 0, the majority class. Early stopping is on validation **AUPRC**,
not loss, because under imbalance the loss is dominated by the negative class
and can improve while the ranking of positives degrades.

**It overfits here, and that is reported rather than hidden**: 1.68M
parameters on 298 training proteins peak at epoch 13 of 25, with training loss
falling to 0.76 while validation loss rises to 1.01. The model is limited by
data at this scale, not architecture.

---

## Commands

```bash
bindsite run           -c configs/homology_split.yaml   # full pipeline
bindsite build-dataset -c configs/homology_split.yaml   # curate, write manifest
bindsite cluster       -c configs/homology_split.yaml   # redundancy + coherence
bindsite label 1STP                                     # one structure
bindsite validate                                       # control experiments
```

Status messages go to **stderr**, JSON to **stdout**, so
`bindsite run -c config.yaml | jq .comparison` works.

Exit codes: `0` ok, `2` bad input, `3` missing dependency, `4` external tool
unavailable, `5` **leakage detected**.

## Configuration

YAML, with **unknown keys and sections rejected as errors**:

```
[homology] identity_threshold is a fraction in (0, 1], not a percentage; got 30
```

| Config | Purpose |
|---|---|
| `configs/homology_split.yaml` | The honest evaluation |
| `configs/random_split.yaml` | Comparison arm; differs in one line and leaks by design |

---

## Layout

```
src/bindsite/
  structure.py          PDB parsing, ligand curation, geometric labelling
  chemcomp.py           Chemical Component Dictionary lookups
  features.py           sequence, physicochemical and backbone features
  homology.py           local-alignment identity, connected-components clustering
  evaluate.py           residue- and protein-level metrics, AUPRC-first
  validation.py         the 7 control experiments
  pipeline.py           orchestration, in the order leakage control requires
  train.py              class-weighted training, AUPRC early stopping
  config.py             typed config; unknown keys are errors
  cli.py                stderr for status, stdout for JSON
  testing.py            synthetic fixtures, including homologous families
  data/dataset.py       RCSB search, curation attrition reporting
  data/splits.py        homology splits and the leakage guards
  models/transformer.py cross-attention architecture, distance bias
  models/baselines.py   prevalence / burial / logistic / forest
scripts/                smoke_test.py, validate_controls.py
docs/                   VALIDATION.md, METHODS.md
hpc/slurm/              GPU training job, MMseqs2 clustering job (unexecuted)
results/reference/      the committed results every number here is drawn from
```

## Documentation

* [docs/VALIDATION.md](docs/VALIDATION.md) — every measurement with its
  provenance tag, what was not reproduced and why, and the ten bugs validation
  caught
* [docs/METHODS.md](docs/METHODS.md) — curation, features, homology
  separation, architecture, metrics
* [data/README.md](data/README.md) — what is downloaded, the curation
  attrition, and why COACH420/HOLO4K are not used
* [examples/quickstart.md](examples/quickstart.md) — a guided tour
* [PROJECT_STATE.md](PROJECT_STATE.md) — status, limitations, next steps

## Attribution

Structures are downloaded from the RCSB PDB (public domain, CC0) and are not
vendored here; cite wwPDB (Berman et al., *Nucleic Acids Res* 28:235, 2000)
and the individual entries. Physicochemical scales and marker definitions are
cited inline in `features.py`. The curation principle follows BioLiP (Yang,
Roy & Zhang, *Nucleic Acids Res* 41:D1096, 2013) and P2Rank (Krivák & Hoksza,
*J Cheminform* 10:39, 2018); this repository contains no code from either.
MMseqs2 (Steinegger & Söding, *Nat Biotechnol* 35:1026, 2017) is invoked as an
external tool, not bundled.

## License

MIT — see [LICENSE](LICENSE).
