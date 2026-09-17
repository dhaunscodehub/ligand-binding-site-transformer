# Quickstart

## Install

```bash
pip install -e ".[all]"
```

## 1. Check the installation

```bash
python scripts/smoke_test.py
```

43 checks, no network, under a minute. Expect `43/43 checks passed`.

## 2. Look at one structure

Streptavidin with biotin, the textbook case:

```bash
bindsite label 1STP | jq -r '.binding_residues[] | "\(.one_letter)\(.seq)"' | tr '\n' ' '
```

```
N23 L25 S27 Y43 S45 V47 G48 N49 W79 A86 S88 T90 W92 W108 L110 D128
```

That is the canonical biotin pocket: the hydrogen-bonding set N23, S27, Y43,
S45, N49, S88, T90, D128 and the hydrophobic box W79, W92, W108. (W120 is
absent because it comes from the adjacent subunit, which is not in this
chain.) Nothing was fitted — this is geometry plus ligand curation.

Try a protein–protein complex with no small-molecule ligand:

```bash
bindsite label 4ZQK; echo "exit=$?"
```

```
no curated ligand: 4ZQK: no het group survives curation (rejected: ['HOH', 'NA']). ...
exit=2
```

Water and sodium are refused rather than labelled as a binding site.

## 3. Run the control experiments

```bash
python scripts/validate_controls.py
```

The synthetic controls only. Add `-c configs/homology_split.yaml` for the
dataset-dependent ones (downloads structures on first use).

Look at the AUROC/AUPRC panel:

```
WHY AUROC IS NOT REPORTED ALONE
  AUROC 0.795 looks strong, but the best achievable F1 is 0.400 with precision
  0.358. The ROC false-positive-rate denominator is the large negative class, so
  many false positives barely move it.
```

## 4. Inspect the homology clustering

```bash
bindsite cluster -c configs/homology_split.yaml | jq '{n_clusters, redundancy, largest_cluster_fraction, linkage}'
```

```json
{
  "n_clusters": 199,
  "redundancy": 0.5216,
  "largest_cluster_fraction": 0.0769,
  "linkage": "single (transitive closure), local alignment"
}
```

52% of this "diverse" 416-protein dataset is redundant at 30% identity. That
redundancy is what a random split leaks through.

## 5. Run the honest evaluation

```bash
bindsite run -c configs/homology_split.yaml -v > homology.json
```

Status goes to stderr, JSON to stdout:

```
[bindsite] 416 proteins, 91692 residues, positive rate 7.80%
[bindsite] 199 clusters, redundancy 52.2%
[bindsite] prevalence     AUPRC 0.070 (1.00x chance)  AUROC 0.500  F1 0.131  P@L/10 0.064
[bindsite] burial         AUPRC 0.090 (1.29x chance)  AUROC 0.617  F1 0.171  P@L/10 0.075
[bindsite] logistic       AUPRC 0.193 (2.76x chance)  AUROC 0.765  F1 0.276  P@L/10 0.217
[bindsite] random_forest  AUPRC 0.221 (3.17x chance)  AUROC 0.777  F1 0.292  P@L/10 0.244
[bindsite] transformer    AUPRC 0.251 (3.59x chance)  AUROC 0.781  F1 0.311  P@L/10 0.262
```

Confirm the split is actually clean:

```bash
jq '.leakage_verification.checks[] | select(.check=="max train-test identity")' homology.json
```

```json
{
  "check": "max train-test identity",
  "passed": true,
  "max_identity": 0.2649,
  "threshold": 0.3,
  "worst_pair": ["3BHY", "2XIR"],
  "source": "precomputed matrix",
  "n_pairs_checked": 16688
}
```

## 6. Measure the leakage yourself

The comparison config differs in exactly one line — `split.strategy`:

```bash
diff configs/homology_split.yaml configs/random_split.yaml | grep strategy
bindsite run -c configs/random_split.yaml -v > random.json
```

```
[bindsite] transformer    AUPRC 0.558 (6.81x chance)  AUROC 0.876  F1 0.570  P@L/10 0.504
```

AUPRC 0.558 against the honest 0.251. Same data, same model, same seed — the
difference is that 33 of 199 homology clusters appear in both train and test.

```bash
jq '.split.shared_clusters_train_test' random.json   # 33
```

## 7. Try to break it

The config rejects typos rather than ignoring them:

```bash
printf 'homology:\n  identity_threshold: 30\n' > /tmp/bad.yaml
bindsite run -c /tmp/bad.yaml; echo "exit=$?"
```

```
config error: [homology] identity_threshold is a fraction in (0, 1], not a percentage; got 30
exit=2
```

A leaking split exits **5**, distinct from a bad config:

```bash
python -c "
from bindsite.data.splits import Split, assert_no_homology_leakage
from bindsite.homology import cluster_connected_components
from bindsite.testing import synthetic_families
s = synthetic_families(n_families=4, per_family=3, seed=0)
cl = cluster_connected_components(s, identity=0.30)
bad = Split(name='bad', train=['F0M0'], val=[], test=['F0M1'], strategy='homology_cluster')
assert_no_homology_leakage(bad, None, sequences=s, max_identity=0.30)
"
```

```
LeakageError: train sequence 'F0M0' and test sequence 'F0M1' share 91.7% identity,
above the 30% threshold this split claims to enforce.
```
