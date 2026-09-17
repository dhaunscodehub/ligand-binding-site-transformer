#!/usr/bin/env python3
"""Run the control experiments and print a PASS/FAIL table.

These controls are what make the reported metrics interpretable. Run this
first on a new machine: if the label-shuffle control scores above chance,
nothing else in the repository can be trusted.

    python scripts/validate_controls.py
    python scripts/validate_controls.py -c configs/homology_split.yaml
    python scripts/validate_controls.py --json results/controls.json

Without ``-c`` only the synthetic controls run, which needs no network and
takes under a minute. With ``-c`` the dataset-dependent controls run too.

Exit status is 0 only if every control passes.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bindsite.io_utils import provenance, write_json  # noqa: E402
from bindsite.validation import run_controls  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    features = clusters = sequences = reference = None
    if args.config:
        from bindsite.config import load_config
        from bindsite.pipeline import (
            prepare_clustering, prepare_dataset, prepare_identity_matrix,
        )

        config = load_config(args.config)
        print(f"building dataset from {args.config} ...", file=sys.stderr)
        dataset = prepare_dataset(config)
        matrix = prepare_identity_matrix(dataset, config)
        clusters = prepare_clustering(dataset, config, matrix=matrix)
        features = {n: dataset.entries[n].features for n in dataset.names}
        sequences = dataset.sequences()
        reference = dataset.entries[dataset.names[0]].structure.source

    report = run_controls(
        features_by_name=features, clusters=clusters, sequences=sequences,
        reference_pdb=reference, seed=args.seed,
    )

    print()
    print("CONTROL EXPERIMENTS")
    print("=" * 82)
    print(f"{'':<4}{'result':<7}{'check':<52}{'measured':>10}{'chance':>9}")
    print("-" * 82)
    for number, check in enumerate(report["checks"], start=1):
        measured = _measured(check)
        chance = _chance(check)
        print(
            f"{number:<4}{'PASS' if check['passed'] else 'FAIL':<7}"
            f"{check['name'][:51]:<52}{measured:>10}{chance:>9}"
        )
    print("-" * 82)
    print(f"{report['n_passed']}/{report['n_checks']} controls passed")
    for name, reason in report["skipped"].items():
        print(f"  skipped: {name} ({reason})")
    print()

    levels = next(
        (c for c in report["checks"] if c["name"].startswith("leakage by split")),
        None,
    )
    if levels:
        print("LEAKAGE BY SPLIT LEVEL")
        print("-" * 82)
        print(f"  model: {levels.get('model', 'unspecified')}")
        print(f"{'  split':<22}{'AUPRC':>9}{'AUROC':>9}{'lift':>9}{'inflation':>12}")
        honest = levels["arms"]["homology_cluster"]["auprc"] or 0.0
        for name, arm in levels["arms"].items():
            auprc = arm["auprc"] or 0.0
            print(
                f"  {name:<20}{auprc:>9.4f}{arm['auroc'] or 0.0:>9.4f}"
                f"{arm['auprc_lift'] or 0.0:>8.2f}x{auprc - honest:>+12.4f}"
            )
        print()

    shuffle = next(
        (c for c in report["checks"] if c["name"].startswith("label shuffle")), None
    )
    if shuffle and "per_seed_auroc" in shuffle:
        print("LABEL SHUFFLE REPLICATES")
        print("-" * 82)
        print(
            f"  mean AUROC {shuffle['auroc']:.4f} "
            f"sd {shuffle['auroc_sd']:.4f} over seeds {shuffle['seeds']}"
        )
        print(f"  per seed: {shuffle['per_seed_auroc']}")
        print()

    gap = next(
        (c for c in report["checks"] if c["name"].startswith("AUROC overstates")), None
    )
    if gap:
        print("WHY AUROC IS NOT REPORTED ALONE")
        print("-" * 82)
        for line in textwrap.wrap(gap["interpretation"], 80):
            print(f"  {line}")
        print()

    if args.json:
        path = write_json(args.json, {**report, "provenance": provenance()})
        print(f"wrote {path}")

    return 0 if report["all_passed"] else 1


def _measured(check: dict) -> str:
    for key in ("auprc", "optimism_balanced_accuracy", "max_train_test_identity"):
        if check.get(key) is not None:
            return f"{check[key]:.4f}"
    if check.get("residue_split_auprc_inflation") is not None:
        return f"{check['residue_split_auprc_inflation']:+.4f}"
    if check.get("max_geometric_difference") is not None:
        return f"{check['max_geometric_difference']:.1e}"
    return "-"


def _chance(check: dict) -> str:
    if check.get("positive_rate") is not None:
        return f"{check['positive_rate']:.4f}"
    if check.get("auprc_chance") is not None:
        return f"{check['auprc_chance']:.4f}"
    if check.get("threshold") is not None:
        return f"{check['threshold']:.2f}"
    return "-"


if __name__ == "__main__":
    raise SystemExit(main())
