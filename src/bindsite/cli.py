"""Command-line interface.

Status messages go to **stderr**, machine-readable JSON to **stdout**, so
``bindsite run -c config.yaml | jq .comparison`` works and redirecting stdout
yields a valid JSON document with no log lines mixed in.

Exit codes:

==== ==========================================================
0    success
2    bad input or invalid configuration
3    a required optional dependency is missing
4    an external tool (MMseqs2) is unavailable or failed
5    data leakage detected
==== ==========================================================

Code 5 is distinct because a leaking split is not a crash and not a bad
config: it is a scientifically invalid run, and a caller should be able to
tell that apart without parsing a message.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .exceptions import (
    ConfigError, DataError, ExternalToolError, LeakageError,
    OptionalDependencyMissing,
)

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_MISSING_DEPENDENCY = 3
EXIT_EXTERNAL_TOOL = 4
EXIT_LEAKAGE = 5


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _emit(payload: dict) -> None:
    from .io_utils import _Encoder, sanitise

    json.dump(sanitise(payload), sys.stdout, indent=2, cls=_Encoder, allow_nan=False)
    sys.stdout.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bindsite",
        description=(
            "Structure-aware prediction of ligand-binding residues, with "
            "homology-separated evaluation."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run the full pipeline from a config")
    run.add_argument("-c", "--config", required=True)
    run.add_argument("-o", "--output-dir", default=None)
    run.add_argument("-v", "--verbose", action="store_true", help="per-epoch progress")

    build = subparsers.add_parser(
        "build-dataset", help="download and curate the dataset, writing a manifest"
    )
    build.add_argument("-c", "--config", required=True)
    build.add_argument("-o", "--manifest", default=None)

    cluster = subparsers.add_parser(
        "cluster", help="cluster sequences and report redundancy and coherence"
    )
    cluster.add_argument("-c", "--config", required=True)

    label = subparsers.add_parser(
        "label", help="show curated ligands and binding residues for one structure"
    )
    label.add_argument("pdb_id", help="4-character PDB id, or a path to a PDB file")
    label.add_argument("--cutoff", type=float, default=4.0)
    label.add_argument(
        "--no-ccd", action="store_true",
        help="skip Chemical Component Dictionary lookups (weaker curation)",
    )

    validate = subparsers.add_parser(
        "validate", help="run the control experiments"
    )
    validate.add_argument("-c", "--config", default=None,
                          help="use this dataset for the data-dependent controls")
    validate.add_argument("--seed", type=int, default=0)

    return parser


def _load(path: str):
    from .config import load_config

    return load_config(path)


def command_run(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline

    try:
        config = _load(args.config)
    except ConfigError as error:
        _log(f"config error: {error}")
        return EXIT_BAD_INPUT

    _log(f"[bindsite] running '{config.name}'")
    _log(
        f"[bindsite] split: {config.split.strategy} at "
        f"{config.homology.identity_threshold:.0%} identity"
    )
    try:
        result = run_pipeline(
            config, output_dir=args.output_dir, progress=args.verbose
        )
    except LeakageError as error:
        _log(f"DATA LEAKAGE DETECTED: {error}")
        return EXIT_LEAKAGE
    except ExternalToolError as error:
        _log(f"external tool unavailable: {error}")
        return EXIT_EXTERNAL_TOOL
    except OptionalDependencyMissing as error:
        _log(f"missing dependency: {error}")
        return EXIT_MISSING_DEPENDENCY
    except (DataError, ConfigError, ValueError) as error:
        _log(f"pipeline error: {error}")
        return EXIT_BAD_INPUT

    stats = result.dataset_statistics
    _log(
        f"[bindsite] {stats['n_proteins']} proteins, {stats['n_residues']} residues, "
        f"positive rate {stats['positive_rate']:.2%}"
    )
    _log(
        f"[bindsite] {result.clustering['n_clusters']} clusters, redundancy "
        f"{result.clustering['redundancy']:.1%}"
    )
    for evaluation in result.evaluations:
        residue = evaluation.residue
        auroc = f"{residue.auroc:.3f}" if residue.auroc is not None else "n/a"
        auprc = f"{residue.auprc:.3f}" if residue.auprc is not None else "n/a"
        lift = f"{residue.auprc_lift:.2f}x" if residue.auprc_lift else "n/a"
        _log(
            f"[bindsite] {evaluation.model:<14} AUPRC {auprc} ({lift} chance)  "
            f"AUROC {auroc}  F1 {residue.best_f1:.3f}  "
            f"P@L/10 {residue.precision_at_top_l10:.3f}"
        )
    for warning in result.warnings:
        _log(f"[bindsite] warning: {warning}")

    _emit(result.to_dict())
    return EXIT_OK


def command_build_dataset(args: argparse.Namespace) -> int:
    from .data.dataset import save_manifest
    from .pipeline import prepare_dataset

    try:
        config = _load(args.config)
    except ConfigError as error:
        _log(f"config error: {error}")
        return EXIT_BAD_INPUT

    _log(f"[bindsite] building dataset from source '{config.data.source}'")
    try:
        dataset = prepare_dataset(config, progress=True)
    except DataError as error:
        _log(f"dataset error: {error}")
        return EXIT_BAD_INPUT

    statistics = dataset.statistics()
    _log(
        f"[bindsite] curated {statistics['n_proteins']} proteins from "
        f"{statistics['n_attempted']} attempted "
        f"({statistics['n_rejected']} rejected)"
    )
    target = args.manifest or str(
        Path(config.data.cache_dir).parent / "dataset_manifest.json"
    )
    save_manifest(dataset, target)
    _log(f"[bindsite] manifest written to {target}")
    _emit(statistics)
    return EXIT_OK


def command_cluster(args: argparse.Namespace) -> int:
    from .pipeline import (
        prepare_clustering, prepare_dataset, prepare_identity_matrix,
    )

    try:
        config = _load(args.config)
    except ConfigError as error:
        _log(f"config error: {error}")
        return EXIT_BAD_INPUT

    try:
        dataset = prepare_dataset(config)
        _log(
            f"[bindsite] clustering {len(dataset)} sequences at "
            f"{config.homology.identity_threshold:.0%} identity"
        )
        matrix = prepare_identity_matrix(dataset, config)
        clustering = prepare_clustering(dataset, config, matrix=matrix)
    except ExternalToolError as error:
        _log(f"external tool unavailable: {error}")
        return EXIT_EXTERNAL_TOOL
    except OptionalDependencyMissing as error:
        _log(f"missing dependency: {error}")
        return EXIT_MISSING_DEPENDENCY
    except DataError as error:
        _log(f"clustering error: {error}")
        return EXIT_BAD_INPUT

    report = {
        **clustering.to_dict(),
        "pairwise_identity_distribution": matrix.distribution(),
    }
    _log(
        f"[bindsite] {report['n_clusters']} clusters from "
        f"{report['n_sequences']} sequences; redundancy "
        f"{report['redundancy']:.1%}; largest cluster "
        f"{report['largest_cluster']} ({report['largest_cluster_fraction']:.1%})"
    )
    _emit(report)
    return EXIT_OK


def command_label(args: argparse.Namespace) -> int:
    from .chemcomp import ComponentCache
    from .exceptions import NoLigandError, StructureError
    from .structure import fetch_pdb, load_structure

    target = Path(args.pdb_id)
    cache = None if args.no_ccd else ComponentCache("data/chemcomp_cache.json")
    try:
        path = target if target.is_file() else fetch_pdb(args.pdb_id)
        structure = load_structure(
            path, identifier=target.stem if target.is_file() else args.pdb_id.upper(),
            cutoff=args.cutoff, component_cache=cache,
        )
    except NoLigandError as error:
        _log(f"no curated ligand: {error}")
        return EXIT_BAD_INPUT
    except (StructureError, DataError) as error:
        _log(f"structure error: {error}")
        return EXIT_BAD_INPUT
    if cache is not None:
        cache.save()

    _log(
        f"[bindsite] {structure.identifier} chain {structure.chain}: "
        f"{structure.n_residues} residues, {structure.n_binding} binding "
        f"({structure.binding_fraction:.1%}) at {args.cutoff} A"
    )
    _log(f"[bindsite] curation: {structure.curation}")
    payload = structure.to_dict()
    payload["binding_residues"] = [
        {
            "residue": residue.name, "one_letter": residue.one_letter,
            "seq": residue.residue_seq,
            "min_ligand_distance": residue.min_ligand_distance,
        }
        for residue in structure.residues if residue.is_binding
    ]
    _emit(payload)
    return EXIT_OK


def command_validate(args: argparse.Namespace) -> int:
    from .validation import run_controls

    features = clustering = sequences = None
    reference = None
    if args.config:
        from .pipeline import prepare_clustering, prepare_dataset

        try:
            config = _load(args.config)
            _log("[bindsite] building dataset for the data-dependent controls")
            dataset = prepare_dataset(config)
            clustering = prepare_clustering(dataset, config)
            features = {n: dataset.entries[n].features for n in dataset.names}
            sequences = dataset.sequences()
            first = dataset.entries[dataset.names[0]].structure.source
            reference = first
        except (ConfigError, DataError) as error:
            _log(f"could not prepare the dataset: {error}")
            return EXIT_BAD_INPUT

    _log("[bindsite] running control experiments")
    report = run_controls(
        features_by_name=features, clusters=clustering, sequences=sequences,
        reference_pdb=reference, seed=args.seed,
    )
    for check in report["checks"]:
        _log(f"  {'PASS' if check['passed'] else 'FAIL'}  {check['name']}")
    for name, reason in report["skipped"].items():
        _log(f"  SKIP  {name}: {reason}")

    _emit(report)
    return EXIT_OK if report["all_passed"] else EXIT_BAD_INPUT


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "run": command_run, "build-dataset": command_build_dataset,
        "cluster": command_cluster, "label": command_label,
        "validate": command_validate,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
