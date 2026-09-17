"""Command-line interface: exit codes and stream discipline."""

from __future__ import annotations

import json

import pytest

from bindsite.cli import (
    EXIT_BAD_INPUT, EXIT_EXTERNAL_TOOL, EXIT_OK, build_parser, main,
)


def test_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


@pytest.mark.parametrize(
    "command", ["run", "build-dataset", "cluster", "label", "validate"]
)
def test_every_documented_subcommand_parses(command):
    argv = [command]
    if command == "label":
        argv.append("1STP")
    elif command != "validate":
        argv += ["-c", "x.yaml"]
    args = build_parser().parse_args(argv)
    assert args.command == command


def test_run_rejects_a_bad_config(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text("split:\n  strategy: nonsense\n")
    assert main(["run", "-c", str(path)]) == EXIT_BAD_INPUT
    assert "config error" in capsys.readouterr().err


def test_run_reports_a_missing_config(tmp_path, capsys):
    assert main(["run", "-c", str(tmp_path / "absent.yaml")]) == EXIT_BAD_INPUT
    assert "not found" in capsys.readouterr().err


def test_validate_emits_json_on_stdout_only(capsys):
    """Status goes to stderr so `bindsite validate | jq` works."""
    code = main(["validate"])
    captured = capsys.readouterr()
    assert code == EXIT_OK
    payload = json.loads(captured.out)
    assert payload["all_passed"] is True
    assert "[bindsite]" in captured.err
    assert "[bindsite]" not in captured.out


def test_validate_logs_each_control(capsys):
    main(["validate"])
    err = capsys.readouterr().err
    assert err.count("PASS") >= 2


def test_validate_reports_skipped_controls(capsys):
    main(["validate"])
    assert "SKIP" in capsys.readouterr().err


def test_label_command_on_a_cached_structure(capsys):
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "data/pdb/1STP.pdb"
    if not path.is_file():
        pytest.skip("reference structure not cached")

    assert main(["label", str(path), "--no-ccd"]) == EXIT_OK
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["n_binding"] > 0
    assert payload["ligands"]
    assert "[bindsite]" not in captured.out
    # Streptavidin's canonical biotin site.
    residues = {r["seq"] for r in payload["binding_residues"]}
    assert {23, 27, 43, 45, 49, 79, 88, 90, 92, 108, 128} <= residues


def test_label_reports_the_cutoff_it_used(capsys):
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "data/pdb/1STP.pdb"
    if not path.is_file():
        pytest.skip("reference structure not cached")
    main(["label", str(path), "--cutoff", "5.0", "--no-ccd"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["contact_cutoff"] == 5.0


def test_label_on_a_structure_without_a_ligand(tmp_path, capsys):
    from bindsite.testing import synthetic_pdb

    path = tmp_path / "nolig.pdb"
    path.write_text(synthetic_pdb(n_residues=20, ligand_atoms=0))
    assert main(["label", str(path), "--no-ccd"]) == EXIT_BAD_INPUT
    assert "no curated ligand" in capsys.readouterr().err


def test_label_on_a_missing_file(tmp_path, capsys):
    assert main(["label", str(tmp_path / "absent.pdb"), "--no-ccd"]) == EXIT_BAD_INPUT


def test_json_output_is_standards_compliant(capsys):
    """No bare NaN or Infinity: a file only Python can read is not portable."""
    main(["validate"])
    text = capsys.readouterr().out
    assert "NaN" not in text
    assert "Infinity" not in text

    def reject(constant):  # pragma: no cover - only on invalid output
        raise AssertionError(f"non-standard JSON constant: {constant}")

    json.loads(text, parse_constant=reject)


def test_exit_codes_are_distinct():
    from bindsite.cli import (
        EXIT_LEAKAGE, EXIT_MISSING_DEPENDENCY, EXIT_OK as ok,
    )

    codes = {ok, EXIT_BAD_INPUT, EXIT_MISSING_DEPENDENCY, EXIT_EXTERNAL_TOOL,
             EXIT_LEAKAGE}
    assert len(codes) == 5
    # Leakage must be distinguishable from a crash or a bad config.
    assert EXIT_LEAKAGE == 5
