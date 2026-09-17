"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from bindsite.config import RunConfig, config_from_dict, load_config
from bindsite.exceptions import ConfigError


def test_defaults_load():
    config = config_from_dict({})
    assert isinstance(config, RunConfig)
    assert config.split.strategy == "homology"
    assert config.homology.identity_threshold == 0.30


def test_unknown_section_is_rejected():
    with pytest.raises(ConfigError, match="unknown top-level section"):
        config_from_dict({"modelz": {}})


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        config_from_dict({"data": {"contact_cutof": 4.0}})


def test_rejection_lists_the_valid_keys():
    with pytest.raises(ConfigError) as info:
        config_from_dict({"data": {"contact_cutof": 4.0}})
    assert "contact_cutoff" in str(info.value)


def test_non_mapping_section_is_rejected():
    with pytest.raises(ConfigError, match="must be a mapping"):
        config_from_dict({"data": ["nope"]})


def test_non_mapping_top_level_is_rejected():
    with pytest.raises(ConfigError, match="mapping at the top level"):
        config_from_dict(["nope"])


@pytest.mark.parametrize(
    "payload",
    [
        {"data": {"source": "parquet"}},
        {"data": {"source": "manifest"}},
        {"data": {"source": "ids"}},
        {"data": {"contact_cutoff": 0}},
        {"data": {"contact_cutoff": -1}},
        {"data": {"contact_cutoff": 15}},
        {"data": {"min_length": 5}},
        {"data": {"min_length": 300, "max_length": 100}},
        {"data": {"max_proteins": 3}},
        {"homology": {"identity_threshold": 30}},
        {"homology": {"identity_threshold": 0}},
        {"homology": {"coverage_threshold": 2}},
        {"homology": {"backend": "magic"}},
        {"split": {"strategy": "random"}},
        {"split": {"test_fraction": 0}},
        {"split": {"test_fraction": 1.5}},
        {"split": {"test_fraction": 0.7, "val_fraction": 0.4}},
        {"split": {"n_folds": 1}},
        {"models": {"baselines": ["magic"]}},
        {"models": {"baselines": [], "transformer": False}},
        {"architecture": {"d_model": 100, "n_heads": 3}},
        {"architecture": {"n_heads": 0}},
        {"architecture": {"use_sequence": False, "use_structure": False}},
        {"architecture": {"dropout": 1.0}},
        {"training": {"epochs": 0}},
        {"training": {"batch_size": 0}},
        {"training": {"learning_rate": 5}},
        {"training": {"warmup_fraction": 1.0}},
        {"training": {"max_pos_weight": 0.5}},
    ],
)
def test_invalid_values_are_rejected(payload):
    with pytest.raises(ConfigError):
        config_from_dict(payload)


def test_identity_threshold_error_explains_the_unit():
    """The most likely mistake is passing a percentage."""
    with pytest.raises(ConfigError, match="percentage"):
        config_from_dict({"homology": {"identity_threshold": 30}})


def test_cutoff_error_names_conventional_values():
    with pytest.raises(ConfigError, match="4.0-5.0"):
        config_from_dict({"data": {"contact_cutoff": 15.0}})


def test_split_strategy_error_says_which_arms_leak():
    with pytest.raises(ConfigError, match="leaks by design"):
        config_from_dict({"split": {"strategy": "random"}})


def test_residue_level_strategy_is_refused_at_config_time():
    """Not after the dataset is built: that would mean minutes of downloading
    before the error appears."""
    with pytest.raises(ConfigError, match="not runnable as a pipeline"):
        config_from_dict({"split": {"strategy": "random_residue"}})


def test_paths_resolve_relative_to_the_config_file(tmp_path):
    directory = tmp_path / "configs"
    directory.mkdir()
    (directory / "run.yaml").write_text(
        "output_dir: ../results\n"
        "data:\n  source: manifest\n  manifest: ../data/m.json\n"
        "  cache_dir: ../data/pdb\n"
        "homology:\n  cache: ../data/clusters.json\n"
    )
    config = load_config(directory / "run.yaml")
    assert config.output_dir == str((tmp_path / "results").resolve())
    assert config.data.manifest == str((tmp_path / "data" / "m.json").resolve())
    assert config.homology.cache == str((tmp_path / "data" / "clusters.json").resolve())


def test_missing_config_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_invalid_yaml_is_reported(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("data: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_empty_yaml_loads_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_config(path).split.strategy == "homology"


def test_shipped_configs_are_valid():
    root = Path(__file__).resolve().parent.parent / "configs"
    files = sorted(root.glob("*.yaml"))
    assert files, "expected shipped configs"
    for path in files:
        load_config(path)


def test_the_two_shipped_configs_differ_only_in_strategy():
    """The comparison is only meaningful if nothing else changes."""
    root = Path(__file__).resolve().parent.parent / "configs"
    homology = load_config(root / "homology_split.yaml")
    random_ = load_config(root / "random_split.yaml")
    assert homology.split.strategy == "homology"
    assert random_.split.strategy == "random_protein"
    assert homology.data.contact_cutoff == random_.data.contact_cutoff
    assert homology.homology.identity_threshold == random_.homology.identity_threshold
    assert homology.architecture.to_dict() == random_.architecture.to_dict()
    assert homology.training.to_dict() == random_.training.to_dict()
    assert homology.split.seed == random_.split.seed
    assert homology.split.test_fraction == random_.split.test_fraction


def test_to_dict_round_trips():
    config = config_from_dict({"data": {"max_proteins": 123}})
    assert config_from_dict(config.to_dict()).data.max_proteins == 123
