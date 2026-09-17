"""Typed configuration loaded from YAML.

Unknown keys and unknown sections are **errors**. A misspelled
``identity_threshold`` that is silently ignored produces a run using the
default 30% while the config file on disk claims something else, and the
resulting split cannot be reconstructed from the record.

Paths resolve relative to the config file, so a config is portable across
working directories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .exceptions import ConfigError
from .models.baselines import BASELINE_MODELS
from .models.transformer import TransformerConfig
from .structure import CONTACT_CUTOFF, MIN_LIGAND_HEAVY_ATOMS
from .train import TrainConfig


def _check_keys(section: str, payload: dict, cls: type) -> None:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ConfigError(
            f"[{section}] unknown key(s) {unknown}; valid keys are "
            f"{sorted(known)}. Unknown keys are rejected rather than ignored so "
            "a typo cannot silently change what a run does."
        )


@dataclass
class DataConfig:
    """Dataset source, curation thresholds and preprocessing."""

    source: str = "rcsb"
    manifest: str | None = None
    pdb_ids: list[str] = field(default_factory=list)
    cache_dir: str = "../data/pdb"
    n_candidates: int = 900
    max_proteins: int = 400
    pool_size: int = 9000
    sample_seed: int = 0
    max_resolution: float = 2.0
    min_length: int = 60
    max_length: int = 400
    contact_cutoff: float = CONTACT_CUTOFF
    min_ligand_heavy_atoms: int = MIN_LIGAND_HEAVY_ATOMS
    min_binding_residues: int = 3

    def __post_init__(self) -> None:
        if self.source not in ("rcsb", "manifest", "ids", "synthetic"):
            raise ConfigError(
                f"[data] source must be rcsb, manifest, ids or synthetic; "
                f"got {self.source!r}"
            )
        if self.source == "manifest" and not self.manifest:
            raise ConfigError("[data] source 'manifest' requires a manifest path")
        if self.source == "ids" and not self.pdb_ids:
            raise ConfigError("[data] source 'ids' requires a non-empty pdb_ids list")
        if self.contact_cutoff <= 0:
            raise ConfigError(
                f"[data] contact_cutoff must be positive, got {self.contact_cutoff}"
            )
        if self.contact_cutoff > 10:
            raise ConfigError(
                f"[data] contact_cutoff of {self.contact_cutoff} A would label most "
                "of the protein as binding; conventional values are 4.0-5.0"
            )
        if self.min_length < 20:
            raise ConfigError(
                f"[data] min_length {self.min_length} admits peptides, which have "
                "no pocket to find; use at least 20"
            )
        if self.max_length <= self.min_length:
            raise ConfigError(
                f"[data] max_length ({self.max_length}) must exceed min_length "
                f"({self.min_length})"
            )
        if self.max_proteins < 10:
            raise ConfigError(
                f"[data] max_proteins {self.max_proteins} is too few to split by "
                "homology cluster; use at least 10"
            )

    def resolve(self, root: Path) -> None:
        self.cache_dir = str((root / self.cache_dir).resolve())
        if self.manifest:
            self.manifest = str((root / self.manifest).resolve())


@dataclass
class HomologyConfig:
    """Sequence-identity clustering settings."""

    identity_threshold: float = 0.30
    coverage_threshold: float = 0.50
    backend: str = "auto"
    work_dir: str = "../data/clustering"
    threads: int = 8
    cache: str | None = "../data/clusters.json"

    def __post_init__(self) -> None:
        if not 0 < self.identity_threshold <= 1:
            raise ConfigError(
                f"[homology] identity_threshold is a fraction in (0, 1], not a "
                f"percentage; got {self.identity_threshold}"
            )
        if not 0 < self.coverage_threshold <= 1:
            raise ConfigError(
                f"[homology] coverage_threshold must be in (0, 1]; got "
                f"{self.coverage_threshold}"
            )
        if self.backend not in ("auto", "mmseqs", "greedy"):
            raise ConfigError(
                f"[homology] backend must be auto, mmseqs or greedy; got "
                f"{self.backend!r}"
            )

    def resolve(self, root: Path) -> None:
        self.work_dir = str((root / self.work_dir).resolve())
        if self.cache:
            self.cache = str((root / self.cache).resolve())


@dataclass
class SplitConfig:
    """How proteins are divided into folds."""

    strategy: str = "homology"
    test_fraction: float = 0.2
    val_fraction: float = 0.15
    seed: int = 0
    n_folds: int = 0
    verify_identity: bool = True

    def __post_init__(self) -> None:
        if self.strategy == "random_residue":
            # Rejected here rather than in the pipeline. The pipeline cannot
            # run a residue-level split, and refusing only after the dataset
            # has been built means several minutes of downloading before the
            # error appears.
            raise ConfigError(
                "[split] strategy 'random_residue' is not runnable as a "
                "pipeline configuration: residues of the same protein would "
                "appear in train and test, which is the most severe leak in "
                "this task. It is implemented only inside the control "
                "experiments, where its purpose is to be measured — see "
                "bindsite.validation.leakage_by_split_level, or run "
                "`bindsite validate`."
            )
        if self.strategy not in ("homology", "random_protein"):
            raise ConfigError(
                f"[split] strategy must be 'homology' or 'random_protein'; got "
                f"{self.strategy!r}. 'random_protein' leaks by design and "
                "exists only as a comparison arm."
            )
        for name in ("test_fraction", "val_fraction"):
            value = getattr(self, name)
            if not 0 < value < 1:
                raise ConfigError(f"[split] {name} must be in (0, 1); got {value}")
        if self.test_fraction + self.val_fraction >= 1:
            raise ConfigError(
                f"[split] test_fraction + val_fraction = "
                f"{self.test_fraction + self.val_fraction:.2f} leaves nothing to "
                "train on"
            )
        if self.n_folds and self.n_folds < 2:
            raise ConfigError(f"[split] n_folds must be 0 or >= 2; got {self.n_folds}")


@dataclass
class ModelsConfig:
    """Which models to fit."""

    baselines: list[str] = field(default_factory=lambda: list(BASELINE_MODELS))
    transformer: bool = True

    def __post_init__(self) -> None:
        unknown = [m for m in self.baselines if m not in BASELINE_MODELS]
        if unknown:
            raise ConfigError(
                f"[models] unknown baseline(s) {unknown}; available: "
                f"{list(BASELINE_MODELS)}"
            )
        if not self.baselines and not self.transformer:
            raise ConfigError("[models] nothing to fit: no baselines and no transformer")


@dataclass
class RunConfig:
    """A complete pipeline configuration."""

    name: str = "bindsite-run"
    output_dir: str = "../results"
    data: DataConfig = field(default_factory=DataConfig)
    homology: HomologyConfig = field(default_factory=HomologyConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    architecture: TransformerConfig = field(default_factory=TransformerConfig)
    training: TrainConfig = field(default_factory=TrainConfig)

    def resolve(self, root: Path) -> None:
        self.output_dir = str((root / self.output_dir).resolve())
        self.data.resolve(root)
        self.homology.resolve(root)

    def to_dict(self) -> dict:
        return asdict(self)


_SECTIONS = {
    "data": DataConfig, "homology": HomologyConfig, "split": SplitConfig,
    "models": ModelsConfig, "architecture": TransformerConfig,
    "training": TrainConfig,
}


def config_from_dict(payload: dict, root: Path | None = None) -> RunConfig:
    """Build a :class:`RunConfig` from a plain mapping."""
    if not isinstance(payload, dict):
        raise ConfigError(
            f"expected a mapping at the top level, got {type(payload).__name__}"
        )
    top_level = {"name", "output_dir", *_SECTIONS}
    unknown = sorted(set(payload) - top_level)
    if unknown:
        raise ConfigError(
            f"unknown top-level section(s) {unknown}; valid sections are "
            f"{sorted(top_level)}"
        )

    sections: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        block = payload.get(name, {}) or {}
        if not isinstance(block, dict):
            raise ConfigError(
                f"[{name}] must be a mapping, got {type(block).__name__}"
            )
        _check_keys(name, block, cls)
        sections[name] = cls(**block)

    config = RunConfig(
        name=str(payload.get("name", "bindsite-run")),
        output_dir=str(payload.get("output_dir", "../results")),
        **sections,
    )
    if root is not None:
        config.resolve(root)
    return config


def load_config(path: str | Path) -> RunConfig:
    """Load a YAML config, resolving paths relative to the file itself."""
    import yaml

    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: invalid YAML: {error}") from error
    return config_from_dict(payload, root=path.resolve().parent)
