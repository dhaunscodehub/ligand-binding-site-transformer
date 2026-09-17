"""Dataset assembly: candidate selection, download, curation and caching.

Candidates come from the RCSB search API, filtered to single-protein-chain
entries with at least one non-polymer group, good resolution and a workable
length. The filters are stated in :func:`search_candidates` and recorded in
the dataset manifest, because "a curated dataset of diverse proteins" means
nothing without them.

Curation happens per structure in :mod:`bindsite.structure`: water, ions and
crystallisation additives are excluded, so a large fraction of candidates
yield no ligand and are dropped. That attrition is reported rather than
hidden — it is the difference between a dataset of binding sites and a dataset
of sulfate contacts.
"""

from __future__ import annotations

import json
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..exceptions import DataError
from ..features import ResidueFeatures, featurise
from ..io_utils import write_json
from ..structure import (
    CONTACT_CUTOFF, MIN_LIGAND_HEAVY_ATOMS, NoLigandError, ProteinStructure,
    StructureError, _ssl_context, fetch_pdb, load_structure,
)

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


@dataclass
class DatasetEntry:
    """One curated, labelled protein."""

    identifier: str
    structure: ProteinStructure
    features: ResidueFeatures

    @property
    def sequence(self) -> str:
        return self.structure.sequence

    @property
    def labels(self) -> np.ndarray:
        return self.features.labels


@dataclass
class Dataset:
    """A collection of curated proteins, with the attrition that produced it."""

    entries: dict[str, DatasetEntry] = field(default_factory=dict)
    attempted: list[str] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)
    contact_cutoff: float = CONTACT_CUTOFF
    filters: dict = field(default_factory=dict)
    curation: str = "curated exclusion lists only (no CCD lookup)"
    exclude_saccharides: bool = True
    component_cache_report: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def names(self) -> list[str]:
        return sorted(self.entries)

    def sequences(self) -> dict[str, str]:
        return {name: self.entries[name].sequence for name in self.names}

    def labels(self) -> dict[str, np.ndarray]:
        return {name: self.entries[name].labels for name in self.names}

    def statistics(self) -> dict:
        if not self.entries:
            return {"n_proteins": 0}
        lengths = np.array([e.structure.n_residues for e in self.entries.values()])
        binding = np.array([e.structure.n_binding for e in self.entries.values()])
        ligand_names = [
            l.residue_name for e in self.entries.values() for l in e.structure.ligands
        ]
        from collections import Counter

        return {
            "n_proteins": len(self.entries),
            "n_residues": int(lengths.sum()),
            "n_binding_residues": int(binding.sum()),
            "positive_rate": float(binding.sum() / lengths.sum()),
            "median_length": float(np.median(lengths)),
            "length_range": [int(lengths.min()), int(lengths.max())],
            "median_binding_per_protein": float(np.median(binding)),
            "n_distinct_ligands": len(set(ligand_names)),
            "most_common_ligands": Counter(ligand_names).most_common(10),
            "contact_cutoff": self.contact_cutoff,
            "curation": self.curation,
            "exclude_saccharides": self.exclude_saccharides,
            "component_cache": self.component_cache_report,
            "n_attempted": len(self.attempted),
            "n_rejected": len(self.rejected),
            "rejection_reasons": _summarise_rejections(self.rejected),
            "filters": self.filters,
        }

    def to_manifest(self) -> dict:
        return {
            "statistics": self.statistics(),
            "proteins": {
                name: entry.structure.to_dict()
                for name, entry in sorted(self.entries.items())
            },
            "rejected": self.rejected,
        }


def _summarise_rejections(rejected: dict[str, str]) -> dict[str, int]:
    from collections import Counter

    # Reasons carry per-entry detail; group by their leading phrase.
    return dict(
        Counter(reason.split(":")[0].split("(")[0].strip() for reason in rejected.values())
    )


def search_candidates(
    limit: int = 600,
    max_resolution: float = 2.0,
    min_length: int = 60,
    max_length: int = 400,
    timeout: float = 60.0,
    pool_size: int = 8000,
    seed: int = 0,
) -> tuple[list[str], dict]:
    """Query RCSB for single-chain protein-ligand complexes.

    Filters, and why each is there:

    * **one protein entity** — binding labels are per chain; a multi-chain
      entry would need a chain choice that changes the labels.
    * **at least one non-polymer entity** — a necessary but far from
      sufficient condition for a ligand, since most non-polymer groups are
      water, ions or additives and are dropped later by curation.
    * **resolution better than** ``max_resolution`` — ligand placement at low
      resolution is unreliable, and a mis-placed ligand mislabels a pocket.
    * **length in range** — very short chains are peptides, and very long ones
      make the O(L^2) attention and the O(L^2) alignment expensive without
      adding diversity.

    Candidates are **randomly sampled** from a large pool rather than taken in
    the order RCSB returns them. This is not a cosmetic choice: the API's
    default ordering is effectively by PDB id, and the first entries matching
    these filters are 102L, 103L, 107L, 108L (all T4 lysozyme mutants) and
    102M, 104M, 106M, 109M (all myoglobin). Taking the first N would build a
    "curated dataset of diverse proteins" containing about four distinct
    proteins. Sampling is seeded, so the selection is reproducible.

    Returns the ids and the filter record, so the manifest states exactly what
    the dataset is.
    """
    filters = {
        "polymer_entity_count_protein": 1,
        "nonpolymer_entity_count": "> 0",
        "max_resolution_angstrom": max_resolution,
        "polymer_monomer_count": [min_length, max_length],
        "experimental_only": True,
        "requested_limit": limit,
        "pool_size": pool_size,
        "sampling": "uniform without replacement from the pool",
        "sampling_seed": seed,
    }
    query = {
        "query": {
            "type": "group", "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                    "operator": "equals", "value": 1}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                    "operator": "greater", "value": 0}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less", "value": max_resolution}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                    "operator": "range",
                    "value": {"from": min_length, "to": max_length}}},
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max(pool_size, limit)},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }

    request = urllib.request.Request(
        RCSB_SEARCH_URL,
        data=json.dumps(query).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "bindsite/0.1"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=_ssl_context()
        ) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, ssl.SSLError) as error:
        raise DataError(
            f"RCSB search failed: {error}. The dataset builder needs network "
            "access; a cached manifest can be used offline instead."
        ) from error

    pool = [r["identifier"] for r in payload.get("result_set", [])]
    if not pool:
        raise DataError("RCSB search returned no entries for these filters")

    rng = np.random.default_rng(seed)
    chosen = (
        list(rng.choice(np.array(pool), size=limit, replace=False))
        if len(pool) > limit else list(pool)
    )
    identifiers = sorted(str(c) for c in chosen)

    filters["total_matching_in_pdb"] = payload.get("total_count")
    filters["pool_returned"] = len(pool)
    filters["n_sampled"] = len(identifiers)
    return identifiers, filters


def build_dataset(
    identifiers: Sequence[str],
    cache_dir: str | Path = "data/pdb",
    cutoff: float = CONTACT_CUTOFF,
    min_ligand_atoms: int = MIN_LIGAND_HEAVY_ATOMS,
    min_length: int = 60,
    max_length: int = 400,
    min_binding_residues: int = 3,
    max_proteins: int | None = None,
    filters: dict | None = None,
    progress: bool = False,
    component_cache=None,
    exclude_saccharides: bool = True,
) -> Dataset:
    """Download, curate and featurise a list of PDB entries.

    ``min_binding_residues`` drops proteins whose curated ligand contacts only
    one or two residues. Such an entry is usually a ligand bound at a crystal
    contact rather than in a pocket, and it contributes a label that no
    structural feature can explain.
    """
    dataset = Dataset(contact_cutoff=cutoff, filters=dict(filters or {}))
    dataset.curation = (
        "chemical component dictionary + curated exclusion lists"
        if component_cache is not None
        else "curated exclusion lists only (no CCD lookup)"
    )
    dataset.exclude_saccharides = exclude_saccharides

    for identifier in identifiers:
        if max_proteins is not None and len(dataset.entries) >= max_proteins:
            break
        dataset.attempted.append(identifier)
        try:
            path = fetch_pdb(identifier, cache_dir=cache_dir)
            structure = load_structure(
                path, identifier=identifier, cutoff=cutoff,
                min_heavy_atoms=min_ligand_atoms, require_ligand=True,
                component_cache=component_cache,
                exclude_saccharides=exclude_saccharides,
            )
        except NoLigandError as error:
            dataset.rejected[identifier] = f"no curated ligand: {error}"[:200]
            continue
        except (StructureError, DataError) as error:
            dataset.rejected[identifier] = f"unreadable: {error}"[:200]
            continue

        if not (min_length <= structure.n_residues <= max_length):
            dataset.rejected[identifier] = (
                f"length {structure.n_residues} outside [{min_length}, {max_length}]"
            )
            continue
        if structure.n_binding < min_binding_residues:
            dataset.rejected[identifier] = (
                f"only {structure.n_binding} binding residues, below "
                f"{min_binding_residues}; likely a crystal-contact ligand"
            )
            continue
        # A structure where nearly everything is "binding" is not a pocket.
        if structure.binding_fraction > 0.5:
            dataset.rejected[identifier] = (
                f"binding fraction {structure.binding_fraction:.0%} exceeds 50%; "
                "the ligand is probably a polymer or spans the whole chain"
            )
            continue

        try:
            features = featurise(structure)
        except DataError as error:
            dataset.rejected[identifier] = f"featurisation failed: {error}"[:200]
            continue

        dataset.entries[identifier] = DatasetEntry(
            identifier=identifier, structure=structure, features=features
        )
        if progress and len(dataset.entries) % 25 == 0:
            # stderr, not stdout: stdout carries the JSON result, and mixing
            # progress into it would make `bindsite run -v | jq` fail.
            print(
                f"  curated {len(dataset.entries)} of "
                f"{len(dataset.attempted)} attempted",
                file=sys.stderr, flush=True,
            )
    return dataset


def save_manifest(dataset: Dataset, path: str | Path) -> Path:
    """Write the dataset manifest, so a build is reproducible and auditable."""
    return write_json(path, dataset.to_manifest())


def load_from_manifest(
    path: str | Path,
    cache_dir: str | Path = "data/pdb",
    cutoff: float | None = None,
) -> Dataset:
    """Rebuild a dataset from a manifest, using cached structure files.

    Re-derives labels and features from the structures rather than trusting
    stored arrays: if the contact cutoff or the curation rules change, a
    manifest built under the old rules must not silently supply old labels.
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"manifest not found: {path}")
    manifest = json.loads(path.read_text())
    identifiers = sorted(manifest.get("proteins", {}))
    if not identifiers:
        raise DataError(f"{path}: manifest lists no proteins")

    statistics = manifest.get("statistics", {})
    return build_dataset(
        identifiers,
        cache_dir=cache_dir,
        cutoff=cutoff if cutoff is not None else statistics.get("contact_cutoff", CONTACT_CUTOFF),
        filters=statistics.get("filters", {}),
    )
