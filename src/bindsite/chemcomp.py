"""Chemical Component Dictionary lookups, for principled ligand curation.

Deciding what counts as a ligand by hand-maintained residue lists does not
work. Two examples found in a 500-protein sample drawn from the PDB:

* ``YOF`` (3-fluorotyrosine) and ``CGU`` (gamma-carboxyglutamate) are
  **modified amino acids**. They appear as HETATM records, so a hand list that
  does not happen to include them treats them as ligands — and then the
  "binding residues" of those proteins are the residues adjacent to their own
  backbone. 28 of 500 proteins were mislabelled this way before this module
  existed.
* ``NAG``, ``MAN``, ``BMA`` are **linking saccharides**: N-glycans covalently
  attached to asparagine. A glycosylation site is a post-translational
  modification, not a ligand-binding pocket.

The PDB's Chemical Component Dictionary already classifies every component,
and it is authoritative. ``chem_comp.type`` distinguishes:

=============================== ===============================================
``non-polymer``                 a genuine ligand
``L-peptide linking``           a modified amino acid — polymer, not a ligand
``D-saccharide, beta linking``  a linking sugar — usually a glycan modification
``RNA linking`` / ``DNA linking`` nucleic acid polymer
=============================== ===============================================

Lookups are cached on disk so a dataset build hits the network once per
component. With no network and no cache entry, :func:`component_type` returns
``None`` and the caller falls back to the built-in lists in
:mod:`bindsite.structure` — conservative, and the fallback is recorded so a
run never silently uses the weaker rule without saying so.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

from .structure import _ssl_context

CHEMCOMP_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{}"
DEFAULT_CACHE = "data/chemcomp_cache.json"

# Component-type substrings that mean "not a free ligand".
POLYMER_TYPE_MARKERS = (
    "peptide linking", "peptide-linking",
    "rna linking", "dna linking",
    "l-peptide nh3 amino terminus", "l-peptide cooh carboxy terminus",
    "peptide-like",
)
SACCHARIDE_TYPE_MARKERS = ("saccharide",)


class ComponentCache:
    """Disk-backed cache of component id to CCD metadata."""

    def __init__(self, path: str | Path = DEFAULT_CACHE, offline: bool = False):
        self.path = Path(path)
        self.offline = offline
        self._entries: dict[str, dict] = {}
        self.misses: set[str] = set()
        if self.path.is_file():
            try:
                self._entries = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                self._entries = {}

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, component_id: str) -> dict | None:
        """Metadata for a component, fetching and caching when needed."""
        key = str(component_id).strip().upper()
        if key in self._entries:
            return self._entries[key]
        if self.offline:
            self.misses.add(key)
            return None

        record = _fetch_component(key)
        if record is None:
            self.misses.add(key)
            return None
        self._entries[key] = record
        return record

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(dict(sorted(self._entries.items())), indent=0) + "\n"
        )
        return self.path

    def report(self) -> dict:
        return {
            "cache_path": str(self.path),
            "n_cached_components": len(self._entries),
            "n_lookup_failures": len(self.misses),
            "lookup_failures": sorted(self.misses)[:20],
            "offline": self.offline,
            "fallback_used": bool(self.misses),
        }


def _fetch_component(component_id: str, timeout: float = 30.0) -> dict | None:
    """Fetch one component's CCD record, or ``None`` on any failure."""
    request = urllib.request.Request(
        CHEMCOMP_URL.format(component_id),
        headers={"User-Agent": "bindsite/0.1"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=_ssl_context()
        ) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None

    component = payload.get("chem_comp", {})
    if not component:
        return None
    return {
        "id": component.get("id", component_id),
        "type": component.get("type", ""),
        "name": component.get("name", ""),
        "formula": component.get("formula", ""),
        "formula_weight": component.get("formula_weight"),
        "parent": component.get("mon_nstd_parent_comp_id"),
    }


def component_type(component_id: str, cache: ComponentCache | None = None) -> str | None:
    """The CCD ``type`` string for a component, or ``None`` if unknown."""
    # `is None`, not `or`: ComponentCache defines __len__, so an empty cache is
    # falsy and `cache or ComponentCache()` would discard the caller's cache on
    # every call — refetching each component and never accumulating anything.
    cache = ComponentCache() if cache is None else cache
    record = cache.get(component_id)
    return record.get("type") if record else None


def classify(component_id: str, cache: ComponentCache | None = None) -> str:
    """Classify a het component as ligand, polymer, saccharide or unknown.

    Returns one of ``"ligand"``, ``"polymer"``, ``"saccharide"`` or
    ``"unknown"``. ``"unknown"`` means the lookup failed and the caller should
    fall back to the built-in exclusion lists rather than guess.
    """
    raw = component_type(component_id, cache)
    if raw is None:
        return "unknown"
    lowered = raw.lower()
    if any(marker in lowered for marker in POLYMER_TYPE_MARKERS):
        return "polymer"
    if any(marker in lowered for marker in SACCHARIDE_TYPE_MARKERS):
        return "saccharide"
    return "ligand"


def parent_residue(component_id: str, cache: ComponentCache | None = None) -> str | None:
    """Parent standard amino acid of a modified residue, when the CCD gives one.

    Used to map a modified residue onto its one-letter code so the polymer
    chain stays contiguous instead of being broken by a selenomethionine.
    """
    cache = ComponentCache() if cache is None else cache
    record = cache.get(component_id)
    if not record:
        return None
    parent = record.get("parent")
    if not parent:
        return None
    # The API returns a list for this field, and a comma-separated string in
    # some older records; both mean "possible parents" and the first is taken.
    if isinstance(parent, (list, tuple)):
        parent = parent[0] if parent else None
    if not parent:
        return None
    first = str(parent).split(",")[0].strip().upper()
    return first or None


def prefetch(component_ids, cache: ComponentCache | None = None) -> ComponentCache:
    """Fetch and cache many components, then save. Returns the cache."""
    cache = ComponentCache() if cache is None else cache
    for component_id in sorted({str(c).strip().upper() for c in component_ids}):
        cache.get(component_id)
    cache.save()
    return cache
