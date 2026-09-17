"""Structure parsing, ligand curation and geometric binding-residue labelling.

The scientifically load-bearing decision in this module is **which HETATM
records count as a ligand**. Most PDB entries contain water, ions and
crystallisation additives, and a naive "any HETATM is a ligand" rule labels
sulfate and glycerol contacts as binding sites. Those are crystallisation
artefacts: a model trained on them learns surface chemistry that happens to
coordinate cryoprotectant, which is not ligand binding.

The curation here follows the exclusion principle used by BioLiP (Yang, Roy &
Zhang, *Nucleic Acids Res* 41:D1096, 2013) and P2Rank (Krivák & Hoksza,
*J Cheminform* 10:39, 2018): drop solvent, ions and known additives, and
require a minimum heavy-atom count so that fragments too small to define a
pocket are not treated as ligands.

A binding residue is a residue with any heavy atom within
:data:`CONTACT_CUTOFF` of any ligand heavy atom. 4.0 Å is the conventional
cutoff for this task; it is a parameter here and recorded in every output,
because the positive rate — and therefore every metric — depends on it.
"""

from __future__ import annotations

import gzip
import ssl
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .exceptions import DataError, NoLigandError, StructureError

# Distance from a ligand heavy atom within which a residue is called binding.
CONTACT_CUTOFF = 4.0

# Minimum ligand heavy atoms. Below this a "ligand" is a fragment or an ion
# cluster that cannot define a pocket; P2Rank and BioLiP both apply a similar
# floor. Set deliberately low enough to keep genuine small fragments.
MIN_LIGAND_HEAVY_ATOMS = 6

# Water, in its several PDB spellings.
WATER = frozenset({"HOH", "DOD", "WAT", "H2O", "TIP", "SOL"})

# Monatomic and simple ions. Excluded as ligands: a lone metal contact is
# coordination chemistry, and including it conflates metal sites with
# small-molecule pockets. Metal-binding prediction is a different task.
IONS = frozenset({
    "NA", "K", "MG", "CA", "MN", "FE", "FE2", "CO", "NI", "CU", "CU1", "ZN",
    "CD", "HG", "BA", "SR", "CS", "RB", "LI", "AL", "AG", "AU", "PT", "PD",
    "CL", "BR", "IOD", "F", "FLO", "OH", "O", "OXY", "NH4", "CO3", "NO3",
    "SO3", "PO3", "CYN", "AZI", "SCN", "MO", "W", "V", "CR", "PB", "TL",
    "SM", "GD", "EU", "YB", "LU", "HO", "ER", "TB", "DY", "CE", "LA", "Y",
})

# Crystallisation additives, cryoprotectants, buffers and detergents. These
# are present because of how the crystal was grown, not because the protein
# binds them in any meaningful sense.
ADDITIVES = frozenset({
    # Cryoprotectants and polyols
    "GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "2PE", "P6G", "PE4", "PE5",
    "PE8", "XPE", "7PE", "12P", "15P", "MPD", "MRD", "DIO", "TRS", "BTB",
    "ETX", "P33", "PGO", "PGR", "PDO", "SGM", "MOH", "EOH", "IPA", "DMS",
    "DMF", "ACN", "CCN", "ACT", "ACY", "FMT", "OXL", "TAR", "MLI", "MLA",
    "CIT", "FLC", "TLA", "SIN", "MES", "EPE", "HEZ", "IMD", "BEZ",
    # Anions and salts commonly added to crystallisation buffers
    "SO4", "PO4", "NO2", "BO3", "BR3", "SCN", "CAC", "CAD", "ARS",
    # Detergents and lipids used for solubilisation
    "LDA", "LMT", "DDQ", "C8E", "BOG", "BNG", "HTG", "OGA", "PLM", "MYR",
    "STE", "OLA", "OLB", "OLC", "D12", "SDS", "TWT", "LI1",
    # Misc buffer and handling components
    "AZI", "BME", "DTT", "DTU", "TCE", "TBU", "IOH", "NHE", "BU1", "BU2",
    "BU3", "URE", "GAI", "SAR", "NH2", "UNX", "UNL", "UNK",
})

EXCLUDED_HETERO = WATER | IONS | ADDITIVES

# Standard amino acids, three-letter to one-letter.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    # Common modified residues, mapped to their parent so the chain stays
    # contiguous rather than being broken by a selenomethionine.
    "MSE": "M", "SEC": "C", "PYL": "K", "HYP": "P", "CSO": "C", "CME": "C",
    "MLY": "K", "M3L": "K", "ALY": "K", "SEP": "S", "TPO": "T", "PTR": "Y",
    "KCX": "K", "LLP": "K", "CSD": "C", "OCS": "C", "CSS": "C", "SMC": "C",
    "PCA": "Q", "ASX": "N", "GLX": "Q", "MEN": "N", "MED": "M", "FME": "M",
}

# Residues that are polymer but whose HETATM records must not be read as
# ligands: modified amino acids appear as HETATM in many entries.
MODIFIED_RESIDUES = frozenset(THREE_TO_ONE) - frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
})


def _ssl_context() -> ssl.SSLContext:
    """TLS context with a working CA bundle.

    python.org and some conda macOS builds ship without a usable CA store
    until ``Install Certificates.command`` has been run, so an RCSB download
    fails with CERTIFICATE_VERIFY_FAILED. Verification is never disabled; a
    certifi bundle is supplied when the interpreter has none.
    """
    context = ssl.create_default_context()
    if context.get_ca_certs():
        return context
    try:
        import certifi
    except ImportError:
        return context
    return ssl.create_default_context(cafile=certifi.where())


@dataclass(frozen=True)
class Atom:
    """One heavy atom."""

    name: str
    element: str
    residue_name: str
    chain: str
    residue_seq: int
    insertion: str
    coord: tuple[float, float, float]
    is_hetero: bool
    altloc: str = ""

    @property
    def residue_key(self) -> tuple[str, int, str]:
        return (self.chain, self.residue_seq, self.insertion)


@dataclass
class Ligand:
    """A curated ligand: a het group that passed every exclusion rule."""

    residue_name: str
    chain: str
    residue_seq: int
    insertion: str
    coords: np.ndarray            # (n_atoms, 3)
    elements: list[str]

    @property
    def n_heavy_atoms(self) -> int:
        return int(self.coords.shape[0])

    @property
    def identifier(self) -> str:
        insertion = self.insertion.strip()
        return f"{self.residue_name}_{self.chain}_{self.residue_seq}{insertion}"

    def to_dict(self) -> dict:
        return {
            "ligand": self.residue_name, "chain": self.chain,
            "residue_seq": self.residue_seq,
            "n_heavy_atoms": self.n_heavy_atoms,
            "identifier": self.identifier,
        }


@dataclass
class Residue:
    """One polymer residue with the geometry the model consumes."""

    name: str
    one_letter: str
    chain: str
    residue_seq: int
    insertion: str
    atom_coords: np.ndarray       # (n_atoms, 3) heavy atoms
    ca: np.ndarray | None = None
    cb: np.ndarray | None = None
    n: np.ndarray | None = None
    c: np.ndarray | None = None
    is_binding: bool = False
    min_ligand_distance: float = float("inf")

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.residue_seq, self.insertion)

    @property
    def has_backbone(self) -> bool:
        return self.ca is not None and self.n is not None and self.c is not None


@dataclass
class ProteinStructure:
    """A single-chain protein with curated ligands and binding labels."""

    identifier: str
    chain: str
    residues: list[Residue]
    ligands: list[Ligand] = field(default_factory=list)
    excluded_hetero: dict[str, int] = field(default_factory=dict)
    contact_cutoff: float = CONTACT_CUTOFF
    source: str | None = None
    curation: str = "curated exclusion lists only (no CCD lookup)"

    @property
    def sequence(self) -> str:
        return "".join(r.one_letter for r in self.residues)

    @property
    def n_residues(self) -> int:
        return len(self.residues)

    @property
    def labels(self) -> np.ndarray:
        return np.array([r.is_binding for r in self.residues], dtype=bool)

    @property
    def n_binding(self) -> int:
        return int(self.labels.sum())

    @property
    def binding_fraction(self) -> float:
        return self.n_binding / self.n_residues if self.n_residues else 0.0

    def to_dict(self) -> dict:
        return {
            "identifier": self.identifier, "chain": self.chain,
            "n_residues": self.n_residues, "n_binding": self.n_binding,
            "binding_fraction": self.binding_fraction,
            "contact_cutoff": self.contact_cutoff,
            "n_ligands": len(self.ligands),
            "ligands": [l.to_dict() for l in self.ligands],
            "excluded_hetero": self.excluded_hetero,
            "sequence_length": len(self.sequence),
            "source": self.source,
            "curation": self.curation,
        }


def parse_pdb(
    text: str, identifier: str = "unknown", chain: str | None = None
) -> tuple[list[Residue], list[Atom], dict[str, int]]:
    """Parse PDB text into polymer residues and candidate het atoms.

    Only the first model is read: an NMR ensemble's later models are
    alternative conformers of the same molecule, and treating them as extra
    structures would duplicate a protein across a homology split.

    Alternate locations are resolved to the highest-occupancy variant per
    atom name, which is the standard convention; keeping both would
    double-count atoms in the distance calculation.
    """
    polymer: dict[tuple[str, int, str], Residue] = {}
    het_atoms: list[Atom] = []
    het_counts: dict[str, int] = {}
    # Atoms are accumulated per residue keyed by atom NAME, so an alternate
    # location replaces the lower-occupancy variant instead of being appended
    # alongside it. Appending would double-count the atom in every distance
    # and neighbour-count calculation downstream.
    atoms_by_name: dict[tuple[str, int, str], dict[str, tuple[np.ndarray, float]]] = {}

    for line in text.splitlines():
        record = line[:6]
        if record == "ENDMDL":
            break
        if record not in ("ATOM  ", "HETATM"):
            continue
        if len(line) < 54:
            raise StructureError(
                f"{identifier}: truncated coordinate record: {line[:30]!r}"
            )

        atom_name = line[12:16].strip()
        altloc = line[16].strip()
        residue_name = line[17:20].strip()
        atom_chain = line[21].strip() or "A"
        try:
            residue_seq = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError as error:
            raise StructureError(
                f"{identifier}: unparseable coordinate record: {line[:54]!r}"
            ) from error
        insertion = line[26].strip()
        occupancy = float(line[54:60]) if len(line) >= 60 and line[54:60].strip() else 1.0
        element = (line[76:78].strip() or atom_name[:1]).upper()

        # Hydrogens are dropped: most crystal structures have none, so
        # including them where present would make contact counts depend on
        # whether the depositor modelled hydrogens.
        if element in ("H", "D"):
            continue

        is_hetero = record == "HETATM"
        # A modified amino acid is polymer even though it is a HETATM record.
        if is_hetero and residue_name in MODIFIED_RESIDUES:
            is_hetero = False

        if not is_hetero and residue_name in THREE_TO_ONE:
            if chain is not None and atom_chain != chain:
                continue
            key = (atom_chain, residue_seq, insertion)
            if key not in polymer:
                polymer[key] = Residue(
                    name=residue_name, one_letter=THREE_TO_ONE[residue_name],
                    chain=atom_chain, residue_seq=residue_seq,
                    insertion=insertion, atom_coords=np.empty((0, 3)),
                )
                atoms_by_name[key] = {}

            existing = atoms_by_name[key].get(atom_name)
            if existing is not None and occupancy <= existing[1]:
                continue
            atoms_by_name[key][atom_name] = (
                np.array([x, y, z], dtype=float), occupancy,
            )
        elif is_hetero:
            het_counts[residue_name] = het_counts.get(residue_name, 0) + 1
            het_atoms.append(
                Atom(
                    name=atom_name, element=element, residue_name=residue_name,
                    chain=atom_chain, residue_seq=residue_seq,
                    insertion=insertion, coord=(x, y, z), is_hetero=True,
                    altloc=altloc,
                )
            )

    if not polymer:
        raise StructureError(
            f"{identifier}: no standard polymer residues found"
            + (f" in chain {chain}" if chain else "")
        )

    # Materialise coordinates once, with a stable atom order so the matrix is
    # reproducible, and assign the named backbone atoms.
    for key, residue in polymer.items():
        named = atoms_by_name[key]
        order = sorted(named)
        residue.atom_coords = (
            np.vstack([named[name][0] for name in order])
            if order else np.empty((0, 3))
        )
        for name, attribute in (("CA", "ca"), ("CB", "cb"), ("N", "n"), ("C", "c")):
            if name in named:
                setattr(residue, attribute, named[name][0])

    # Sequence order, with insertion codes breaking ties.
    residues = sorted(polymer.values(), key=lambda r: (r.chain, r.residue_seq, r.insertion))
    return residues, het_atoms, het_counts


def curate_ligands(
    het_atoms: Sequence[Atom],
    min_heavy_atoms: int = MIN_LIGAND_HEAVY_ATOMS,
    extra_excluded: Iterable[str] = (),
    component_cache=None,
    exclude_saccharides: bool = True,
) -> tuple[list[Ligand], dict[str, int]]:
    """Group het atoms into ligands, dropping everything that is not a ligand.

    Two complementary mechanisms, because neither suffices alone:

    * The **Chemical Component Dictionary** (via ``component_cache``)
      identifies modified amino acids and linking saccharides. A hand-written
      list cannot: ``YOF`` (3-fluorotyrosine) and ``CGU``
      (gamma-carboxyglutamate) are modified residues that appear as HETATM,
      and treating them as ligands labels a protein's own backbone neighbours
      as its binding site.
    * The **curated exclusion lists** in this module identify solvent, ions
      and crystallisation additives. The CCD cannot: water, sulfate, glycerol
      and zinc are all legitimately typed ``non-polymer``, so the CCD has no
      basis to distinguish a cryoprotectant from a substrate.

    With no ``component_cache`` the CCD step is skipped and only the lists
    apply. That is the weaker rule, and the caller records that it was used.

    Returns the accepted ligands and a count of what was rejected, by residue
    name. The rejection tally is reported rather than discarded: if a dataset
    turns out to be labelled mostly from sulfate contacts, that has to be
    visible.
    """
    excluded = EXCLUDED_HETERO | {str(name).upper() for name in extra_excluded}

    groups: dict[tuple[str, str, int, str], list[Atom]] = {}
    rejected: dict[str, int] = {}
    classifications: dict[str, str] = {}
    for atom in het_atoms:
        name = atom.residue_name.upper()
        if name in excluded:
            rejected[name] = rejected.get(name, 0) + 1
            continue
        if component_cache is not None:
            if name not in classifications:
                from .chemcomp import classify

                classifications[name] = classify(name, component_cache)
            kind = classifications[name]
            if kind == "polymer":
                rejected[f"{name} (modified residue, not a ligand)"] = (
                    rejected.get(f"{name} (modified residue, not a ligand)", 0) + 1
                )
                continue
            if kind == "saccharide" and exclude_saccharides:
                key = f"{name} (linking saccharide, a glycan modification)"
                rejected[key] = rejected.get(key, 0) + 1
                continue
        groups.setdefault(
            (name, atom.chain, atom.residue_seq, atom.insertion), []
        ).append(atom)

    ligands: list[Ligand] = []
    for (name, chain, seq, insertion), atoms in sorted(groups.items()):
        if len(atoms) < min_heavy_atoms:
            rejected[f"{name} (only {len(atoms)} heavy atoms)"] = len(atoms)
            continue
        ligands.append(
            Ligand(
                residue_name=name, chain=chain, residue_seq=seq,
                insertion=insertion,
                coords=np.array([a.coord for a in atoms], dtype=float),
                elements=[a.element for a in atoms],
            )
        )
    return ligands, rejected


def label_binding_residues(
    residues: Sequence[Residue],
    ligands: Sequence[Ligand],
    cutoff: float = CONTACT_CUTOFF,
) -> None:
    """Mark residues within ``cutoff`` of any ligand heavy atom, in place.

    Distances are computed over **all** heavy atoms of the residue, not just
    CA or CB. A lysine side chain reaches ~6 Å from its CA, so a CA-based
    cutoff would miss side-chain contacts that are the actual interaction, and
    a residue's label would depend on its size rather than its contacts.
    """
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")

    for residue in residues:
        residue.is_binding = False
        residue.min_ligand_distance = float("inf")
    if not ligands:
        return

    ligand_coords = np.vstack([l.coords for l in ligands])
    for residue in residues:
        if residue.atom_coords.size == 0:
            continue
        # (n_residue_atoms, n_ligand_atoms) pairwise distances.
        deltas = residue.atom_coords[:, None, :] - ligand_coords[None, :, :]
        distance = float(np.sqrt((deltas ** 2).sum(axis=-1)).min())
        residue.min_ligand_distance = distance
        residue.is_binding = distance <= cutoff


def structure_from_pdb_text(
    text: str,
    identifier: str = "unknown",
    chain: str | None = None,
    cutoff: float = CONTACT_CUTOFF,
    min_heavy_atoms: int = MIN_LIGAND_HEAVY_ATOMS,
    extra_excluded: Iterable[str] = (),
    require_ligand: bool = True,
    component_cache=None,
    exclude_saccharides: bool = True,
) -> ProteinStructure:
    """Parse, curate and label a structure in one step."""
    residues, het_atoms, _ = parse_pdb(text, identifier, chain)
    ligands, rejected = curate_ligands(
        het_atoms, min_heavy_atoms, extra_excluded,
        component_cache=component_cache,
        exclude_saccharides=exclude_saccharides,
    )

    if require_ligand and not ligands:
        raise NoLigandError(
            f"{identifier}: no het group survives curation "
            f"(rejected: {sorted(rejected)[:8]}). Water, ions and "
            "crystallisation additives are excluded deliberately — labelling "
            "their contacts as binding sites would train the model on "
            "crystallisation artefacts."
        )

    # Restrict to one chain if not already: binding labels are per chain, and
    # a multi-chain structure would otherwise mix sequences.
    selected = chain or (residues[0].chain if residues else "A")
    residues = [r for r in residues if r.chain == selected]
    if not residues:
        raise StructureError(f"{identifier}: chain {selected!r} has no residues")

    label_binding_residues(residues, ligands, cutoff)
    return ProteinStructure(
        identifier=identifier, chain=selected, residues=residues,
        ligands=ligands, excluded_hetero=rejected, contact_cutoff=cutoff,
        curation=(
            "chemical component dictionary + curated exclusion lists"
            if component_cache is not None
            else "curated exclusion lists only (no CCD lookup)"
        ),
    )


def fetch_pdb(
    pdb_id: str, cache_dir: str | Path = "data/pdb", timeout: float = 60.0
) -> Path:
    """Download a PDB entry from RCSB, caching it.

    Downloads the gzipped file and stores it decompressed, so a cached entry
    is readable without this package.
    """
    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4 or not pdb_id[0].isdigit():
        raise DataError(
            f"{pdb_id!r} is not a 4-character PDB id beginning with a digit"
        )
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{pdb_id}.pdb"
    if target.is_file() and target.stat().st_size > 0:
        return target

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
    request = urllib.request.Request(url, headers={"User-Agent": "bindsite/0.1"})
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=_ssl_context()
        ) as response:
            payload = gzip.decompress(response.read()).decode("utf-8", "replace")
    except Exception as error:  # noqa: BLE001 - network and gzip both possible
        raise DataError(f"could not download {pdb_id} from RCSB: {error}") from error

    target.write_text(payload)
    return target


def load_structure(
    path: str | Path,
    identifier: str | None = None,
    chain: str | None = None,
    cutoff: float = CONTACT_CUTOFF,
    **kwargs,
) -> ProteinStructure:
    """Load and label a structure from a local PDB file."""
    path = Path(path)
    if not path.is_file():
        raise StructureError(f"structure file not found: {path}")
    text = (
        gzip.decompress(path.read_bytes()).decode("utf-8", "replace")
        if path.suffix == ".gz" else path.read_text(errors="replace")
    )
    structure = structure_from_pdb_text(
        text, identifier or path.stem, chain=chain, cutoff=cutoff, **kwargs
    )
    structure.source = str(path)
    return structure
