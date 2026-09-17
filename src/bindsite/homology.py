"""Sequence-identity clustering, for homology-separated splits.

This is the module the whole evaluation rests on. Binding-site predictors are
routinely reported on random splits, and because the PDB is enormously
redundant — many entries are the same protein with a different ligand, or a
close homolog — a random split puts near-identical sequences in train and
test. The reported number then measures memorisation of protein families, not
generalisation to a new one.

Two backends:

**MMseqs2** (Steinegger & Söding, *Nat Biotechnol* 35:1026, 2017) is what
scales. ``cluster_mmseqs`` builds and runs the documented command. It is the
right tool for 145k sequences and is what a real training run would use.

**Built-in greedy clustering** (:func:`cluster_greedy`) actually runs here. It
computes true pairwise alignment identity and applies the same greedy
incremental-centroid rule MMseqs2's linclust step approximates. It is exact
rather than approximate, but it is O(n²) in sequences and O(L²) in length, so
it is suitable for hundreds of proteins and not for hundreds of thousands.
That limit is stated rather than hidden, and :func:`cluster_sequences` reports
which backend produced a clustering.

Identity is computed as matches divided by the length of the **shorter**
sequence. This is the convention MMseqs2 uses by default (``--seq-id-mode 1``
is coverage-based; mode 0 is over the alignment). Dividing by the alignment
length would let a short high-identity local match count as globally similar,
which understates redundancy and therefore leaks.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .exceptions import DataError, ExternalToolError, OptionalDependencyMissing

# Default identity threshold. 30% is the conventional boundary for "different
# family" in this literature and is what the homology-separated split uses.
DEFAULT_IDENTITY = 0.30
# Minimum alignment coverage of the shorter sequence for a pair to be
# considered homologous at all. Without a coverage requirement, a short
# conserved motif shared by unrelated proteins would merge their clusters.
DEFAULT_COVERAGE = 0.50


@dataclass
class Clustering:
    """Assignment of sequences to homology clusters."""

    labels: dict[str, int]
    identity_threshold: float
    coverage_threshold: float
    backend: str
    representatives: dict[int, str] = field(default_factory=dict)
    exact: bool = True

    coherence: dict | None = None
    linkage: str = "centroid"
    n_edges: int | None = None

    @property
    def n_clusters(self) -> int:
        return len(set(self.labels.values()))

    @property
    def n_sequences(self) -> int:
        return len(self.labels)

    def members(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for name, cluster in self.labels.items():
            out.setdefault(cluster, []).append(name)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def cluster_sizes(self) -> dict[int, int]:
        return {k: len(v) for k, v in self.members().items()}

    def to_dict(self) -> dict:
        sizes = sorted(self.cluster_sizes().values(), reverse=True)
        return {
            "backend": self.backend, "exact": self.exact,
            "linkage": self.linkage, "n_similarity_edges": self.n_edges,
            "identity_threshold": self.identity_threshold,
            "coverage_threshold": self.coverage_threshold,
            "n_sequences": self.n_sequences, "n_clusters": self.n_clusters,
            "largest_cluster": sizes[0] if sizes else 0,
            "largest_cluster_fraction": (
                sizes[0] / self.n_sequences if sizes and self.n_sequences else 0.0
            ),
            "n_singletons": sum(1 for s in sizes if s == 1),
            "redundancy": (
                1.0 - self.n_clusters / self.n_sequences if self.n_sequences else 0.0
            ),
            "coherence": self.coherence,
        }


def pairwise_identity(
    left: str, right: str, mode: str = "local"
) -> tuple[float, float]:
    """Alignment identity and coverage, relative to the shorter sequence.

    ``mode="local"`` (Smith-Waterman) is the default, and the choice matters
    more than it appears. BLAST and MMseqs2 both assess homology from
    **local** alignments, so a "30% identity" threshold in this literature
    means 30% over a locally significant region.

    Forcing a **global** end-to-end alignment of two unrelated proteins
    inflates identity: with affine gaps and BLOSUM62 it finds matches
    throughout, and on the 416-protein sample here enough unrelated pairs
    exceeded 30% that the transitive closure of the similarity graph merged
    the entire dataset into a single component — leaving no family to hold
    out. Local alignment does not have this behaviour.

    ``mode="global"`` is retained so that failure is reproducible.

    Returns (identity, coverage), both divided by the length of the **shorter**
    sequence. Dividing identity by the alignment length instead would let a
    short high-identity local match count as globally similar, which
    understates redundancy and therefore leaks.
    """
    try:
        from Bio import Align
        from Bio.Align import substitution_matrices
    except ImportError as error:
        raise OptionalDependencyMissing(
            "Biopython is required for the built-in clustering backend: "
            "pip install biopython. Alternatively install MMseqs2, which is "
            "the backend a real training run should use."
        ) from error

    if not left or not right:
        raise DataError("cannot align an empty sequence")

    if mode not in ("local", "global"):
        raise DataError(f"mode must be local or global, got {mode!r}")

    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    # BLAST's default protein gap costs.
    aligner.open_gap_score = -11.0
    aligner.extend_gap_score = -1.0
    aligner.mode = mode

    # Unknown characters are not in BLOSUM62; map them to X, which is.
    alphabet = set(aligner.substitution_matrix.alphabet)
    left = "".join(c if c in alphabet else "X" for c in left.upper())
    right = "".join(c if c in alphabet else "X" for c in right.upper())

    alignment = aligner.align(left, right)[0]
    top, bottom = alignment[0], alignment[1]
    matches = sum(
        1 for a, b in zip(top, bottom) if a == b and a != "-"
    )
    aligned = sum(1 for a, b in zip(top, bottom) if a != "-" and b != "-")

    shorter = min(len(left), len(right))
    return matches / shorter, aligned / shorter


def cluster_greedy(
    sequences: dict[str, str],
    identity: float = DEFAULT_IDENTITY,
    coverage: float = DEFAULT_COVERAGE,
    mode: str = "local",
) -> Clustering:
    """Greedy incremental clustering by true pairwise identity.

    Sequences are processed longest-first, matching MMseqs2's convention of
    preferring longer representatives: a long sequence is more likely to
    contain a short homolog than the reverse, so this yields fewer spurious
    singletons.

    Exact, and O(n^2) alignments. Fine for hundreds of sequences, unusable for
    hundreds of thousands — use MMseqs2 there.
    """
    if not sequences:
        raise DataError("no sequences to cluster")
    if not 0 < identity <= 1:
        raise DataError(f"identity must be in (0, 1], got {identity}")
    if not 0 < coverage <= 1:
        raise DataError(f"coverage must be in (0, 1], got {coverage}")

    order = sorted(sequences, key=lambda k: (-len(sequences[k]), k))
    labels: dict[str, int] = {}
    representatives: dict[int, str] = {}

    for name in order:
        assigned = None
        for cluster, representative in representatives.items():
            score, covered = pairwise_identity(
                sequences[name], sequences[representative], mode=mode
            )
            if score >= identity and covered >= coverage:
                assigned = cluster
                break
        if assigned is None:
            assigned = len(representatives)
            representatives[assigned] = name
        labels[name] = assigned

    return Clustering(
        labels=labels, identity_threshold=identity, coverage_threshold=coverage,
        backend="greedy_pairwise", representatives=representatives, exact=True,
        linkage=f"centroid, {mode} alignment",
    )


def cluster_coherence(
    clustering: Clustering,
    sequences: dict[str, str],
    max_pairs_per_cluster: int = 40,
    seed: int = 0,
) -> dict:
    """Measure whether clusters are internally coherent, and report it.

    Greedy centroid clustering compares each sequence only to a cluster's
    **representative**, so it chains: A at 31% identity to representative R
    and B at 31% to R land in one cluster even if A and B share 9%. CD-HIT and
    MMseqs2 behave the same way; it is a property of the algorithm, not a bug
    in this implementation.

    For leakage control, chaining is the safe direction — over-merging puts
    unrelated proteins in the same fold, which never splits a homologous pair.
    It does, however, coarsen the split: one cluster holding 20% of the data
    forces 20% of the proteins into the same fold, which makes realised fold
    fractions deviate from the requested ones and reduces the effective number
    of independent test families.

    That trade-off is real and is reported rather than presented as clean
    family separation. ``min_within_cluster_identity`` well below the
    threshold is the signature of chaining.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    per_cluster: dict[int, dict] = {}
    for cluster, names in clustering.members().items():
        if len(names) < 2:
            continue
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
        if len(pairs) > max_pairs_per_cluster:
            index = rng.choice(len(pairs), size=max_pairs_per_cluster, replace=False)
            pairs = [pairs[i] for i in index]
        identities = [
            pairwise_identity(sequences[a], sequences[b])[0]
            for a, b in pairs
            if a in sequences and b in sequences
        ]
        if not identities:
            continue
        per_cluster[cluster] = {
            "size": len(names),
            "n_pairs_sampled": len(identities),
            "min_identity": float(min(identities)),
            "mean_identity": float(np.mean(identities)),
        }

    if not per_cluster:
        return {"n_multi_member_clusters": 0, "note": "all clusters are singletons"}

    minima = np.array([v["min_identity"] for v in per_cluster.values()])
    chained = {
        c: v for c, v in per_cluster.items()
        if v["min_identity"] < clustering.identity_threshold
    }
    return {
        "n_multi_member_clusters": len(per_cluster),
        "min_within_cluster_identity": float(minima.min()),
        "mean_within_cluster_min_identity": float(minima.mean()),
        "n_clusters_showing_chaining": len(chained),
        "fraction_clusters_chained": len(chained) / len(per_cluster),
        "identity_threshold": clustering.identity_threshold,
        "interpretation": (
            "members within a cluster can fall below the identity threshold to "
            "each other because centroid linkage compares only to the "
            "representative. This over-merges, which is conservative for "
            "leakage control but coarsens the split."
        ),
        "per_cluster": dict(sorted(per_cluster.items())),
    }


@dataclass
class IdentityMatrix:
    """All pairwise identities and coverages for a set of sequences.

    Computing this once and reusing it is what makes the direct split
    verification affordable. Recomputing alignments on every run costs
    O(n_train x n_test) local alignments — about 22000 for a 416-protein
    dataset, several minutes — which is enough that the verification would
    end up switched off, which defeats its purpose.
    """

    names: list[str]
    identity: "np.ndarray"
    coverage: "np.ndarray"
    mode: str = "local"

    def index_of(self, name: str) -> int:
        return self._index[name]

    def __post_init__(self) -> None:
        self._index = {name: i for i, name in enumerate(self.names)}

    def get(self, left: str, right: str) -> tuple[float, float]:
        i, j = self._index[left], self._index[right]
        return float(self.identity[i, j]), float(self.coverage[i, j])

    def covers(self, names) -> bool:
        return set(names) <= set(self._index)

    def save(self, path) -> "Path":
        """Store as compressed .npz.

        Not JSON: a 416-protein matrix is 173000 floats, which is several
        megabytes of text and slow to parse. The file is a derived artifact
        and is regenerable from the sequences, so it is not committed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, identity=self.identity, coverage=self.coverage,
            names=np.array(self.names), mode=np.array(self.mode),
        )
        return path

    @classmethod
    def load(cls, path) -> "IdentityMatrix":
        payload = np.load(Path(path), allow_pickle=False)
        return cls(
            names=[str(n) for n in payload["names"]],
            identity=np.asarray(payload["identity"], dtype=float),
            coverage=np.asarray(payload["coverage"], dtype=float),
            mode=str(payload["mode"]),
        )

    def distribution(self) -> dict:
        """Percentiles of the off-diagonal identities.

        Reported because the threshold's meaning depends on this: if the
        median identity between arbitrary pairs is close to the threshold,
        the threshold is not separating homologs from background.
        """
        import numpy as np

        n = len(self.names)
        if n < 2:
            return {}
        off = self.identity[np.triu_indices(n, 1)]
        return {
            "mode": self.mode,
            "n_pairs": int(off.size),
            "mean": float(off.mean()),
            **{f"p{q}": float(np.percentile(off, q)) for q in (50, 75, 90, 95, 99)},
            "max": float(off.max()),
        }


def identity_matrix(
    sequences: dict[str, str], mode: str = "local"
) -> IdentityMatrix:
    """Compute every pairwise identity. O(n^2) alignments."""
    import numpy as np

    if not sequences:
        raise DataError("no sequences supplied")
    names = sorted(sequences)
    n = len(names)
    ident = np.eye(n)
    cov = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = pairwise_identity(sequences[names[i]], sequences[names[j]], mode)
            ident[i, j] = ident[j, i] = a
            cov[i, j] = cov[j, i] = b
    return IdentityMatrix(names=names, identity=ident, coverage=cov, mode=mode)


def cluster_from_matrix(
    matrix: IdentityMatrix,
    identity: float = DEFAULT_IDENTITY,
    coverage: float = DEFAULT_COVERAGE,
) -> Clustering:
    """Connected-components clustering from a precomputed identity matrix.

    Separating this from the alignment work means a threshold can be changed,
    or a sweep run, without recomputing 86000 alignments.
    """
    import numpy as np

    names = matrix.names
    n = len(names)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_edges = 0
    above = (matrix.identity >= identity) & (matrix.coverage >= coverage)
    for i in range(n):
        for j in range(i + 1, n):
            if above[i, j]:
                n_edges += 1
                a, b = find(i), find(j)
                if a != b:
                    parent[b] = a

    roots: dict[int, int] = {}
    labels: dict[str, int] = {}
    for i, name in enumerate(names):
        root = find(i)
        if root not in roots:
            roots[root] = len(roots)
        labels[name] = roots[root]

    return Clustering(
        labels=labels, identity_threshold=identity, coverage_threshold=coverage,
        backend="connected_components", exact=True,
        representatives={index: names[root] for root, index in roots.items()},
        linkage=f"single (transitive closure), {matrix.mode} alignment",
        n_edges=n_edges,
    )


def cluster_connected_components(
    sequences: dict[str, str],
    identity: float = DEFAULT_IDENTITY,
    coverage: float = DEFAULT_COVERAGE,
    mode: str = "local",
) -> Clustering:
    """Cluster as connected components of the "identity >= threshold" graph.

    **This is the correct algorithm for building a homology-separated split**,
    and it is the default for that reason.

    A split is only valid if no train-test pair exceeds the identity
    threshold. That property is exactly "clusters are the connected components
    of the graph whose edges join pairs above the threshold". Any clustering
    that does not compute the transitive closure can leave such a pair in
    different clusters, and then the split leaks.

    Greedy centroid clustering (:func:`cluster_greedy`, and the same scheme
    CD-HIT and MMseqs2's linclust use) compares each sequence only to a
    cluster **representative**. It therefore fails in both directions:

    * it **over-merges** — A and B both within threshold of representative R
      join one cluster even when A and B are unrelated to each other; and
    * it **under-merges** — A can be 47% identical to B while being below
      threshold to B's representative, so they end up in different clusters.

    The second failure is the dangerous one. Measured on the 416-protein
    sample in this repository, greedy clustering placed ``2VKF`` and ``1LFK``
    — 46.8% identical — in different clusters, and the resulting split was
    rejected by :func:`~bindsite.data.splits.assert_no_homology_leakage`.

    The cost is all pairs: O(n^2) alignments, about 86000 for 416 sequences.
    That is a few minutes once, and it is cached. For 145k sequences it is
    impossible and MMseqs2 is required — with the caveat that MMseqs2's own
    clustering has the same transitive-closure gap, so a split built from it
    should still be verified directly against the sequences.

    Over-merging is retained and is the conservative direction: unrelated
    proteins sharing a fold costs split granularity, never validity.
    """
    if not sequences:
        raise DataError("no sequences to cluster")
    if not 0 < identity <= 1:
        raise DataError(f"identity must be in (0, 1], got {identity}")
    if not 0 < coverage <= 1:
        raise DataError(f"coverage must be in (0, 1], got {coverage}")

    names = sorted(sequences)
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    n_edges = 0
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            # Skip the alignment when the two are already in one component:
            # the edge cannot change the partition, and this saves a large
            # fraction of the work on a redundant dataset.
            if find(left) == find(right):
                continue
            score, covered = pairwise_identity(
                sequences[left], sequences[right], mode=mode
            )
            if score >= identity and covered >= coverage:
                union(left, right)
                n_edges += 1

    roots: dict[str, int] = {}
    labels: dict[str, int] = {}
    for name in names:
        root = find(name)
        if root not in roots:
            roots[root] = len(roots)
        labels[name] = roots[root]

    return Clustering(
        labels=labels, identity_threshold=identity, coverage_threshold=coverage,
        backend="connected_components", exact=True,
        representatives={index: root for root, index in roots.items()},
        linkage="single (transitive closure)",
        n_edges=n_edges,
    )


def mmseqs_available() -> bool:
    """Whether an ``mmseqs`` executable is on PATH."""
    return shutil.which("mmseqs") is not None


def mmseqs_command(
    fasta: str | Path,
    output_prefix: str | Path,
    identity: float = DEFAULT_IDENTITY,
    coverage: float = DEFAULT_COVERAGE,
    threads: int = 8,
) -> list[str]:
    """Build the MMseqs2 ``easy-cluster`` command.

    ``--cov-mode 0`` requires the coverage on both sequences, and
    ``--cluster-mode 0`` is greedy set-cover, which is the setting that
    matches the intent of a homology-separated split: every member of a
    cluster is within the threshold of its representative.
    """
    return [
        "mmseqs", "easy-cluster", str(fasta), str(output_prefix),
        str(Path(output_prefix).parent / "mmseqs_tmp"),
        "--min-seq-id", f"{identity}",
        "-c", f"{coverage}",
        "--cov-mode", "0",
        "--cluster-mode", "0",
        "--threads", str(threads),
    ]


def write_fasta(sequences: dict[str, str], path: str | Path) -> Path:
    """Write sequences as FASTA, wrapped at 60 columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, sequence in sequences.items():
        lines.append(f">{name}")
        lines.extend(sequence[i:i + 60] for i in range(0, len(sequence), 60))
    path.write_text("\n".join(lines) + "\n")
    return path


def cluster_mmseqs(
    sequences: dict[str, str],
    work_dir: str | Path,
    identity: float = DEFAULT_IDENTITY,
    coverage: float = DEFAULT_COVERAGE,
    threads: int = 8,
    dry_run: bool = False,
) -> Clustering:
    """Cluster with MMseqs2.

    With ``dry_run`` the command is constructed and returned as metadata
    without being executed, so the intended invocation is inspectable on a
    machine without MMseqs2 installed.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    fasta = write_fasta(sequences, work / "sequences.fasta")
    prefix = work / "clusters"
    command = mmseqs_command(fasta, prefix, identity, coverage, threads)

    if dry_run:
        return Clustering(
            labels={}, identity_threshold=identity, coverage_threshold=coverage,
            backend="mmseqs2 (dry run, not executed)", exact=True,
        )
    if not mmseqs_available():
        raise ExternalToolError(
            "mmseqs is not on PATH. Install MMseqs2 "
            "(https://github.com/soedinglab/MMseqs2), or use the built-in "
            "greedy backend, which is exact but O(n^2) and therefore suitable "
            f"only for hundreds of sequences. Intended command: {' '.join(command)}"
        )

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ExternalToolError(
            f"mmseqs easy-cluster failed (exit {result.returncode}): "
            f"{result.stderr.strip()[-500:]}"
        )

    tsv = prefix.with_name(prefix.name + "_cluster.tsv")
    if not tsv.is_file():
        raise ExternalToolError(
            f"mmseqs reported success but {tsv} is absent; cannot read clusters"
        )

    # Each line is "representative\tmember".
    assignment: dict[str, int] = {}
    representatives: dict[int, str] = {}
    index: dict[str, int] = {}
    for line in tsv.read_text().splitlines():
        if not line.strip():
            continue
        representative, member = line.split("\t")[:2]
        if representative not in index:
            index[representative] = len(index)
            representatives[index[representative]] = representative
        assignment[member] = index[representative]

    missing = sorted(set(sequences) - set(assignment))
    if missing:
        raise ExternalToolError(
            f"mmseqs did not assign {len(missing)} sequence(s), e.g. {missing[:5]}"
        )

    return Clustering(
        labels=assignment, identity_threshold=identity,
        coverage_threshold=coverage, backend="mmseqs2",
        representatives=representatives, exact=True,
    )


def cluster_sequences(
    sequences: dict[str, str],
    identity: float = DEFAULT_IDENTITY,
    coverage: float = DEFAULT_COVERAGE,
    backend: str = "auto",
    work_dir: str | Path = "data/clustering",
    threads: int = 8,
) -> Clustering:
    """Cluster sequences for a homology-separated split.

    ``backend="auto"`` uses :func:`cluster_connected_components`, which is the
    only backend here that guarantees the property a split needs: no pair
    above the identity threshold ends up in different clusters. It is O(n^2)
    and therefore practical for hundreds to a few thousand sequences.

    ``"mmseqs"`` scales to the full PDB but its clustering, like greedy
    centroid clustering, does not compute the transitive closure, so a split
    built from it must be verified directly against the sequences.
    ``"greedy"`` is retained for comparison and to demonstrate the failure.

    The result always records which backend and linkage produced it, because
    they are not interchangeable and a reader needs to know.
    """
    if backend not in ("auto", "mmseqs", "greedy", "connected_components"):
        raise DataError(
            f"backend must be auto, mmseqs, greedy or connected_components; "
            f"got {backend!r}"
        )
    if backend == "mmseqs":
        return cluster_mmseqs(sequences, work_dir, identity, coverage, threads)
    if backend == "greedy":
        return cluster_greedy(sequences, identity, coverage)
    return cluster_connected_components(sequences, identity, coverage)


def max_identity_between(
    left: Sequence[str], right: Sequence[str], sequences: dict[str, str]
) -> tuple[float, str, str]:
    """Highest pairwise identity between two groups, and which pair achieved it.

    Used to verify a split rather than trust it: a clustering can be correct
    while the split built from it is not, so the invariant is checked directly
    on the sequences that ended up in each fold.
    """
    best, best_pair = 0.0, ("", "")
    for a in left:
        for b in right:
            score, _ = pairwise_identity(sequences[a], sequences[b])
            if score > best:
                best, best_pair = score, (a, b)
    return best, best_pair[0], best_pair[1]
