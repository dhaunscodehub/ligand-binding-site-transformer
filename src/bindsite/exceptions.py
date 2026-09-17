"""Exception types, and the exit codes they map to.

Separated so the CLI can distinguish a scientifically invalid run from a bad
input from a missing tool, and so a caller in a batch script can act on the
difference without parsing messages.
"""

from __future__ import annotations


class BindSiteError(Exception):
    """Base class for every error this package raises deliberately."""


class DataError(BindSiteError, ValueError):
    """Input data is missing, malformed, or unusable as requested."""


class StructureError(DataError):
    """A structure file cannot be parsed or lacks what is needed."""


class NoLigandError(DataError):
    """A structure contains no ligand that passes curation.

    Distinct from :class:`StructureError` because it is an expected outcome,
    not a failure: most PDB entries contain only water, ions and
    crystallisation additives, and those must not be labelled as binding sites.
    """


class ConfigError(BindSiteError, ValueError):
    """Configuration is invalid or contains unrecognised keys."""


class OptionalDependencyMissing(BindSiteError, ImportError):
    """An optional dependency is required for the requested operation."""


class ExternalToolError(BindSiteError, RuntimeError):
    """An external tool (MMseqs2) is unavailable or failed."""


class LeakageError(BindSiteError, AssertionError):
    """A split that promised homology separation does not have it.

    An ``AssertionError`` subclass because that is what it is: a violated
    invariant, not a recoverable condition. Maps to exit code 5.
    """
