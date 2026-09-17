"""JSON, CSV helpers and run provenance."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class _Encoder(json.JSONEncoder):
    """Encoder for the numpy and dataclass types that appear in results.

    ``default`` is only consulted for objects the base encoder cannot handle,
    which is why non-finite floats cannot be fixed here: ``np.float64`` is a
    subclass of ``float``, so the encoder serialises it natively and raises on
    NaN under ``allow_nan=False`` before ``default`` is ever called. Non-finite
    values are therefore handled in :func:`sanitise` instead.
    """

    def default(self, o: Any) -> Any:
        # np.bool_ must come first: it is not a subclass of np.integer but
        # json has no encoder for it either, so it would raise.
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def sanitise(payload: Any) -> Any:
    """Replace non-finite floats with ``None`` throughout a payload.

    JSON has no NaN or Infinity. Writing them anyway (``allow_nan=True``)
    produces a file that ``json.load`` accepts but every other language's
    parser rejects, so results become unreadable outside Python. Emitting
    ``null`` instead keeps the file valid and keeps "not computable" visibly
    distinct from a number.

    This has to happen before ``json.dumps`` rather than inside the encoder,
    because both ``float`` and ``np.float64`` are handled natively by the C
    encoder and never reach ``JSONEncoder.default``.
    """
    if isinstance(payload, dict):
        return {k: sanitise(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [sanitise(v) for v in payload]
    if isinstance(payload, np.ndarray):
        return sanitise(payload.tolist())
    # bool before float: numpy bools are not float subclasses, but Python
    # bools are ints, and isinstance checks on numbers are order-sensitive.
    if isinstance(payload, (bool, np.bool_)):
        return bool(payload)
    if isinstance(payload, (float, np.floating)):
        value = float(payload)
        return value if np.isfinite(value) else None
    if is_dataclass(payload) and not isinstance(payload, type):
        return sanitise(asdict(payload))
    return payload


def write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    """Write a payload as JSON, with non-finite floats emitted as ``null``.

    ``allow_nan=False`` is deliberate: it turns a NaN that :func:`sanitise`
    somehow missed into a loud failure rather than a file that only Python can
    read.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitise(payload), indent=indent, cls=_Encoder, allow_nan=False)
        + "\n"
    )
    return path


def write_table(path: str | Path, rows: Iterable[dict], columns: list[str] | None = None) -> Path:
    """Write records as CSV with stable column ordering."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    if columns:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise KeyError(f"requested columns absent from records: {missing}")
        frame = frame[columns]
    frame.to_csv(path, index=False)
    return path


def provenance(extra: dict | None = None) -> dict:
    """Timestamp, interpreter, platform, git commit and package versions.

    Package versions are recorded because a trained model's numbers depend
    on them: torch changes default initialisations and reduction kernels
    between releases, so a stored metric is only interpretable alongside the
    versions that produced it.
    """
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "packages": package_versions(),
    }
    if extra:
        record.update(extra)
    return record


def package_versions(
    names: Iterable[str] = ("numpy", "scipy", "pandas", "scikit-learn", "torch", "biopython"),
) -> dict[str, str | None]:
    """Installed versions of the packages that affect results.

    A package that is absent records ``None`` rather than being omitted, so a
    run on a machine without torch is distinguishable from one where the
    version simply was not asked for.
    """
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None
