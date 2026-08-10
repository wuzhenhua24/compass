"""Crash-safe file writes.

Every record Compass keeps on disk — the checkpoint index, per-case and
per-trial results, traces, grade records — is written all-at-once or not at
all. A plain ``write_text`` leaves a truncated file if the process dies
mid-write, and a truncated JSON file is worse than a missing one: a missing
checkpoint means "start fresh", a corrupt one means the resume path reads
garbage. The window is not theoretical here — the checkpoint index is
rewritten after every single case and trial.

The mechanism is the usual one: write to a temp file beside the target (same
directory, so ``os.replace`` stays within one filesystem and is atomic), fsync
it, then rename over the target. Readers therefore only ever observe the old
complete file or the new complete file.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def atomic_path(path: Path | str, suffix: str = "") -> Iterator[Path]:
    """Yield a temp path to write; rename it over *path* on clean exit.

    For writers that need a real filename rather than bytes — PIL's
    ``Image.save``, say. The temp file is removed if the block raises, so a
    failed write never leaves a half-finished file behind under either name.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=suffix or ".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path | str, data: bytes) -> Path:
    """Write *data* to *path* atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> Path:
    """Write *text* to *path* atomically."""
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(
    path: Path | str,
    payload: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
    default: Any = str,
) -> Path:
    """Serialize *payload* and write it atomically.

    Serializing before opening anything matters: a payload that fails to
    encode raises without having touched the existing file.
    """
    return atomic_write_text(
        path,
        json.dumps(payload, indent=indent, ensure_ascii=ensure_ascii, default=default),
    )
