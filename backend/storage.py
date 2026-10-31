"""Uploaded-file housekeeping that tolerates Windows file locking."""
from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from .models import Document

log = logging.getLogger(__name__)


def remove_file(path: str | Path) -> bool:
    """Delete a stored upload. Returns False (and logs) instead of raising if the file is locked,
    e.g. open in a PDF viewer on Windows. Leftovers are removed by `remove_orphan_uploads`."""
    try:
        Path(path).unlink(missing_ok=True)
        return True
    except OSError as exc:
        log.warning("Could not delete %s yet (%s); it will be removed at next startup.", path, exc)
        return False


def remove_orphan_uploads(db: Session, uploads_dir: Path) -> int:
    """Delete files in uploads_dir that no document row references. Returns how many were removed."""
    if not uploads_dir.is_dir():
        return 0
    known = {Path(p).name for (p,) in db.query(Document.stored_path)}
    return sum(remove_file(f) for f in uploads_dir.glob("*.pdf") if f.name not in known)
