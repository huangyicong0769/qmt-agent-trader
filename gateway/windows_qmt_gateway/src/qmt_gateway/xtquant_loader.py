"""Load xtquant from the local QMT installation path."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def add_xtquant_path(path: Path | None) -> None:
    if path and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def can_import_xtquant(path: Path | None = None) -> bool:
    add_xtquant_path(path)
    try:
        importlib.import_module("xtquant")
    except ImportError:
        return False
    return True
