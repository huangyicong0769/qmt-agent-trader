"""Dataset catalog metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetVersion:
    name: str
    layer: str
    path: Path
    version: str
    source: str


def visible_dataset_names(layer: str, names: list[str]) -> list[str]:
    return names
