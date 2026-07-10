"""Minimal repository protocols shared by later migration phases."""

from __future__ import annotations

from typing import Protocol, TypeVar

T = TypeVar("T")


class Repository(Protocol[T]):
    def get(self, identity: str) -> T | None: ...
    def save(self, value: T) -> None: ...
