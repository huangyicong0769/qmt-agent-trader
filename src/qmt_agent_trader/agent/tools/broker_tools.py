"""Broker tools intentionally exclude live submit for LLM usage."""

from __future__ import annotations


def query_gateway_health() -> dict[str, str]:
    return {"status": "unknown"}
