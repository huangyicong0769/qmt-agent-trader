"""Validation helpers for declarative universe specs."""

from __future__ import annotations

from typing import Any

ALLOWED_RULE_OPERATORS = {
    "eq",
    "ne",
    "in",
    "not_in",
    "gt",
    "gte",
    "lt",
    "lte",
    "between",
    "contains",
    "starts_with",
    "ends_with",
}


def normalize_symbol(value: Any) -> str | None:
    text = str(value).strip().upper()
    if not text:
        return None
    if "." not in text and text.isdigit() and len(text) == 6:
        suffix = "SZ" if text.startswith(("0", "1", "2", "3")) else "SH"
        text = f"{text}.{suffix}"
    if "." not in text:
        return None
    code, suffix = text.split(".", 1)
    if not (code.isdigit() and len(code) == 6 and suffix in {"SZ", "SH", "BJ"}):
        return None
    return f"{code}.{suffix}"


def normalize_symbols(values: list[Any]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        symbol = normalize_symbol(value)
        if symbol is None:
            raise ValueError(f"invalid explicit symbol: {value!r}")
        if symbol not in normalized:
            normalized.append(symbol)
    return normalized


def looks_like_code_expression(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return any(
        token in lowered
        for token in (
            "__",
            "import(",
            "__import__",
            "eval(",
            "exec(",
            "lambda ",
            "os.",
            "subprocess",
        )
    )
