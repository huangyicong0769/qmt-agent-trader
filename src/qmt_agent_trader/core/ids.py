"""Stable ID helpers."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def utcnow_iso() -> str:
    return datetime.now(tz=ZoneInfo("UTC")).isoformat()


def shanghai_now_iso() -> str:
    return datetime.now(tz=SHANGHAI_TZ).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(tz=SHANGHAI_TZ):%Y%m%d}_{uuid4().hex[:12]}"


def new_nonce() -> str:
    return uuid4().hex


def new_idempotency_key() -> str:
    return str(uuid4())
