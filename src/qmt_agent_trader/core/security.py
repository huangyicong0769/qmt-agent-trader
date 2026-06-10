"""API key, HMAC, timestamp, and nonce helpers."""

from __future__ import annotations

import hmac
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256

from qmt_agent_trader.core.errors import SecurityError
from qmt_agent_trader.core.ids import new_nonce

API_KEY_HEADER = "X-QMT-API-Key"
SIGNATURE_HEADER = "X-QMT-Signature"
TIMESTAMP_HEADER = "X-QMT-Timestamp"
NONCE_HEADER = "X-QMT-Nonce"


def canonical_json(payload: object) -> str:
    return json.dumps(
        payload if payload is not None else {},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_request(method: str, path: str, timestamp: str, nonce: str, body: object) -> str:
    return "\n".join([method.upper(), path, timestamp, nonce, canonical_json(body)])


def sign_request(
    *,
    secret: str,
    method: str,
    path: str,
    timestamp: str,
    nonce: str,
    body: object,
) -> str:
    message = canonical_request(method, path, timestamp, nonce, body)
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), sha256).hexdigest()


def build_auth_headers(
    *,
    api_key: str,
    secret: str,
    method: str,
    path: str,
    body: object | None = None,
    timestamp: int | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    ts = str(timestamp or int(time.time()))
    request_nonce = nonce or new_nonce()
    signature = sign_request(
        secret=secret,
        method=method,
        path=path,
        timestamp=ts,
        nonce=request_nonce,
        body=body or {},
    )
    return {
        API_KEY_HEADER: api_key,
        TIMESTAMP_HEADER: ts,
        NONCE_HEADER: request_nonce,
        SIGNATURE_HEADER: signature,
    }


@dataclass
class NonceStore:
    ttl_seconds: int = 300
    _seen: dict[str, int] = field(default_factory=dict)

    def check_and_store(self, nonce: str, now: int | None = None) -> None:
        current = now or int(time.time())
        self._seen = {item: expires for item, expires in self._seen.items() if expires > current}
        if nonce in self._seen:
            raise SecurityError("replayed nonce")
        self._seen[nonce] = current + self.ttl_seconds


def verify_auth_headers(
    *,
    expected_api_key: str,
    secret: str,
    method: str,
    path: str,
    headers: Mapping[str, str],
    body: object | None = None,
    nonce_store: NonceStore | None = None,
    timestamp_tolerance_seconds: int = 300,
    now: int | None = None,
) -> None:
    current = now or int(time.time())
    api_key = headers.get(API_KEY_HEADER) or headers.get(API_KEY_HEADER.lower())
    timestamp = headers.get(TIMESTAMP_HEADER) or headers.get(TIMESTAMP_HEADER.lower())
    nonce = headers.get(NONCE_HEADER) or headers.get(NONCE_HEADER.lower())
    signature = headers.get(SIGNATURE_HEADER) or headers.get(SIGNATURE_HEADER.lower())

    if not api_key or api_key != expected_api_key:
        raise SecurityError("invalid api key")
    if not timestamp or not timestamp.isdigit():
        raise SecurityError("invalid timestamp")
    if abs(current - int(timestamp)) > timestamp_tolerance_seconds:
        raise SecurityError("expired timestamp")
    if not nonce:
        raise SecurityError("missing nonce")
    if not signature:
        raise SecurityError("missing signature")

    expected_signature = sign_request(
        secret=secret,
        method=method,
        path=path,
        timestamp=timestamp,
        nonce=nonce,
        body=body or {},
    )
    if not hmac.compare_digest(signature, expected_signature):
        raise SecurityError("invalid signature")

    if nonce_store is not None:
        nonce_store.check_and_store(nonce, now=current)
