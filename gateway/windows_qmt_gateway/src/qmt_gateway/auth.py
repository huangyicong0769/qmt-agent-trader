"""Gateway authentication and replay protection."""

from __future__ import annotations

import hmac
import ipaddress
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256

from fastapi import HTTPException, Request, status

API_KEY_HEADER = "X-QMT-API-Key"
SIGNATURE_HEADER = "X-QMT-Signature"
TIMESTAMP_HEADER = "X-QMT-Timestamp"
NONCE_HEADER = "X-QMT-Nonce"


class GatewayAuthError(Exception):
    pass


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
    secret: str, method: str, path: str, timestamp: str, nonce: str, body: object
) -> str:
    message = canonical_request(method, path, timestamp, nonce, body)
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), sha256).hexdigest()


@dataclass
class NonceStore:
    ttl_seconds: int = 300
    _seen: dict[str, int] = field(default_factory=dict)

    def check_and_store(self, nonce: str, now: int | None = None) -> None:
        current = now or int(time.time())
        self._seen = {key: expires for key, expires in self._seen.items() if expires > current}
        if nonce in self._seen:
            raise GatewayAuthError("replayed nonce")
        self._seen[nonce] = current + self.ttl_seconds


def client_ip_allowed(client_ip: str, cidrs: list[str]) -> bool:
    ip = ipaddress.ip_address(client_ip)
    return any(ip in ipaddress.ip_network(cidr, strict=False) for cidr in cidrs)


def verify_headers(
    *,
    expected_api_key: str,
    secret: str,
    method: str,
    path: str,
    headers: Mapping[str, str],
    body: object,
    nonce_store: NonceStore,
    now: int | None = None,
) -> None:
    current = now or int(time.time())
    api_key = headers.get(API_KEY_HEADER) or headers.get(API_KEY_HEADER.lower())
    timestamp = headers.get(TIMESTAMP_HEADER) or headers.get(TIMESTAMP_HEADER.lower())
    nonce = headers.get(NONCE_HEADER) or headers.get(NONCE_HEADER.lower())
    signature = headers.get(SIGNATURE_HEADER) or headers.get(SIGNATURE_HEADER.lower())
    if not expected_api_key or not secret:
        raise GatewayAuthError("gateway auth secret is not configured")
    if api_key != expected_api_key:
        raise GatewayAuthError("invalid api key")
    if timestamp is None or not timestamp.isdigit() or abs(current - int(timestamp)) > 300:
        raise GatewayAuthError("invalid timestamp")
    if not nonce:
        raise GatewayAuthError("missing nonce")
    if not signature:
        raise GatewayAuthError("missing signature")
    expected = sign_request(secret, method, path, timestamp, nonce, body)
    if not hmac.compare_digest(signature, expected):
        raise GatewayAuthError("invalid signature")
    nonce_store.check_and_store(nonce, now=current)


async def require_auth(request: Request) -> None:
    settings = request.app.state.settings
    nonce_store = request.app.state.nonce_store
    client_host = request.client.host if request.client else ""
    if not client_host or not client_ip_allowed(client_host, settings.allowed_client_cidrs):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="client ip not allowed")
    body: object = {}
    if request.method.upper() in {"POST", "PUT", "PATCH"}:
        raw_body = await request.body()
        body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
    signed_path = request.url.path
    if request.url.query:
        signed_path = f"{signed_path}?{request.url.query}"
    try:
        verify_headers(
            expected_api_key=settings.gateway_api_key,
            secret=settings.gateway_hmac_secret,
            method=request.method,
            path=signed_path,
            headers=request.headers,
            body=body,
            nonce_store=nonce_store,
        )
    except GatewayAuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
