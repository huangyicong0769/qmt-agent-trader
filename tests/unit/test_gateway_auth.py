import time

import pytest
from fastapi.testclient import TestClient
from qmt_gateway.app import create_app
from qmt_gateway.auth import GatewayAuthError, NonceStore, client_ip_allowed, verify_headers
from qmt_gateway.config import GatewaySettings

from qmt_agent_trader.core.security import build_auth_headers


def test_gateway_ip_allowlist() -> None:
    assert client_ip_allowed("192.168.1.5", ["192.168.1.0/24"])
    assert not client_ip_allowed("10.0.0.5", ["192.168.1.0/24"])


def test_gateway_nonce_replay() -> None:
    body = {"plan_hash": "h"}
    headers = build_auth_headers(
        api_key="key",
        secret="secret",
        method="POST",
        path="/orders/precheck",
        body=body,
        timestamp=1_800_000_000,
        nonce="nonce",
    )
    store = NonceStore()
    verify_headers(
        expected_api_key="key",
        secret="secret",
        method="POST",
        path="/orders/precheck",
        headers=headers,
        body=body,
        nonce_store=store,
        now=1_800_000_001,
    )
    with pytest.raises(GatewayAuthError, match="replayed nonce"):
        verify_headers(
            expected_api_key="key",
            secret="secret",
            method="POST",
            path="/orders/precheck",
            headers=headers,
            body=body,
            nonce_store=store,
            now=1_800_000_002,
        )


def test_gateway_submit_plan_default_rejects_live_gate(tmp_path) -> None:
    settings = GatewaySettings(
        gateway_api_key="key",
        gateway_hmac_secret="secret",
        allowed_client_cidrs=["192.168.1.0/24"],
        audit_jsonl_path=tmp_path / "audit.jsonl",
    )
    app = create_app(settings)
    client = TestClient(app, client=("192.168.1.10", 12345))
    body = {
        "strategy_approval_status": "APPROVED",
        "approval": {"status": "APPROVED"},
        "risk_checks": {"status": "PASSED"},
        "plan_hash": "hash",
        "idempotency_key": "idem",
    }
    headers = build_auth_headers(
        api_key="key",
        secret="secret",
        method="POST",
        path="/orders/submit-plan",
        body=body,
        timestamp=int(time.time()),
        nonce="n1",
    )
    response = client.post("/orders/submit-plan", json=body, headers=headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is False
    assert "LIVE_TRADING_ENABLED is false" in payload["reasons"]
    assert "ALLOW_ORDER_ENDPOINT is false" in payload["reasons"]
