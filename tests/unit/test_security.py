import pytest

from qmt_agent_trader.core.errors import SecurityError
from qmt_agent_trader.core.security import NonceStore, build_auth_headers, verify_auth_headers


def test_hmac_headers_verify_and_nonce_replay_fails() -> None:
    headers = build_auth_headers(
        api_key="key",
        secret="secret",
        method="POST",
        path="/orders/precheck",
        body={"a": 1},
        timestamp=1_800_000_000,
        nonce="nonce-1",
    )
    store = NonceStore()
    verify_auth_headers(
        expected_api_key="key",
        secret="secret",
        method="POST",
        path="/orders/precheck",
        headers=headers,
        body={"a": 1},
        nonce_store=store,
        now=1_800_000_001,
    )
    with pytest.raises(SecurityError, match="replayed nonce"):
        verify_auth_headers(
            expected_api_key="key",
            secret="secret",
            method="POST",
            path="/orders/precheck",
            headers=headers,
            body={"a": 1},
            nonce_store=store,
            now=1_800_000_002,
        )
