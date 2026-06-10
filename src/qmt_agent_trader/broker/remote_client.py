"""Remote client for the Windows QMT Gateway."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx

from qmt_agent_trader.broker.order_plan import OrderPlan
from qmt_agent_trader.core.security import build_auth_headers


class RemoteQMTBrokerClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        hmac_secret: str,
        *,
        timeout: float = 10.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.hmac_secret = hmac_secret
        self._client = http_client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def _path(self, path: str, query: str | None = None) -> str:
        return f"{path}?{query}" if query else path

    def _headers(self, method: str, path: str, body: object | None = None) -> dict[str, str]:
        return build_auth_headers(
            api_key=self.api_key,
            secret=self.hmac_secret,
            method=method,
            path=path,
            body=body or {},
        )

    def _request(self, method: str, path: str, json_body: object | None = None) -> dict[str, Any]:
        parsed = urlparse(path)
        signed_path = self._path(parsed.path, parsed.query or None)
        response = self._client.request(
            method,
            path,
            json=json_body,
            headers=self._headers(method, signed_path, json_body),
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError("gateway returned non-object json")
        return data

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def qmt_status(self) -> dict[str, Any]:
        return self._request("GET", "/qmt/status")

    def query_asset(self) -> dict[str, Any]:
        return self._request("GET", "/account/asset")

    def query_positions(self) -> dict[str, Any]:
        return self._request("GET", "/account/positions")

    def query_orders(self) -> dict[str, Any]:
        return self._request("GET", "/account/orders")

    def query_trades(self) -> dict[str, Any]:
        return self._request("GET", "/account/trades")

    def get_latest_quotes(self, symbols: list[str]) -> dict[str, Any]:
        return self._request("GET", f"/market/latest?symbols={','.join(symbols)}")

    def get_bars(self, symbols: list[str], start: str, end: str, freq: str) -> dict[str, Any]:
        query = f"symbols={','.join(symbols)}&start={start}&end={end}&freq={freq}"
        return self._request("GET", f"/market/bars?{query}")

    def precheck_order_plan(self, order_plan: OrderPlan) -> dict[str, Any]:
        body = order_plan.model_dump(mode="json")
        return self._request("POST", "/orders/precheck", body)

    def submit_order_plan(self, order_plan: OrderPlan) -> dict[str, Any]:
        order_plan.assert_submittable(live=not order_plan.dry_run)
        body = order_plan.model_dump(mode="json")
        body["idempotency_key"] = order_plan.idempotency_key
        return self._request("POST", "/orders/submit-plan", body)
