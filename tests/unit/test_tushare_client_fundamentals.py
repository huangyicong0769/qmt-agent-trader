from __future__ import annotations

import pytest

from qmt_agent_trader.data.tushare_client import TushareClient


@pytest.fixture
def client() -> TushareClient:
    return TushareClient(token="fake")


def test_build_daily_basic_request_normalizes_dates(client: TushareClient) -> None:
    request = client.build_daily_basic_request(
        trade_date="2024-01-31",
        ts_code="000001.SZ",
    )

    assert request.api_name == "daily_basic"
    assert request.params == {"trade_date": "20240131", "ts_code": "000001.SZ"}
    assert "pe_ttm" in str(request.fields)
    assert "dv_ttm" in str(request.fields)


@pytest.mark.parametrize(
    ("builder_name", "api_name", "field"),
    [
        ("build_income_request", "income", "total_revenue"),
        ("build_balancesheet_request", "balancesheet", "total_assets"),
        ("build_cashflow_request", "cashflow", "n_cashflow_act"),
        ("build_fina_indicator_request", "fina_indicator", "roe"),
        ("build_dividend_request", "dividend", "cash_div_tax"),
    ],
)
def test_build_financial_requests_normalize_dates_and_fields(
    client: TushareClient,
    builder_name: str,
    api_name: str,
    field: str,
) -> None:
    builder = getattr(client, builder_name)

    request = builder(
        start_date="2024-01-01",
        end_date="2024-01-31",
        period="2023-12-31",
        ts_code="000001.SZ",
    )

    assert request.api_name == api_name
    assert request.params == {
        "start_date": "20240101",
        "end_date": "20240131",
        "period": "20231231",
        "ts_code": "000001.SZ",
    }
    assert field in str(request.fields)
    assert "ts_code" in str(request.fields)


def test_build_macro_request_normalizes_common_date_params(
    client: TushareClient,
) -> None:
    request = client.build_macro_request(
        api_name="shibor",
        start_date="2024-01-01",
        end_date="2024-01-31",
        fields="date,on,1w",
        market="CN",
        unused=None,
    )

    assert request.api_name == "shibor"
    assert request.params == {
        "market": "CN",
        "start_date": "20240101",
        "end_date": "20240131",
    }
    assert request.fields == "date,on,1w"
