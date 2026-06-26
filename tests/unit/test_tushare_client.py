from qmt_agent_trader.data.tushare_client import TushareClient


def test_tushare_daily_request_params() -> None:
    request = TushareClient(token="token").build_daily_request(
        start_date="20200101",
        end_date="20200131",
        ts_code="000001.SZ",
    )
    assert request.api_name == "daily"
    assert request.params == {
        "start_date": "20200101",
        "end_date": "20200131",
        "ts_code": "000001.SZ",
    }


def test_tushare_fund_daily_request_params() -> None:
    request = TushareClient(token="token").build_fund_daily_request(
        ts_code="159259.SZ",
        start_date="20250828",
        end_date="20250829",
    )

    assert request.api_name == "fund_daily"
    assert request.params == {
        "ts_code": "159259.SZ",
        "start_date": "20250828",
        "end_date": "20250829",
    }


def test_tushare_fund_daily_request_normalizes_hyphenated_dates() -> None:
    request = TushareClient(token="token").build_fund_daily_request(
        ts_code="159259.SZ",
        start_date="2026-01-01",
        end_date="2026-06-26",
    )

    assert request.params == {
        "ts_code": "159259.SZ",
        "start_date": "20260101",
        "end_date": "20260626",
    }
