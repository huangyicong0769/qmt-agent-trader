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
