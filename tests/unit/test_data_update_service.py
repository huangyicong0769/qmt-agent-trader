from __future__ import annotations

import pandas as pd

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.tushare_client import TushareClient, TushareRequest
from qmt_agent_trader.services.data_update_service import RequestLimiter, TushareDataUpdateService


class FakeTushareClient(TushareClient):
    def __init__(self) -> None:
        super().__init__(token="fake")
        self.seen: list[str] = []
        self.requests: list[TushareRequest] = []

    def execute(self, request: TushareRequest) -> pd.DataFrame:
        self.seen.append(request.api_name)
        self.requests.append(request)
        if request.api_name == "trade_cal":
            if request.params["start_date"] == "EMPTY":
                return pd.DataFrame()
            return pd.DataFrame(
                [
                    {"cal_date": "20260609", "is_open": 1},
                    {"cal_date": "20260610", "is_open": 0},
                ]
            )
        if request.api_name == "stock_basic":
            return pd.DataFrame([{"ts_code": "000001.SZ", "name": "Ping An Bank"}])
        if request.api_name == "fund_basic":
            return pd.DataFrame(
                [
                    {"ts_code": "510300.SH", "name": "ETF", "list_date": "20120528"},
                    {"ts_code": "159259.SZ", "name": "New ETF", "list_date": "20250828"},
                ]
            )
        if request.api_name == "namechange":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "name": "Ping An Bank",
                        "start_date": "19910403",
                        "end_date": "",
                    }
                ]
            )
        if request.api_name == "daily":
            trade_date = request.params.get("trade_date", request.params.get("start_date"))
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": trade_date,
                        "close": 10.0,
                    }
                ]
            )
        if request.api_name == "fund_daily":
            return pd.DataFrame(
                [
                    {
                        "ts_code": request.params["ts_code"],
                        "trade_date": request.params["start_date"],
                        "open": 1.0,
                        "high": 1.1,
                        "low": 0.9,
                        "close": 1.05,
                        "vol": 100.0,
                        "amount": 1000.0,
                    }
                ]
            )
        if request.api_name == "suspend_d":
            return pd.DataFrame(
                [{"ts_code": "000001.SZ", "trade_date": "20260609", "suspend_type": "N"}]
            )
        if request.api_name == "stk_limit":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": request.params["trade_date"],
                        "up_limit": 11.0,
                        "down_limit": 9.0,
                    }
                ]
            )
        raise AssertionError(request.api_name)


class PagingNamechangeClient(TushareClient):
    def __init__(self) -> None:
        super().__init__(token="fake")
        self.offsets: list[int] = []

    def execute(self, request: TushareRequest) -> pd.DataFrame:
        assert request.api_name == "namechange"
        limit = int(request.params["limit"])
        offset = int(request.params["offset"])
        self.offsets.append(offset)
        rows = [
            {"ts_code": "000001.SZ", "name": "A", "start_date": "20200101", "end_date": ""},
            {"ts_code": "000002.SZ", "name": "B", "start_date": "20200101", "end_date": ""},
            {"ts_code": "000003.SZ", "name": "C", "start_date": "20200101", "end_date": ""},
        ]
        return pd.DataFrame(rows[offset : offset + limit])


class FlakyCalendarClient(FakeTushareClient):
    def __init__(self) -> None:
        super().__init__()
        self.calendar_attempts = 0

    def execute(self, request: TushareRequest) -> pd.DataFrame:
        if request.api_name == "trade_cal":
            self.calendar_attempts += 1
            if self.calendar_attempts == 1:
                raise TimeoutError("temporary timeout")
        return super().execute(request)


def test_tushare_data_update_writes_lake(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = FakeTushareClient()
    result = TushareDataUpdateService(client, lake).update("20260609", "20260610")

    assert result.open_dates == ["20260609"]
    assert {write.name for write in result.writes} == {
        "tushare_trade_calendar",
        "tushare_stock_basic",
        "tushare_etf_basic",
        "tushare_namechange",
        "tushare_daily",
        "tushare_suspend",
        "tushare_stk_limit",
    }
    assert lake.dataset_path("raw", "tushare_daily").exists()
    assert lake.fetch_state("tushare", "tushare_daily")[0]["status"] == "success"


def test_tushare_data_update_is_idempotent_for_overlapping_ranges(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = FakeTushareClient()
    service = TushareDataUpdateService(client, lake)

    service.update("20260609", "20260610")
    service.update("20260609", "20260610")

    daily = lake.read_parquet("raw", "tushare_daily")
    assert daily.drop_duplicates(["ts_code", "trade_date"]).shape[0] == len(daily)
    assert not lake.dataset_path("raw", "tushare_daily_20260609_20260610").exists()


def test_tushare_data_update_refetches_dates_with_required_symbol_gaps(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20260609"}]),
        "raw",
        "tushare_daily",
    )
    client = FakeTushareClient()

    TushareDataUpdateService(client, lake).update(
        "20260609",
        "20260610",
        include_basics=False,
        required_symbols=["000001.SZ", "000002.SZ"],
    )

    daily_requests = [request for request in client.requests if request.api_name == "daily"]
    assert [request.params for request in daily_requests] == [{"trade_date": "20260609"}]


def test_tushare_data_update_falls_back_when_calendar_empty(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = FakeTushareClient()
    result = TushareDataUpdateService(client, lake).update("EMPTY", "20260610")

    daily_write = next(write for write in result.writes if write.name.startswith("tushare_daily"))
    assert result.open_dates == []
    assert daily_write.rows == 1


def test_tushare_data_update_fetches_single_stock_by_ts_code_range(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = FakeTushareClient()

    TushareDataUpdateService(client, lake).update(
        "20260609",
        "20260610",
        include_basics=False,
        ts_code="000001.SZ",
        asset_type="stock",
    )

    daily_requests = [request for request in client.requests if request.api_name == "daily"]
    assert len(daily_requests) == 1
    assert daily_requests[0].params == {
        "start_date": "20260609",
        "end_date": "20260610",
        "ts_code": "000001.SZ",
    }


def test_tushare_namechange_uses_pagination(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = PagingNamechangeClient()

    frame = TushareDataUpdateService(client, lake)._fetch_namechange_pages(page_size=2)

    assert len(frame) == 3
    assert client.offsets == [0, 2]


def test_request_limiter_enforces_minimum_interval() -> None:
    now = [100.0]
    sleeps: list[float] = []

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RequestLimiter(min_interval_seconds=0.5, clock=clock, sleep=sleep)

    limiter.wait()
    limiter.wait()

    assert sleeps == [0.5]


def test_tushare_data_update_retries_transient_request_errors(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = FlakyCalendarClient()

    result = TushareDataUpdateService(
        client,
        lake,
        retry_attempts=2,
        retry_backoff_seconds=0,
    ).update("20260609", "20260610", include_daily=False)

    assert result.open_dates == ["20260609"]
    assert client.calendar_attempts == 2


def test_tushare_data_update_fetches_etf_daily_from_list_date(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = FakeTushareClient()

    result = TushareDataUpdateService(client, lake).update(
        "20240101",
        "20250829",
        ts_code="159259.SZ",
        asset_type="auto",
    )

    assert "daily" not in client.seen
    assert "fund_daily" in client.seen
    assert result.start == "20250828"
    assert lake.dataset_path("raw", "tushare_fund_daily").exists()
    fund_daily = lake.read_parquet("raw", "tushare_fund_daily")
    assert fund_daily["ts_code"].tolist() == ["159259.SZ"]
    assert fund_daily["trade_date"].tolist() == ["20250828"]
