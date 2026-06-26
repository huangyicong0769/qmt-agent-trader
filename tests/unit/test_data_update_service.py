from __future__ import annotations

import pandas as pd

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.tushare_client import TushareClient, TushareRequest
from qmt_agent_trader.services.data_update_service import RequestLimiter, TushareDataUpdateService


class FakeTushareClient(TushareClient):
    def __init__(self) -> None:
        super().__init__(token="fake")
        self.seen: list[str] = []

    def execute(self, request: TushareRequest) -> pd.DataFrame:
        self.seen.append(request.api_name)
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
            return pd.DataFrame([{"ts_code": "510300.SH", "name": "ETF"}])
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


def test_tushare_data_update_falls_back_when_calendar_empty(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = FakeTushareClient()
    result = TushareDataUpdateService(client, lake).update("EMPTY", "20260610")

    daily_write = next(write for write in result.writes if write.name.startswith("tushare_daily"))
    assert result.open_dates == []
    assert daily_write.rows == 1


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
