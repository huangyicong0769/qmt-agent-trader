from __future__ import annotations

import pandas as pd
from pydantic import SecretStr

from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools.remote_data_tools import (
    build_remote_data_tools,
    run_remote_data_update_tool,
    wire,
)
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.tushare_client import TushareClient, TushareRequest


class ExplodingClient(TushareClient):
    def __init__(self) -> None:
        super().__init__(token="secret-token")

    def execute(self, request: TushareRequest) -> pd.DataFrame:
        raise AssertionError(f"unexpected live request: {request.api_name}")


class RecordingClient(TushareClient):
    def __init__(self) -> None:
        super().__init__(token="secret-token")
        self.seen: list[str] = []

    def execute(self, request: TushareRequest) -> pd.DataFrame:
        self.seen.append(request.api_name)
        if request.api_name == "trade_cal":
            return pd.DataFrame([{"cal_date": "20240102", "is_open": 1}])
        if request.api_name == "daily":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": request.params["trade_date"],
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.0,
                        "close": 10.5,
                    }
                ]
            )
        if request.api_name == "suspend_d":
            return pd.DataFrame()
        if request.api_name == "stk_limit":
            return pd.DataFrame()
        raise AssertionError(f"unexpected request: {request.api_name}")


class FailingClient(TushareClient):
    def __init__(self) -> None:
        super().__init__(token="secret-token")

    def execute(self, request: TushareRequest) -> pd.DataFrame:
        raise RuntimeError("upstream rejected request with secret-token")


class RecordingEtfClient(TushareClient):
    def __init__(self) -> None:
        super().__init__(token="secret-token")
        self.requests: list[TushareRequest] = []

    def execute(self, request: TushareRequest) -> pd.DataFrame:
        self.requests.append(request)
        if request.api_name == "fund_basic":
            return pd.DataFrame(
                [{"ts_code": "159259.SZ", "name": "ETF", "list_date": "20200101"}]
            )
        if request.api_name == "fund_daily":
            return pd.DataFrame()
        raise AssertionError(f"unexpected request: {request.api_name}")


def test_run_remote_data_update_fetches_even_when_agent_context_is_dry_run(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = RecordingClient()
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path),
        client_factory=lambda: client,
    )

    result = run_remote_data_update_tool.run(
        {
            "source": "tushare",
            "start_date": "20240102",
            "end_date": "20240102",
            "include_basics": False,
        },
        ToolContext(run_id="r-live-agent", dry_run=True),
    )

    assert result["status"] == "updated"
    assert "daily" in client.seen
    assert lake.dataset_path("raw", "tushare_daily").exists()


def test_run_remote_data_update_skips_live_fetch_when_range_is_covered(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame([{"cal_date": "20240102", "is_open": 1}]),
        "raw",
        "tushare_trade_calendar",
    )
    lake.write_parquet(
        pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20240102"}]),
        "raw",
        "tushare_daily",
    )
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path, tushare_token=None),
        client_factory=lambda: ExplodingClient(),
    )

    result = run_remote_data_update_tool.run(
        {
            "source": "tushare",
            "start_date": "20240102",
            "end_date": "20240102",
            "include_daily": True,
        },
        ToolContext(run_id="r-covered-live", dry_run=False),
    )

    assert result["status"] == "up_to_date"
    assert result["data_update_needed"] is False
    assert result["dry_run"] is False


def test_run_remote_data_update_normalizes_hyphenated_dates_for_tushare(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = RecordingEtfClient()
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path),
        client_factory=lambda: client,
    )

    result = run_remote_data_update_tool.run(
        {
            "source": "tushare",
            "start_date": "2026-01-01",
            "end_date": "2026-06-26",
            "ts_code": "159259.SZ",
            "asset_type": "etf",
            "include_basics": False,
        },
        ToolContext(run_id="r-hyphenated-dates", dry_run=False),
    )

    fund_daily_request = next(item for item in client.requests if item.api_name == "fund_daily")
    assert result["status"] == "PARTIAL_COVERAGE"
    assert fund_daily_request.params["start_date"] == "20260101"
    assert fund_daily_request.params["end_date"] == "20260626"


def test_run_remote_data_update_supports_explicit_dry_run_plan(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path, tushare_token=None),
        client_factory=lambda: ExplodingClient(),
    )

    result = run_remote_data_update_tool.run(
        {
            "source": "tushare",
            "start_date": "20240101",
            "end_date": "20240103",
            "dry_run": True,
        },
        ToolContext(run_id="r-dry", dry_run=True),
    )

    assert result["status"] == "CALENDAR_VALIDATION_REQUIRED"
    assert result["dry_run"] is True
    assert result["data_update_needed"] is True
    assert result["metadata"]["plan_meaning"] == "dry_run_only_no_remote_fetch_performed"
    assert result["metadata"]["requires_trade_calendar_validation"] is True
    assert "requests" in result
    assert not lake.dataset_path("raw", "tushare_daily").exists()


def test_run_remote_data_update_dry_run_uses_trade_calendar_when_available(
    tmp_path,
) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {"cal_date": "20240101", "is_open": 0},
                {"cal_date": "20240102", "is_open": 1},
                {"cal_date": "20240103", "is_open": 1},
            ]
        ),
        "raw",
        "tushare_trade_calendar",
    )
    lake.write_parquet(
        pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20240102"}]),
        "raw",
        "tushare_daily",
    )
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path, tushare_token=None),
        client_factory=lambda: ExplodingClient(),
    )

    result = run_remote_data_update_tool.run(
        {
            "source": "tushare",
            "start_date": "20240101",
            "end_date": "20240103",
            "dry_run": True,
        },
        ToolContext(run_id="r-calendar-dry", dry_run=True),
    )

    assert result["metadata"]["missing_ranges_are_calendar_days"] is False
    assert result["metadata"]["requires_trade_calendar_validation"] is False
    assert result["missing_ranges"] == [{"start_date": "20240103", "end_date": "20240103"}]
    assert result["requested_end_date"] == "20240103"
    assert result["actual_data_end"] == "20240102"
    assert result["coverage_end_date"] == "20240102"
    assert result["data_freshness"] == "missing_expected_trading_dates"


def test_run_remote_data_update_dry_run_does_not_treat_empty_calendar_window_as_covered(
    tmp_path,
) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame([{"cal_date": "20240102", "is_open": 1}]),
        "raw",
        "tushare_trade_calendar",
    )
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path, tushare_token=None),
        client_factory=lambda: ExplodingClient(),
    )

    result = run_remote_data_update_tool.run(
        {
            "source": "tushare",
            "start_date": "20240408",
            "end_date": "20240410",
            "symbols": ["000001.SZ"],
            "dry_run": True,
        },
        ToolContext(run_id="r-empty-calendar-window", dry_run=True),
    )

    assert result["status"] == "CALENDAR_VALIDATION_REQUIRED"
    assert result["data_update_needed"] is True
    assert result["missing_ranges"] == [{"start_date": "20240408", "end_date": "20240410"}]
    assert result["metadata"]["calendar_source"] == "calendar_days"


def test_run_remote_data_update_dry_run_detects_symbol_specific_gaps(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {"cal_date": "20240102", "is_open": 1},
                {"cal_date": "20240103", "is_open": 1},
            ]
        ),
        "raw",
        "tushare_trade_calendar",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20240102"},
                {"ts_code": "000002.SZ", "trade_date": "20240103"},
            ]
        ),
        "raw",
        "tushare_daily",
    )
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path, tushare_token=None),
        client_factory=lambda: ExplodingClient(),
    )

    result = run_remote_data_update_tool.run(
        {
            "source": "tushare",
            "start_date": "20240102",
            "end_date": "20240103",
            "ts_code": "000001.SZ",
            "asset_type": "stock",
            "dry_run": True,
        },
        ToolContext(run_id="r-symbol-calendar-dry", dry_run=True),
    )

    assert result["data_update_needed"] is True
    assert result["missing_ranges"] == [{"start_date": "20240103", "end_date": "20240103"}]
    assert result["actual_data_end"] == "20240102"
    assert result["data_freshness"] == "missing_expected_trading_dates"


def test_run_remote_data_update_dry_run_detects_basket_symbol_gaps(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {"cal_date": "20240102", "is_open": 1},
                {"cal_date": "20240103", "is_open": 1},
            ]
        ),
        "raw",
        "tushare_trade_calendar",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20240102"},
                {"ts_code": "000002.SZ", "trade_date": "20240102"},
                {"ts_code": "000001.SZ", "trade_date": "20240103"},
                {"ts_code": "000999.SZ", "trade_date": "20240103"},
            ]
        ),
        "raw",
        "tushare_daily",
    )
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path, tushare_token=None),
        client_factory=lambda: ExplodingClient(),
    )

    result = run_remote_data_update_tool.run(
        {
            "source": "tushare",
            "start_date": "20240102",
            "end_date": "20240103",
            "symbols": ["000001.SZ", "000002.SZ"],
            "asset_type": "stock",
            "dry_run": True,
        },
        ToolContext(run_id="r-basket-calendar-dry", dry_run=True),
    )

    assert result["data_update_needed"] is True
    assert result["missing_ranges"] == [{"start_date": "20240103", "end_date": "20240103"}]
    assert result["metadata"]["requested_symbols_count"] == 2
    assert result["data_freshness"] == "missing_expected_trading_dates"
    assert result["coverage_by_symbol"]["000001.SZ"]["data_freshness"] == (
        "covers_expected_trading_dates"
    )
    assert result["coverage_by_symbol"]["000002.SZ"]["missing_ranges"] == [
        {"start_date": "20240103", "end_date": "20240103"}
    ]
    assert result["covered_symbols"] == ["000001.SZ"]
    assert result["missing_symbols"] == []
    assert result["stale_symbols"] == ["000002.SZ"]
    assert result["estimated_request_count"] >= 1


def test_run_remote_data_update_dynamic_timeout_scales_with_estimated_requests(
    tmp_path,
) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {"cal_date": f"202401{day:02d}", "is_open": 1}
                for day in range(2, 23)
            ]
        ),
        "raw",
        "tushare_trade_calendar",
    )
    settings = Settings(project_root=tmp_path, tushare_token=None)
    wire(
        data_lake=lake,
        settings=settings,
        client_factory=lambda: ExplodingClient(),
    )

    resolver = run_remote_data_update_tool.timeout_seconds_for_call  # type: ignore[attr-defined]
    timeout_seconds = resolver(
        {
            "source": "tushare",
            "start_date": "20240102",
            "end_date": "20240122",
            "symbols": ["000001.SZ", "000002.SZ"],
            "dry_run": False,
        },
        ToolContext(run_id="r-timeout-estimate"),
    )

    assert timeout_seconds > 300
    assert timeout_seconds <= 3600

    lake.write_parquet(
        pd.DataFrame(
            [
                {"cal_date": f"2024{month:02d}{day:02d}", "is_open": 1}
                for month in range(1, 13)
                for day in range(1, 29)
            ]
        ),
        "raw",
        "tushare_trade_calendar",
    )

    capped_timeout_seconds = resolver(
        {
            "source": "tushare",
            "start_date": "20240101",
            "end_date": "20241228",
            "symbols": ["000001.SZ", "000002.SZ"],
            "dry_run": False,
        },
        ToolContext(run_id="r-timeout-cap"),
    )

    assert capped_timeout_seconds == 3600


def test_build_remote_data_tools_dynamic_timeout_uses_injected_dependencies(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [{"cal_date": f"202401{day:02d}", "is_open": 1} for day in range(2, 23)]
        ),
        "raw",
        "tushare_trade_calendar",
    )
    settings = Settings(project_root=tmp_path, tushare_token=None)
    deps = type("Deps", (), {"data_lake": lake, "settings": settings})()
    built_tool = build_remote_data_tools(deps)[0]

    resolver = built_tool.timeout_seconds_for_call  # type: ignore[attr-defined]
    timeout_seconds = resolver(
        {
            "source": "tushare",
            "start_date": "20240102",
            "end_date": "20240122",
            "symbols": ["000001.SZ", "000002.SZ"],
        },
        ToolContext(run_id="r-built-timeout"),
    )

    assert timeout_seconds > 300


def test_run_remote_data_update_reports_partial_coverage_after_live_fetch(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    client = RecordingClient()
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path),
        client_factory=lambda: client,
    )

    result = run_remote_data_update_tool.run(
        {
            "source": "tushare",
            "start_date": "20240102",
            "end_date": "20240102",
            "symbols": ["000001.SZ", "000002.SZ"],
            "include_basics": False,
        },
        ToolContext(run_id="r-live-partial", dry_run=False),
    )

    assert result["status"] == "PARTIAL_COVERAGE"
    assert result["data_update_needed"] is True
    assert result["covered_symbols"] == ["000001.SZ"]
    assert result["missing_symbols"] == ["000002.SZ"]
    assert result["coverage_by_symbol"]["000002.SZ"]["missing_ranges"] == [
        {"start_date": "20240102", "end_date": "20240102"}
    ]
    assert result["metadata"]["post_update_status"] == "PARTIAL_COVERAGE"
    assert any(write["name"] == "tushare_daily" for write in result["writes"])


def test_run_remote_data_update_dry_run_uses_observed_market_dates_without_calendar(
    tmp_path,
) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {"ts_code": "159259.SZ", "trade_date": "20260626"},
                {"ts_code": "159259.SZ", "trade_date": "20260629"},
            ]
        ),
        "raw",
        "tushare_fund_daily",
    )
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path, tushare_token=None),
        client_factory=lambda: ExplodingClient(),
    )

    result = run_remote_data_update_tool.run(
        {
            "source": "tushare",
            "start_date": "20260626",
            "end_date": "20260630",
            "dry_run": True,
        },
        ToolContext(run_id="r-observed-calendar", dry_run=True),
    )

    assert result["status"] == "planned"
    assert result["metadata"]["calendar_source"] == "observed_market_daily_dates"
    assert result["metadata"]["missing_ranges_are_calendar_days"] is False
    assert result["missing_ranges"] == []
    assert result["requested_end_date"] == "20260630"
    assert result["actual_data_end"] == "20260629"
    assert result["coverage_end_date"] == "20260629"
    assert result["data_freshness"] == "covers_expected_trading_dates"


def test_run_remote_data_update_reports_no_data_before_etf_listing(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [{"ts_code": "159259.SZ", "name": "ETF", "list_date": "20250828"}]
        ),
        "raw",
        "tushare_etf_basic",
    )
    wire(
        data_lake=lake,
        settings=Settings(project_root=tmp_path, tushare_token=None),
        client_factory=lambda: ExplodingClient(),
    )

    result = run_remote_data_update_tool.run(
        {
            "source": "tushare",
            "start_date": "20240101",
            "end_date": "2025-07-10",
            "ts_code": "159259.SZ",
            "asset_type": "etf",
            "dry_run": True,
        },
        ToolContext(run_id="r-pre-listing", dry_run=True),
    )

    assert result["status"] == "NO_DATA_EXPECTED"
    assert result["data_update_needed"] is False
    assert result["metadata"]["reason"] == "requested_end_before_listing"
    assert result["metadata"]["list_date"] == "20250828"


def test_run_remote_data_update_rejects_missing_token_for_live_fetch(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    wire(data_lake=lake, settings=Settings(project_root=tmp_path, tushare_token=None))

    result = run_remote_data_update_tool.run(
        {"source": "tushare", "start_date": "20240101", "end_date": "20240103"},
        ToolContext(run_id="r-live", dry_run=False),
    )

    assert result["status"] == "NOT_CONFIGURED"
    assert "token" in result["message"].lower()


def test_run_remote_data_update_rejects_oversized_ranges(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    wire(
        data_lake=lake,
        settings=Settings(
            project_root=tmp_path,
            tushare_token=SecretStr("secret-token"),
            remote_data_max_days_per_call=1,
        ),
    )

    result = run_remote_data_update_tool.run(
        {"source": "tushare", "start_date": "20240101", "end_date": "20240103"},
        ToolContext(run_id="r-large", dry_run=False),
    )

    assert result["status"] == "INVALID_REQUEST"
    assert "remote_data_max_days_per_call" in result["message"]


def test_run_remote_data_update_sanitizes_service_errors(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    wire(
        data_lake=lake,
        settings=Settings(
            project_root=tmp_path,
            tushare_token=SecretStr("secret-token"),
        ),
        client_factory=lambda: FailingClient(),
    )

    result = run_remote_data_update_tool.run(
        {"source": "tushare", "start_date": "20240101", "end_date": "20240103"},
        ToolContext(run_id="r-error", dry_run=False),
    )

    assert result["status"] == "error"
    assert "secret-token" not in result["message"]
