import pandas as pd

from qmt_agent_trader.data.bars import (
    CANONICAL_BAR_COLUMNS,
    _apply_historical_st_flags,
    column_quality,
    enrich_trade_states,
    load_daily_bars,
    normalize_tushare_daily,
)
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.library.price_volume import turnover_20d


def test_normalize_tushare_daily_ignores_empty_marker_column() -> None:
    frame = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "vol": 1000.0,
                "amount": 10000.0,
                "_empty": None,
            }
        ]
    )

    bars = normalize_tushare_daily(frame)

    assert bars.iloc[0]["symbol"] == "000001.SZ"
    assert str(bars.iloc[0]["trade_date"]) == "2024-01-02"
    assert "turnover" in bars.columns


def test_missing_turnover_is_marked_unusable_not_filled_with_zero() -> None:
    frame = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
            }
        ]
    )

    bars = normalize_tushare_daily(frame)

    assert pd.isna(bars.iloc[0]["turnover"])
    assert column_quality(bars, "turnover") == {
        "source": "missing_from_raw",
        "imputed": True,
        "usable_for_factor": False,
    }


def test_turnover_factor_requires_real_turnover() -> None:
    bars = normalize_tushare_daily(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": f"202401{day:02d}",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                }
                for day in range(1, 22)
            ]
        )
    )

    try:
        turnover_20d(bars)
    except ValueError as exc:
        assert "TURNOVER_NOT_REAL_OR_INSUFFICIENT" in str(exc)
    else:
        raise AssertionError("turnover_20d should reject imputed turnover")


def test_load_daily_bars_ignores_legacy_daily_batches(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                }
            ]
        ),
        "raw",
        "tushare_daily_20240101_20240103",
    )

    bars = load_daily_bars(lake, include_trade_state=False)

    assert bars.empty


def test_load_daily_bars_reads_new_registry_daily_dataset_only(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    legacy = {
        "ts_code": "000001.SZ",
        "trade_date": "20240102",
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.0,
    }
    stable = dict(legacy)
    stable["close"] = 10.5

    lake.write_parquet(pd.DataFrame([legacy]), "raw", "tushare_daily")
    lake.write_parquet(pd.DataFrame([stable]), "raw", "tushare/daily")

    bars = load_daily_bars(lake, include_trade_state=False)

    assert len(bars) == 1
    assert bars.iloc[0]["close"] == 10.5


def test_enrich_trade_states_uses_suspend_limit_and_st_sources() -> None:
    bars = normalize_tushare_daily(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 9.5,
                    "close": 10.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "trade_date": "20240102",
                    "open": 4.5,
                    "high": 5.0,
                    "low": 4.5,
                    "close": 4.5,
                },
                {
                    "ts_code": "000003.SZ",
                    "trade_date": "20240102",
                    "open": 8.0,
                    "high": 8.1,
                    "low": 7.9,
                    "close": 8.0,
                },
                {
                    "ts_code": "000004.SZ",
                    "trade_date": "20240102",
                    "open": 12.0,
                    "high": 12.1,
                    "low": 11.8,
                    "close": 12.0,
                },
            ]
        )
    )

    enriched = enrich_trade_states(
        bars,
        suspend=pd.DataFrame(
            [{"ts_code": "000003.SZ", "trade_date": "20240102", "suspend_type": "S"}]
        ),
        stk_limit=pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "up_limit": 10.0,
                    "down_limit": 9.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "trade_date": "20240102",
                    "up_limit": 5.5,
                    "down_limit": 4.5,
                },
            ]
        ),
        namechange=pd.DataFrame(
            [
                {
                    "ts_code": "000003.SZ",
                    "name": "ST Historical",
                    "start_date": "20230101",
                    "end_date": "",
                },
                {
                    "ts_code": "000004.SZ",
                    "name": "*ST Example",
                    "start_date": "20230101",
                    "end_date": "",
                },
            ]
        ),
    ).set_index("symbol")

    assert enriched.loc["000001.SZ", "limit_up"]
    assert enriched.loc["000002.SZ", "limit_down"]
    assert enriched.loc["000003.SZ", "suspended"]
    assert enriched.loc["000003.SZ", "st"]
    assert enriched.loc["000004.SZ", "st"]


def test_load_daily_bars_enriches_trade_states_from_lake(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 9.5,
                    "close": 10.0,
                }
            ]
        ),
        "raw",
        "tushare/daily",
    )
    lake.write_parquet(
        pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20240102", "suspend_type": "S"}]),
        "raw",
        "tushare/suspend_d",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "up_limit": 10.0,
                    "down_limit": 9.0,
                }
            ]
        ),
        "raw",
        "tushare/stk_limit",
    )
    lake.write_parquet(
        pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"]),
        "raw",
        "tushare/namechange",
    )

    bars = load_daily_bars(lake)

    assert bars.iloc[0]["suspended"]
    assert bars.iloc[0]["limit_up"]


def test_load_daily_bars_uses_filtered_reads_for_bars_and_trade_state(
    tmp_path,
    monkeypatch,
) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"]),
        "raw",
        "tushare/suspend_d",
    )
    lake.write_parquet(
        pd.DataFrame(columns=["ts_code", "trade_date", "up_limit", "down_limit"]),
        "raw",
        "tushare/stk_limit",
    )
    lake.write_parquet(
        pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"]),
        "raw",
        "tushare/namechange",
    )
    calls: list[dict[str, object]] = []

    def fake_read_filtered(layer, name, **kwargs):
        calls.append({"layer": layer, "name": name, **kwargs})
        if name == "tushare/daily":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20240102",
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.5,
                        "close": 10.2,
                    }
                ]
            )
        if name == "tushare/fund_daily":
            return pd.DataFrame(columns=["ts_code", "trade_date", "open", "high", "low", "close"])
        if name == "tushare/suspend_d":
            return pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20240102"}])
        if name == "tushare/stk_limit":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20240102",
                        "up_limit": 10.2,
                        "down_limit": 9.0,
                    }
                ]
            )
        if name == "tushare/namechange":
            return pd.DataFrame()
        raise AssertionError(name)

    monkeypatch.setattr(lake, "read_parquet_filtered", fake_read_filtered)

    bars = load_daily_bars(
        lake,
        start="20240101",
        end="20240131",
        symbols=["000001.SZ"],
    )

    assert bars.iloc[0]["suspended"]
    assert bars.iloc[0]["limit_up"]
    assert all(call["start"] == "20240101" for call in calls[:4])
    assert all(call["end"] == "20240131" for call in calls[:4])
    assert all(call["symbols"] == ["000001.SZ"] for call in calls)


def test_load_daily_bars_can_skip_trade_state_enrichment(tmp_path, monkeypatch) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    calls: list[str] = []

    def fake_read_filtered(layer, name, **kwargs):
        calls.append(name)
        if name == "tushare/daily":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20240102",
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.5,
                        "close": 10.2,
                    }
                ]
            )
        return pd.DataFrame(columns=CANONICAL_BAR_COLUMNS)

    monkeypatch.setattr(lake, "read_parquet_filtered", fake_read_filtered)

    bars = load_daily_bars(lake, include_trade_state=False)

    assert not bars.empty
    assert calls == ["tushare/daily", "tushare/fund_daily"]


def test_historical_st_flags_handle_multiple_periods_without_cross_symbol_bleed() -> None:
    bars = normalize_tushare_daily(
        pd.DataFrame(
            [
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                }
                for symbol in ["000001.SZ", "000002.SZ"]
                for trade_date in ["20240102", "20240103", "20240201"]
            ]
        )
    )
    namechange = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "ST First",
                "start_date": "20240101",
                "end_date": "20240115",
            },
            {
                "ts_code": "000001.SZ",
                "name": "*ST Second",
                "start_date": "20240201",
                "end_date": "",
            },
            {
                "ts_code": "000002.SZ",
                "name": "Normal Name",
                "start_date": "20240101",
                "end_date": "",
            },
        ]
    )
    bars["st"] = False

    enriched = _apply_historical_st_flags(bars, namechange).set_index(["symbol", "trade_date"])

    assert enriched.loc[("000001.SZ", pd.Timestamp("2024-01-02").date()), "st"]
    assert enriched.loc[("000001.SZ", pd.Timestamp("2024-02-01").date()), "st"]
    assert not enriched.loc[("000002.SZ", pd.Timestamp("2024-01-02").date()), "st"]
