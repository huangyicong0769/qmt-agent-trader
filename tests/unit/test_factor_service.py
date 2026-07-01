from datetime import date, timedelta

import pandas as pd

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.service import (
    compute_factor_to_lake,
    evaluate_factor,
    validate_factor,
    walk_forward_factor_validation,
)


def test_compute_factor_to_lake(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    start = date(2024, 1, 1)
    rows = [
        {
            "ts_code": "000001.SZ",
            "trade_date": f"{start + timedelta(days=offset):%Y%m%d}",
            "open": 10.0 + offset,
            "high": 11.0 + offset,
            "low": 9.0 + offset,
            "close": 10.0 + offset,
            "vol": 1000.0,
            "amount": 10000.0,
        }
        for offset in range(21)
    ]
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare_daily")

    result = compute_factor_to_lake(lake, name="momentum_20d", date="20240121")

    assert result.rows == 1
    assert result.non_null == 1
    assert lake.dataset_path("gold", "factor_momentum_20d_20240121").exists()


def test_compute_fundamental_factor_to_lake_uses_pit_context(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240131",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                },
                {
                    "ts_code": "000002.SZ",
                    "trade_date": "20240131",
                    "open": 20.0,
                    "high": 21.0,
                    "low": 19.0,
                    "close": 20.5,
                },
            ]
        ),
        "raw",
        "tushare_daily",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20240131", "pe_ttm": 4.0},
                {"ts_code": "000002.SZ", "trade_date": "20240131", "pe_ttm": 8.0},
            ]
        ),
        "raw",
        "tushare_daily_basic",
    )

    result = compute_factor_to_lake(lake, name="pe_ttm_rank", date="20240131")

    assert result.rows == 2
    assert result.non_null == 2
    output = lake.read_parquet("gold", "factor_pe_ttm_rank_20240131")
    assert output["factor_value"].tolist() == [0.5, 1.0]


def test_validate_factor_computes_ic(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    start = date(2024, 1, 1)
    rows = []
    for offset in range(22):
        trade_date = f"{start + timedelta(days=offset):%Y%m%d}"
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "open": 10.0 + offset,
                "high": 11.0 + offset,
                "low": 9.0 + offset,
                "close": 10.0 + offset,
            }
        )
        rows.append(
            {
                "ts_code": "000002.SZ",
                "trade_date": trade_date,
                "open": 20.0 + offset,
                "high": 21.0 + offset,
                "low": 19.0 + offset,
                "close": 20.0 + offset,
            }
        )
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare_daily")

    result = validate_factor(lake, name="momentum_20d", start="20240121", end="20240121")

    assert result.observations == 2
    assert result.non_null == 2


def test_walk_forward_factor_validation_slices_history(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    start = date(2024, 1, 1)
    rows = []
    for offset in range(35):
        trade_date = f"{start + timedelta(days=offset):%Y%m%d}"
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "open": 10.0 + offset,
                "high": 11.0 + offset,
                "low": 9.0 + offset,
                "close": 10.0 + offset,
            }
        )
        rows.append(
            {
                "ts_code": "000002.SZ",
                "trade_date": trade_date,
                "open": 20.0 + offset * 0.1,
                "high": 21.0 + offset * 0.1,
                "low": 19.0 + offset * 0.1,
                "close": 20.0 + offset * 0.1,
            }
        )
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare_daily")

    result = walk_forward_factor_validation(
        lake,
        name="momentum_20d",
        start="20240121",
        end="20240202",
        window_days=5,
        step_days=5,
        quantile=0.5,
    )
    payload = result.as_dict()

    assert payload["slice_count"] == 3
    assert payload["positive_slice_ratio"] == 1.0
    assert result.slices[0].mean_ic is not None
    assert result.slices[0].mean_ic > 0
    assert result.slices[0].long_short_spread is not None
    assert result.slices[0].long_short_spread > 0


def test_validate_factor_loads_only_requested_window_with_lookback(
    tmp_path,
    monkeypatch,
) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    calls: list[dict[str, object]] = []
    start = date(2024, 1, 1)
    rows = []
    for offset in range(25):
        trade_date = start + timedelta(days=offset)
        for symbol, base in [("000001.SZ", 10.0), ("000002.SZ", 20.0)]:
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "open": base + offset,
                    "high": base + offset,
                    "low": base + offset,
                    "close": base + offset,
                    "volume": 1000.0,
                    "amount": 10000.0,
                    "turnover": 0.0,
                    "suspended": False,
                    "limit_up": False,
                    "limit_down": False,
                    "st": False,
                }
            )

    def fake_loader(lake_arg, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame(rows)

    monkeypatch.setattr("qmt_agent_trader.factors.service.load_daily_bars", fake_loader)

    result = validate_factor(
        lake,
        name="momentum_20d",
        start="20240121",
        end="20240122",
        symbols=["000001.SZ"],
    )

    assert result.start == "20240121"
    assert calls == [
        {
            "start": "20240101",
            "end": "20240123",
            "symbols": ["000001.SZ"],
        }
    ]


def test_evaluate_factor_loads_bars_once_and_reuses_validation_frame(
    tmp_path,
    monkeypatch,
) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    calls = 0
    start = date(2024, 1, 1)
    rows = []
    for offset in range(35):
        trade_date = start + timedelta(days=offset)
        for symbol, base in [("000001.SZ", 10.0), ("000002.SZ", 20.0)]:
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "open": base + offset,
                    "high": base + offset,
                    "low": base + offset,
                    "close": base + offset,
                    "volume": 1000.0,
                    "amount": 10000.0,
                    "turnover": 0.0,
                    "suspended": False,
                    "limit_up": False,
                    "limit_down": False,
                    "st": False,
                }
            )

    def fake_loader(lake_arg, **kwargs):
        nonlocal calls
        calls += 1
        return pd.DataFrame(rows)

    monkeypatch.setattr("qmt_agent_trader.factors.service.load_daily_bars", fake_loader)

    bundle = evaluate_factor(
        lake,
        name="momentum_20d",
        start="20240121",
        end="20240202",
        symbols=["000001.SZ", "000002.SZ"],
        window_days=5,
        step_days=5,
        quantile=0.5,
    )

    assert calls == 1
    assert bundle.validation.observations > 0
    assert bundle.walk_forward.slices
    assert bundle.quantile_returns["walk_forward_slices"] == len(bundle.walk_forward.slices)
