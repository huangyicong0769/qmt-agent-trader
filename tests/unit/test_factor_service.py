from datetime import date, timedelta

import pandas as pd

from qmt_agent_trader.data.frequency import Frequency
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.factors.service import (
    check_factor_input_readiness,
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
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare/daily")

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
        "tushare/daily",
    )
    lake.write_parquet(
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20240131", "pe_ttm": 4.0},
                {"ts_code": "000002.SZ", "trade_date": "20240131", "pe_ttm": 8.0},
            ]
        ),
        "raw",
        "tushare/daily_basic",
    )

    result = compute_factor_to_lake(lake, name="pe_ttm_rank", date="20240131")

    assert result.rows == 2
    assert result.non_null == 2
    output = lake.read_parquet("gold", "factor_pe_ttm_rank_20240131")
    assert output["factor_value"].tolist() == [0.5, 1.0]


def test_compute_low_frequency_factor_to_lake_uses_daily_asof_panel(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    rows = []
    for offset in range(3):
        trade_date = date(2024, 1, 1) + timedelta(days=offset)
        for symbol, base in [("000001.SZ", 10.0), ("000002.SZ", 20.0)]:
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": f"{trade_date:%Y%m%d}",
                    "open": base + offset,
                    "high": base + offset + 1,
                    "low": base + offset - 1,
                    "close": base + offset,
                }
            )
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare/daily")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "end_date": "20231231",
                    "ann_date": "20240101",
                    "debt_to_assets": 0.40,
                },
                {
                    "ts_code": "000002.SZ",
                    "end_date": "20231231",
                    "ann_date": "20240101",
                    "debt_to_assets": 0.70,
                },
            ]
        ),
        "raw",
        "tushare/fina_indicator",
    )

    result = compute_factor_to_lake(lake, name="debt_to_assets_rank", date="20240102")

    assert result.rows == 2
    assert result.non_null == 2
    output = lake.read_parquet("gold", "factor_debt_to_assets_rank_20240102")
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
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare/daily")

    result = validate_factor(lake, name="momentum_20d", start="20240121", end="20240121")

    assert result.observations == 2
    assert result.non_null == 2


def test_validate_low_frequency_factor_uses_daily_asof_panel(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    rows = []
    for offset in range(4):
        trade_date = date(2024, 1, 1) + timedelta(days=offset)
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": f"{trade_date:%Y%m%d}",
                "open": 10.0 + offset,
                "high": 11.0 + offset,
                "low": 9.0 + offset,
                "close": 10.0 + offset,
            }
        )
        rows.append(
            {
                "ts_code": "000002.SZ",
                "trade_date": f"{trade_date:%Y%m%d}",
                "open": 20.0 + offset,
                "high": 21.0 + offset,
                "low": 19.0 + offset,
                "close": 20.0 - offset,
            }
        )
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare/daily")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "end_date": "20231231",
                    "ann_date": "20240102",
                    "debt_to_assets": 0.40,
                },
                {
                    "ts_code": "000002.SZ",
                    "end_date": "20231231",
                    "ann_date": "20240102",
                    "debt_to_assets": 0.70,
                },
            ]
        ),
        "raw",
        "tushare/fina_indicator",
    )

    result = validate_factor(
        lake,
        name="debt_to_assets_rank",
        start="20240102",
        end="20240103",
    )

    assert result.observations == 4
    assert result.non_null == 4


def test_financial_current_wide_alone_does_not_satisfy_pit_factor_input(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    rows = []
    for offset in range(3):
        trade_date = date(2024, 1, 1) + timedelta(days=offset)
        for symbol in ["000001.SZ", "000002.SZ"]:
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": f"{trade_date:%Y%m%d}",
                    "open": 10.0 + offset,
                    "high": 11.0 + offset,
                    "low": 9.0 + offset,
                    "close": 10.0 + offset,
                }
            )
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare/daily")
    lake.write_parquet(
        pd.DataFrame(
            [
                {
                    "symbol": "000001.SZ",
                    "snapshot_as_of_date": "20240102",
                    "debt_to_assets": 0.40,
                }
            ]
        ),
        "silver",
        "financial_current_wide",
    )

    readiness = check_factor_input_readiness(
        lake,
        factor_name="debt_to_assets_rank",
        start="20240101",
        end="20240102",
    )

    assert readiness["status"] == "PARTIAL_COVERAGE"
    assert readiness["missing_fields"]["debt_to_assets"]["reason"] == "raw_dataset_missing"
    assert readiness["repair_action"]["fetch_items"][0]["api_name"] == "fina_indicator"


def test_readiness_reports_repair_action_for_missing_exact_source(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    rows = []
    for offset in range(3):
        trade_date = date(2024, 1, 1) + timedelta(days=offset)
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": f"{trade_date:%Y%m%d}",
                "open": 10.0 + offset,
                "high": 11.0 + offset,
                "low": 9.0 + offset,
                "close": 10.0 + offset,
            }
        )
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare/daily")

    readiness = check_factor_input_readiness(
        lake,
        factor_name="dividend_yield",
        start="20240101",
        end="20240102",
        symbols=["000001.SZ"],
    )

    assert readiness["status"] == "PARTIAL_COVERAGE"
    assert readiness["fill_policy_by_field"]["dv_ttm"] == "exact"
    assert readiness["repair_action"]["fetch_items"] == [
        {
            "api_name": "daily_basic",
            "symbols": ["000001.SZ"],
            "fields": ["ts_code", "trade_date", "dv_ttm"],
            "start_date": "20240101",
            "end_date": "20240102",
        }
    ]


def test_readiness_uses_contract_grid_for_partial_exact_daily_coverage(tmp_path) -> None:
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "db.duckdb")
    start = date(2024, 1, 1)
    daily_rows = []
    daily_basic_rows = []
    for offset in range(10):
        trade_date = start + timedelta(days=offset)
        for symbol, base in [("000001.SZ", 10.0), ("000002.SZ", 20.0)]:
            daily_rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": f"{trade_date:%Y%m%d}",
                    "open": base + offset,
                    "high": base + offset + 1,
                    "low": base + offset - 1,
                    "close": base + offset,
                }
            )
            if offset in {0, 1}:
                daily_basic_rows.append(
                    {
                        "ts_code": symbol,
                        "trade_date": f"{trade_date:%Y%m%d}",
                        "pb": 1.0 + offset,
                    }
                )
    lake.write_parquet(pd.DataFrame(daily_rows), "raw", "tushare/daily")
    lake.write_parquet(pd.DataFrame(daily_basic_rows), "raw", "tushare/daily_basic")

    readiness = check_factor_input_readiness(
        lake,
        factor_name="pb_rank",
        start="20240101",
        end="20240110",
        symbols=["000001.SZ", "000002.SZ"],
    )

    evidence = readiness["coverage_evidence"][0]
    repair_plan = readiness["repair_plans"][0]
    fetch_plan = repair_plan["fetch_plan"]

    assert readiness["status"] == "PARTIAL_COVERAGE"
    assert readiness["contract_status"] == "PARTIAL_REPAIRABLE"
    assert readiness["reason"] == "UNSATISFIED_DATA_REQUIREMENT"
    assert evidence["field"] == "pb"
    assert evidence["required_cells"] == 20
    assert evidence["observed_non_null_cells"] == 4
    assert evidence["field_coverage"] == 0.2
    assert evidence["status"] == "PARTIAL"
    assert repair_plan["matched_source"]["source_id"] == "tushare.daily_basic"
    assert fetch_plan["fetch_shape"] == "marketwide_time_slice"
    assert fetch_plan["expansion"] == "trade_calendar"
    assert fetch_plan["estimated_requests"] == 8
    assert fetch_plan["missing_time_points"] == [
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-06",
        "2024-01-07",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
    ]
    assert readiness["repair_action"]["reason"] == "UNSATISFIED_OBSERVATION_COVERAGE"
    assert (
        readiness["repair_action"]["contract_repair_plans"][0]["fetch_plan"]["fetch_shape"]
        == "marketwide_time_slice"
    )


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
    lake.write_parquet(pd.DataFrame(rows), "raw", "tushare/daily")

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

    def fake_panel_builder(lake_arg, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame(rows), {
            "unresolved_fields": [],
            "missing_fields": {},
            "field_sources": {},
            "coverage_by_field": {},
        }

    monkeypatch.setattr(
        "qmt_agent_trader.factors.service.build_target_frequency_panel",
        fake_panel_builder,
    )

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
            "target_frequency": Frequency.DAILY,
            "target_start": "20240101",
            "target_end": "20240123",
            "required_fields": ["symbol", "trade_date", "close"],
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

    def fake_panel_builder(lake_arg, **kwargs):
        nonlocal calls
        calls += 1
        return pd.DataFrame(rows), {
            "unresolved_fields": [],
            "missing_fields": {},
            "field_sources": {},
            "coverage_by_field": {},
        }

    monkeypatch.setattr(
        "qmt_agent_trader.factors.service.build_target_frequency_panel",
        fake_panel_builder,
    )

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
