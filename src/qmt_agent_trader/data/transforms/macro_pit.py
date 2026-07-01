"""Point-in-time macro transforms."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from qmt_agent_trader.data.macro import get_macro_dataset, macro_visible_date, parse_date
from qmt_agent_trader.data.storage import DataLake


def load_macro_series_asof(
    lake: DataLake,
    *,
    dataset: str,
    as_of_date: str | date,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    fields: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_macro_dataset(dataset)
    if spec is None:
        return pd.DataFrame(), {"status": "INVALID_REQUEST", "message": "unknown macro dataset"}
    if not lake.dataset_path("raw", spec.raw_dataset).exists():
        return pd.DataFrame(), {"status": "NO_DATA", "dataset": dataset}

    frame = lake.read_parquet("raw", spec.raw_dataset)
    if frame.empty or spec.date_column not in frame.columns:
        return pd.DataFrame(), {"status": "NO_DATA", "dataset": dataset}

    as_of = parse_date(as_of_date)
    start = parse_date(start_date) if start_date is not None else None
    end = parse_date(end_date) if end_date is not None else as_of
    data = frame.copy()
    data["visible_date"] = macro_visible_date(data[spec.date_column], spec)
    data["period_date"] = (
        pd.to_datetime(data["visible_date"], errors="coerce")
        - pd.to_timedelta(spec.default_lag_days, unit="D")
    ).dt.date
    data["visible_date"] = pd.to_datetime(data["visible_date"], errors="coerce").dt.date
    data = data[data["visible_date"] <= as_of]
    if end is not None:
        data = data[data["period_date"] <= end]
    if start is not None:
        data = data[data["period_date"] >= start]
    if data.empty:
        return pd.DataFrame(), {"status": "NO_DATA", "dataset": dataset}

    identity = [spec.date_column, "period_date", "visible_date"]
    requested = fields or [
        column
        for column in data.columns
        if column not in {"visible_date", "period_date", "_empty"}
    ]
    columns = [column for column in [*identity, *requested] if column in data.columns]
    output = data[columns].drop_duplicates().sort_values("visible_date").reset_index(drop=True)
    return output, {
        "status": "OK",
        "dataset": dataset,
        "point_in_time": True,
        "pit_safe": spec.pit_safe,
        "visibility_rule": spec.visibility_rule,
        "frequency": spec.frequency,
    }


def records_jsonable(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        rows.append(
            {
                key: (f"{value:%Y-%m-%d}" if isinstance(value, date) else value)
                for key, value in record.items()
                if not pd.isna(value)
            }
        )
    return rows
