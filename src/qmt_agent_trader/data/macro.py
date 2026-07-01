"""Macro dataset registry and point-in-time visibility rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd


@dataclass(frozen=True)
class MacroDatasetSpec:
    dataset_id: str
    api_name: str
    frequency: str
    date_column: str
    value_columns: tuple[str, ...] | None
    pit_safe: bool
    default_lag_days: int
    description: str
    default_fields: str | None = None

    @property
    def raw_dataset(self) -> str:
        return f"tushare_macro_{self.dataset_id}"

    @property
    def key_columns(self) -> list[str]:
        return [self.date_column]

    @property
    def visibility_rule(self) -> str:
        if self.default_lag_days == 0:
            return f"{self.date_column} same-day visibility"
        return f"{self.date_column} + {self.default_lag_days} calendar days"


MACRO_DATASETS: dict[str, MacroDatasetSpec] = {
    "cn_gdp": MacroDatasetSpec(
        dataset_id="cn_gdp",
        api_name="cn_gdp",
        frequency="quarterly",
        date_column="quarter",
        value_columns=None,
        pit_safe=False,
        default_lag_days=45,
        description="China GDP quarterly indicators.",
    ),
    "cn_cpi": MacroDatasetSpec(
        dataset_id="cn_cpi",
        api_name="cn_cpi",
        frequency="monthly",
        date_column="month",
        value_columns=None,
        pit_safe=False,
        default_lag_days=15,
        description="China CPI monthly indicators.",
    ),
    "cn_ppi": MacroDatasetSpec(
        dataset_id="cn_ppi",
        api_name="cn_ppi",
        frequency="monthly",
        date_column="month",
        value_columns=None,
        pit_safe=False,
        default_lag_days=15,
        description="China PPI monthly indicators.",
    ),
    "shibor": MacroDatasetSpec(
        dataset_id="shibor",
        api_name="shibor",
        frequency="daily",
        date_column="date",
        value_columns=None,
        pit_safe=True,
        default_lag_days=0,
        description="Shanghai interbank offered rate.",
    ),
}


def get_macro_dataset(dataset_id: str) -> MacroDatasetSpec | None:
    return MACRO_DATASETS.get(dataset_id)


def macro_visible_date(values: pd.Series, spec: MacroDatasetSpec) -> pd.Series:
    parsed = _parse_macro_dates(values, spec)
    return (parsed + pd.to_timedelta(spec.default_lag_days, unit="D")).dt.date


def _parse_macro_dates(values: pd.Series, spec: MacroDatasetSpec) -> pd.Series:
    text = values.astype(str).str.strip()
    if spec.frequency == "monthly":
        normalized = text.str.replace("-", "", regex=False).str[:6] + "01"
        return pd.to_datetime(normalized, format="%Y%m%d", errors="coerce") + pd.offsets.MonthEnd(0)
    if spec.frequency == "quarterly":
        normalized = text.str.replace("-", "", regex=False)
        quarters = normalized.str.extract(r"(?P<year>\d{4})Q?(?P<quarter>[1-4])")
        if quarters.notna().all(axis=None):
            month = quarters["quarter"].astype(int).map({1: 3, 2: 6, 3: 9, 4: 12})
            date_text = quarters["year"] + month.astype(str).str.zfill(2) + "01"
            return (
                pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
                + pd.offsets.MonthEnd(0)
            )
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce").fillna(
        pd.to_datetime(text, errors="coerce")
    )


def parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return datetime.fromisoformat(value).date()
