"""Data update service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.data.tushare_client import TushareClient


def build_data_update_plan(client: TushareClient, start: str, end: str) -> list[dict[str, object]]:
    plan = [
        client.build_trade_calendar_request(start_date=start, end_date=end).__dict__,
        client.build_stock_basic_request().__dict__,
        client.build_etf_basic_request().__dict__,
        client.build_namechange_request().__dict__,
        client.build_daily_request(start_date=start, end_date=end).__dict__,
        client.build_suspend_request(start_date=start, end_date=end).__dict__,
    ]
    plan.append(
        {
            "api_name": "stk_limit",
            "params": {"trade_dates": "derived from trade_cal open dates"},
            "fields": "ts_code,trade_date,up_limit,down_limit",
        }
    )
    return plan


@dataclass(frozen=True)
class DatasetWrite:
    name: str
    layer: str
    path: Path
    rows: int


@dataclass(frozen=True)
class DataUpdateResult:
    start: str
    end: str
    writes: list[DatasetWrite]
    open_dates: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": "updated",
            "start": self.start,
            "end": self.end,
            "open_dates": self.open_dates,
            "writes": [
                {
                    "name": write.name,
                    "layer": write.layer,
                    "path": str(write.path),
                    "rows": write.rows,
                }
                for write in self.writes
            ],
        }


class TushareDataUpdateService:
    def __init__(self, client: TushareClient, lake: DataLake) -> None:
        self.client = client
        self.lake = lake

    def update(
        self,
        start: str,
        end: str,
        *,
        include_daily: bool = True,
        include_basics: bool = True,
    ) -> DataUpdateResult:
        writes: list[DatasetWrite] = []
        calendar = self.client.execute(
            self.client.build_trade_calendar_request(start_date=start, end_date=end)
        )
        writes.append(self._write("tushare_trade_calendar", calendar))

        if include_basics:
            stock_basic = self.client.execute(self.client.build_stock_basic_request())
            writes.append(self._write("tushare_stock_basic", stock_basic))

            etf_basic = self.client.execute(self.client.build_etf_basic_request())
            writes.append(self._write("tushare_etf_basic", etf_basic))

            namechange = self._fetch_namechange_pages()
            writes.append(self._write("tushare_namechange", namechange))

        open_dates = self._open_dates(calendar)
        if include_daily:
            daily = (
                self._fetch_daily_by_open_dates(open_dates)
                if open_dates
                else self.client.execute(
                    self.client.build_daily_request(start_date=start, end_date=end)
                )
            )
            writes.append(self._write(f"tushare_daily_{start}_{end}", daily))

            suspend = self.client.execute(
                self.client.build_suspend_request(start_date=start, end_date=end)
            )
            writes.append(self._write(f"tushare_suspend_{start}_{end}", suspend))

            limits = self._fetch_stk_limit_by_open_dates(open_dates)
            writes.append(self._write(f"tushare_stk_limit_{start}_{end}", limits))

        return DataUpdateResult(start=start, end=end, writes=writes, open_dates=open_dates)

    def _write(self, name: str, frame: pd.DataFrame) -> DatasetWrite:
        path = self.lake.write_parquet(frame, "raw", name)
        self.lake.register_parquet(name, "raw", name)
        return DatasetWrite(name=name, layer="raw", path=path, rows=len(frame))

    @staticmethod
    def _open_dates(calendar: pd.DataFrame) -> list[str]:
        if calendar.empty or "cal_date" not in calendar.columns:
            return []
        data = calendar.copy()
        if "is_open" in data.columns:
            data = data[data["is_open"].astype(int) == 1]
        return [str(item) for item in data["cal_date"].dropna().sort_values().tolist()]

    def _fetch_daily_by_open_dates(self, open_dates: list[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for trade_date in open_dates:
            frame = self.client.execute(self.client.build_daily_by_trade_date_request(trade_date))
            if not frame.empty:
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _fetch_namechange_pages(self, page_size: int = 5000) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        offset = 0
        while True:
            frame = self.client.execute(
                self.client.build_namechange_request(limit=page_size, offset=offset)
            )
            if frame.empty:
                break
            if frames and _same_page(frame, frames[-1]):
                break
            frames.append(frame)
            if len(frame) < page_size:
                break
            offset += page_size
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _fetch_stk_limit_by_open_dates(self, open_dates: list[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for trade_date in open_dates:
            frame = self.client.execute(
                self.client.build_stk_limit_by_trade_date_request(trade_date)
            )
            if not frame.empty:
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _same_page(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if len(left) != len(right) or list(left.columns) != list(right.columns):
        return False
    return bool(left.reset_index(drop=True).equals(right.reset_index(drop=True)))
