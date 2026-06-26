"""Data update service."""

from __future__ import annotations

import fcntl
import hashlib
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import TextIO

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


class RequestLimiter:
    def __init__(
        self,
        *,
        min_interval_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval_seconds = min_interval_seconds
        self.clock = clock
        self.sleep = sleep
        self._last_request_at: float | None = None

    def wait(self) -> None:
        now = self.clock()
        if self._last_request_at is not None:
            elapsed = now - self._last_request_at
            remaining = self.min_interval_seconds - elapsed
            if remaining > 0:
                self.sleep(remaining)
                now = self.clock()
        self._last_request_at = now


class DataUpdateLock(AbstractContextManager["DataUpdateLock"]):
    def __init__(self, path: Path, *, timeout_seconds: float) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self._handle: TextIO | None = None

    def __enter__(self) -> DataUpdateLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("w", encoding="utf-8")
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._handle = handle
                return self
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    handle.close()
                    raise TimeoutError("remote data update lock timeout") from exc
                time.sleep(0.05)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._handle is not None:
            handle = self._handle
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            self._handle = None
        return None


class TushareDataUpdateService:
    def __init__(
        self,
        client: TushareClient,
        lake: DataLake,
        *,
        limiter: RequestLimiter | None = None,
        lock_timeout_seconds: float = 30.0,
    ) -> None:
        self.client = client
        self.lake = lake
        self.limiter = limiter or RequestLimiter(min_interval_seconds=0.5)
        self.lock_timeout_seconds = lock_timeout_seconds

    def update(
        self,
        start: str,
        end: str,
        *,
        include_daily: bool = True,
        include_basics: bool = True,
    ) -> DataUpdateResult:
        with DataUpdateLock(
            self.lake.root / "_locks" / "remote_data.lock",
            timeout_seconds=self.lock_timeout_seconds,
        ):
            writes: list[DatasetWrite] = []
            calendar = self._execute(
                self.client.build_trade_calendar_request(start_date=start, end_date=end)
            )
            writes.append(self._write("tushare_trade_calendar", calendar, start=start, end=end))

            if include_basics:
                stock_basic = self._execute(self.client.build_stock_basic_request())
                writes.append(
                    self._write("tushare_stock_basic", stock_basic, start=start, end=end)
                )

                etf_basic = self._execute(self.client.build_etf_basic_request())
                writes.append(self._write("tushare_etf_basic", etf_basic, start=start, end=end))

                namechange = self._fetch_namechange_pages()
                writes.append(self._write("tushare_namechange", namechange, start=start, end=end))

            open_dates = self._open_dates(calendar)
            if include_daily:
                missing_daily_dates = self._missing_trade_dates("tushare_daily", open_dates)
                if open_dates:
                    daily = self._fetch_daily_by_open_dates(missing_daily_dates)
                else:
                    daily = self._execute(
                        self.client.build_daily_request(start_date=start, end_date=end)
                    )
                if not daily.empty or not open_dates:
                    writes.append(
                        self._write_incremental(
                            "tushare_daily",
                            daily,
                            start=start,
                            end=end,
                            key_columns=["ts_code", "trade_date"],
                        )
                    )

                missing_suspend_dates = self._missing_trade_dates("tushare_suspend", open_dates)
                if missing_suspend_dates or not open_dates:
                    suspend = self._execute(
                        self.client.build_suspend_request(start_date=start, end_date=end)
                    )
                    writes.append(
                        self._write_incremental(
                            "tushare_suspend",
                            suspend,
                            start=start,
                            end=end,
                            key_columns=["ts_code", "trade_date"],
                        )
                    )

                missing_limit_dates = self._missing_trade_dates("tushare_stk_limit", open_dates)
                limits = self._fetch_stk_limit_by_open_dates(missing_limit_dates)
                if not limits.empty:
                    writes.append(
                        self._write_incremental(
                            "tushare_stk_limit",
                            limits,
                            start=start,
                            end=end,
                            key_columns=["ts_code", "trade_date"],
                        )
                    )

            return DataUpdateResult(start=start, end=end, writes=writes, open_dates=open_dates)

    def _execute(self, request: object) -> pd.DataFrame:
        self.limiter.wait()
        return self.client.execute(request)  # type: ignore[arg-type]

    def _write(self, name: str, frame: pd.DataFrame, *, start: str, end: str) -> DatasetWrite:
        path = self.lake.write_parquet(frame, "raw", name)
        self.lake.register_parquet(name, "raw", name)
        self._record_success(name, frame, start=start, end=end)
        return DatasetWrite(name=name, layer="raw", path=path, rows=len(frame))

    def _write_incremental(
        self,
        name: str,
        frame: pd.DataFrame,
        *,
        start: str,
        end: str,
        key_columns: list[str],
    ) -> DatasetWrite:
        path = self.lake.write_incremental_parquet(
            frame,
            "raw",
            name,
            key_columns=key_columns,
        )
        self._record_success(name, frame, start=start, end=end)
        return DatasetWrite(name=name, layer="raw", path=path, rows=len(frame))

    def _record_success(self, name: str, frame: pd.DataFrame, *, start: str, end: str) -> None:
        self.lake.record_fetch_result(
            source="tushare",
            dataset=name,
            start_date=start,
            end_date=end,
            status="success",
            row_count=len(frame),
            checksum=_checksum_frame(frame),
            error=None,
        )

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
            frame = self._execute(self.client.build_daily_by_trade_date_request(trade_date))
            if not frame.empty:
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _fetch_namechange_pages(self, page_size: int = 5000) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        offset = 0
        while True:
            frame = self._execute(
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
            frame = self._execute(
                self.client.build_stk_limit_by_trade_date_request(trade_date)
            )
            if not frame.empty:
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _missing_trade_dates(self, dataset_name: str, open_dates: list[str]) -> list[str]:
        if not open_dates:
            return []
        covered = self._covered_trade_dates(dataset_name)
        return [item for item in open_dates if item not in covered]

    def _covered_trade_dates(self, dataset_name: str) -> set[str]:
        if not self.lake.dataset_path("raw", dataset_name).exists():
            return set()
        frame = self.lake.read_parquet("raw", dataset_name)
        if "trade_date" not in frame.columns:
            return set()
        return {_format_trade_date(item) for item in frame["trade_date"].dropna().tolist()}


def _same_page(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if len(left) != len(right) or list(left.columns) != list(right.columns):
        return False
    return bool(left.reset_index(drop=True).equals(right.reset_index(drop=True)))


def _format_trade_date(value: object) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y%m%d")  # type: ignore[no-any-return]
    text = str(value)
    if "-" in text:
        return datetime.fromisoformat(text).strftime("%Y%m%d")
    return text


def _checksum_frame(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "empty"
    payload = frame.sort_index(axis=1).to_json(orient="records", date_format="iso")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
