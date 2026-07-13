"""Summarize the evidence in a canonical factor-rank backtest report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qmt_agent_trader.persistence.artifacts import (
    ArtifactManifest,
    artifact_store_for_root,
)
from qmt_agent_trader.persistence.locks import LockManager


def require_mapping(payload: dict[str, object], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"report field {key!r} must be an object")
    return value


def require_list(payload: dict[str, object], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"report field {key!r} must be an array")
    return value


def worst_daily_changes(points: list[Any], *, limit: int = 5) -> list[dict[str, object]]:
    changes: list[dict[str, object]] = []
    previous: float | None = None
    for item in points:
        if not isinstance(item, dict):
            continue
        equity = float(item.get("equity", 0.0))
        if previous is not None and previous > 0:
            changes.append(
                {
                    "trade_date": item.get("trade_date"),
                    "change": equity / previous - 1.0,
                }
            )
        previous = equity
    return sorted(changes, key=lambda item: float(item["change"]))[:limit]


def max_turnover(points: list[Any]) -> float:
    values = [
        float(item.get("one_way_turnover", 0.0))
        for item in points
        if isinstance(item, dict)
    ]
    return max(values, default=0.0)


def diagnose(report: dict[str, object]) -> dict[str, object]:
    metrics = require_mapping(report, "metrics")
    equity_points = require_list(report, "equity_points")
    rebalance_points = require_list(report, "rebalance_points")
    config = report.get("config") if isinstance(report.get("config"), dict) else {}
    universe = (
        report.get("universe_resolution")
        if isinstance(report.get("universe_resolution"), dict)
        else {}
    )
    return {
        "run_id": report.get("run_id"),
        "strategy_id": report.get("strategy_id"),
        "requested_data_window": {
            "start": config.get("start_date"),
            "end": config.get("end_date"),
        },
        "actual_data_window": report.get("data_window", {}),
        "universe": universe.get("metadata", {}),
        "net_total_return": metrics.get("net_total_return"),
        "same_trade_gross_return": metrics.get("same_trade_gross_return"),
        "cost_drag": metrics.get("cost_drag"),
        "average_one_way_turnover": metrics.get("average_one_way_turnover"),
        "max_rebalance_turnover": max_turnover(rebalance_points),
        "average_top_n_overlap": metrics.get("average_top_n_overlap"),
        "worst_daily_changes": worst_daily_changes(equity_points, limit=5),
        "data_quality": report.get("data_quality", {}),
        "adapter_capability_warnings": report.get("adapter_limitations", []),
    }


def _read_report(path: Path, *, unsafe_direct_json: bool) -> dict[str, object]:
    manifests_dir = path.parent / ".manifests"
    if manifests_dir.exists():
        manifests = [
            ArtifactManifest.model_validate_json(item.read_text(encoding="utf-8"))
            for item in manifests_dir.glob("*.json")
        ]
        manifest = next((item for item in manifests if item.relative_path == path.name), None)
        if manifest is not None:
            store = artifact_store_for_root(
                path.parent,
                lock_manager=LockManager(path.parent.parent / "locks"),
            )
            return json.loads(
                store.read_verified(manifest.artifact_id, expected_relative_path=path.name)
            )
    if not unsafe_direct_json:
        raise ValueError(
            "governed artifact manifest not found; pass --unsafe-direct-json for offline data"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--unsafe-direct-json", action="store_true")
    args = parser.parse_args()
    report = _read_report(args.report.resolve(), unsafe_direct_json=args.unsafe_direct_json)
    print(json.dumps(diagnose(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
