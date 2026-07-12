from __future__ import annotations

import inspect
from pathlib import Path

from qmt_agent_trader.services.order_plan_service import (
    append_order_plan_event,
    load_order_plan,
    load_order_plan_events,
    save_order_plan,
)
from qmt_agent_trader.strategy.approval import read_approval_file, write_approval_file


def _python_sources() -> list[Path]:
    root = Path(__file__).parents[3]
    return [
        path
        for directory in (root / "src", root / "scripts", root / "tests")
        for path in directory.rglob("*.py")
    ]


def test_removed_query_api_has_no_executable_callers() -> None:
    forbidden = ".query_" + "parquet("
    offenders = [path for path in _python_sources() if forbidden in path.read_text()]
    assert offenders == []


def test_production_code_has_no_private_artifact_lock_root() -> None:
    root = Path(__file__).parents[3]
    forbidden = ".artifact-" + "locks"
    offenders = [
        path for path in (root / "src").rglob("*.py") if forbidden in path.read_text()
    ]
    assert offenders == []


def test_governed_apis_have_one_root_authority() -> None:
    for function in (
        save_order_plan,
        load_order_plan,
        append_order_plan_event,
        load_order_plan_events,
        write_approval_file,
        read_approval_file,
    ):
        parameters = inspect.signature(function).parameters
        assert "directory" not in parameters
        assert "artifact_store" in parameters
