from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools import build_agent_registry
from qmt_agent_trader.data.storage import DataLake


@pytest.fixture
def registry(tmp_path: Path):
    lake = DataLake(root=tmp_path / "lake", duckdb_path=tmp_path / "test.duckdb")
    return build_agent_registry(
        data_lake=lake,
        audit_path=tmp_path / "audit.jsonl",
        experiment_root=tmp_path / "experiments",
        sandbox=CodeSandbox(tmp_path / "generated"),
    )


def test_agent_authored_python_factor_is_saved_checked_and_sample_run(registry) -> None:
    context = ToolContext(run_id="factor-authoring")
    python_function = """
def compute_factor(data: pd.DataFrame, context: FactorContext) -> pd.Series:
    ordered = data.sort_values(["symbol", "trade_date"])
    values = ordered.groupby("symbol")["close"].transform(lambda item: item.pct_change())
    return values.reindex(data.index)
""".strip()

    result = registry.run_tool(
        "generate_factor_code",
        {
            "factor_name": "custom_price_change",
            "factor_description": "agent-authored grouped price change",
            "python_function": python_function,
        },
        context,
    )

    assert result["status"] == "generated"
    assert result["static_check_status"] == "PASSED"
    assert result["sample_test_status"] == "PASSED"
    assert result["review_required"] is True

    static_result = registry.run_tool(
        "run_factor_static_checks",
        {"code_path": result["code_path"]},
        context,
    )
    assert static_result["status"] == "PASSED"

    spec = importlib.util.spec_from_file_location("generated_factor", result["code_path"])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.compute_factor)
    assert callable(module.compute)


def test_formula_fallback_requests_python_function_when_not_supported(registry) -> None:
    result = registry.run_tool(
        "generate_factor_code",
        {
            "factor_spec": {
                "factor_id": "factor_formula_fallback",
                "name": "custom_signal",
                "formula": "combine several proprietary qualitative channels",
            }
        },
        ToolContext(run_id="factor-needs-python"),
    )

    assert result["status"] == "NEEDS_PYTHON_FUNCTION"
    assert result["next_required_input"] == "python_function"
    assert "code_path" not in result


@pytest.mark.parametrize(
    "python_function",
    [
        "import os\n\ndef compute_factor(data, context):\n    return data['close']",
        "def compute_factor(data, context):\n    return data.groupby('symbol')['close'].shift(-1)",
    ],
)
def test_agent_authored_factor_rejects_dangerous_code(registry, python_function: str) -> None:
    result = registry.run_tool(
        "generate_factor_code",
        {
            "factor_name": "unsafe_factor",
            "factor_description": "unsafe",
            "python_function": python_function,
        },
        ToolContext(run_id="factor-rejects-danger"),
    )

    assert result["status"] == "STATIC_CHECK_FAILED"
    assert result["issues"]
    assert "code_path" not in result
