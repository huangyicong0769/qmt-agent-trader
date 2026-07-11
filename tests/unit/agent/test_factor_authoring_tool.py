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


def test_agent_authored_factor_failure_guides_dataframe_return_repair(registry) -> None:
    python_function = """
def compute_factor(data: pd.DataFrame, context: FactorContext) -> pd.DataFrame:
    values = data["close"].pct_change()
    return pd.DataFrame({"factor_value": values}, index=data.index)
""".strip()

    result = registry.run_tool(
        "generate_factor_code",
        {
            "factor_name": "bad_dataframe_factor",
            "factor_description": "incorrectly returns a DataFrame",
            "python_function": python_function,
        },
        ToolContext(run_id="factor-dataframe-failure"),
    )

    assert result["status"] == "SAMPLE_TEST_FAILED"
    assert result["next_repair_tool"] == "generate_factor_code"
    assert "return pandas Series" in " ".join(result["sample_test"]["issues"])
    assert "pd.Series" in result["suggested_repair"]
    assert result["authoring_contract"]["return_type"] == "pd.Series"


def test_agent_authored_factor_failure_warns_against_trade_date_index(registry) -> None:
    python_function = """
def compute_factor(data: pd.DataFrame, context: FactorContext) -> pd.Series:
    factor_values = data.groupby("symbol")["close"].pct_change()
    return pd.Series(factor_values.to_numpy(), index=data["trade_date"], name="factor_value")
""".strip()

    result = registry.run_tool(
        "generate_factor_code",
        {
            "factor_name": "bad_trade_date_index_factor",
            "factor_description": "incorrectly indexes by trade_date",
            "python_function": python_function,
        },
        ToolContext(run_id="factor-trade-date-index-failure"),
    )

    assert result["status"] == "SAMPLE_TEST_FAILED"
    issues = " ".join(result["sample_test"]["issues"])
    assert "duplicate" in issues
    assert "trade_date" in issues
    assert result["next_repair_tool"] == "generate_factor_code"


def test_agent_authored_factor_accepts_series_aligned_to_input_index(registry) -> None:
    python_function = """
def compute_factor(data: pd.DataFrame, context: FactorContext) -> pd.Series:
    ordered = data.sort_values(["symbol", "trade_date"])
    values = ordered.groupby("symbol")["close"].transform(lambda item: item.pct_change())
    aligned = values.reindex(data.index)
    result = pd.Series(aligned, index=data.index, name="factor_value")
    return pd.to_numeric(result, errors="coerce")
""".strip()

    result = registry.run_tool(
        "generate_factor_code",
        {
            "factor_name": "aligned_factor",
            "factor_description": "returns a row-aligned Series",
            "python_function": python_function,
        },
        ToolContext(run_id="factor-aligned-success"),
    )

    assert result["status"] == "generated"
    assert result["sample_test_status"] == "PASSED"


def test_agent_authored_factor_reports_real_schema_blocker_for_industry(registry) -> None:
    python_function = """
def compute_factor(data: pd.DataFrame, context: FactorContext) -> pd.Series:
    return data["close"] - data.groupby("industry")["close"].transform("mean")
""".strip()

    result = registry.run_tool(
        "generate_factor_code",
        {
            "factor_spec": {
                "factor_id": "industry_neutral_factor",
                "name": "industry neutral factor",
                "formula": "行业中性 close",
                "required_columns": ["daily_bars", "close", "industry"],
            },
            "python_function": python_function,
        },
        ToolContext(run_id="factor-real-schema"),
    )

    assert result["status"] == "generated"
    assert result["synthetic_contract_test"]["contract_test_only"] is True
    assert result["real_schema_status"] == "BLOCKED"
    assert result["domain_status"] == "BLOCKED"
    assert result["evidence_status"] == "BLOCKED"
    assert result["missing_columns"] == ["industry"]


def test_agent_authored_factor_sample_test_rejects_input_mutation(registry) -> None:
    python_function = """
def compute_factor(data: pd.DataFrame, context: FactorContext) -> pd.Series:
    data["mutated"] = 1
    return pd.Series(data["close"].pct_change(), index=data.index, name="factor_value")
""".strip()

    result = registry.run_tool(
        "generate_factor_code",
        {
            "factor_name": "mutating_factor",
            "factor_description": "mutates input data",
            "python_function": python_function,
        },
        ToolContext(run_id="factor-mutation-failure"),
    )

    assert result["status"] == "SAMPLE_TEST_FAILED"
    assert "mutated input DataFrame" in " ".join(result["sample_test"]["issues"])


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


def test_same_factor_identity_can_generate_two_immutable_run_versions(registry) -> None:
    spec = {"factor_id": "factor_regenerated", "name": "regenerated", "formula": "close"}
    python_function = (
        "def compute_factor(data: pd.DataFrame, context: FactorContext) -> pd.Series:\n"
        "    return pd.Series(data['close'], index=data.index, name='factor_value')"
    )
    first = registry.run_tool(
        "generate_factor_code",
        {"factor_spec": spec, "python_function": python_function},
        ToolContext(run_id="run_one"),
    )
    second = registry.run_tool(
        "generate_factor_code",
        {"factor_spec": spec, "python_function": python_function},
        ToolContext(run_id="run_two"),
    )

    assert first["status"] == second["status"] == "generated"
    assert first["code_path"] != second["code_path"]
    assert Path(first["code_path"]).exists() and Path(second["code_path"]).exists()
    assert "run_one" in first["code_path"] and "run_two" in second["code_path"]


def test_run_directory_encoding_does_not_collapse_distinct_run_ids(registry) -> None:
    python_function = (
        "def compute_factor(data: pd.DataFrame, context: FactorContext) -> pd.Series:\n"
        "    return pd.Series(data['close'], index=data.index, name='factor_value')"
    )
    payload = {
        "factor_spec": {"factor_id": "factor_collision", "name": "collision"},
        "python_function": python_function,
    }
    first = registry.run_tool("generate_factor_code", payload, ToolContext(run_id="a/b"))
    second = registry.run_tool("generate_factor_code", payload, ToolContext(run_id="a?b"))

    assert first["status"] == second["status"] == "generated"
    assert Path(first["code_path"]).parent != Path(second["code_path"]).parent
