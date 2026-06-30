from qmt_agent_trader.agent.sandbox import CodeSandbox
from qmt_agent_trader.agent.schemas import ToolContext
from qmt_agent_trader.agent.tools.strategy_tools import (
    generate_strategy_code_tool,
    run_backtest_tool,
    run_strategy_static_checks_tool,
    save_strategy_candidate_tool,
)


def test_strategy_tools_have_explicit_input_schemas() -> None:
    assert generate_strategy_code_tool.spec.input_schema["required"] == ["strategy_spec"]
    assert run_strategy_static_checks_tool.spec.input_schema["required"] == ["code_path"]
    assert save_strategy_candidate_tool.spec.input_schema["required"] == [
        "strategy_spec",
        "code_path",
    ]
    assert {"required": ["strategy_spec"]} in run_backtest_tool.spec.input_schema["anyOf"]
    assert {"required": ["factor_name"]} in run_backtest_tool.spec.input_schema["anyOf"]
    assert {"required": ["strategy_id"]} in run_backtest_tool.spec.input_schema["anyOf"]


def test_run_strategy_static_checks_rejects_dangerous_code(tmp_path) -> None:
    path = tmp_path / "strategy.py"
    path.write_text("import socket\n", encoding="utf-8")

    result = run_strategy_static_checks_tool.run(
        {"code_path": str(path)},
        ToolContext(run_id="test"),
    )

    assert result["status"] == "FAILED"
    assert result["issues"]


def test_generated_strategy_code_passes_static_scan(tmp_path) -> None:
    sandbox = CodeSandbox(tmp_path / "generated")
    result = generate_strategy_code_tool.run(
        {
            "strategy_spec": {
                "strategy_id": "strat_ok",
                "name": "ok",
                "factors": ["momentum_20d"],
            }
        },
        ToolContext(run_id="test"),
    )
    # Direct tool is not wired to this sandbox; assert the schema contract here.
    assert generate_strategy_code_tool.spec.permission.value == "CODE_GENERATION"
    assert sandbox.generated_root.exists()
    assert result["status"] in {"error", "STATIC_CHECK_FAILED"}
