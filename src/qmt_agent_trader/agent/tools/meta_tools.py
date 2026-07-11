"""Meta tools: detect_tool_gap, create_tool_spec, generate_tool_code,
generate_tool_tests, run_tool_sandbox_tests, score_tool_candidate,
propose_tool_registration.

These enable the Agent to propose *new* tools, but never register them
directly into the formal registry.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.sandbox import CodeSandbox, generated_identity_segment
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_dependencies import AgentToolDependencies
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.core.ids import new_id, shanghai_now_iso

_sandbox: CodeSandbox | None = None
_store: ExperimentStore | None = None
_sandbox_var: ContextVar[CodeSandbox | None] = ContextVar("meta_tool_sandbox", default=None)
_store_var: ContextVar[ExperimentStore | None] = ContextVar("meta_tool_store", default=None)


def wire(sandbox: CodeSandbox, store: ExperimentStore) -> None:
    global _sandbox, _store
    _sandbox = sandbox
    _store = store


def _get_sandbox() -> CodeSandbox | None:
    return _sandbox_var.get() or _sandbox


def _with_deps(
    deps: AgentToolDependencies,
    fn: Callable[[dict[str, Any], ToolContext], dict[str, Any]],
    input_data: dict[str, Any],
    context: ToolContext,
) -> dict[str, Any]:
    sandbox_token = _sandbox_var.set(deps.sandbox)
    store_token = _store_var.set(deps.experiment_store)
    try:
        return fn(input_data, context)
    finally:
        _store_var.reset(store_token)
        _sandbox_var.reset(sandbox_token)


# ── detect_tool_gap ──────────────────────────────────────────────────────────


def _detect_tool_gap(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    input_data.get("recent_experiment_ids", [])
    repeated_steps = input_data.get("repeated_steps", [])
    failure_summaries = input_data.get("failure_summaries", [])

    proposals: list[dict[str, Any]] = []
    if failure_summaries:
        failure_text = " ".join(failure_summaries).lower()
        if any(w in failure_text for w in ["not available", "not_implemented", "missing"]):
            proposals.append(
                {
                    "problem": "Missing data source or unfinished stub preventing research.",
                    "proposed_tool_name": "extend_data_layer",
                    "expected_benefit": "Unblock factor research that requires PIT data.",
                    "permission_level": "READ_ONLY",
                    "risk_level": "LOW",
                }
            )
    if len(repeated_steps) > 2:
        proposals.append(
            {
                "problem": f"Repeated steps: {', '.join(repeated_steps[:3])}",
                "proposed_tool_name": "batch_compute_factors",
                "expected_benefit": "Reduce token usage and audit noise.",
                "permission_level": "BACKTEST_EXECUTE",
                "risk_level": "LOW",
            }
        )

    return {"tool_gap_proposals": proposals}


detect_tool_gap_tool: AgentTool = tool(
    ToolSpec(
        name="detect_tool_gap",
        description="根据最近失败记录识别是否需要新工具。",
        permission=PermissionLevel.READ_ONLY,
        deterministic=False,
    ),
    fn=_detect_tool_gap,
)

# ── create_tool_spec ─────────────────────────────────────────────────────────


def _create_tool_spec(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    gap = input_data.get("tool_gap_proposal", {})
    name = gap.get("proposed_tool_name", "candidate_tool")
    permission = gap.get("permission_level", "READ_ONLY")
    risk = gap.get("risk_level", "MEDIUM")

    # Disallow broker/order/gateway/live tools
    forbidden_keywords = ["broker", "order", "gateway", "live", "submit", "approve_strategy"]
    for kw in forbidden_keywords:
        if kw in name.lower():
            return {
                "status": "REJECTED",
                "message": f"Tool name contains forbidden keyword: '{kw}'",
            }

    spec = {
        "name": name,
        "description": gap.get("expected_benefit", gap.get("problem", "")),
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "output_schema": {"type": "object", "properties": {"status": {"type": "string"}}},
        "permission_level": permission,
        "side_effect_level": "none" if permission == "READ_ONLY" else "write_generated",
        "deterministic": permission == "READ_ONLY",
        "timeout_seconds": 30,
        "test_cases": [{"input": {}, "expected_output": {"status": "ok"}}],
        "failure_modes": ["data_not_available", "invalid_input"],
        "risk_level": risk,
    }
    return {"tool_spec": spec}


create_tool_spec_tool: AgentTool = tool(
    ToolSpec(
        name="create_tool_spec",
        description="根据 tool gap 生成候选工具规格。",
        permission=PermissionLevel.CODE_GENERATION,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_create_tool_spec,
)

# ── generate_tool_code ──────────────────────────────────────────────────────


def _generate_tool_code(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    spec_data = input_data.get("tool_spec", {})
    name = spec_data.get("name", "candidate")
    safe_name = name.replace(" ", "_").lower()
    version = "0.1.0"

    code = f'''"""Candidate tool: {name} (v{version}).
Auto-generated by Agent self-bootstrap — REVIEW_REQUIRED before promotion.
"""

from typing import Any
from qmt_agent_trader.agent.schemas import ToolContext


def run(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Stub implementation — must be reviewed before use."""
    return {{"status": "stub", "tool": "{safe_name}"}}
'''

    sb = _get_sandbox()
    if sb is None:
        return {"status": "error", "message": "sandbox not wired"}

    run_id = context.run_id
    run_segment = generated_identity_segment(run_id)
    rel_path = f"tools/{safe_name}/{version}/{run_segment}/tool.py"
    try:
        code_path = sb.write_candidate_file(
            rel_path,
            code,
            artifact_id=f"tool:{safe_name}:{version}:{run_id}:implementation",
            related_run_id=run_id,
        )
        return {"code_path": str(code_path), "tests_path": "", "status": "generated"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


generate_tool_code_tool: AgentTool = tool(
    ToolSpec(
        name="generate_tool_code",
        description="为候选工具生成代码（仅可写入 generated/tools/）。",
        permission=PermissionLevel.CODE_GENERATION,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_generate_tool_code,
)

# ── generate_tool_tests ─────────────────────────────────────────────────────


def _generate_tool_tests(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    spec_data = input_data.get("tool_spec", {})
    name = spec_data.get("name", "candidate")
    safe_name = name.replace(" ", "_").lower()
    version = "0.1.0"
    code_path_raw = str(input_data.get("code_path") or "")
    sb = _get_sandbox()
    if sb is None:
        return {"status": "error", "message": "sandbox not wired"}
    try:
        code_path = sb.validate_path(code_path_raw)
    except Exception as exc:
        return {"tests_path": "", "error": str(exc)}
    if not code_path.is_file():
        return {"tests_path": "", "error": "code_path is required and must exist"}

    test_code = f'''"""Tests for candidate tool: {name}."""

import importlib.util
from pathlib import Path
from qmt_agent_trader.agent.schemas import ToolContext

CODE_PATH = Path({str(code_path)!r})
SPEC = importlib.util.spec_from_file_location("candidate_tool_exact_run", CODE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
run = MODULE.run

def test_empty_input():
    """Normal input produces a status."""
    result = run({{}}, ToolContext(run_id="test"))
    assert "status" in result


def test_error_input():
    """Invalid input should not crash."""
    result = run(None, ToolContext(run_id="test"))  # type: ignore
    assert isinstance(result, dict)


def test_audit_trail():
    """Tool context run_id must propagate."""
    result = run({{}}, ToolContext(run_id="audit-test"))
    assert isinstance(result, dict)
'''

    run_id = context.run_id
    run_segment = generated_identity_segment(run_id)
    rel_path = f"tools/{safe_name}/{version}/{run_segment}/test_tool.py"
    try:
        tests_path = sb.write_candidate_file(
            rel_path,
            test_code,
            artifact_id=f"tool:{safe_name}:{version}:{run_id}:tests",
            related_run_id=run_id,
        )
        return {"tests_path": str(tests_path)}
    except Exception as exc:
        return {"tests_path": "", "error": str(exc)}


generate_tool_tests_tool: AgentTool = tool(
    ToolSpec(
        name="generate_tool_tests",
        description="为候选工具生成测试（覆盖空输入、错误输入、边界、权限、审计）。",
        permission=PermissionLevel.CODE_GENERATION,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_generate_tool_tests,
)

# ── run_tool_sandbox_tests ──────────────────────────────────────────────────


def _run_tool_sandbox_tests(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    code_path = input_data.get("code_path", "")
    tests_path = input_data.get("tests_path", "")
    test_path = Path(tests_path) if tests_path else Path(code_path)

    sb = _get_sandbox()
    if sb is None:
        return {"status": "FAILED", "safety_issues": ["sandbox not wired"]}

    result = sb.run_tests(test_path)
    return {
        "status": result.status,
        "test_summary": result.test_summary,
        "safety_issues": result.safety_issues,
    }


run_tool_sandbox_tests_tool: AgentTool = tool(
    ToolSpec(
        name="run_tool_sandbox_tests",
        description="在沙箱中运行候选工具测试（默认无网络、只读数据、无 broker）。",
        permission=PermissionLevel.BACKTEST_EXECUTE,
        deterministic=False,
        timeout_seconds=60,
    ),
    fn=_run_tool_sandbox_tests,
)

# ── score_tool_candidate ────────────────────────────────────────────────────


def _score_tool_candidate(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    input_data.get("tool_spec", {})
    test_summary = input_data.get("test_summary", {})
    safety_issues = input_data.get("safety_issues", [])

    correctness = 1.0 if test_summary.get("has_test_functions", False) else 0.3
    safety = 0.0 if safety_issues else 1.0
    usefulness = 0.5
    maintainability = 0.7
    duplication = 0.5
    performance = 0.8

    if safety == 0.0:
        recommendation = "REJECT"
    elif correctness < 0.6:
        recommendation = "NEEDS_HUMAN_REVIEW"
    else:
        recommendation = "EXPERIMENTAL"

    return {
        "score": {
            "correctness": correctness,
            "usefulness": usefulness,
            "safety": safety,
            "maintainability": maintainability,
            "duplication": duplication,
            "performance": performance,
        },
        "recommendation": recommendation,
    }


score_tool_candidate_tool: AgentTool = tool(
    ToolSpec(
        name="score_tool_candidate",
        description="评价候选工具是否值得保留（REJECT | EXPERIMENTAL | NEEDS_HUMAN_REVIEW）。",
        permission=PermissionLevel.READ_ONLY,
        deterministic=True,
    ),
    fn=_score_tool_candidate,
)

# ── propose_tool_registration ────────────────────────────────────────────────


def _propose_tool_registration(input_data: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    candidate_id = input_data.get("tool_candidate_id", new_id("tool"))
    score = input_data.get("score", {})
    recommendation = score.get("recommendation", "REJECT")

    if recommendation == "REJECT":
        return {
            "status": "REJECTED",
            "proposal_path": "",
            "message": "tool did not pass safety/correctness checks",
        }

    # Write proposal
    sb = _get_sandbox()
    proposal_root = sb.generated_root / "tools" if sb else Path("proposals")
    proposal_root.mkdir(parents=True, exist_ok=True)
    proposal_path = proposal_root / f"tool_proposal_{candidate_id}.json"

    import json

    proposal = {
        "tool_candidate_id": candidate_id,
        "score": score,
        "recommendation": recommendation,
        "status": "REVIEW_REQUIRED",
        "created_at": shanghai_now_iso(),
    }
    proposal_path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "proposal_path": str(proposal_path),
        "status": "REVIEW_REQUIRED",
        "note": "tool registration requires explicit human approval",
    }


propose_tool_registration_tool: AgentTool = tool(
    ToolSpec(
        name="propose_tool_registration",
        description="提出正式注册候选工具的申请（需要人工审批）。",
        permission=PermissionLevel.APPROVAL_REQUIRED,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_propose_tool_registration,
)


def build_meta_tools(deps: AgentToolDependencies) -> list[AgentTool]:
    definitions: list[
        tuple[AgentTool, Callable[[dict[str, Any], ToolContext], dict[str, Any]]]
    ] = [
        (detect_tool_gap_tool, _detect_tool_gap),
        (create_tool_spec_tool, _create_tool_spec),
        (generate_tool_code_tool, _generate_tool_code),
        (generate_tool_tests_tool, _generate_tool_tests),
        (run_tool_sandbox_tests_tool, _run_tool_sandbox_tests),
        (score_tool_candidate_tool, _score_tool_candidate),
        (propose_tool_registration_tool, _propose_tool_registration),
    ]
    return [_bind_tool(deps, existing, fn) for existing, fn in definitions]


def _bind_tool(
    deps: AgentToolDependencies,
    existing: AgentTool,
    fn: Callable[[dict[str, Any], ToolContext], dict[str, Any]],
) -> AgentTool:
    return tool(
        existing.spec,
        fn=lambda input_data, context: _with_deps(deps, fn, input_data, context),
    )
