"""Agent tool registry and DeepSeek function-tool adapter.

`AgentToolRegistry` is the runtime registry. `ToolRegistry` is retained as a
small adapter that converts runtime tools into the DeepSeek client tool shape.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from qmt_agent_trader.agent.audit import AuditLogger
from qmt_agent_trader.agent.errors import ToolDuplicateError, ToolExecutionError, ToolNotFoundError
from qmt_agent_trader.agent.llm_client import DeepSeekTool
from qmt_agent_trader.agent.permissions import (
    ToolCallMode,
    ToolCapability,
    assert_llm_tool_allowed,
    can_call_tool,
    can_llm_call,
    require_permission,
    to_capability,
)
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_result import (
    DomainStatus,
    EvidenceStatus,
    ExecutionStatus,
    RecommendationStatus,
    audit_status_from_result,
    normalize_tool_result,
)
from qmt_agent_trader.agent.tools.base import AgentTool
from qmt_agent_trader.core.config import get_settings
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.errors import StorageError
from qmt_agent_trader.persistence.health import storage_error_health_payload
from qmt_agent_trader.persistence.locks import LockManager
from qmt_agent_trader.persistence.paths import PersistencePaths

_PROCESS_PAYLOAD_SPILL_BYTES = 1_000_000

# ── Original ToolDefinition + ToolRegistry (preserved) ───────────────────────


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    capability: ToolCapability
    fn: Callable[..., Any]
    description: str = ""
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
    )

    def as_deepseek_tool(self) -> DeepSeekTool:
        def guarded_fn(**kwargs: Any) -> Any:
            assert_llm_tool_allowed(self.capability)
            return self.fn(**kwargs)

        return DeepSeekTool(
            name=self.name,
            description=self.description or self.name,
            parameters=self.parameters,
            fn=guarded_fn,
        )


@dataclass
class ToolRegistry:
    tools: dict[str, ToolDefinition] = field(default_factory=dict)

    def register(self, definition: ToolDefinition) -> None:
        self.tools[definition.name] = definition

    def list_tools(self) -> list[dict[str, object]]:
        return [
            {
                "name": definition.name,
                "capability": definition.capability.value,
                "description": definition.description,
                "parameters": definition.parameters,
            }
            for definition in sorted(self.tools.values(), key=lambda item: item.name)
        ]

    def call_as_llm(self, tool_name: str, **kwargs: Any) -> Any:
        definition = self.tools[tool_name]
        assert_llm_tool_allowed(definition.capability)
        return definition.fn(**kwargs)

    def deepseek_tools_for_llm(self) -> list[DeepSeekTool]:
        tools: list[DeepSeekTool] = []
        for definition in sorted(self.tools.values(), key=lambda item: item.name):
            assert_llm_tool_allowed(definition.capability)
            tools.append(definition.as_deepseek_tool())
        return tools


# ── New AgentToolRegistry ────────────────────────────────────────────────────


@dataclass
class AgentToolRegistry:
    """Tool registry built around the `AgentTool` protocol and `PermissionLevel`.

    Every invocation is permission-checked and audit-logged.
    """

    tools: dict[str, AgentTool] = field(default_factory=dict)
    audit_logger: AuditLogger | None = None
    _audit_path: Path | None = None

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, tool: AgentTool) -> None:
        name = tool.spec.name
        if name in self.tools:
            raise ToolDuplicateError(f"tool '{name}' is already registered")
        self.tools[name] = tool

    def register_all(self, *tools: AgentTool) -> None:
        for entry in tools:
            self.register(entry)

    # ── Discovery ─────────────────────────────────────────────────────────

    def list_tools(
        self,
        *,
        permission: str | None = None,
        agent_callable_only: bool = False,
        call_mode: ToolCallMode = ToolCallMode.AUTONOMOUS_AGENT,
    ) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for _name, tool in sorted(self.tools.items()):
            spec = tool.spec
            if permission is not None and spec.permission.value != permission:
                continue
            agent_callable = spec.llm_callable and can_call_tool(spec.permission, call_mode)
            if agent_callable_only and not agent_callable:
                continue
            result.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "permission": spec.permission.value,
                    "input_schema": spec.input_schema,
                    "output_schema": spec.output_schema,
                    "side_effect_level": spec.side_effect_level,
                    "deterministic": spec.deterministic,
                    "llm_callable": spec.llm_callable and can_llm_call(spec.permission),
                    "agent_callable": agent_callable,
                }
            )
        return result

    def describe_tool(self, name: str) -> ToolSpec:
        tool = self._require_tool(name)
        return tool.spec

    # ── Execution ─────────────────────────────────────────────────────────

    def run_tool(
        self,
        name: str,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> dict[str, Any]:
        tool = self._require_tool(name)
        spec = tool.spec
        call_mode = context.call_mode or (
            ToolCallMode.AUTONOMOUS_AGENT
            if context.requested_by_llm
            else ToolCallMode.TRUSTED_INTERNAL_WORKFLOW
        )

        # 1. Permissions
        require_permission(
            spec.permission,
            requested_by_llm=context.requested_by_llm,
            call_mode=call_mode,
            tool_name=name,
        )

        # 2. Audit (before)
        start_ms = int(time.monotonic() * 1000)
        timeout_seconds = _timeout_seconds_for_call(
            tool,
            input_data,
            context,
            default=spec.timeout_seconds,
        )
        if context.requested_by_llm and (context.session_id or context.experiment_id):
            self._audit_entry(
                tool_name=name,
                run_id=context.run_id,
                session_id=context.session_id,
                experiment_id=context.experiment_id,
                permission=spec.permission.value,
                requested_by_llm=context.requested_by_llm,
                call_mode=call_mode.value,
                input_data=input_data,
                output_data={
                    "status": "STARTED",
                    "tool_name": name,
                    "timeout_seconds": timeout_seconds,
                    "execution_status": ExecutionStatus.STARTED.value,
                    "domain_status": DomainStatus.UNKNOWN.value,
                    "evidence_status": EvidenceStatus.UNKNOWN.value,
                    "recommendation_status": RecommendationStatus.UNKNOWN.value,
                },
                status="started",
                error_message=None,
                duration_ms=0,
            )

        # 3. Execute
        execution_status = ExecutionStatus.OK
        error_message = None
        result: dict[str, Any] = {}
        try:
            result = _run_with_timeout(tool, input_data, context, timeout_seconds)
            if not isinstance(result, dict):
                result = {"value": result}
            if result.get("status") == "TIMEOUT":
                execution_status = ExecutionStatus.TIMEOUT
        except FutureTimeoutError:
            execution_status = ExecutionStatus.TIMEOUT
            result = {
                "status": "TIMEOUT",
                "tool_name": name,
                "timeout_seconds": timeout_seconds,
                "duration_ms": int(time.monotonic() * 1000) - start_ms,
                "kill_attempted": False,
            }
        except StorageError as exc:
            execution_status = ExecutionStatus.ERROR
            result = storage_error_health_payload(exc)
        except Exception as exc:
            execution_status = (
                ExecutionStatus.PERMISSION_DENIED
                if "PermissionDenied" in type(exc).__name__
                else ExecutionStatus.ERROR
            )
            error_message = str(exc)
            result = {"error": True, "message": error_message}

        # 4. Audit (after)
        duration_ms = int(time.monotonic() * 1000) - start_ms
        result = normalize_tool_result(
            name,
            result,
            execution_status=execution_status,
            duration_ms=duration_ms,
        )
        status = audit_status_from_result(result)
        self._audit_entry(
            tool_name=name,
            run_id=context.run_id,
            session_id=context.session_id,
            experiment_id=context.experiment_id,
            permission=spec.permission.value,
            requested_by_llm=context.requested_by_llm,
            call_mode=call_mode.value,
            input_data=input_data,
            output_data=result,
            status=status,
            error_message=error_message,
            duration_ms=duration_ms,
        )

        if error_message is not None:
            raise ToolExecutionError(name, Exception(error_message))

        return result

    # ── Bridge: expose as original ToolRegistry (for LLM client) ──────────

    def to_legacy_registry(
        self,
        *,
        context_factory: Callable[[], ToolContext] | None = None,
        llm_callable_only: bool = True,
    ) -> ToolRegistry:
        legacy = ToolRegistry()
        for name, tool in sorted(self.tools.items()):
            spec = tool.spec
            if llm_callable_only and not can_llm_call(spec.permission):
                continue
            if llm_callable_only and not spec.llm_callable:
                continue
            capability = to_capability(spec.permission)

            def build_fn(nt: str) -> Callable[..., Any]:
                def fn(**kwargs: Any) -> dict[str, Any]:
                    context = (
                        context_factory()
                        if context_factory is not None
                        else ToolContext(run_id="legacy")
                    )
                    return self.run_tool(nt, kwargs, context)

                return fn

            legacy.register(
                ToolDefinition(
                    name=name,
                    capability=capability,
                    fn=build_fn(name),
                    description=spec.description,
                    parameters=_llm_input_schema(spec.input_schema),
                )
            )
        return legacy

    # ── Internal helpers ──────────────────────────────────────────────────

    def _require_tool(self, name: str) -> AgentTool:
        if name not in self.tools:
            raise ToolNotFoundError(f"tool '{name}' is not registered")
        return self.tools[name]

    def _audit_entry(
        self,
        *,
        tool_name: str,
        run_id: str,
        session_id: str | None,
        experiment_id: str | None,
        permission: str,
        requested_by_llm: bool,
        call_mode: str,
        input_data: dict[str, Any] | None,
        output_data: dict[str, Any] | None,
        status: str,
        error_message: str | None,
        duration_ms: int,
    ) -> None:
        if self.audit_logger is not None:
            try:
                output = output_data or {}
                self.audit_logger.append(
                    tool_name=tool_name,
                    run_id=run_id,
                    session_id=session_id,
                    experiment_id=experiment_id,
                    permission=permission,
                    requested_by_llm=requested_by_llm,
                    call_mode=call_mode,
                    input_data=input_data,
                    output_data=output_data,
                    status=status,
                    error_message=error_message,
                    duration_ms=duration_ms,
                    execution_status=str(output.get("execution_status", "UNKNOWN")),
                    domain_status=str(output.get("domain_status", "UNKNOWN")),
                    evidence_status=str(output.get("evidence_status", "UNKNOWN")),
                    recommendation_status=str(output.get("recommendation_status", "UNKNOWN")),
                    raw_status=(
                        str(output.get("raw_status"))
                        if output.get("raw_status") is not None
                        else None
                    ),
                    diagnostic_status=(
                        str(output.get("diagnostic_status"))
                        if output.get("diagnostic_status") is not None
                        else None
                    ),
                    blockers=[str(item) for item in output.get("blockers", []) if str(item)]
                    if isinstance(output.get("blockers"), list)
                    else [],
                    warnings=[str(item) for item in output.get("warnings", []) if str(item)]
                    if isinstance(output.get("warnings"), list)
                    else [],
                    next_repair_tool=(
                        str(output.get("next_repair_tool"))
                        if output.get("next_repair_tool") is not None
                        else None
                    ),
                )
            except Exception:
                pass  # audit failure must not break tool execution


def _llm_input_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return an OpenAI function-tool compatible object schema."""
    if schema.get("type") == "object":
        return schema
    if not schema:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
        "additionalProperties": schema.get("additionalProperties", False),
    }


def _run_with_timeout(
    tool: AgentTool,
    input_data: dict[str, Any],
    context: ToolContext,
    timeout_seconds: int | float,
) -> dict[str, Any]:
    if _should_use_process_timeout(tool, timeout_seconds):
        return _run_with_process_timeout(tool, input_data, context, timeout_seconds)

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(tool.run, input_data, context)
        return future.result(timeout=max(timeout_seconds, 0))
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _should_use_process_timeout(tool: AgentTool, timeout_seconds: int | float) -> bool:
    heavy_tools = {"run_backtest", "evaluate_factor_candidate", "query_bars"}
    return tool.spec.name in heavy_tools or timeout_seconds <= 0


def _run_with_process_timeout(
    tool: AgentTool,
    input_data: dict[str, Any],
    context: ToolContext,
    timeout_seconds: int | float,
) -> dict[str, Any]:
    try:
        process_context: Any = get_context("fork")
    except ValueError:
        return _run_with_thread_timeout_payload(tool, input_data, context, timeout_seconds)
    recv_conn, send_conn = process_context.Pipe(duplex=False)
    process = process_context.Process(
        target=_process_tool_runner,
        args=(tool, input_data, context, send_conn),
    )
    started_at = time.monotonic()
    process.start()
    send_conn.close()
    timeout_at = started_at + max(float(timeout_seconds), 0.0)
    payload: dict[str, Any] | None = None
    while True:
        if recv_conn.poll(0.05):
            payload = recv_conn.recv()
            break
        if not process.is_alive():
            break
        if time.monotonic() >= timeout_at:
            duration_ms = int((time.monotonic() - started_at) * 1000)
            process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join(1)
            recv_conn.close()
            return {
                "status": "TIMEOUT",
                "tool_name": tool.spec.name,
                "timeout_seconds": timeout_seconds,
                "duration_ms": duration_ms,
                "kill_attempted": True,
                "exitcode": process.exitcode,
                "partial_payload": payload is not None,
            }

    duration_ms = int((time.monotonic() - started_at) * 1000)
    process.join(1)
    recv_conn.close()
    if payload is None:
        if process.exitcode == 0:
            return {}
        raise RuntimeError(f"tool process exited with code {process.exitcode}")
    try:
        payload = _resolve_process_payload(payload)
    except Exception as exc:
        raise RuntimeError(f"failed to read tool process payload: {exc}") from exc
    if not isinstance(payload, dict):
        return {"value": payload}
    if payload.get("__error__"):
        raise RuntimeError(str(payload.get("message", "tool process failed")))
    result = payload.get("result")
    if isinstance(result, dict):
        if payload.get("__payload_file__"):
            result.setdefault("payload_file", payload["__payload_file__"])
            result.setdefault("payload_size_bytes", payload.get("payload_size_bytes"))
        result.setdefault("duration_ms", duration_ms)
        return result
    return {"value": result, "duration_ms": duration_ms}


def _run_with_thread_timeout_payload(
    tool: AgentTool,
    input_data: dict[str, Any],
    context: ToolContext,
    timeout_seconds: int | float,
) -> dict[str, Any]:
    started_at = time.monotonic()
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(tool.run, input_data, context)
        return future.result(timeout=max(timeout_seconds, 0))
    except FutureTimeoutError:
        return {
            "status": "TIMEOUT",
            "tool_name": tool.spec.name,
            "timeout_seconds": timeout_seconds,
            "duration_ms": int((time.monotonic() - started_at) * 1000),
            "kill_attempted": False,
        }
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _prepare_process_payload(
    payload: dict[str, Any],
    *,
    tool: AgentTool,
    context: ToolContext,
) -> dict[str, Any]:
    raw = json.dumps(payload, ensure_ascii=False, default=str)
    size = len(raw.encode("utf-8"))
    if size <= _PROCESS_PAYLOAD_SPILL_BYTES:
        return payload
    path = _process_payload_path(tool.spec.name, context.run_id)
    locks = PersistencePaths.from_settings(get_settings()).locks_root
    AtomicFileStore(LockManager(locks)).write_text(path, raw)
    result = payload.get("result")
    summary = _payload_summary(result)
    return {
        "__payload_file__": str(path),
        "payload_size_bytes": size,
        "summary": summary,
    }


def _resolve_process_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload_file = payload.get("__payload_file__")
    if not payload_file:
        return payload
    path = Path(str(payload_file))
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        loaded["__payload_file__"] = str(path)
        loaded["payload_size_bytes"] = payload.get("payload_size_bytes")
        loaded["summary"] = payload.get("summary")
        return loaded
    return {"result": loaded, "__payload_file__": str(path)}


def _process_payload_path(tool_name: str, run_id: str) -> Path:
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id or "run")
    safe_tool = re.sub(r"[^A-Za-z0-9_.-]+", "_", tool_name or "tool")
    root = PersistencePaths.from_settings(get_settings()).reports_root / "tool_payloads"
    return root / f"{safe_run_id}_{safe_tool}.json"


def _payload_summary(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return {
            "type": "dict",
            "keys": sorted(str(key) for key in result.keys())[:20],
            "key_count": len(result),
        }
    if isinstance(result, list):
        return {"type": "list", "length": len(result)}
    return {"type": type(result).__name__}


def _process_tool_runner(
    tool: AgentTool,
    input_data: dict[str, Any],
    context: ToolContext,
    conn: Any,
) -> None:
    try:
        payload = {"result": tool.run(input_data, context)}
        conn.send(_prepare_process_payload(payload, tool=tool, context=context))
    except Exception as exc:
        conn.send(
            {
                "__error__": True,
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )
    finally:
        conn.close()


def _timeout_seconds_for_call(
    tool: AgentTool,
    input_data: dict[str, Any],
    context: ToolContext,
    *,
    default: int,
) -> int | float:
    resolver = getattr(tool, "timeout_seconds_for_call", None)
    if resolver is None:
        return default
    resolved = resolver(input_data, context)
    if resolved is None:
        return default
    if not isinstance(resolved, int | float):
        return default
    return resolved
