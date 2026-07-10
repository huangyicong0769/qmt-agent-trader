"""Universe research tools."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from pydantic import ValidationError

from qmt_agent_trader.agent.permissions import PermissionLevel
from qmt_agent_trader.agent.schemas import ToolContext, ToolSpec
from qmt_agent_trader.agent.tool_dependencies import AgentToolDependencies
from qmt_agent_trader.agent.tool_result import (
    DomainStatus,
    EvidenceStatus,
    ExecutionStatus,
    RecommendationStatus,
)
from qmt_agent_trader.agent.tools.base import AgentTool, tool
from qmt_agent_trader.core.ids import new_id, shanghai_now_iso
from qmt_agent_trader.data.storage import DataLake
from qmt_agent_trader.universe.builtins import broad_universe_spec
from qmt_agent_trader.universe.fingerprints import fingerprint_spec
from qmt_agent_trader.universe.models import UniverseSpec
from qmt_agent_trader.universe.registry import UniverseRegistry, registry_root_from_payload
from qmt_agent_trader.universe.resolver import UniverseResolver

_lake: DataLake | None = None
_lake_var: ContextVar[DataLake | None] = ContextVar("universe_tool_lake", default=None)


def set_data_lake(lake: DataLake) -> None:
    global _lake
    _lake = lake


def _get_lake() -> DataLake | None:
    return _lake_var.get() or _lake


def _with_deps(
    deps: AgentToolDependencies,
    fn: Callable[[dict[str, Any], ToolContext], dict[str, Any]],
    input_data: dict[str, Any],
    context: ToolContext,
) -> dict[str, Any]:
    token = _lake_var.set(deps.data_lake)
    try:
        return fn(input_data, context)
    finally:
        _lake_var.reset(token)


def _create_universe_spec(input_data: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    payload = dict(input_data)
    payload.setdefault("universe_id", new_id("universe"))
    payload.setdefault("source", "agent_generated")
    payload.setdefault("asset_types", ["stock"])
    payload.setdefault("selection", {"mode": "all"})
    payload.setdefault("mode", "snapshot")
    payload.setdefault("rebalance_frequency", "daily")
    payload.setdefault("created_at", shanghai_now_iso())
    if payload.get("source") == "agent_generated":
        payload.update(
            {
                "research_only": True,
                "live_trading_allowed": False,
                "approval_required": True,
            }
        )
    try:
        spec = UniverseSpec.model_validate(payload)
    except ValidationError as exc:
        return _invalid_request("INVALID_UNIVERSE_SPEC", exc)
    return _with_universe_evidence_status(
        {
            "status": "created",
            "universe_spec": spec.model_dump(mode="json"),
            "research_only": spec.research_only,
            "live_trading_allowed": spec.live_trading_allowed,
            "suggested_next_tools": [
                "validate_universe_spec",
                "build_universe",
                "save_universe_spec",
            ],
        }
    )


def _validate_universe_spec(input_data: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    raw = input_data.get("universe_spec", input_data)
    try:
        spec = UniverseSpec.model_validate(raw)
    except ValidationError as exc:
        return _with_universe_evidence_status(
            {
                "status": "INVALID_REQUEST",
                "reason": "INVALID_UNIVERSE_SPEC",
                "valid": False,
                "normalized_spec": None,
                "errors": _format_validation_errors(exc),
            }
        )
    return _with_universe_evidence_status(
        {
            "status": "OK",
            "valid": True,
            "normalized_spec": spec.model_dump(mode="json"),
            "spec_fingerprint": fingerprint_spec(spec),
            "research_only": spec.research_only,
            "live_trading_allowed": spec.live_trading_allowed,
        }
    )


def _build_universe(input_data: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    if lake is None:
        return _with_universe_evidence_status(
            {"status": "NOT_AVAILABLE", "message": "data lake not wired"}
        )
    spec_result = _resolve_spec(input_data, lake)
    if spec_result.get("status") != "OK":
        return _with_universe_evidence_status(spec_result)
    spec = spec_result["spec"]
    result = UniverseResolver(lake).build(
        spec,
        as_of_date=_optional_string(input_data.get("as_of_date")),
        mode=_optional_string(input_data.get("mode")) or spec.mode,
        start_date=_optional_string(input_data.get("start_date")),
        end_date=_optional_string(input_data.get("end_date")),
        rebalance_frequency=_optional_string(input_data.get("rebalance_frequency"))
        or spec.rebalance_frequency,
        limit=int(input_data.get("limit", 2000)),
        include_exclusions=bool(input_data.get("include_exclusions", False)),
    )
    return _with_universe_evidence_status(result)


def _save_universe_spec(input_data: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    raw = input_data.get("universe_spec", input_data)
    try:
        spec = UniverseSpec.model_validate(raw)
    except ValidationError as exc:
        return _invalid_request("INVALID_UNIVERSE_SPEC", exc)
    try:
        registry = UniverseRegistry(registry_root_from_payload(input_data, lake))
    except ValueError as exc:
        return _with_universe_evidence_status({"status": "INVALID_REQUEST", "reason": str(exc)})
    path = registry.save(spec, expected_revision=input_data.get("expected_revision"))
    return _with_universe_evidence_status(
        {
            "status": "saved",
            "path": str(path),
            "universe_spec": spec.model_dump(mode="json"),
            "research_only": spec.research_only,
            "live_trading_allowed": spec.live_trading_allowed,
        }
    )


def _list_universes(input_data: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    try:
        registry = UniverseRegistry(registry_root_from_payload(input_data, lake))
    except ValueError as exc:
        return _with_universe_evidence_status({"status": "INVALID_REQUEST", "reason": str(exc)})
    specs = registry.list(
        source=_optional_string(input_data.get("source")),
        query=_optional_string(input_data.get("query")),
        asset_type=_optional_string(input_data.get("asset_type")),
        mode=_optional_string(input_data.get("mode")),
    )
    return _with_universe_evidence_status(
        {
            "status": "DEGRADED" if registry.last_diagnostics else "OK",
            "diagnostics": [
                {"path": str(item.path), "reason": item.error.reason}
                for item in registry.last_diagnostics
            ],
            "universes": [
                {
                    "universe_id": spec.universe_id,
                    "name": spec.name,
                    "description": spec.description,
                    "source": spec.source,
                    "asset_types": spec.asset_types,
                    "mode": spec.mode,
                    "research_only": spec.research_only,
                    "live_trading_allowed": spec.live_trading_allowed,
                }
                for spec in specs
            ],
        }
    )


def _inspect_universe(input_data: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    lake = _get_lake()
    if not input_data.get("universe_id"):
        return _with_universe_evidence_status(
            {"status": "INVALID_REQUEST", "reason": "UNIVERSE_ID_REQUIRED"}
        )
    registry = UniverseRegistry(registry_root_from_payload(input_data, lake))
    spec = registry.load(str(input_data["universe_id"]))
    if spec is None:
        return _with_universe_evidence_status(
            {
                "status": "NOT_FOUND",
                "reason": "UNIVERSE_NOT_FOUND",
                "universe_id": str(input_data["universe_id"]),
            }
        )
    payload: dict[str, Any] = {
        "status": "OK",
        "universe_spec": spec.model_dump(mode="json"),
        "spec_fingerprint": fingerprint_spec(spec),
    }
    if bool(input_data.get("preview")):
        if lake is None:
            return _with_universe_evidence_status(
                {
                    **payload,
                    "preview": {
                        "status": "NOT_AVAILABLE",
                        "message": "data lake not wired",
                    },
                }
            )
        payload["preview"] = UniverseResolver(lake).build(
            spec,
            as_of_date=_optional_string(input_data.get("as_of_date")),
            mode=_optional_string(input_data.get("mode")) or spec.mode,
            start_date=_optional_string(input_data.get("start_date")),
            end_date=_optional_string(input_data.get("end_date")),
            rebalance_frequency=_optional_string(input_data.get("rebalance_frequency"))
            or spec.rebalance_frequency,
            limit=int(input_data.get("limit", 2000)),
            include_exclusions=bool(input_data.get("include_exclusions", False)),
        )
    return _with_universe_evidence_status(payload)


def _query_universe(input_data: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
    legacy = _legacy_theme_filter(input_data)
    if legacy is not None:
        return _with_universe_evidence_status(legacy)
    if input_data.get("filters"):
        return _with_universe_evidence_status(
            {
                "status": "INVALID_REQUEST",
                "reason": "UNSUPPORTED_QUERY_FILTERS",
                "message": "query_universe no longer accepts ad hoc filters. Use universe_spec.",
                "suggested_next_tools": [
                    "create_universe_spec",
                    "validate_universe_spec",
                    "build_universe",
                ],
            }
        )
    lake = _get_lake()
    if lake is None:
        return _with_universe_evidence_status(
            {"status": "NOT_AVAILABLE", "message": "data lake not wired"}
        )
    build_input = dict(input_data)
    if "universe_spec" not in build_input and "universe_id" not in build_input:
        mode = str(build_input.get("mode") or "snapshot")
        frequency = str(build_input.get("rebalance_frequency") or "daily")
        try:
            spec = broad_universe_spec(
                str(build_input.get("universe_type") or "stock"),
                mode=mode,
                rebalance_frequency=frequency,
            )
        except ValueError as exc:
            return _with_universe_evidence_status(
                {
                    "status": "INVALID_REQUEST",
                    "reason": "UNSUPPORTED_UNIVERSE_TYPE",
                    "message": str(exc),
                    "allowed_universe_types": ["stock", "etf", "mixed"],
                }
            )
        build_input["universe_spec"] = spec.model_dump(mode="json")
    return _build_universe(build_input, _context)


def _resolve_spec(input_data: dict[str, Any], lake: DataLake) -> dict[str, Any]:
    if input_data.get("universe_spec") is not None:
        try:
            return {
                "status": "OK",
                "spec": UniverseSpec.model_validate(input_data["universe_spec"]),
            }
        except ValidationError as exc:
            return {
                "status": "INVALID_REQUEST",
                "reason": "INVALID_UNIVERSE_SPEC",
                "errors": _format_validation_errors(exc),
            }
    if input_data.get("universe_id") is not None:
        registry = UniverseRegistry(registry_root_from_payload(input_data, lake))
        spec = registry.load(str(input_data["universe_id"]))
        if spec is None:
            return {
                "status": "NOT_FOUND",
                "reason": "UNIVERSE_NOT_FOUND",
                "universe_id": str(input_data["universe_id"]),
            }
        return {"status": "OK", "spec": spec}
    try:
        return {
            "status": "OK",
            "spec": broad_universe_spec(
                str(input_data.get("universe_type") or "stock"),
                mode=str(input_data.get("mode") or "snapshot"),
                rebalance_frequency=str(input_data.get("rebalance_frequency") or "daily"),
            ),
        }
    except ValueError as exc:
        return {
            "status": "INVALID_REQUEST",
            "reason": "UNSUPPORTED_UNIVERSE_TYPE",
            "message": str(exc),
        }


def _legacy_theme_filter(input_data: dict[str, Any]) -> dict[str, Any] | None:
    legacy_payload = input_data.get("filters")
    if isinstance(legacy_payload, dict) and "theme" in legacy_payload:
        legacy_label = "filters."
        legacy_label += "theme"
        return {
            "status": "INVALID_REQUEST",
            "reason": "LEGACY_THEME_FILTER_REMOVED",
            "message": (
                f"{legacy_label} has been removed. "
                "Use create_universe_spec + build_universe instead."
            ),
            "suggested_next_tools": [
                "create_universe_spec",
                "validate_universe_spec",
                "build_universe",
            ],
        }
    if "theme" in input_data:
        return {
            "status": "INVALID_REQUEST",
            "reason": "LEGACY_THEME_FILTER_REMOVED",
            "message": "theme has been removed. Use create_universe_spec + build_universe instead.",
            "suggested_next_tools": [
                "create_universe_spec",
                "validate_universe_spec",
                "build_universe",
            ],
        }
    return None


def _invalid_request(reason: str, exc: ValidationError) -> dict[str, Any]:
    return _with_universe_evidence_status(
        {
            "status": "INVALID_REQUEST",
            "reason": reason,
            "errors": _format_validation_errors(exc),
        }
    )


def _format_validation_errors(exc: ValidationError) -> list[dict[str, str]]:
    return [
        {
            "field": ".".join(str(part) for part in error.get("loc", ())),
            "message": (
                f"{'.'.join(str(part) for part in error.get('loc', ())) or 'spec'}: "
                f"{error.get('msg', 'invalid value')}"
            ),
        }
        for error in exc.errors()
    ]


def _with_universe_evidence_status(payload: dict[str, Any]) -> dict[str, Any]:
    status = str(payload.get("status", "UNKNOWN"))
    enriched = dict(payload)
    enriched["execution_status"] = ExecutionStatus.OK.value
    enriched["raw_status"] = status
    if status in {"OK", "created", "saved"}:
        enriched.setdefault("domain_status", DomainStatus.OK.value)
        enriched.setdefault("evidence_status", EvidenceStatus.VALID.value)
        enriched.setdefault("recommendation_status", RecommendationStatus.RESEARCH_ONLY.value)
    elif status in {"INVALID_REQUEST"}:
        enriched.setdefault("domain_status", DomainStatus.INVALID_REQUEST.value)
        enriched.setdefault("evidence_status", EvidenceStatus.INVALID.value)
        enriched.setdefault("recommendation_status", RecommendationStatus.BLOCKED.value)
    elif status in {"NOT_AVAILABLE", "NOT_FOUND"}:
        enriched.setdefault("domain_status", DomainStatus.NOT_CONFIGURED.value)
        enriched.setdefault("evidence_status", EvidenceStatus.BLOCKED.value)
        enriched.setdefault("recommendation_status", RecommendationStatus.BLOCKED.value)
    elif status in {"ERROR"}:
        enriched.setdefault("domain_status", DomainStatus.FAILED.value)
        enriched.setdefault("evidence_status", EvidenceStatus.INVALID.value)
        enriched.setdefault("recommendation_status", RecommendationStatus.BLOCKED.value)
    else:
        enriched.setdefault("domain_status", DomainStatus.UNKNOWN.value)
        enriched.setdefault("evidence_status", EvidenceStatus.UNKNOWN.value)
        enriched.setdefault("recommendation_status", RecommendationStatus.UNKNOWN.value)
    return enriched


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


_UNIVERSE_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
}


create_universe_spec_tool: AgentTool = tool(
    ToolSpec(
        name="create_universe_spec",
        description="Create a research-only declarative universe spec without resolving symbols.",
        input_schema={
            "type": "object",
            "properties": {
                "universe_id": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "source": {"type": "string"},
                "asset_types": {"type": "array", "items": {"type": "string"}},
                "selection": _UNIVERSE_SPEC_SCHEMA,
                "filters": _UNIVERSE_SPEC_SCHEMA,
                "ranking": _UNIVERSE_SPEC_SCHEMA,
                "max_symbols": {"type": "integer"},
                "mode": {"type": "string"},
                "rebalance_frequency": {"type": "string"},
            },
            "required": ["name"],
            "additionalProperties": True,
        },
        permission=PermissionLevel.RESEARCH_WRITE,
        deterministic=False,
    ),
    fn=_create_universe_spec,
)

validate_universe_spec_tool: AgentTool = tool(
    ToolSpec(
        name="validate_universe_spec",
        description="Statically validate a declarative universe spec and return a normalized copy.",
        input_schema={
            "type": "object",
            "properties": {"universe_spec": _UNIVERSE_SPEC_SCHEMA},
            "required": ["universe_spec"],
        },
        permission=PermissionLevel.READ_ONLY,
        deterministic=True,
    ),
    fn=_validate_universe_spec,
)

build_universe_tool: AgentTool = tool(
    ToolSpec(
        name="build_universe",
        description="Resolve a universe spec into snapshot symbols or rolling per-date symbols.",
        input_schema={
            "type": "object",
            "properties": {
                "universe_spec": _UNIVERSE_SPEC_SCHEMA,
                "universe_id": {"type": "string"},
                "as_of_date": {"type": "string"},
                "mode": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "rebalance_frequency": {"type": "string"},
                "limit": {"type": "integer"},
                "include_exclusions": {"type": "boolean"},
            },
            "additionalProperties": True,
        },
        permission=PermissionLevel.READ_ONLY,
        deterministic=False,
    ),
    fn=_build_universe,
)

save_universe_spec_tool: AgentTool = tool(
    ToolSpec(
        name="save_universe_spec",
        description="Persist a validated research-only universe spec.",
        input_schema={
            "type": "object",
            "properties": {"universe_spec": _UNIVERSE_SPEC_SCHEMA},
            "required": ["universe_spec"],
        },
        permission=PermissionLevel.RESEARCH_WRITE,
        side_effect_level="write_generated",
        deterministic=False,
    ),
    fn=_save_universe_spec,
)

list_universes_tool: AgentTool = tool(
    ToolSpec(
        name="list_universes",
        description="List saved universe specs with optional filters.",
        input_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "query": {"type": "string"},
                "asset_type": {"type": "string"},
                "mode": {"type": "string"},
            },
        },
        permission=PermissionLevel.READ_ONLY,
        deterministic=False,
    ),
    fn=_list_universes,
)

inspect_universe_tool: AgentTool = tool(
    ToolSpec(
        name="inspect_universe",
        description="Inspect a saved universe spec and optionally build a preview.",
        input_schema={
            "type": "object",
            "properties": {
                "universe_id": {"type": "string"},
                "preview": {"type": "boolean"},
                "as_of_date": {"type": "string"},
                "mode": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "rebalance_frequency": {"type": "string"},
                "limit": {"type": "integer"},
                "include_exclusions": {"type": "boolean"},
            },
            "required": ["universe_id"],
        },
        permission=PermissionLevel.READ_ONLY,
        deterministic=False,
    ),
    fn=_inspect_universe,
)

query_universe_tool: AgentTool = tool(
    ToolSpec(
        name="query_universe",
        description=(
            "Resolve a saved or inline first-class universe spec. Legacy theme-filter "
            "requests return INVALID_REQUEST."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "as_of_date": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "universe_type": {"type": "string"},
                "universe_id": {"type": "string"},
                "universe_spec": _UNIVERSE_SPEC_SCHEMA,
                "mode": {"type": "string"},
                "limit": {"type": "integer"},
                "include_exclusions": {"type": "boolean"},
            },
            "additionalProperties": True,
        },
        permission=PermissionLevel.READ_ONLY,
        deterministic=False,
    ),
    fn=_query_universe,
)


def build_universe_tools(deps: AgentToolDependencies) -> list[AgentTool]:
    return [
        tool(
            create_universe_spec_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps, _create_universe_spec, input_data, context
            ),
        ),
        tool(
            validate_universe_spec_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps, _validate_universe_spec, input_data, context
            ),
        ),
        tool(
            build_universe_tool.spec,
            fn=lambda input_data, context: _with_deps(deps, _build_universe, input_data, context),
        ),
        tool(
            save_universe_spec_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps, _save_universe_spec, input_data, context
            ),
        ),
        tool(
            list_universes_tool.spec,
            fn=lambda input_data, context: _with_deps(deps, _list_universes, input_data, context),
        ),
        tool(
            inspect_universe_tool.spec,
            fn=lambda input_data, context: _with_deps(
                deps, _inspect_universe, input_data, context
            ),
        ),
        tool(
            query_universe_tool.spec,
            fn=lambda input_data, context: _with_deps(deps, _query_universe, input_data, context),
        ),
    ]
