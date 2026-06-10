"""Normalize LLM drafts into strict internal schemas."""

from __future__ import annotations

import re
from typing import Any

from qmt_agent_trader.agent.schemas import ResearchSpec


def normalize_research_spec(
    payload: dict[str, Any],
    *,
    fallback_name: str,
    fallback_description: str,
    universe: list[str],
) -> tuple[ResearchSpec, bool]:
    if "hypothesis" in payload and "implementation_plan" in payload:
        spec = ResearchSpec.model_validate(payload)
        return _force_paper_only(spec), True

    name = str(payload.get("name") or payload.get("id") or fallback_name)
    description = str(
        payload.get("description")
        or payload.get("summary")
        or payload.get("strategy")
        or fallback_description
    )
    intuition = str(payload.get("intuition") or payload.get("rationale") or description)
    required_data = payload.get("required_data") or payload.get("data") or ["daily_bars"]
    known_risks = (
        payload.get("known_risks") or payload.get("risks") or ["LLM draft requires review"]
    )

    spec = ResearchSpec.model_validate(
        {
            "hypothesis": {
                "name": _to_snake_case(name),
                "description": description,
                "intuition": intuition,
                "required_data": _string_list(required_data),
                "lookback": int(payload.get("lookback") or 20),
                "universe": universe,
                "expected_behavior": str(
                    payload.get("expected_behavior")
                    or payload.get("expected_return_driver")
                    or "Candidate should be evaluated by walk-forward paper backtests."
                ),
                "known_risks": _string_list(known_risks),
            },
            "implementation_plan": {
                "factor_code_allowed": True,
                "strategy_code_allowed": True,
                "live_trading_allowed": False,
            },
        }
    )
    return spec, False


def _force_paper_only(spec: ResearchSpec) -> ResearchSpec:
    if spec.implementation_plan.live_trading_allowed is False:
        return spec
    data = spec.model_dump(mode="json")
    data["implementation_plan"]["live_trading_allowed"] = False
    return ResearchSpec.model_validate(data)


def _to_snake_case(value: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").lower()
    return value or "llm_candidate"


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if value is None:
        return []
    return [str(value)]
