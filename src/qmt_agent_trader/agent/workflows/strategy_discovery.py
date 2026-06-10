"""Strategy discovery workflow."""

from __future__ import annotations

from typing import Any

from qmt_agent_trader.agent.llm_client import DeepSeekClient, _parse_json_object
from qmt_agent_trader.agent.prompts import STRATEGY_DISCOVERY_PROMPT
from qmt_agent_trader.agent.tools.research_context import build_research_context_tool
from qmt_agent_trader.agent.workflows.normalization import normalize_research_spec
from qmt_agent_trader.core.config import Settings


def run_strategy_discovery(universe: str, settings: Settings | None = None) -> dict[str, Any]:
    if settings is None or settings.deepseek_api_key is None:
        return {"universe": universe, "status": "REVIEW_REQUIRED", "mode": "offline_skeleton"}

    client = DeepSeekClient(
        api_key=settings.deepseek_api_key.get_secret_value(),
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
    )
    result = client.run_tool_loop(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a quant research assistant. Call get_research_context before "
                    "drafting the candidate. Return only JSON matching ResearchSpec."
                ),
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        STRATEGY_DISCOVERY_PROMPT,
                        f"Universe: {universe}",
                        "Return one candidate only. It must require human review before approval.",
                    ]
                ),
            },
        ],
        tools=[build_research_context_tool()],
    )
    payload = _parse_json_object(result.content)
    spec, schema_valid = normalize_research_spec(
        payload,
        fallback_name=f"{universe}_strategy_candidate",
        fallback_description="LLM strategy candidate for human review.",
        universe=[item.strip() for item in universe.split(",") if item.strip()],
    )
    return {
        "status": "REVIEW_REQUIRED",
        "mode": "deepseek",
        "model": settings.deepseek_model,
        "tool_use": {
            "enabled": True,
            "tool_calls": [call.name for call in result.tool_calls],
        },
        "schema_valid": schema_valid,
        "spec": spec.model_dump(mode="json"),
    }
