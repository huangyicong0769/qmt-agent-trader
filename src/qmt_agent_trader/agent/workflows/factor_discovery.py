"""Factor discovery workflow.

Two entry points:
1.  `run_factor_discovery`: LLM-driven discovery (original, preserved).
2.  `FactorDiscoveryWorkflow`: tool-chain pipeline (new).
"""

from __future__ import annotations

from typing import Any

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.llm_client import DeepSeekClient, _parse_json_object
from qmt_agent_trader.agent.prompts import FACTOR_DISCOVERY_PROMPT
from qmt_agent_trader.agent.schemas import ExperimentRecord, ExperimentStatus, ToolContext
from qmt_agent_trader.agent.tool_registry import AgentToolRegistry
from qmt_agent_trader.agent.tools.research_context import build_research_context_tool
from qmt_agent_trader.agent.workflows.normalization import normalize_research_spec
from qmt_agent_trader.core.config import Settings
from qmt_agent_trader.core.ids import new_id

# ── Original LLM-driven discovery (preserved) ────────────────────────────────


def run_factor_discovery(theme: str, settings: Settings | None = None) -> dict[str, Any]:
    if settings is None or settings.deepseek_api_key is None:
        return {"theme": theme, "status": "REVIEW_REQUIRED", "mode": "offline_skeleton"}

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
                        FACTOR_DISCOVERY_PROMPT,
                        f"Theme: {theme}",
                        "Focus on A-share stocks and ETFs, daily frequency, paper research only.",
                    ]
                ),
            },
        ],
        tools=[build_research_context_tool()],
    )
    payload = _parse_json_object(result.content)
    spec, schema_valid = normalize_research_spec(
        payload,
        fallback_name=theme,
        fallback_description="LLM factor candidate for human review.",
        universe=["A_SHARE_STOCK", "ETF"],
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


# ── New tool-chain pipeline ──────────────────────────────────────────────────


class FactorDiscoveryWorkflow:
    """Execute the end-to-end factor discovery pipeline.

    hypothesis → factor spec → code generation → static checks →
    factor evaluation → research report → REVIEW_REQUIRED.

    Every step creates artifacts in the experiment store. The final state
    is never greater than REVIEW_REQUIRED.
    """

    def __init__(self, registry: AgentToolRegistry, store: ExperimentStore) -> None:
        self.registry = registry
        self.store = store

    def run(
        self,
        theme: str,
        universe: str,
        start_date: str,
        end_date: str,
    ) -> ExperimentRecord:
        run_id = new_id("run")
        exp = self.store.create_experiment(
            kind="factor_discovery",
            hypothesis={"theme": theme, "universe": universe},
            tags=["factor_discovery", universe],
        )

        def _ctx() -> ToolContext:
            return ToolContext(run_id=run_id, experiment_id=exp.experiment_id)

        try:
            self.store.update_experiment(exp.experiment_id, status=ExperimentStatus.RUNNING)

            # Step 1: create_factor_spec
            spec_result = self.registry.run_tool(
                "create_factor_spec",
                {
                    "hypothesis": {
                        "name": theme.replace(" ", "_")[:40],
                        "intuition": theme,
                        "formula_sketch": "candidate",
                    }
                },
                _ctx(),
            )
            self.store.add_artifact(exp.experiment_id, "factor_spec_created")
            factor_spec = spec_result.get("factor_spec", {})
            factor_id = factor_spec.get("factor_id", "unknown")

            # Step 2: generate_factor_code
            code_result = self.registry.run_tool(
                "generate_factor_code",
                {"factor_spec": factor_spec},
                _ctx(),
            )
            code_path = code_result.get("code_path", "")
            spec_path = code_result.get("spec_path", "")
            if code_path:
                self.store.add_artifact(exp.experiment_id, code_path)

            # Step 3: run_factor_static_checks
            static_result = self.registry.run_tool(
                "run_factor_static_checks",
                {"code_path": code_path},
                _ctx(),
            )
            if static_result.get("status") == "FAILED":
                issues = static_result.get("issues", [])
                self.store.add_lesson(
                    exp.experiment_id, f"static checks failed: {issues}"
                )
                self.store.update_experiment(
                    exp.experiment_id,
                    status=ExperimentStatus.FAILED,
                    metrics={"static_issues": issues},
                )
                return self.store.get_experiment(exp.experiment_id)

            # Step 4: save_factor
            save_result = self.registry.run_tool(
                "save_factor",
                {
                    "factor_id": factor_id,
                    "code_path": code_path,
                    "spec_path": spec_path,
                },
                _ctx(),
            )
            registry_path = save_result.get("registry_path", "")
            if registry_path:
                self.store.add_artifact(exp.experiment_id, registry_path)

            # Step 5: evaluate_factor_candidate
            eval_result = self.registry.run_tool(
                "evaluate_factor_candidate",
                {
                    "factor_id": factor_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "universe": universe,
                },
                _ctx(),
            )
            report_path = eval_result.get("report_path", "")
            if report_path:
                self.store.add_artifact(exp.experiment_id, report_path)

            # Step 6: generate_research_report
            report_result = self.registry.run_tool(
                "generate_research_report",
                {
                    "experiment_id": exp.experiment_id,
                    "run_ids": [run_id],
                    "include_sections": ["summary", "metrics", "lessons"],
                },
                _ctx(),
            )
            rp = report_result.get("report_path", "")
            if rp:
                self.store.add_artifact(exp.experiment_id, rp)

            self.store.update_experiment(
                exp.experiment_id,
                status=ExperimentStatus.REVIEW_REQUIRED,
                metrics={
                    "ic_mean": eval_result.get("ic_mean"),
                    "coverage": eval_result.get("coverage"),
                },
            )
            return self.store.get_experiment(exp.experiment_id)

        except Exception as exc:
            self.store.add_lesson(exp.experiment_id, f"workflow failed: {exc}")
            self.store.update_experiment(
                exp.experiment_id,
                status=ExperimentStatus.FAILED,
                metrics={"error": str(exc)},
            )
            return self.store.get_experiment(exp.experiment_id)
