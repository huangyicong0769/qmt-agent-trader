"""Strategy engineering workflow.

Pipelines: select factors → strategy spec → code generation → backtest →
research report → REVIEW_REQUIRED or FAILED.
"""

from __future__ import annotations

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.schemas import ExperimentRecord, ExperimentStatus, ToolContext
from qmt_agent_trader.agent.tool_registry import AgentToolRegistry
from qmt_agent_trader.core.ids import new_id


class StrategyEngineeringWorkflow:
    """Execute the end-to-end strategy engineering pipeline."""

    def __init__(self, registry: AgentToolRegistry, store: ExperimentStore) -> None:
        self.registry = registry
        self.store = store

    def run(
        self,
        strategy_idea: str,
        selected_factors: list[str],
        universe: str,
        start_date: str,
        end_date: str,
    ) -> ExperimentRecord:
        run_id = new_id("run")
        exp = self.store.create_experiment(
            kind="strategy_engineering",
            hypothesis={
                "idea": strategy_idea,
                "factors": selected_factors,
                "universe": universe,
            },
            tags=["strategy", universe],
        )

        def _ctx() -> ToolContext:
            return ToolContext(run_id=run_id, experiment_id=exp.experiment_id)

        try:
            self.store.update_experiment(exp.experiment_id, status=ExperimentStatus.RUNNING)

            # Step 1: create_strategy_spec
            spec_result = self.registry.run_tool(
                "create_strategy_spec",
                {
                    "strategy_idea": strategy_idea,
                    "selected_factors": selected_factors,
                    "universe": universe,
                    "rebalance_frequency": "daily",
                    "constraints": {},
                },
                _ctx(),
            )
            self.store.add_artifact(exp.experiment_id, "strategy_spec_created")
            strategy_spec = spec_result.get("strategy_spec", {})
            strategy_id = strategy_spec.get("strategy_id", "unknown")

            # Step 2: generate_strategy_code
            code_result = self.registry.run_tool(
                "generate_strategy_code",
                {"strategy_spec": strategy_spec},
                _ctx(),
            )
            code_path = code_result.get("code_path", "")
            if code_path:
                self.store.add_artifact(exp.experiment_id, code_path)

            # Step 3: run_backtest
            bt_result = self.registry.run_tool(
                "run_backtest",
                {
                    "strategy_id": strategy_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "universe": universe,
                    "initial_cash": 1_000_000,
                },
                _ctx(),
            )
            report_path = bt_result.get("report_path", "")
            if report_path:
                self.store.add_artifact(exp.experiment_id, report_path)

            if bt_result.get("status") != "completed":
                self.store.add_lesson(
                    exp.experiment_id,
                    f"backtest failed: {bt_result.get('message', 'unknown')}",
                )
                self.store.update_experiment(
                    exp.experiment_id,
                    status=ExperimentStatus.FAILED,
                    metrics=bt_result.get("metrics", {}),
                )
                return self.store.get_experiment(exp.experiment_id)

            # Step 4: generate_research_report
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
                metrics=bt_result.get("metrics", {}),
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
