"""Strategy engineering workflow.

Pipelines: select factors → strategy spec → code generation → backtest →
research report → REVIEW_REQUIRED or FAILED.
"""

from __future__ import annotations

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.schemas import ExperimentRecord, ExperimentStatus, ToolContext
from qmt_agent_trader.agent.tool_registry import AgentToolRegistry
from qmt_agent_trader.core.ids import new_id
from qmt_agent_trader.core.types import ApprovalStatus
from qmt_agent_trader.strategy.models import (
    SavedStrategy,
    StrategySource,
    strategy_spec_from_agent_spec,
)
from qmt_agent_trader.strategy.registry import StrategyRegistry


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
            factor_name = (
                strategy_spec.get("factors", [{}])[0].get("factor_id")
                if strategy_spec.get("factors")
                else None
            )

            # Step 2: generate_strategy_code
            code_result = self.registry.run_tool(
                "generate_strategy_code",
                {"strategy_spec": strategy_spec},
                _ctx(),
            )
            code_path = code_result.get("code_path", "")
            tests_path = code_result.get("tests_path", "")
            if code_path:
                self.store.add_artifact(exp.experiment_id, code_path)
            if tests_path:
                self.store.add_artifact(exp.experiment_id, tests_path)
            if code_result.get("status") != "generated":
                self.store.add_lesson(
                    exp.experiment_id,
                    f"code generation failed: {code_result.get('message', 'unknown')}",
                )
                self.store.update_experiment(exp.experiment_id, status=ExperimentStatus.FAILED)
                return self.store.get_experiment(exp.experiment_id)

            # Step 3: static checks
            static_result = self.registry.run_tool(
                "run_strategy_static_checks",
                {"code_path": code_path},
                _ctx(),
            )
            self.store.add_artifact(
                exp.experiment_id,
                f"static_check:{static_result.get('status')}",
            )
            if static_result.get("status") != "PASSED":
                self.store.add_lesson(
                    exp.experiment_id,
                    f"static check failed: {static_result.get('issues', [])}",
                )
                self.store.update_experiment(exp.experiment_id, status=ExperimentStatus.FAILED)
                return self.store.get_experiment(exp.experiment_id)

            # Step 4: save candidate registry record
            strategy_registry = StrategyRegistry(self.store.root.parent / "strategies")
            canonical_spec = strategy_spec_from_agent_spec(strategy_spec)
            try:
                saved = strategy_registry.save_candidate(
                    SavedStrategy(
                        strategy_id=canonical_spec.strategy_id,
                        name=canonical_spec.name,
                        version=canonical_spec.version,
                        source=StrategySource.AGENT_GENERATED,
                        status=ApprovalStatus.GENERATED_BY_LLM,
                        spec=canonical_spec,
                        implementation_ref=f"file:{code_path}",
                        code_path=code_path,
                        tests_path=tests_path or None,
                        created_by="agent",
                    )
                )
                self.store.add_artifact(exp.experiment_id, f"strategy_registry:{saved.strategy_id}")
            except ValueError as exc:
                self.store.add_lesson(exp.experiment_id, f"registry save skipped: {exc}")

            # Step 5: run_backtest
            bt_result = self.registry.run_tool(
                "run_backtest",
                {
                    "strategy_id": strategy_id,
                    "strategy_spec": strategy_spec,
                    "code_path": code_path,
                    "factor_name": factor_name,
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
                try:
                    strategy_registry.attach_report(strategy_id, report_path)
                    strategy_registry.update_status(strategy_id, ApprovalStatus.BACKTESTED)
                except ValueError as exc:
                    self.store.add_lesson(
                        exp.experiment_id,
                        f"registry report attach skipped: {exc}",
                    )

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

            diagnostics = bt_result.get("diagnostics") or {}

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
                try:
                    strategy_registry.attach_report(strategy_id, rp)
                except ValueError as exc:
                    self.store.add_lesson(
                        exp.experiment_id,
                        f"registry report attach skipped: {exc}",
                    )

            if diagnostics.get("status") == "FAIL":
                self.store.add_lesson(
                    exp.experiment_id,
                    "diagnostics failed; candidate not review-ready",
                )
                self.store.update_experiment(
                    exp.experiment_id,
                    status=ExperimentStatus.FAILED,
                    metrics=bt_result.get("metrics", {}),
                )
                return self.store.get_experiment(exp.experiment_id)

            try:
                strategy_registry.update_status(strategy_id, ApprovalStatus.REVIEW_REQUIRED)
            except ValueError as exc:
                self.store.add_lesson(exp.experiment_id, f"registry status update skipped: {exc}")

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
