"""Self-bootstrap workflow.

Orchestrates: search failures → detect tool gaps → create spec → generate code
→ generate tests → sandbox tests → score → propose registration.

Cannot create broker/order/gateway/live tools.
"""

from __future__ import annotations

from typing import Any

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.permissions import ToolCallMode
from qmt_agent_trader.agent.schemas import ExperimentRecord, ExperimentStatus, ToolContext
from qmt_agent_trader.agent.tool_registry import AgentToolRegistry
from qmt_agent_trader.core.ids import new_id

FORBIDDEN_TOOL_KEYWORDS = [
    "broker",
    "order",
    "gateway",
    "live",
    "submit",
    "approve_strategy",
    "register_production",
    "modify_risk",
    "modify_config",
    "delete_",
    "query_account_secret",
    "read_env",
    "write_env",
]


class SelfBootstrapWorkflow:
    """Search failures, propose and evaluate new tool candidates."""

    def __init__(self, registry: AgentToolRegistry, store: ExperimentStore) -> None:
        self.registry = registry
        self.store = store

    def run(self, recent_experiment_ids: list[str]) -> ExperimentRecord:
        run_id = new_id("run")
        exp = self.store.create_experiment(
            kind="self_bootstrap",
            hypothesis={"recent_experiment_ids": recent_experiment_ids},
            tags=["self_bootstrap"],
        )

        def _ctx() -> ToolContext:
            return ToolContext(run_id=run_id, experiment_id=exp.experiment_id)

        try:
            self.store.update_experiment(exp.experiment_id, status=ExperimentStatus.RUNNING)

            # Gather failure context
            failures = self.store.list_recent_failures()
            failure_summaries = [
                ", ".join(f.lessons[-2:]) if f.lessons else f.kind
                for f in failures[:5]
            ]

            # Step 1: detect_tool_gap
            gap_result = self.registry.run_tool(
                "detect_tool_gap",
                {
                    "recent_experiment_ids": recent_experiment_ids or [],
                    "repeated_steps": ["factor_evaluation"],
                    "failure_summaries": failure_summaries,
                },
                _ctx(),
            )
            proposals = gap_result.get("tool_gap_proposals", [])
            self.store.add_artifact(exp.experiment_id, "tool_gaps_detected")

            if not proposals:
                self.store.add_lesson(exp.experiment_id, "no tool gaps detected")
                self.store.update_experiment(
                    exp.experiment_id, status=ExperimentStatus.COMPLETED
                )
                return self.store.get_experiment(exp.experiment_id)

            results: list[dict[str, Any]] = []
            for proposal in proposals:
                # Reject dangerous proposals
                name = proposal.get("proposed_tool_name", "").lower()
                if any(kw in name for kw in FORBIDDEN_TOOL_KEYWORDS):
                    self.store.add_lesson(
                        exp.experiment_id,
                        f"rejected forbidden tool proposal: {name}",
                    )
                    results.append({"tool": name, "status": "REJECTED_FORBIDDEN"})
                    continue

                # Step 2: create_tool_spec
                spec_result = self.registry.run_tool(
                    "create_tool_spec",
                    {"tool_gap_proposal": proposal},
                    _ctx(),
                )
                if spec_result.get("status") == "REJECTED":
                    results.append({"tool": name, "status": "REJECTED"})
                    continue

                tool_spec = spec_result.get("tool_spec", {})

                # Step 3: generate_tool_code
                code_result = self.registry.run_tool(
                    "generate_tool_code",
                    {"tool_spec": tool_spec},
                    _ctx(),
                )
                code_path = code_result.get("code_path", "")

                # Step 4: generate_tool_tests
                tests_result = self.registry.run_tool(
                    "generate_tool_tests",
                    {"tool_spec": tool_spec, "code_path": code_path},
                    _ctx(),
                )
                tests_path = tests_result.get("tests_path", "")

                # Step 5: run_tool_sandbox_tests
                sandbox_result = self.registry.run_tool(
                    "run_tool_sandbox_tests",
                    {
                        "code_path": code_path,
                        "tests_path": tests_path,
                    },
                    _ctx(),
                )

                # Step 6: score_tool_candidate
                score_result = self.registry.run_tool(
                    "score_tool_candidate",
                    {
                        "tool_spec": tool_spec,
                        "test_summary": sandbox_result.get("test_summary", {}),
                        "safety_issues": sandbox_result.get("safety_issues", []),
                    },
                    _ctx(),
                )
                recommendation = score_result.get("recommendation", "REJECT")

                if recommendation == "REJECT":
                    self.store.add_lesson(exp.experiment_id, f"tool '{name}' rejected by scorer")
                    results.append(
                        {"tool": name, "status": "REJECTED", "score": score_result.get("score")}
                    )
                    continue

                # Step 7: propose_tool_registration (for human review)
                try:
                    proposal_result = self.registry.run_tool(
                        "propose_tool_registration",
                        {
                            "tool_candidate_id": name,
                            "score": score_result.get("score", {}),
                        },
                        ToolContext(
                            run_id=run_id,
                            experiment_id=exp.experiment_id,
                            requested_by_llm=False,
                            call_mode=ToolCallMode.TRUSTED_INTERNAL_WORKFLOW,
                        ),
                    )
                    proposal_path = proposal_result.get("proposal_path", "")
                    if proposal_path:
                        self.store.add_artifact(exp.experiment_id, proposal_path)
                    results.append(
                        {
                            "tool": name,
                            "status": "REVIEW_REQUIRED",
                            "recommendation": recommendation,
                        }
                    )
                except Exception:
                    results.append({"tool": name, "status": "NEEDS_HUMAN_REVIEW"})

            self.store.update_experiment(
                exp.experiment_id,
                status=ExperimentStatus.REVIEW_REQUIRED,
                metrics={"tool_proposals": results},
            )
            return self.store.get_experiment(exp.experiment_id)

        except Exception as exc:
            self.store.add_lesson(exp.experiment_id, f"self_bootstrap failed: {exc}")
            self.store.update_experiment(
                exp.experiment_id,
                status=ExperimentStatus.FAILED,
                metrics={"error": str(exc)},
            )
            return self.store.get_experiment(exp.experiment_id)
