"""Agent Router — natural language intent classification.

Supports rule-based fallback for when the LLM is unavailable.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AgentIntent(StrEnum):
    GENERAL_RESEARCH = "GENERAL_RESEARCH"
    FACTOR_DISCOVERY = "FACTOR_DISCOVERY"
    STRATEGY_ENGINEERING = "STRATEGY_ENGINEERING"
    BACKTEST_ANALYSIS = "BACKTEST_ANALYSIS"
    SELF_BOOTSTRAP = "SELF_BOOTSTRAP"
    TOOL_EXPLORATION = "TOOL_EXPLORATION"
    EXPERIMENT_REVIEW = "EXPERIMENT_REVIEW"
    ARTIFACT_REVIEW = "ARTIFACT_REVIEW"
    UNKNOWN = "UNKNOWN"


class RoutingDecision(BaseModel):
    intent: AgentIntent
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    required_tools: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    proposed_workflow: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    needs_user_clarification: bool = False
    clarification_question: str | None = None


class AgentRouter:
    """Route user messages to agent intents using rules, with LLM enhancement optional."""

    def route(
        self,
        message: str,
        session_context: dict[str, Any] | None = None,
        available_tools: list[str] | None = None,
        recent_experiments: list[dict[str, Any]] | None = None,
    ) -> RoutingDecision:
        """Classify user intent from natural language message."""
        if session_context is None:
            session_context = {}
        if available_tools is None:
            available_tools = []
        if recent_experiments is None:
            recent_experiments = []

        return self._rule_based_route(message)

    def _rule_based_route(self, message: str) -> RoutingDecision:
        msg = message.lower()

        # ── SELF_BOOTSTRAP ──
        bootstrap_keywords = [
            "改进你自己", "创建工具", "缺少工具", "优化流程",
            "自举", "自动提升", "反思失败", "减少重复",
            "缺什么工具", "为自己创建", "升级你自己",
            "提高自己", "扩展能力", "学习新模式",
            "bootstrap", "少了", "还不够", "再加一个工具",
        ]
        if any(kw in msg for kw in bootstrap_keywords):
            return RoutingDecision(
                intent=AgentIntent.SELF_BOOTSTRAP,
                confidence=0.88,
                rationale="用户要求分析系统缺口并为 Agent 创建新工具或能力。",
                required_tools=[
                    "search_experiments", "detect_tool_gap", "create_tool_spec",
                    "generate_tool_code", "generate_tool_tests",
                    "run_tool_sandbox_tests", "score_tool_candidate",
                ],
                proposed_workflow="self_bootstrap",
                parameters={"budget_mode": "balanced"},
            )

        # ── BACKTEST_ANALYSIS ──
        backtest_keywords = [
            "解释回测", "为什么回撤", "收益来源", "绩效归因",
            "风险暴露", "交易成本", "leakage", "未来函数",
            "回测结果怎么", "看回测",
        ]
        if any(kw in msg for kw in backtest_keywords):
            return RoutingDecision(
                intent=AgentIntent.BACKTEST_ANALYSIS,
                confidence=0.85,
                rationale="用户想要理解或分析已有的回测结果。",
                required_tools=[
                    "search_experiments", "list_data_catalog",
                    "generate_research_report",
                ],
                proposed_workflow=None,
                parameters={},
            )

        # ── EXPERIMENT_REVIEW ──
        experiment_keywords = [
            "上一个实验", "最近实验", "失败记录", "历史结果",
            "实验报告", "实验怎么样了", "看看实验",
            "实验列表", "实验结果",
        ]
        if any(kw in msg for kw in experiment_keywords):
            return RoutingDecision(
                intent=AgentIntent.EXPERIMENT_REVIEW,
                confidence=0.87,
                rationale="用户想查看或回顾历史实验。",
                required_tools=["search_experiments"],
                proposed_workflow=None,
                parameters={},
            )

        # ── ARTIFACT_REVIEW ── (check before FACTOR to catch "因子代码"/"策略代码")
        artifact_keywords = [
            "生成的代码", "回测报告", "因子代码", "看代码", "代码在哪",
            "策略代码", "工具代码", "测试代码",
        ]
        if any(kw in msg for kw in artifact_keywords):
            return RoutingDecision(
                intent=AgentIntent.ARTIFACT_REVIEW,
                confidence=0.86,
                rationale="用户想查看已生成的 artifact。",
                required_tools=["search_experiments"],
                proposed_workflow=None,
                parameters={},
            )

        # ── FACTOR_DISCOVERY and STRATEGY_ENGINEERING ──
        factor_keywords = [
            "因子", "alpha", "选股信号", "指标",
            "ic", "rankic", "分组收益",
            "低波动", "动量", "反转", "价值", "质量", "拥挤度",
            "发现", "挖掘",
        ]
        strategy_keywords = [
            "策略", "组合", "调仓", "轮动", "持仓", "仓位",
            "回测", "交易规则", "止损", "风控",
        ]
        has_factor = any(kw in msg for kw in factor_keywords)
        has_strategy = any(kw in msg for kw in strategy_keywords)

        if has_factor and has_strategy:
            return RoutingDecision(
                intent=AgentIntent.STRATEGY_ENGINEERING,
                confidence=0.82,
                rationale="用户提到了因子和策略/组合相关概念，推断为策略编写。",
                required_tools=[
                    "list_data_catalog", "create_strategy_spec",
                    "generate_strategy_code", "run_backtest",
                    "generate_research_report",
                ],
                proposed_workflow="strategy_engineering",
                parameters={"budget_mode": "balanced"},
            )

        if has_factor:
            return RoutingDecision(
                intent=AgentIntent.FACTOR_DISCOVERY,
                confidence=0.88,
                rationale="用户要求发现或分析因子/alpha/选股信号。",
                required_tools=[
                    "list_data_catalog", "create_factor_spec",
                    "generate_factor_code", "run_factor_static_checks",
                    "evaluate_factor_candidate", "generate_research_report",
                ],
                proposed_workflow="factor_discovery",
                parameters={"budget_mode": "balanced"},
            )

        if has_strategy:
            return RoutingDecision(
                intent=AgentIntent.STRATEGY_ENGINEERING,
                confidence=0.83,
                rationale="用户要求编写或分析交易策略。",
                required_tools=[
                    "list_data_catalog", "create_strategy_spec",
                    "generate_strategy_code", "run_backtest",
                    "generate_research_report",
                ],
                proposed_workflow="strategy_engineering",
                parameters={"budget_mode": "balanced"},
            )

        # ── TOOL_EXPLORATION ──
        tool_keywords = ["有什么工具", "工具列表", "能做什么", "功能列表", "help", "帮助"]
        if any(kw in msg for kw in tool_keywords):
            return RoutingDecision(
                intent=AgentIntent.TOOL_EXPLORATION,
                confidence=0.90,
                rationale="用户在询问可用的工具和功能。",
                required_tools=["list_tools"],
                proposed_workflow=None,
                parameters={},
            )

        # ── GENERAL_RESEARCH (default) ──
        return RoutingDecision(
            intent=AgentIntent.GENERAL_RESEARCH,
            confidence=0.70,
            rationale="未匹配到特定意图，将作为通用研究问题处理。",
            required_tools=["list_data_catalog", "search_experiments"],
            proposed_workflow=None,
            parameters={},
        )


# Singleton
agent_router = AgentRouter()
