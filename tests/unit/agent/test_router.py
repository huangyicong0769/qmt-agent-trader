"""Tests for Agent Router — natural language intent classification."""

from __future__ import annotations

import pytest

from qmt_agent_trader.agent.router import AgentIntent, AgentRouter


@pytest.fixture
def router() -> AgentRouter:
    return AgentRouter()


def test_factor_discovery_intent(router: AgentRouter) -> None:
    decision = router.route(
        "帮我发现几个适合A股个股和ETF的低波动高胜率因子，并自动跑初步验证。"
    )
    assert decision.intent == AgentIntent.FACTOR_DISCOVERY
    assert decision.confidence > 0.8
    assert decision.proposed_workflow == "factor_discovery"


def test_strategy_engineering_intent(router: AgentRouter) -> None:
    decision = router.route("写一个ETF轮动策略并回测")
    assert decision.intent == AgentIntent.STRATEGY_ENGINEERING
    assert decision.confidence > 0.8


def test_self_bootstrap_intent(router: AgentRouter) -> None:
    decision = router.route("分析最近失败实验，看你缺什么工具")
    assert decision.intent == AgentIntent.SELF_BOOTSTRAP
    assert decision.confidence > 0.8
    assert decision.proposed_workflow == "self_bootstrap"


def test_backtest_analysis_intent(router: AgentRouter) -> None:
    decision = router.route("解释上一次回测为什么回撤大")
    assert decision.intent == AgentIntent.BACKTEST_ANALYSIS
    assert decision.confidence > 0.8


def test_experiment_review_intent(router: AgentRouter) -> None:
    decision = router.route("最近实验结果怎么样")
    assert decision.intent == AgentIntent.EXPERIMENT_REVIEW
    assert decision.confidence > 0.8


def test_general_research_fallback(router: AgentRouter) -> None:
    decision = router.route("你好，今天天气怎么样")
    assert decision.intent in (AgentIntent.GENERAL_RESEARCH, AgentIntent.UNKNOWN)


def test_routing_decision_has_required_fields(router: AgentRouter) -> None:
    decision = router.route("发现动量因子")
    assert decision.intent is not None
    assert decision.confidence is not None
    assert isinstance(decision.rationale, str)
    assert isinstance(decision.required_tools, list)
    assert isinstance(decision.needs_user_clarification, bool)


def test_routing_confidence_in_range(router: AgentRouter) -> None:
    messages = [
        "因子发现",
        "写一个策略",
        "看看实验",
        "分析回测",
        "今天天气如何",
    ]
    for msg in messages:
        decision = router.route(msg)
        assert 0.0 <= decision.confidence <= 1.0


def test_factor_plus_strategy_keywords(router: AgentRouter) -> None:
    """When both factor and strategy keywords are present, router picks strategy."""
    decision = router.route("发现动量因子并做成轮动策略回测")
    assert decision.intent == AgentIntent.STRATEGY_ENGINEERING


def test_self_bootstrap_keywords(router: AgentRouter) -> None:
    decision = router.route("看看少了什么工具，为你自己创建")
    assert decision.intent == AgentIntent.SELF_BOOTSTRAP


def test_tool_exploration_intent(router: AgentRouter) -> None:
    decision = router.route("你有什么工具可以用")
    assert decision.intent == AgentIntent.TOOL_EXPLORATION


def test_artifact_review_intent(router: AgentRouter) -> None:
    decision = router.route("看看生成的因子代码")
    assert decision.intent == AgentIntent.ARTIFACT_REVIEW


def test_low_volatility_factor_chinese(router: AgentRouter) -> None:
    decision = router.route("低波动因子")
    assert decision.intent == AgentIntent.FACTOR_DISCOVERY


def test_momentum_strategy_chinese(router: AgentRouter) -> None:
    decision = router.route("动量策略回测")
    assert decision.intent == AgentIntent.STRATEGY_ENGINEERING
