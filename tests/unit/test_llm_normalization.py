from qmt_agent_trader.agent.workflows.normalization import normalize_research_spec


def test_normalize_research_spec_forces_paper_only() -> None:
    spec, schema_valid = normalize_research_spec(
        {"id": "Candidate 001", "summary": "Rank ETFs by trend.", "risks": ["crowding"]},
        fallback_name="fallback",
        fallback_description="fallback description",
        universe=["ETF"],
    )

    assert schema_valid is False
    assert spec.hypothesis.name == "candidate_001"
    assert spec.implementation_plan.live_trading_allowed is False
