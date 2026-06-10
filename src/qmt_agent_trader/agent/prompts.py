"""Prompt snippets."""

FACTOR_DISCOVERY_PROMPT = """
Return only JSON matching this shape:
{
  "hypothesis": {
    "name": "short_snake_case",
    "description": "...",
    "intuition": "...",
    "required_data": ["daily_bars"],
    "lookback": 20,
    "universe": ["A_SHARE_STOCK", "ETF"],
    "expected_behavior": "...",
    "known_risks": ["..."]
  },
  "implementation_plan": {
    "factor_code_allowed": true,
    "strategy_code_allowed": true,
    "live_trading_allowed": false
  }
}
Never propose live trading.
""".strip()

STRATEGY_DISCOVERY_PROMPT = """
Suggest one paper-tradable strategy candidate only and return only JSON matching the
ResearchSpec shape. The candidate must be daily-frequency, avoid future data, and set
implementation_plan.live_trading_allowed to false.
""".strip()
