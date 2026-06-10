from qmt_agent_trader.core.config import Settings


def test_config_defaults_are_safe() -> None:
    settings = Settings()
    assert settings.dry_run is True
    assert settings.live_trading_enabled is False
    assert settings.deepseek_model
