from qmt_agent_trader.core.config import Settings


def test_config_defaults_are_safe() -> None:
    settings = Settings()
    assert settings.dry_run is True
    assert settings.live_trading_enabled is False
    assert settings.deepseek_model
    assert settings.remote_data_max_concurrency == 200
    assert settings.remote_data_min_interval_seconds == 0.3
