from qmt_agent_trader.core.config import Settings


def test_config_defaults_are_safe() -> None:
    settings = Settings()
    assert settings.dry_run is True
    assert settings.live_trading_enabled is False
    assert settings.deepseek_model
    assert settings.remote_data_max_concurrency == 200
    assert settings.remote_data_min_interval_seconds == 0.3
    assert settings.remote_data_http_timeout_seconds == 300.0
    assert settings.remote_data_retry_attempts == 3
    assert settings.remote_data_retry_backoff_seconds == 2.0
