from qmt_agent_trader.core.config import Settings


def test_config_defaults_are_safe() -> None:
    settings = Settings()
    assert settings.dry_run is True
    assert settings.live_trading_enabled is False
    assert settings.deepseek_model
    assert settings.remote_data_max_concurrency == 200
    assert settings.remote_data_min_interval_seconds == 0.3
    assert settings.tushare_quota_profile_source == "official_table"
    assert settings.tushare_points == 2000
    assert settings.tushare_max_requests_per_minute == 200
    assert settings.tushare_max_requests_per_day_per_api == 100000
    assert settings.remote_data_http_timeout_seconds == 300.0
    assert settings.remote_data_tool_base_timeout_seconds == 120
    assert settings.remote_data_tool_timeout_seconds_per_request == 15
    assert settings.remote_data_tool_max_timeout_seconds == 3600
    assert settings.remote_data_retry_attempts == 3
    assert settings.remote_data_retry_backoff_seconds == 2.0
