from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from qmt_agent_trader.data.providers.tushare.quota import (
    DEFAULT_TUSHARE_2000_POINT_PROFILE,
    TushareAccountQuotaProfile,
    TushareFetchCostEstimate,
    TushareQuotaManager,
    TushareUsageLedger,
    new_usage_record,
    normalized_request_hash,
)


def test_default_tushare_quota_profile_is_2000_point_account() -> None:
    profile = DEFAULT_TUSHARE_2000_POINT_PROFILE

    assert profile.source == "official_table"
    assert profile.points == 2000
    assert profile.tier == "2000+"
    assert profile.max_requests_per_minute == 200
    assert profile.max_requests_per_day_per_api == 100000
    assert profile.minute_quota_scope == "account_global"
    assert profile.daily_quota_scope == "per_api"


def test_quota_manager_approves_116_requests_with_zero_usage() -> None:
    manager = TushareQuotaManager(profile=DEFAULT_TUSHARE_2000_POINT_PROFILE)
    state = manager.current_state()
    cost = _cost("daily_basic", 116)

    decision = manager.evaluate(cost, state, execution_mode="autonomous")

    assert decision.status == "APPROVED_BY_ACCOUNT_QUOTA"
    assert decision.safe_to_execute_now is True


def test_quota_manager_recommends_rate_pacing_when_minute_remainder_is_low(
    tmp_path,
) -> None:
    ledger = TushareUsageLedger.from_lake_root(tmp_path / "lake")
    _write_usage_rows(ledger, api_name="daily_basic", count=190)
    manager = TushareQuotaManager(
        profile=DEFAULT_TUSHARE_2000_POINT_PROFILE,
        ledger=ledger,
    )

    decision = manager.evaluate(
        _cost("fina_indicator", 49),
        manager.current_state(),
        execution_mode="autonomous",
    )

    assert decision.status == "APPROVED_WITH_RATE_PACING"
    assert decision.safe_to_execute_now is True
    assert decision.recommended_batches[0]["request_count"] == 10


def test_quota_manager_blocks_daily_quota_per_api_without_cross_api_bleed(
    tmp_path,
) -> None:
    ledger = TushareUsageLedger.from_lake_root(tmp_path / "lake")
    _write_usage_rows(ledger, api_name="daily_basic", count=99990)
    manager = TushareQuotaManager(
        profile=DEFAULT_TUSHARE_2000_POINT_PROFILE,
        ledger=ledger,
    )
    state = manager.current_state()

    blocked = manager.evaluate(
        _cost("daily_basic", 116),
        state,
        execution_mode="autonomous",
    )
    allowed_other_api = manager.evaluate(
        _cost("fina_indicator", 49),
        state,
        execution_mode="autonomous",
    )

    assert blocked.status == "BLOCKED_BY_DAILY_QUOTA"
    assert allowed_other_api.status in {
        "APPROVED_BY_ACCOUNT_QUOTA",
        "APPROVED_WITH_RATE_PACING",
    }


def test_unknown_quota_requires_approval() -> None:
    manager = TushareQuotaManager(
        profile=TushareAccountQuotaProfile(source="unknown", confidence="LOW")
    )

    decision = manager.evaluate(
        _cost("daily_basic", 1),
        manager.current_state(),
        execution_mode="autonomous",
    )

    assert decision.status == "UNKNOWN_QUOTA_REQUIRES_APPROVAL"
    assert decision.safe_to_execute_now is False


def test_usage_ledger_records_success_failure_rate_limit_and_ignores_dry_run(
    tmp_path,
) -> None:
    ledger = TushareUsageLedger.from_lake_root(tmp_path / "lake")
    params = {"trade_date": "20240102"}
    fields = ["ts_code", "trade_date"]

    ledger.append(
        new_usage_record(
            api_name="daily_basic",
            params=params,
            fields=fields,
            status="SUCCESS",
            execution_mode="manual",
            row_count=1,
            finished_at=datetime.now(tz=UTC),
        )
    )
    ledger.append(
        new_usage_record(
            api_name="daily_basic",
            params={"trade_date": "20240103"},
            fields=fields,
            status="FAILED",
            execution_mode="manual",
            error_type="RuntimeError",
            error_message="boom",
            finished_at=datetime.now(tz=UTC),
        )
    )
    ledger.append(
        new_usage_record(
            api_name="daily_basic",
            params={"trade_date": "20240104"},
            fields=fields,
            status="RATE_LIMITED",
            execution_mode="manual",
            error_type="RateLimit",
            error_message="rate limit",
            finished_at=datetime.now(tz=UTC),
        )
    )
    ledger.append(
        new_usage_record(
            api_name="daily_basic",
            params={"trade_date": "20240105"},
            fields=fields,
            status="PLANNED",
            execution_mode="dry_run",
        )
    )

    request_hash = normalized_request_hash(
        api_name="daily_basic",
        params=params,
        fields=list(reversed(fields)),
    )
    assert ledger.usage_last_minute() == 3
    assert ledger.usage_today("daily_basic") == 3
    assert len(ledger.recent_rate_limit_errors()) == 1
    assert ledger.request_seen("daily_basic", request_hash) is True


def test_equivalent_request_hash_is_stable() -> None:
    left = normalized_request_hash(
        api_name="daily_basic",
        params={"b": 2, "a": 1},
        fields=["trade_date", "ts_code"],
    )
    right = normalized_request_hash(
        api_name="daily_basic",
        params={"a": 1, "b": 2},
        fields=["ts_code", "trade_date"],
    )

    assert left == right


def _cost(api_name: str, count: int) -> TushareFetchCostEstimate:
    manager = TushareQuotaManager(profile=DEFAULT_TUSHARE_2000_POINT_PROFILE)
    return manager.estimate_cost(
        [
            {
                "api_name": api_name,
                "batches": [
                    {
                        "api_name": api_name,
                        "params": {"request": index},
                        "fields": ["ts_code"],
                    }
                    for index in range(count)
                ],
            }
        ]
    )


def _write_usage_rows(
    ledger: TushareUsageLedger,
    *,
    api_name: str,
    count: int,
) -> None:
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=UTC).replace(tzinfo=None)
    old = now - timedelta(hours=2)
    pd.DataFrame(
        [
            {
                "request_id": f"{api_name}-{index}",
                "run_id": "quota-test",
                "api_name": api_name,
                "params_hash": f"hash-{index}",
                "params_redacted": "{}",
                "fields": '["ts_code"]',
                "planned_at": old,
                "started_at": now,
                "finished_at": now,
                "status": "SUCCESS",
                "row_count": 1,
                "error_type": None,
                "error_message": None,
                "token_hash": None,
                "execution_mode": "manual",
            }
            for index in range(count)
        ]
    ).to_parquet(ledger.path, index=False)
