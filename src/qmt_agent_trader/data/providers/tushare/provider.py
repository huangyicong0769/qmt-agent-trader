"""Tushare provider facade."""

from __future__ import annotations

from qmt_agent_trader.data.providers.base import (
    FetchItem,
    FetchPlan,
    FetchResult,
    ProviderCapability,
)
from qmt_agent_trader.data.providers.tushare.fetcher import TushareFetcher
from qmt_agent_trader.data.providers.tushare.planner import (
    TushareFetchPlanner,
    TusharePlannerConfig,
)
from qmt_agent_trader.data.providers.tushare.quota import TushareUsageLedger
from qmt_agent_trader.data.providers.tushare.registry import (
    TushareEndpointRegistry,
    default_tushare_registry,
)


class TushareProvider:
    source_name = "tushare"

    def __init__(
        self,
        *,
        registry: TushareEndpointRegistry | None = None,
        fetcher: TushareFetcher | None = None,
        planner_config: TusharePlannerConfig | None = None,
        usage_ledger: TushareUsageLedger | None = None,
    ) -> None:
        self.registry = registry or default_tushare_registry()
        self.fetcher = fetcher
        self.planner = TushareFetchPlanner(
            self.registry,
            config=planner_config,
            usage_ledger=usage_ledger,
        )

    def list_capabilities(
        self,
        *,
        category: str | None = None,
        asset_type: str | None = None,
    ) -> ProviderCapability:
        return ProviderCapability(
            source=self.source_name,
            endpoints=self.registry.as_capabilities(category=category, asset_type=asset_type),
        )

    def plan_fetch(
        self,
        items: list[FetchItem],
        *,
        requested_by_llm: bool = False,
        storage_mode: str = "persistent",
    ) -> FetchPlan:
        return self.planner.plan(
            items,
            requested_by_llm=requested_by_llm,
            storage_mode=storage_mode,
        )

    def run_fetch(
        self,
        plan: FetchPlan,
        *,
        execute_plan: bool = False,
        dry_run: bool = False,
    ) -> FetchResult:
        if self.fetcher is None:
            return FetchResult(
                status="NOT_AVAILABLE",
                source=self.source_name,
                metadata={"message": "Tushare fetcher is not wired"},
                execution_status="OK",
                domain_status="NOT_CONFIGURED",
                evidence_status="BLOCKED",
                recommendation_status="BLOCKED",
                coverage_status="BLOCKED",
                blockers=["Tushare fetcher is not wired"],
            )
        return self.fetcher.run(plan, execute_plan=execute_plan, dry_run=dry_run)
