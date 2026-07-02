"""Reserved QMT data provider placeholder."""

from __future__ import annotations

from qmt_agent_trader.data.providers.base import (
    FetchItem,
    FetchPlan,
    FetchResult,
    ProviderCapability,
)


class QMTProvider:
    source_name = "qmt"

    def list_capabilities(
        self,
        *,
        category: str | None = None,
        asset_type: str | None = None,
    ) -> ProviderCapability:
        _ = category, asset_type
        return ProviderCapability(
            source=self.source_name,
            endpoints=[
                {
                    "source": "qmt",
                    "implemented": False,
                    "status": "NOT_IMPLEMENTED",
                    "message": "QMT data provider is reserved but not implemented yet.",
                }
            ],
        )

    def plan_fetch(
        self,
        items: list[FetchItem],
        *,
        requested_by_llm: bool = False,
        storage_mode: str = "persistent",
    ) -> FetchPlan:
        _ = items, requested_by_llm, storage_mode
        return FetchPlan(
            status="NOT_IMPLEMENTED",
            source=self.source_name,
            reason="qmt_provider_reserved",
            message="QMT data provider is reserved but not implemented yet.",
        )

    def run_fetch(
        self,
        plan: FetchPlan,
        *,
        execute_plan: bool = False,
        dry_run: bool = False,
    ) -> FetchResult:
        _ = plan, execute_plan, dry_run
        return FetchResult(
            status="NOT_IMPLEMENTED",
            source=self.source_name,
            metadata={"message": "QMT data provider is reserved but not implemented yet."},
        )
