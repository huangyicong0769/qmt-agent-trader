"""Tests for workflow API routes."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from qmt_agent_trader.agent.experiment_store import ExperimentStore
from qmt_agent_trader.agent.schemas import ExperimentRecord, ExperimentStatus
from qmt_agent_trader.web.routes import workflows


class _FakeFactorWorkflow:
    def __init__(self, _registry: object, _store: ExperimentStore) -> None:
        pass

    def run(
        self,
        theme: str,
        universe: str,
        start_date: str,
        end_date: str,
    ) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id="exp_factor",
            kind="factor_discovery",
            status=ExperimentStatus.REVIEW_REQUIRED,
            hypothesis={
                "theme": theme,
                "universe": universe,
                "start": start_date,
                "end": end_date,
            },
        )


def test_workflow_api_creates_run_and_returns_status(tmp_path, monkeypatch) -> None:
    workflows._workflow_runs.clear()
    monkeypatch.setattr(workflows, "get_registry", lambda: object())
    monkeypatch.setattr(workflows, "get_experiment_store", lambda: ExperimentStore(tmp_path))
    monkeypatch.setattr(workflows, "FactorDiscoveryWorkflow", _FakeFactorWorkflow)
    app = FastAPI()
    app.include_router(workflows.router)
    client = TestClient(app)

    response = client.post("/factor-discovery", json={"theme": "momentum"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "REVIEW_REQUIRED"
    run_response = client.get(f"/runs/{payload['run_id']}")
    assert run_response.status_code == 200
    assert run_response.json()["experiment_id"] == "exp_factor"
