"""Experiment store — versioned one-JSON-record-per-file experiment memory.

Records live under the injected experiment root and share canonical lock and
quarantine infrastructure with the other mutable-state repositories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qmt_agent_trader.agent.errors import ExperimentNotFoundError
from qmt_agent_trader.agent.schemas import ExperimentRecord, ExperimentStatus
from qmt_agent_trader.core.ids import new_id
from qmt_agent_trader.persistence.repositories.versioned_record import (
    RecordDiagnostic,
    VersionedRecordRepository,
)


class ExperimentStore:
    """Create, update, search, and recall Agent experiments."""

    def __init__(
        self, root: Path, *, locks_root: Path | None = None, quarantine_root: Path | None = None
    ) -> None:
        self.root = root.expanduser().resolve()
        self.repository = VersionedRecordRepository(
            self.root,
            ExperimentRecord,
            store_name="experiments",
            locks_root=locks_root,
            quarantine_root=quarantine_root,
            identity=lambda record: record.experiment_id,
        )
        self.last_diagnostics: list[RecordDiagnostic] = []

    # ── CRUD ──────────────────────────────────────────────────────────────

    def create_experiment(
        self,
        kind: str,
        *,
        experiment_id: str | None = None,
        hypothesis: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> ExperimentRecord:
        experiment_id = experiment_id or new_id("exp")
        record = ExperimentRecord(
            experiment_id=experiment_id,
            kind=kind,
            hypothesis=hypothesis,
            tags=tags or [],
        )
        return self.repository.create(experiment_id, record)

    def update_experiment(self, experiment_id: str, **updates: Any) -> ExperimentRecord:
        expected_revision = updates.pop("expected_revision", None)

        def apply(record: ExperimentRecord) -> ExperimentRecord:
            data = record.model_dump(mode="json")
            data.update(updates)
            return ExperimentRecord.model_validate(data)

        return self.repository.mutate(experiment_id, apply, expected_revision=expected_revision)

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        try:
            return self.repository.load(experiment_id)
        except FileNotFoundError:
            raise ExperimentNotFoundError(f"experiment '{experiment_id}' not found") from None

    def add_lesson(self, experiment_id: str, lesson: str) -> None:
        self.repository.mutate(
            experiment_id,
            lambda record: record.model_copy(update={"lessons": [*record.lessons, lesson]}),
        )

    def add_artifact(self, experiment_id: str, artifact: str) -> None:
        self.repository.mutate(
            experiment_id,
            lambda record: record.model_copy(update={"artifacts": [*record.artifacts, artifact]}),
        )

    # ── Search ────────────────────────────────────────────────────────────

    def search_experiments(
        self,
        *,
        query: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[ExperimentRecord]:
        results: list[ExperimentRecord] = []
        records, self.last_diagnostics = self.repository.list_with_diagnostics()
        records.sort(key=lambda record: record.updated_at, reverse=True)
        for record in records:
            if len(results) >= limit:
                break
            if not self._matches_filter(record, query=query, tags=tags):
                continue
            results.append(record)
        return results

    def list_recent_failures(self, limit: int = 10) -> list[ExperimentRecord]:
        return [
            record
            for record in self.search_experiments(limit=limit * 3)
            if record.status == ExperimentStatus.FAILED
        ][:limit]

    # ── Internals ─────────────────────────────────────────────────────────

    def _path(self, experiment_id: str) -> Path:
        return self.repository.path_for(experiment_id)

    @staticmethod
    def _matches_filter(
        record: ExperimentRecord,
        *,
        query: str | None,
        tags: list[str] | None,
    ) -> bool:
        if tags:
            record_tags = set(record.tags)
            if not record_tags.issuperset(tags):
                return False
        if query:
            text = json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                default=str,
            ).lower()
            if query.lower() not in text:
                return False
        return True
