"""Experiment store — local JSONL-backed experiment memory.

First version uses JSONL under `data/experiments/`. The directory is within the
project data root so it stays alongside the data lake.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qmt_agent_trader.agent.errors import ExperimentNotFoundError
from qmt_agent_trader.agent.schemas import ExperimentRecord, ExperimentStatus
from qmt_agent_trader.core.ids import new_id, shanghai_now_iso


class ExperimentStore:
    """Create, update, search, and recall Agent experiments."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

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
        self._write(record)
        return record

    def update_experiment(
        self, experiment_id: str, **updates: Any
    ) -> ExperimentRecord:
        record = self.get_experiment(experiment_id)
        data = record.model_dump(mode="json")
        data.update(updates)
        data["updated_at"] = datetime.now(tz=UTC)
        updated = ExperimentRecord.model_validate(data)
        self._write(updated)
        return updated

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        path = self._path(experiment_id)
        if not path.exists():
            raise ExperimentNotFoundError(
                f"experiment '{experiment_id}' not found"
            )
        return ExperimentRecord.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )

    def add_lesson(self, experiment_id: str, lesson: str) -> None:
        record = self.get_experiment(experiment_id)
        lessons = list(record.lessons)
        lessons.append(lesson)
        self.update_experiment(experiment_id, lessons=lessons)

    def add_artifact(self, experiment_id: str, artifact: str) -> None:
        record = self.get_experiment(experiment_id)
        artifacts = list(record.artifacts)
        artifacts.append(artifact)
        self.update_experiment(experiment_id, artifacts=artifacts)

    # ── Search ────────────────────────────────────────────────────────────

    def search_experiments(
        self,
        *,
        query: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[ExperimentRecord]:
        results: list[ExperimentRecord] = []
        for path in sorted(
            self.root.glob("exp_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            if len(results) >= limit:
                break
            try:
                record = ExperimentRecord.model_validate(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except Exception:
                continue
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
        return self.root / f"{experiment_id}.json"

    def _write(self, record: ExperimentRecord) -> None:
        data = record.model_dump(mode="json")
        data["updated_at"] = shanghai_now_iso()
        self._path(record.experiment_id).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

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
