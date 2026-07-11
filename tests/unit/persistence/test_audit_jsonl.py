from __future__ import annotations

import json
import multiprocessing
from datetime import UTC, datetime
from pathlib import Path

from qmt_agent_trader.agent.audit import AuditLogger as AgentAuditLogger
from qmt_agent_trader.core.audit import AuditLogger as CoreAuditLogger
from qmt_agent_trader.persistence.atomic_files import AtomicFileStore
from qmt_agent_trader.persistence.audit import AuditJsonlStore, verify_audit_jsonl
from qmt_agent_trader.persistence.locks import LockManager


def _append_rows(path: str, locks: str, worker: int) -> None:
    store = AuditJsonlStore(
        Path(path), AtomicFileStore(LockManager(Path(locks))), schema_version=2
    )
    for index in range(40):
        store.append({"worker": worker, "index": index})


def _append_rotating_rows(path: str, locks: str, worker: int) -> None:
    store = AuditJsonlStore(
        Path(path),
        AtomicFileStore(LockManager(Path(locks))),
        rotation_bytes=180,
    )
    for index in range(20):
        store.append({"worker": worker, "index": index, "padding": "x" * 40})


def test_audit_multi_process_append_is_complete(tmp_path: Path) -> None:
    path = tmp_path / "audit/events.jsonl"
    processes = [
        multiprocessing.Process(
            target=_append_rows, args=(str(path), str(tmp_path / "locks"), worker)
        )
        for worker in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 160
    assert all(row["schema_version"] == 2 for row in rows)


def test_verifier_distinguishes_half_tail_from_mid_file_corruption(tmp_path: Path) -> None:
    half_tail = tmp_path / "half.jsonl"
    half_tail.write_bytes(b'{"legacy":1}\n{"broken"')
    result = verify_audit_jsonl(half_tail)
    assert result.valid_records == 1
    assert result.tail_truncated is True
    assert result.corruptions == []

    middle = tmp_path / "middle.jsonl"
    middle.write_bytes(b'{"legacy":1}\nnot-json\n{"schema_version":2,"ok":true}\n')
    result = verify_audit_jsonl(middle)
    assert result.tail_truncated is False
    assert result.valid_records == 2
    assert result.corruptions[0].line_number == 2


def test_repeated_rotation_preserves_every_event_in_order(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    audit = AuditJsonlStore(
        path,
        AtomicFileStore(LockManager(tmp_path / "locks")),
        rotation_bytes=90,
    )
    for index in range(6):
        audit.append({"event": index, "padding": "x" * 40})

    events = [row["event"] for row in audit.read_records()]
    assert events == list(range(6))
    assert len(list(tmp_path.glob("audit.jsonl.*"))) >= 3
    assert audit.verify().valid_records == 6


def test_multiprocess_rotation_preserves_all_boundary_events(tmp_path: Path) -> None:
    path = tmp_path / "audit/events.jsonl"
    locks = tmp_path / "locks"
    processes = [
        multiprocessing.Process(
            target=_append_rotating_rows, args=(str(path), str(locks), worker)
        )
        for worker in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    audit = AuditJsonlStore(path, AtomicFileStore(LockManager(locks)), rotation_bytes=180)
    rows = audit.read_records()
    assert len(rows) == 80
    assert {(row["worker"], row["index"]) for row in rows} == {
        (worker, index) for worker in range(4) for index in range(20)
    }
    assert audit.verify().valid_records == 80


def test_store_schema_version_cannot_be_overridden_by_record(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    store = AuditJsonlStore(path, AtomicFileStore(LockManager(tmp_path / "locks")))

    store.append({"schema_version": 1, "event": "current"})

    assert json.loads(path.read_text())["schema_version"] == 2


def test_fsync_policy_is_forwarded(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr("qmt_agent_trader.persistence.atomic_files.os.fsync", calls.append)
    store = AuditJsonlStore(
        tmp_path / "audit.jsonl",
        AtomicFileStore(LockManager(tmp_path / "locks")),
        fsync=False,
    )
    store.append({"event": "no-sync"})
    assert calls == []


def test_public_loggers_preserve_contract_scrub_secrets_and_ignore_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    locks = LockManager(tmp_path / "locks")
    atomic = AtomicFileStore(locks)
    agent_path = tmp_path / "canonical/agent.jsonl"
    agent = AgentAuditLogger(agent_path, atomic_store=atomic)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    agent.append(
        "tool",
        "run",
        error_message="token sk-secret leaked",
        output_data={
            "nested": {"api_key": "sk-secret"},
            "safe": "visible",
            "token_count": 42,
            "token_budget": 100,
            "description": "token usage is within budget",
            "credentials": {
                "password": "hunter2",
                "authorization": "Bearer abc.def.ghi",
                "qmt_gateway_hmac_secret": "gateway-secret",
                "hmac_secret": "hmac-value",
                "api_secret": "api-value",
                "access_key": "access-value",
                "broker-api-secret": "broker-value",
                "providerAccessKey": "provider-value",
                "providerAPISecret": "provider-api-value",
                "qmtGatewayHMACSecret": "gateway-hmac-value",
            },
            "benign_prose": "token usage and hmac checks are healthy; secret strategy note",
            "assignment": "hmac_secret=gateway-value",
            "prefixed_assignments": [
                "qmt_gateway_hmac_secret=one",
                "provider_hmac_secret=two",
                "QMTGatewayHMACSecret=three",
                "providerAPISecret=four",
            ],
            "benign_assignments": [
                "token_count=42",
                "token_budget=100",
                "status=healthy",
            ],
            "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
        },
        warnings=["Bearer abc.def.ghi"],
    )
    row = json.loads(agent_path.read_text())
    assert row["schema_version"] == 2
    assert row["error_message"] == "[scrubbed]"
    assert row["output_data"] == {
        "nested": {"api_key": "[scrubbed]"},
        "safe": "visible",
        "token_count": 42,
        "token_budget": 100,
        "description": "token usage is within budget",
        "credentials": {
            "password": "[scrubbed]",
            "authorization": "[scrubbed]",
            "qmt_gateway_hmac_secret": "[scrubbed]",
            "hmac_secret": "[scrubbed]",
            "api_secret": "[scrubbed]",
            "access_key": "[scrubbed]",
            "broker-api-secret": "[scrubbed]",
            "providerAccessKey": "[scrubbed]",
            "providerAPISecret": "[scrubbed]",
            "qmtGatewayHMACSecret": "[scrubbed]",
        },
        "benign_prose": "token usage and hmac checks are healthy; secret strategy note",
        "assignment": "[scrubbed]",
        "prefixed_assignments": ["[scrubbed]"] * 4,
        "benign_assignments": [
            "token_count=42",
            "token_budget=100",
            "status=healthy",
        ],
        "timestamp": "2026-01-01 00:00:00+00:00",
    }
    assert row["warnings"] == ["[scrubbed]"]

    core_path = tmp_path / "canonical/trade.jsonl"
    core = CoreAuditLogger(core_path, atomic_store=atomic)
    event = core.append("risk", "tester", {"ok": True})
    assert event.event_type == "risk"
    assert core.read_latest()[0]["payload"] == {"ok": True}
