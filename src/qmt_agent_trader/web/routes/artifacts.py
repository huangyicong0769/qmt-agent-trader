"""Artifact browsing API routes."""

from __future__ import annotations

import base64
import binascii
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from qmt_agent_trader.web.config import WebConfig, get_web_config
from qmt_agent_trader.web.schemas import ArtifactDetail, ArtifactSummary

router = APIRouter()

MAX_TEXT_BYTES = 1_000_000


@router.get("/", response_model=list[ArtifactSummary])
async def list_artifacts() -> list[ArtifactSummary]:
    config = get_web_config()
    artifacts: list[ArtifactSummary] = []
    for root in config.artifact_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and config.is_path_safe(path):
                artifacts.append(_summary(path))
    return sorted(artifacts, key=lambda artifact: artifact.modified_at, reverse=True)


@router.get("/{artifact_id}/content", response_model=ArtifactDetail)
async def get_artifact_content(artifact_id: str) -> ArtifactDetail:
    path = _artifact_path_or_404(artifact_id, get_web_config())
    if path.stat().st_size > MAX_TEXT_BYTES:
        raise HTTPException(status_code=413, detail="artifact is too large for inline display")
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=415, detail="artifact is not UTF-8 text") from exc
    return ArtifactDetail(artifact=_summary(path), content=content)


@router.get("/{artifact_id}/download")
async def download_artifact(artifact_id: str) -> FileResponse:
    path = _artifact_path_or_404(artifact_id, get_web_config())
    return FileResponse(path, filename=path.name)


def encode_artifact_id(path: Path) -> str:
    raw = str(path.resolve()).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_artifact_id(artifact_id: str) -> Path:
    padding = "=" * (-len(artifact_id) % 4)
    try:
        decoded = base64.urlsafe_b64decode(f"{artifact_id}{padding}".encode("ascii"))
    except (ValueError, UnicodeDecodeError, binascii.Error) as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    return Path(decoded.decode("utf-8"))


def _artifact_path_or_404(artifact_id: str, config: WebConfig) -> Path:
    path = decode_artifact_id(artifact_id).resolve()
    if not path.exists() or not path.is_file() or not config.is_path_safe(path):
        raise HTTPException(status_code=404, detail="artifact not found")
    return path


def _summary(path: Path) -> ArtifactSummary:
    stat = path.stat()
    return ArtifactSummary(
        artifact_id=encode_artifact_id(path),
        name=path.name,
        path=str(path.resolve()),
        size_bytes=stat.st_size,
        modified_at=datetime_from_timestamp(stat.st_mtime),
    )


def datetime_from_timestamp(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp).astimezone()
