from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import uuid

from .utils import atomic_write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def write_run_manifest(path: str | Path, payload: dict) -> None:
    atomic_write_json(path, {**payload, "updated_at": utc_now()})

