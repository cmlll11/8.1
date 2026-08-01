from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import numpy
        record["numpy"] = numpy.__version__
    except Exception as exc:
        record["numpy_error"] = repr(exc)
    try:
        import torch
        record.update(
            torch=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_version=torch.version.cuda,
            gpu_count=torch.cuda.device_count(),
            gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        )
    except Exception as exc:
        record["torch_error"] = repr(exc)
    return record
