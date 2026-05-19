"""In-process API state (upload flags, last job metadata)."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_STATE_PATH = Path(__file__).resolve().parents[1] / "datasets" / "mun_data" / ".api_state.json"
_lock = threading.Lock()


@dataclass
class ApiState:
    territories_uploaded: bool = False
    target_column: str | None = None
    target_indicator_ids: list[int] = field(default_factory=list)
    last_upload_job: str | None = None
    last_embedding_job: str | None = None
    same_file_inference: bool = True


def _load() -> ApiState:
    if not _STATE_PATH.is_file():
        return ApiState()
    try:
        raw = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        return ApiState(**{k: raw[k] for k in asdict(ApiState()) if k in raw})
    except Exception:
        return ApiState()


def _save(st: ApiState) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(asdict(st), ensure_ascii=False, indent=2), encoding="utf-8")


def get_state() -> ApiState:
    with _lock:
        return _load()


def update_state(**kwargs: Any) -> ApiState:
    with _lock:
        st = _load()
        for k, v in kwargs.items():
            if hasattr(st, k):
                setattr(st, k, v)
        _save(st)
        return st
