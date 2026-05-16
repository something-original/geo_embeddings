"""Spawn CPU-heavy API work in a separate process."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable

from api.state import update_state


def spawn_background(
    target: Callable[..., None],
    *args: Any,
    job_key: str | None = None,
    **kwargs: Any,
) -> None:
  ctx = mp.get_context("spawn")
  proc = ctx.Process(target=target, args=args, kwargs=kwargs, daemon=True)
  proc.start()
  if job_key:
    update_state(**{job_key: "running"})
