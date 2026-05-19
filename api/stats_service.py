"""Read-only stats for GET /stats/* endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from api.db_ops import fetch_indicators, table_row_count
from embedding_qdrant import make_qdrant_client
from config import QDRANT_COLLECTION
from utils import PathBuilder


def _load_best_summary() -> dict[str, Any] | None:
    p = Path("logs") / "best_embedding_summary.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


async def get_model_info() -> dict[str, Any]:
    summary = _load_best_summary()
    if not summary:
        return {"model": None, "embedding_dim": None}
    return {
        "model": summary.get("best_model"),
        "embedding_dim": summary.get("emb_dim"),
    }


async def get_dataset_sizes(engine: AsyncEngine) -> dict[str, int]:
    train_n = await table_row_count(engine, "indicator_values_train")
    inf_n = await table_row_count(engine, "indicator_values_inference")
    if train_n == 0 and inf_n == 0:
        inf_n = await table_row_count(engine, "indicator_values")
    paths = PathBuilder.build_emb_datasets_paths()
    train_csv = paths["train_path"]
    inf_csv = paths["inference_full_path"]
    train_csv_n = 0
    inf_csv_n = 0
    if Path(train_csv).is_file():
        train_csv_n = max(0, len(pd.read_csv(train_csv, sep=";")) - 0)
    if Path(inf_csv).is_file():
        inf_csv_n = max(0, len(pd.read_csv(inf_csv, sep=";")))
    return {
        "train_rows_db": train_n,
        "inference_rows_db": inf_n,
        "train_rows_local": train_csv_n,
        "inference_rows_local": inf_csv_n,
    }


async def get_indicators_list(engine: AsyncEngine) -> list[dict[str, Any]]:
    return await fetch_indicators(engine)


async def get_embedding_stats(engine: AsyncEngine) -> dict[str, Any]:
    model_info = await get_model_info()
    sizes = await get_dataset_sizes(engine)
    indicators = await get_indicators_list(engine)

    qdrant_points = None
    try:
        client = make_qdrant_client()
        info = client.get_collection(QDRANT_COLLECTION)
        qdrant_points = info.points_count
    except Exception:
        pass

    return {
        **model_info,
        "train_size": sizes["train_rows_db"] or sizes["train_rows_local"],
        "inference_size": sizes["inference_rows_db"] or sizes["inference_rows_local"],
        "indicators": indicators,
        "qdrant_points": qdrant_points,
    }
