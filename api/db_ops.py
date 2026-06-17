"""Async PostgreSQL helpers for API dataset uploads."""

from __future__ import annotations

from typing import Any

import polars as pl
from sqlalchemy import Integer, String, column, insert, table, text
from sqlalchemy.ext.asyncio import AsyncEngine

from parsers.osm_municipal.db import (
    _clean_value,
    _infer_column_sql_types,
    _qi,
    _relation_exists,
    _sqlalchemy_type,
    sync_municipalities_geometry_postgis,
)


def _pl_schema_to_sql_types(schema: pl.Schema, stem: str) -> dict[str, str]:
    return _infer_column_sql_types(tuple(schema.names()), stem, schema)


async def _table_columns(conn, table_name: str) -> set[str]:
    rows = (
        await conn.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = :t
                """
            ),
            {"t": table_name},
        )
    ).fetchall()
    return {r[0] for r in rows}


async def sync_table_columns(
    engine: AsyncEngine,
    table_name: str,
    desired_columns: list[str],
    col_types: dict[str, str],
    *,
    protected: frozenset[str] = frozenset(),
) -> None:
    """Drop columns not in desired_columns; add missing ones. Never drops protected."""
    async with engine.begin() as conn:
        if not await _relation_exists(conn, table_name):
            cols_sql_parts: list[str] = []
            for col in desired_columns:
                typ = col_types[col]
                if col == "id":
                    cols_sql_parts.append(f'{_qi(col)} {typ} PRIMARY KEY')
                else:
                    cols_sql_parts.append(f"{_qi(col)} {typ}")
            await conn.execute(
                text(f'CREATE TABLE {_qi(table_name)} ({", ".join(cols_sql_parts)})')
            )
            return

        existing = await _table_columns(conn, table_name)
        desired_set = set(desired_columns)
        to_drop = existing - desired_set - protected
        for col in sorted(to_drop):
            await conn.execute(
                text(f'ALTER TABLE {_qi(table_name)} DROP COLUMN IF EXISTS {_qi(col)}')
            )
        for col in desired_columns:
            if col not in existing:
                await conn.execute(
                    text(
                        f'ALTER TABLE {_qi(table_name)} '
                        f"ADD COLUMN IF NOT EXISTS {_qi(col)} {col_types[col]}"
                    )
                )


async def replace_table_rows(
    engine: AsyncEngine,
    table_name: str,
    df: pl.DataFrame,
    col_types: dict[str, str],
) -> None:
    columns = [c for c in df.columns if c in col_types]
    if not columns:
        return

    sub = df.select(columns)
    exprs = []
    for c in columns:
        if sub.schema[c] in (pl.Float32, pl.Float64):
            exprs.append(pl.col(c).fill_nan(None).alias(c))
        else:
            exprs.append(pl.col(c))
    sub = sub.select(exprs)
    rows = [{k: _clean_value(r.get(k)) for k in columns} for r in sub.to_dicts()]

    t = table(
        table_name,
        *[column(c, _sqlalchemy_type(col_types[c])) for c in columns],
    )
    max_params = 30000
    chunk_rows = max(1, max_params // max(len(columns), 1))

    async with engine.begin() as conn:
        await conn.execute(text(f'TRUNCATE TABLE {_qi(table_name)}'))
        for i in range(0, len(rows), chunk_rows):
            chunk = rows[i:i + chunk_rows]
            if chunk:
                await conn.execute(insert(t), chunk)


async def load_municipalities_from_frame(
    engine: AsyncEngine,
    df: pl.DataFrame,
    id_wkb: list[tuple[int, bytes]],
) -> None:
    """Sync municipalities attributes + PostGIS geometry (EPSG:4326)."""
    attr_cols = [c for c in df.columns if c not in ("geometry",)]
    col_types = _pl_schema_to_sql_types(df.schema, "municipalities")
    if "id" not in col_types:
        col_types["id"] = "INTEGER"

    await sync_table_columns(
        engine,
        "municipalities",
        attr_cols,
        col_types,
        protected=frozenset({"geometry"}),
    )
    await replace_table_rows(engine, "municipalities", df.select(attr_cols), col_types)
    await sync_municipalities_geometry_postgis(engine, id_wkb, srid=4326)


async def load_indicators(
    engine: AsyncEngine,
    names: list[str],
) -> dict[str, int]:
    """Replace indicators table; return mapping original_name -> id."""
    rows = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    mapping = {n: i + 1 for i, n in enumerate(names)}

    async with engine.begin() as conn:
        if await _relation_exists(conn, "indicators"):
            await conn.execute(text('TRUNCATE TABLE "indicators" RESTART IDENTITY CASCADE'))
        else:
            await conn.execute(
                text(
                    'CREATE TABLE "indicators" ('
                    '"id" INTEGER PRIMARY KEY, "name" VARCHAR)'
                )
            )
        if rows:
            t = table("indicators", column("id", Integer()), column("name", String()))
            await conn.execute(insert(t), rows)

    return mapping


def rename_feature_columns(
    df: pl.DataFrame,
    id_col: str,
    target_col: str | None,
    name_to_id: dict[str, int],
) -> pl.DataFrame:
    """Rename feature columns to ind_{id} / target_{id}; keep municipality_id."""
    renames: dict[str, str] = {}
    for col in df.columns:
        if col == id_col:
            continue
        if col not in name_to_id:
            continue
        ind_id = name_to_id[col]
        if target_col and col == target_col:
            renames[col] = f"target_{ind_id}"
        else:
            renames[col] = f"ind_{ind_id}"
    out = df.rename(renames)

    return out


async def load_indicator_values_table(
    engine: AsyncEngine,
    table_name: str,
    df: pl.DataFrame,
) -> None:
    col_types = _pl_schema_to_sql_types(df.schema, "indicator_values")
    if "municipality_id" not in col_types:
        col_types["municipality_id"] = "INTEGER"
    for c in df.columns:
        if c != "municipality_id" and c not in col_types:
            col_types[c] = "DOUBLE PRECISION"

    await sync_table_columns(engine, table_name, list(df.columns), col_types)
    await replace_table_rows(engine, table_name, df, col_types)


async def filter_municipalities_by_ids(
    engine: AsyncEngine,
    municipality_ids: list[int],
) -> None:
    if not municipality_ids:
        return
    ids_sql = ",".join(str(int(i)) for i in sorted(set(municipality_ids)))
    async with engine.begin() as conn:
        if await _relation_exists(conn, "municipalities"):
            await conn.execute(
                text(f'DELETE FROM "municipalities" WHERE id NOT IN ({ids_sql})')
            )


async def table_row_count(engine: AsyncEngine, table_name: str) -> int:
    async with engine.connect() as conn:
        if not await _relation_exists(conn, table_name):
            return 0
        n = await conn.scalar(text(f'SELECT COUNT(*) FROM {_qi(table_name)}'))
        return int(n or 0)


async def fetch_indicators(engine: AsyncEngine) -> list[dict[str, Any]]:
    async with engine.connect() as conn:
        if not await _relation_exists(conn, "indicators"):
            return []
        rows = (
            await conn.execute(text('SELECT id, name FROM "indicators" ORDER BY id'))
        ).mappings().all()
        return [dict(r) for r in rows]
