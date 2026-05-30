import asyncio
import math

from pathlib import Path

import polars as pl
from sqlalchemy import Boolean, Float, Integer, String, Text, column, insert, table, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncEngine


SAVE_TO_DB = {
    "municipalities",
    "regions"
}

_INDICATOR_STEMS = frozenset({
    "indicator_values",
    "indicator_values_old",
    "indicator_values_train",
    "indicator_values_inference",
})


def _qi(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _infer_column_sql_types(
    columns: tuple[str, ...],
    stem: str,
    schema: pl.Schema,
) -> dict[str, str]:
    stem_l = stem.lower()
    out: dict[str, str] = {}
    for col in columns:
        if col == "geometry":
            continue
        if stem_l in _INDICATOR_STEMS:
            if col == "municipality_id":
                out[col] = "INTEGER"
            else:
                out[col] = "DOUBLE PRECISION"
            continue
        if col in ("name", "code"):
            out[col] = "VARCHAR"
            continue
        if "id" in col.lower():
            out[col] = "INTEGER"
            continue
        dt = schema[col]
        if dt in (pl.Utf8, pl.String):
            out[col] = "TEXT"
        elif dt in (pl.Float32, pl.Float64):
            out[col] = "DOUBLE PRECISION"
        elif dt in (
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ):
            out[col] = "INTEGER"
        elif dt == pl.Boolean:
            out[col] = "BOOLEAN"
        else:
            out[col] = "TEXT"
    return out


def _sqlalchemy_type(sql_t: str):
    if sql_t == "INTEGER":
        return Integer()
    if sql_t == "DOUBLE PRECISION":
        return Float()
    if sql_t == "VARCHAR":
        return String()
    if sql_t == "BOOLEAN":
        return Boolean()
    return Text()


def _read_mun_csv(path: Path) -> pl.DataFrame:
    try:
        return pl.read_csv(
            path,
            separator=";",
            infer_schema_length=None,
            try_parse_dates=False,
        )
    except Exception:
        return pl.read_csv(
            path,
            separator=",",
            infer_schema_length=None,
            try_parse_dates=False,
        )


def _clean_value(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _build_create_sql(table_name: str, columns: list[str], col_types: dict[str, str]) -> str:
    parts: list[str] = []
    for col in columns:
        typ = col_types[col]
        if col == "id":
            parts.append(f"{_qi(col)} {typ} PRIMARY KEY")
        else:
            parts.append(f"{_qi(col)} {typ}")

    tn = _qi(table_name)
    cols_sql = ", ".join(parts)
    return f"CREATE TABLE {tn} ({cols_sql})"


def _is_duplicate_table_error(exc: ProgrammingError) -> bool:
    orig = getattr(exc, "orig", None)
    if orig is not None:
        code = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
        if code == "42P07":
            return True
    msg = str(exc).lower()
    return "already exists" in msg and "relation" in msg


async def municipalities_has_geometry_column(engine: AsyncEngine) -> bool:
    async with engine.connect() as conn:
        if not await _relation_exists(conn, "municipalities"):
            return False
        row = await conn.scalar(
            text(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'municipalities'
                  AND column_name = 'geometry'
                LIMIT 1
                """
            )
        )
        return bool(row)


async def _relation_exists(conn, table_name: str) -> bool:
    row = await conn.scalar(
        text(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public'
                  AND c.relkind = 'r'
                  AND c.relname = :relname
            )
            """
        ),
        {"relname": table_name},
    )
    return bool(row)


async def create_table_and_insert(
    engine: AsyncEngine,
    table_name: str,
    df: pl.DataFrame,
    column_types: dict[str, str],
) -> None:

    columns = [c for c in df.columns if c != "geometry" and c in column_types]
    if not columns:
        return

    col_types = {c: column_types[c] for c in columns}
    create_sql = _build_create_sql(table_name, columns, col_types)

    async with engine.connect() as conn:
        if await _relation_exists(conn, table_name):
            return

    sub = df.select(columns)
    exprs = []

    for c in columns:
        if sub.schema[c] in (pl.Float32, pl.Float64):
            exprs.append(pl.col(c).fill_nan(None).alias(c))
        else:
            exprs.append(pl.col(c))

    sub = sub.select(exprs)
    rows_raw = sub.to_dicts()
    rows = [
        {k: _clean_value(r.get(k)) for k in columns}
        for r in rows_raw
    ]
    t = table(
        table_name,
        *[column(c, _sqlalchemy_type(col_types[c])) for c in columns],
    )

    max_params = 30000
    chunk_rows = max(1, max_params // max(len(columns), 1))

    async with engine.begin() as conn:
        if await _relation_exists(conn, table_name):
            return
        try:
            await conn.execute(text(create_sql))
        except ProgrammingError as e:
            if _is_duplicate_table_error(e):
                return
            raise
        if not rows:
            return
        for i in range(0, len(rows), chunk_rows):
            chunk = rows[i:i + chunk_rows]
            await conn.execute(insert(t), chunk)


async def sync_municipalities_geometry_postgis(
    engine: AsyncEngine,
    id_wkb: list[tuple[int, bytes]],
    srid: int,
) -> None:
    """Write WKB geometries into municipalities.geometry and drop rows not in id_wkb."""
    if not id_wkb:
        return

    allowed_ids = sorted({int(i) for i, _ in id_wkb})

    async with engine.begin() as conn:
        if not await _relation_exists(conn, "municipalities"):
            return

        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        except ProgrammingError:
            pass

        col = await conn.scalar(
            text(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'municipalities'
                  AND column_name = 'geometry'
                """
            )
        )
        if not col:
            await conn.execute(
                text(
                    'ALTER TABLE "municipalities" '
                    "ADD COLUMN IF NOT EXISTS geometry geometry"
                )
            )

        upd = text(
            """
            UPDATE "municipalities"
            SET geometry = ST_SetSRID(ST_GeomFromWKB(:wkb), :srid)
            WHERE id = :id
            """
        )
        for mid, wkb in id_wkb:
            await conn.execute(upd, {"wkb": wkb, "srid": srid, "id": int(mid)})

        ids_sql = ",".join(str(i) for i in allowed_ids)
        await conn.execute(
            text(f'DELETE FROM "municipalities" WHERE id NOT IN ({ids_sql})')
        )


async def load_mun_csvs_to_database(engine: AsyncEngine, folder_path: Path) -> None:
    paths = sorted(folder_path.glob("*.csv"))
    tasks: list[asyncio.Task] = []
    for path in paths:
        stem = path.stem
        if stem not in SAVE_TO_DB:
            continue

        async def _one(p: Path = path) -> None:
            df = _read_mun_csv(p)
            columns = tuple(c for c in df.columns if c != "geometry")
            df = df.select(list(columns))
            col_types = _infer_column_sql_types(columns, p.stem, df.schema)
            await create_table_and_insert(engine, p.stem, df, col_types)

        tasks.append(asyncio.create_task(_one()))
    if tasks:
        await asyncio.gather(*tasks)
