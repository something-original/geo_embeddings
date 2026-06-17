"""Background workers for dataset upload endpoints (run in subprocess)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import polars as pl
from shapely import wkt
from shapely.geometry.base import BaseGeometry
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from api.db_ops import (
    filter_municipalities_by_ids,
    load_indicator_values_table,
    load_indicators,
    load_municipalities_from_frame,
    rename_feature_columns,
)
from api.state import update_state
from config import DB_URL
from emb_fit.utils import prepare_and_save_dataset
from parsers.osm_municipal.db import municipalities_has_geometry_column
from sklearn.preprocessing import StandardScaler
from utils import PathBuilder, setup_logging

logger = logging.getLogger(__name__)
CSV_SEP = ";"


def _read_csv_fast(path: Path) -> pl.DataFrame:
    for sep in (";", ","):
        try:
            return pl.read_csv(path, separator=sep, infer_schema_length=10_000, try_parse_dates=False)
        except Exception:
            continue
    return pl.read_csv(path, infer_schema_length=10_000, try_parse_dates=False)


def _resolve_id_series(df: pl.DataFrame, index_col: str | None) -> pl.DataFrame:
    if index_col and index_col in df.columns:
        if index_col == "municipality_id":
            return df.with_columns(pl.col(index_col).cast(pl.Int64))
        return (
            df.with_columns(pl.col(index_col).cast(pl.Int64).alias("municipality_id"))
            .drop(index_col)
        )
    if "municipality_id" in df.columns:
        return df.with_columns(pl.col("municipality_id").cast(pl.Int64))
    return df.with_row_index("municipality_id").with_columns(
        (pl.col("municipality_id") + 1).cast(pl.Int64)
    )


def _geometries_to_wkb_list(
    gdf: gpd.GeoDataFrame,
    id_col: str = "id",
) -> list[tuple[int, bytes]]:
    out: list[tuple[int, bytes]] = []
    for row in gdf.itertuples(index=False):
        mid = int(getattr(row, id_col))
        geom: BaseGeometry = getattr(row, "geometry")
        if geom is None or geom.is_empty:
            continue
        out.append((mid, geom.wkb))
    return out


def _load_geometries_from_csv(
    path: Path,
    geom_col: str,
    crs: str,
    index_col: str | None,
) -> tuple[gpd.GeoDataFrame, pl.DataFrame]:
    pdf = _read_csv_fast(path).to_pandas()
    if geom_col not in pdf.columns:
        raise ValueError(f"Geometry column {geom_col!r} not found in {path.name}")

    if pdf[geom_col].dtype == object and isinstance(pdf[geom_col].iloc[0], str):
        pdf[geom_col] = pdf[geom_col].apply(wkt.loads)

    gdf = gpd.GeoDataFrame(pdf, geometry=geom_col, crs=crs).to_crs("EPSG:4326")
    gdf = gdf.rename(columns={geom_col: "geometry"})

    if index_col and index_col in gdf.columns:
        gdf["id"] = gdf[index_col].astype(int)
    else:
        gdf["id"] = (gdf.index + 1).astype(int)

    attr_pl = pl.from_pandas(gdf.drop(columns=["geometry"], errors="ignore"))
    return gdf, attr_pl


async def _process_upload_territories(path: Path, params: dict[str, Any]) -> None:
    crs = params.get("crs", "EPSG:4326")
    index_col: str | None = params.get("index_col")
    geom_col: str = params["geom_col"]

    gdf, attr_pl = _load_geometries_from_csv(path, geom_col, crs, index_col)
    id_wkb = _geometries_to_wkb_list(gdf, "id")

    attr_pl = attr_pl.with_columns(pl.col("id").cast(pl.Int64))
    if "geometry" in attr_pl.columns:
        attr_pl = attr_pl.drop("geometry")

    engine = create_async_engine(url=DB_URL)
    try:
        await load_municipalities_from_frame(engine, attr_pl, id_wkb)
        update_state(territories_uploaded=True)
        logger.info("upload_territories: loaded %s municipalities", len(id_wkb))
    finally:
        await engine.dispose()


def _feature_columns(df: pl.DataFrame, id_col: str, geom_col: str | None, target_col: str | None) -> list[str]:
    skip = {id_col, "municipality_id"}
    if geom_col:
        skip.add(geom_col)
    if target_col:
        skip.add(target_col)
    return [c for c in df.columns if c not in skip and not c.startswith("_")]


async def _write_municipalities_from_inference(
    inf_path: Path,
    geom_col: str,
    crs: str,
    index_col: str | None,
) -> None:
    gdf, attr_pl = _load_geometries_from_csv(inf_path, geom_col, crs, index_col)
    id_wkb = _geometries_to_wkb_list(gdf, "id")
    engine = create_async_engine(url=DB_URL)
    try:
        await load_municipalities_from_frame(engine, attr_pl, id_wkb)
        update_state(territories_uploaded=True)
    finally:
        await engine.dispose()


def _save_local_csv(df: pl.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(dest, separator=CSV_SEP)


async def _process_upload_feature_values(
    file_paths: dict[str, Path],
    params: dict[str, Any],
) -> None:
    same_file = bool(params.get("same_file_inference", True))
    index_col: str | None = params.get("index_col")
    target_col: str | None = params.get("target_col")
    geom_col: str | None = params.get("geom_col")
    crs: str = params.get("crs", "EPSG:4326")

    if same_file:
        if len(file_paths) != 1:
            raise ValueError("same_file_inference=true requires exactly one uploaded file")
        train_name = inf_name = next(iter(file_paths))
    else:
        train_name = params.get("train_file_name")
        inf_name = params.get("inference_file_name")
        if not train_name or not inf_name:
            raise ValueError("train_file_name and inference_file_name are required when same_file_inference=false")
        if train_name not in file_paths or inf_name not in file_paths:
            raise ValueError("Uploaded files must include train_file_name and inference_file_name")

    train_path = file_paths[train_name]
    inf_path = file_paths[inf_name if not same_file else train_name]

    st = __import__("api.state", fromlist=["get_state"]).get_state()
    territories_ready = st.territories_uploaded

    if geom_col:
        await _write_municipalities_from_inference(inf_path, geom_col, crs, index_col)
        territories_ready = True
    elif not territories_ready:
        engine = create_async_engine(url=DB_URL)
        try:
            territories_ready = await municipalities_has_geometry_column(engine)
        finally:
            await engine.dispose()

    if not territories_ready and geom_col is None:
        raise ValueError(
            "Territories not loaded: call POST /datasets/upload_territories first "
            "or provide geom_col + crs in upload_feature_values"
        )

    train_df = _read_csv_fast(train_path)
    inf_df = _read_csv_fast(inf_path)

    if index_col:
        train_df = _resolve_id_series(train_df, index_col)
        inf_df = _resolve_id_series(inf_df, index_col)
    else:
        train_df = _resolve_id_series(train_df, None)
        inf_df = _resolve_id_series(inf_df, None)

    if geom_col:
        train_df = train_df.drop(geom_col, strict=False)
        inf_df = inf_df.drop(geom_col, strict=False)

    if not same_file and target_col and target_col in inf_df.columns:
        inf_df = inf_df.drop(target_col)

    feature_names_train = _feature_columns(train_df, "municipality_id", geom_col, None)
    feature_names_inf = _feature_columns(
        inf_df, "municipality_id", geom_col, target_col if not same_file else None
    )

    all_feature_names = list(dict.fromkeys(feature_names_train + feature_names_inf))
    if target_col and target_col not in all_feature_names:
        all_feature_names.append(target_col)

    mun_index_col = params.get('index_col', None)
    if not mun_index_col:
        mun_index_col = "id"
        inf_df = inf_df.with_row_index(mun_index_col)
        train_df = train_df.with_row_index(mun_index_col)

    inf_ids = inf_df[mun_index_col].to_list()
    name_to_id = {n: i + 1 for i, n in enumerate(all_feature_names)}

    inf_df = rename_feature_columns(inf_df, mun_index_col, target_col, name_to_id)
    train_df = rename_feature_columns(train_df, mun_index_col, target_col, name_to_id)

    update_state(
        target_column=target_col,
        target_indicator_ids=[],
        same_file_inference=same_file,
        territories_uploaded=territories_ready,
    )

    try:
        engine = create_async_engine(url=DB_URL)
        await filter_municipalities_by_ids(engine, inf_ids)

    finally:
        await engine.dispose()

    paths = PathBuilder.build_emb_datasets_paths()
    mun_dir = Path(paths["dataset_path"]).parent

    emb_paths = {
        "dataset_path_old": str(paths["dataset_path_old"]),
        "dataset_path": str(paths["inference_full_path"]),
        "indicators_path": str(paths["indicators_path"]),
        "train_path": str(paths["train_path"]),
        "val_path": str(paths["val_path"]),
        "inference_full_path": str(paths["inference_full_path"])
    }

    pl.DataFrame(
        {"id": list(name_to_id.values()), "name": list(name_to_id.keys())}
    ).write_csv(Path(paths["indicators_path"]), separator=CSV_SEP)

    _save_local_csv(inf_df, Path(paths["inference_full_path"]))
    _save_local_csv(inf_df, Path(paths["dataset_path"]))    

    if not same_file:
        _save_local_csv(train_df, mun_dir / "indicator_values_old.csv")

    prepare_and_save_dataset(
        emb_dataset_paths=emb_paths,
        separate_inference=not same_file,
        features_to_drop=[],
        index_feature=mun_index_col,
        experiment_target_features=[],
        csv_sep=CSV_SEP,
        use_scaler=True,
        scaler=StandardScaler(),
    )

    logger.info(f"upload_feature_values complete (same_file_inference={same_file})")


def run_upload_territories(csv_path: str, params: dict[str, Any]) -> None:
    setup_logging()
    asyncio.run(_process_upload_territories(Path(csv_path), params))


def run_upload_feature_values(file_paths: dict[str, str], params: dict[str, Any]) -> None:
    setup_logging()
    paths = {k: Path(v) for k, v in file_paths.items()}
    asyncio.run(_process_upload_feature_values(paths, params))
