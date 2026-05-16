import asyncio
import aiohttp
import os
import pandas as pd
import geopandas as gpd

from pathlib import Path
import shutil
import subprocess

from config import MUN_GEOMS_LINK, SEVEN_ZIP_BIN

import ssl
import certifi

from sqlalchemy.ext.asyncio import AsyncEngine

from .db import municipalities_has_geometry_column, sync_municipalities_geometry_postgis


def resolve_seven_zip_bin() -> str:
    """Return path to 7z executable (SEVEN_ZIP_BIN env or PATH)."""
    if SEVEN_ZIP_BIN:
        p = Path(SEVEN_ZIP_BIN)
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(f"SEVEN_ZIP_BIN is set but not found: {SEVEN_ZIP_BIN}")
    for name in ("7z", "7zz", "7za"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError(
        "7z not found. Install p7zip (e.g. apt install p7zip-full) "
        "or set SEVEN_ZIP_BIN to the binary path."
    )


async def load_mun_geometry(mun_data_geom_link, save_path):
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(mun_data_geom_link, ssl=ssl_context) as resp:
            with open(save_path, 'wb') as f:
                f.write(await resp.read())


async def form_mun_geometry(engine: AsyncEngine | None = None):

    if engine is not None and await municipalities_has_geometry_column(engine):
        return

    root_dir = Path(__file__).resolve().parents[2]

    path_parts = [root_dir, 'datasets', 'mun_data']
    mun_districts_df_path = os.path.join(*path_parts, 'municipalities.csv')
    indicators_df_path = os.path.join(*path_parts, 'indicator_values.csv')
    old_indicators_df_path = os.path.join(*path_parts, 'indicator_values_old.csv')

    df_attrs = {'sep': ';', 'encoding': 'utf-8'}

    mun_districts_df = pd.read_csv(mun_districts_df_path, **df_attrs)
    indicators_df = pd.read_csv(indicators_df_path, **df_attrs)
    indicators_df_old = pd.read_csv(old_indicators_df_path, **df_attrs)

    save_folder = Path(*path_parts)
    save_path = os.path.join(*path_parts, 'mun_data_geom.rar')

    if not Path(save_path).exists():
        await load_mun_geometry(MUN_GEOMS_LINK, save_path)

    files_before = set(save_folder.rglob('*'))
    seven_zip = resolve_seven_zip_bin()
    unpack_cmd = [seven_zip, "x", save_path, f"-o{save_folder}", "-y"]
    result = subprocess.run(unpack_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"7z failed (exit {result.returncode}): {result.stderr.decode(errors='replace')}"
        )

    os.remove(save_path)
    files_after = set(save_folder.rglob('*'))
    new_files_paths = files_after - files_before

    mun_geometry = None
    mun_points = None

    for file in new_files_paths:
        file_str = str(file)
        if file_str.endswith('.gpkg'):
            mun_geometry = gpd.read_file(file_str)
            os.remove(file_str)
        elif file_str.endswith('.xlsx'):
            mun_points = gpd.read_file(file_str)
            os.remove(file_str)

    mun_districts_df.drop(columns=['geometry'], inplace=True, errors='ignore')
    mun_districts_df = mun_districts_df[~mun_districts_df['oktmo'].isin(["CD", "ND"])]

    mun_points = mun_points[['oktmo', 'territory_id']]
    mun_points['oktmo'] = mun_points['oktmo'].str.replace('-','').str.slice(stop=-3)

    mun_geometry = mun_geometry[['territory_id', 'geometry']]
    mun_geometry['territory_id'] = mun_geometry['territory_id'].apply(int)
    mun_geometry_crs = mun_geometry.crs

    mun_geometry = mun_points.merge(mun_geometry, how='left')
    mun_geometry.drop(columns=['territory_id'], inplace=True)

    mun_districts_df['oktmo'] = mun_districts_df['oktmo'].apply(str)
    mun_districts_df.loc[mun_districts_df['oktmo'].str.len() == 7, 'oktmo'] = '0' + mun_districts_df['oktmo']

    mun_districts_df = mun_districts_df.merge(mun_geometry, how='left')

    mun_districts_df = mun_districts_df[~mun_districts_df['geometry'].isna()]
    indicators_df = indicators_df[indicators_df['municipality_id'].isin(mun_districts_df['id'])]
    indicators_df_old = indicators_df_old[indicators_df_old['municipality_id'].isin(mun_districts_df['id'])]

    mun_districts_df = mun_districts_df.drop_duplicates(subset=['id'])

    if engine is not None and 'geometry' in mun_districts_df.columns:
        gdf = gpd.GeoDataFrame(
            mun_districts_df,
            geometry='geometry',
            crs=mun_geometry_crs,
        )
        epsg = gdf.crs.to_epsg() if gdf.crs is not None else None
        srid = int(epsg) if epsg is not None else 4326
        id_wkb = [
            (int(r['id']), r['geometry'].wkb)
            for _, r in gdf.iterrows()
            if r['geometry'] is not None and not r['geometry'].is_empty
        ]
        await sync_municipalities_geometry_postgis(engine, id_wkb, srid)

    mun_districts_df.to_csv(mun_districts_df_path, index=False, **df_attrs)
    indicators_df.to_csv(indicators_df_path, index=False, **df_attrs)
    indicators_df_old.to_csv(old_indicators_df_path, index=False, **df_attrs)
