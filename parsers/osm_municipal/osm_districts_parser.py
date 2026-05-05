import aiohttp
import asyncio
import os
import pandas as pd
import geopandas as gpd
from shapely import Point, wkt

from pathlib import Path
from config import MUN_GEOMS_LINK
import subprocess

import ssl
import certifi


async def load_mun_geometry(mun_data_geom_link, save_path):
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(mun_data_geom_link, ssl=ssl_context) as resp:
            with open(save_path, 'wb') as f:
                f.write(await resp.read())


def form_mun_geometry():

    root_dir = Path(__file__).resolve().parents[2]

    path_parts = [root_dir, 'datasets', 'mun_data']
    mun_districts_df_path = os.path.join(*path_parts, 'municipalities.csv')
    indicators_df_path = os.path.join(*path_parts, 'indicator_values.csv')

    df_attrs = {'sep': ';', 'encoding': 'utf-8'}

    mun_districts_df = pd.read_csv(mun_districts_df_path, **df_attrs)
    indicators_df = pd.read_csv(indicators_df_path, **df_attrs)

    save_folder = Path(*path_parts)
    save_path = os.path.join(*path_parts, 'mun_data_geom.rar')

    if not Path(save_path).exists():
        asyncio.run(load_mun_geometry(MUN_GEOMS_LINK, save_path))

    files_before = set(save_folder.rglob('*'))
    unpack_cmd = ["7z", "x", save_path, f"-o{save_folder}", "-y"]
    subprocess.run(unpack_cmd, stdout=subprocess.PIPE)

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

    mun_df_orig = mun_districts_df.copy()
    mun_districts_df = mun_districts_df[~mun_districts_df['oktmo'].isin(["CD", "ND"])]

    mun_points = mun_points[['oktmo', 'territory_id']]
    mun_points['oktmo'] = mun_points['oktmo'].str.replace('-','').str.slice(stop=-3)
    
    mun_geometry = mun_geometry[['territory_id', 'geometry']]
    mun_geometry['territory_id'] = mun_geometry['territory_id'].apply(int)
    
    mun_geometry = mun_points.merge(mun_geometry, how='left')
    mun_geometry.drop(columns=['territory_id'], inplace=True)
    mun_districts_df = mun_districts_df.merge(mun_geometry, how='left')
    
    mun_districts_df = mun_districts_df[~mun_districts_df['geometry'].isna()]
    indicators_df = indicators_df[indicators_df['municipality_id'].isin(mun_districts_df['id'])]

    mun_districts_df.to_csv(mun_districts_df_path, index=False, **df_attrs)
    indicators_df.to_csv(indicators_df_path, index=False, **df_attrs)


if __name__ == '__main__':
    form_mun_geometry()
