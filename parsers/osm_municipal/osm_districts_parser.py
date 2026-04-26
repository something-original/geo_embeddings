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


def main():

    root_dir = Path(__file__).resolve().parents[2]

    path_parts = [root_dir, 'datasets', 'mun_data']
    mun_districts_df_path = os.path.join(*path_parts, 'municipalities.csv')

    df_attrs = {'sep': ';', 'encoding': 'utf-8'}

    mun_districts_df = pd.read_csv(mun_districts_df_path, **df_attrs)

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

    mun_points_cols = ['oktmo', 'municipal_district_center_lat', 'municipal_district_center_lon']
    mun_points = mun_points[mun_points_cols]
    lat_col = 'municipal_district_center_lat'
    lon_col = 'municipal_district_center_lon'

    mun_points['geometry'] = [Point(lon, lat) for lon, lat in zip(mun_points[lon_col], mun_points[lat_col])]
    mun_points.drop(columns=[lat_col, lon_col], inplace=True)
    mun_points = gpd.GeoDataFrame(mun_points).set_geometry('geometry').set_crs('EPSG:4326')

    mun_geometry = (
        mun_geometry.groupby('osm_ref')
        .agg({'year_from': 'max', 'geometry': 'first'})
        .reset_index()
        .drop(columns=['year_from', 'osm_ref'])
    )

    mun_geometry = gpd.GeoDataFrame(mun_geometry).set_geometry('geometry').set_crs('EPSG:4326')

    mun_geometry = mun_geometry.sjoin(mun_points, how='left')
    mun_geometry['oktmo'] = mun_geometry['oktmo'].str.replace('-', '').str.slice(stop=-3)

    mun_districts_df = mun_districts_df.merge(mun_geometry, on=['oktmo'], how='inner')
    mun_districts_df = gpd.GeoDataFrame(mun_districts_df).set_geometry('geometry').set_crs('EPSG:4326')

    mun_districts_df.to_csv(mun_districts_df_path, index=False, **df_attrs)


if __name__ == '__main__':
    main()
