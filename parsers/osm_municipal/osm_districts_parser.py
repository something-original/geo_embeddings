import aiohttp
import asyncio
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
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

    load_dotenv()
    root_dir = Path(__file__).resolve().parents[2]
    mun_data_geom_link = os.getenv('MUN_GEOMS_LINK')

    path_parts = [root_dir, 'datasets', 'mun_data']
    mun_districts_df_path = os.path.join(*path_parts, 'municipalities.csv')
    regions_df_path = os.path.join(*path_parts, 'regions.csv')

    df_attrs = {'sep': ';', 'encoding': 'utf-8'}
    mun_districts_df = pd.read_csv(mun_districts_df_path, **df_attrs)
    regions_df = pd.read_csv(regions_df_path, **df_attrs)

    mun_districts_df = mun_districts_df.merge(
        regions_df, how='left', left_on='region_id', right_on='id', suffixes=('_mun', '_reg')
    ).drop(columns=['id_mun', 'region_id', 'id_reg'])

    save_path = os.path.join(*path_parts, 'mun_data_geom.rar')

    if not Path(save_path).exists():
        asyncio.run(load_mun_geometry(mun_data_geom_link, save_path))

    subprocess.run("")
    debug = 1


if __name__ == '__main__':
    main()
