import wget
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from zipfile import ZipFile


def main():

    load_dotenv()
    root_dir = Path(__file__).resolve().parents[2]
    overpass_link = os.getenv('OVERPASS_API_LINK')

    path_parts = [root_dir, 'datasets', 'mun_data']
    mun_districts_df_path = os.path.join(*path_parts, 'municipalities.csv')
    regions_df_path = os.path.join(*path_parts, 'regions.csv')
    
    df_attrs = {'sep': ';', 'encoding': 'utf-8'}
    mun_districts_df = pd.read_csv(mun_districts_df_path, **df_attrs)
    regions_df = pd.read_csv(regions_df_path, **df_attrs)

    debug = 1

if __name__ == '__main__':
    main()
