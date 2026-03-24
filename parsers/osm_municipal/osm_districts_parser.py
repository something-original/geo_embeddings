import wget
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from zipfile import ZipFile


def main():

    load_dotenv()
    root_dir = Path(__file__).resolve().parents[2]
    save_path = os.path.join(root_dir, "datasets", "raw_mun_data")
    os.makedirs(save_path, exist_ok=True)

    download_url = os.getenv("MUN_DATA_LINK")
    zip_file_name = download_url.split("/")[-1]
    full_zip_path = os.path.join(save_path, zip_file_name)

    if not os.path.exists(full_zip_path):
        wget.download(download_url, out=save_path)

    with ZipFile(full_zip_path, "r") as zip_ref:
        zip_ref.extractall(save_path)

    pq_file_name = [file for file in zip_ref.namelist() if file.endswith(".parquet")][0]
    csv_file_name = [file for file in zip_ref.namelist() if file.endswith(".csv")][0]

    pq_file = pd.read_parquet(os.path.join(save_path, pq_file_name))
    csv_file = pd.read_csv(os.path.join(save_path, csv_file_name), sep=';')
   
    debug = 1


if __name__ == '__main__':
    main()
