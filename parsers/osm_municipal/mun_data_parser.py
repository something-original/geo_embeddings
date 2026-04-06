import asyncio
import aiohttp
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from zipfile import ZipFile
from tqdm import tqdm


async def download_all_files(base_download_url, mun_datasets_df):
    tasks = []
    timeout = aiohttp.ClientTimeout(total=600)
    semaphore = asyncio.Semaphore(10)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for section_id, dataset_code in zip(
            mun_datasets_df['Код раздела'],
            mun_datasets_df['Код показателя']
        ):
            task = asyncio.create_task(
                download_one_file(session, semaphore, base_download_url, str(section_id), str(dataset_code))
            )
            tasks.append(task)
        await asyncio.gather(*tasks)


async def download_one_file(session, semaphore, base_download_url, section_id, dataset_code):
    async with semaphore:
        download_url = base_download_url.replace("<section_id>", section_id)
        download_url = download_url.replace("<dataset_code>", dataset_code)
        print(f"URL: {download_url}")

        root_dir = Path(__file__).resolve().parents[2]
        save_folder_path = os.path.join(root_dir, "datasets", "mun_data")
        os.makedirs(save_folder_path, exist_ok=True)
        save_file_path = os.path.join(save_folder_path, f"{dataset_code}.zip")

        async with session.get(download_url) as response:
            with open(save_file_path, 'wb') as f:
                f.write(await response.read())


def download_mun_data(root_dir):
    load_dotenv()
    base_download_url = os.getenv("TOCHNO_ST_BASE_LINK")

    mun_datasets_df_path = os.path.join(root_dir, "datasets", "mun_datasets_metadata.csv")
    mun_datasets_df = pd.read_csv(mun_datasets_df_path, sep=';')

    asyncio.run(download_all_files(base_download_url, mun_datasets_df))


def add_new_values(
    df_from: pd.DataFrame,
    df_to: pd.DataFrame,
    col_names: str | list[str],
    cols_to_rename: dict[str, str] | None = None
) -> None:

    known_values = set(df_to[col_names])
    coming_values = df_from[col_names].drop_duplicates().reset_index(drop=True)
    new_values = coming_values[~coming_values.isin(known_values)]

    if not new_values.empty:
        last_id = df_to['id'].max() if not df_to.empty else 0

        values_to_add = {
            col: new_values[col].values
            for col in col_names
            if col != "id"
        }

        df_to_add = pd.DataFrame({
            **values_to_add,
            "id": range(last_id + 1, last_id + 1 + len(new_values))
        })

        if cols_to_rename:
            df_to_add.rename(columns=cols_to_rename, inplace=True)

        df_to = pd.concat([df_to, df_to_add], ignore_index=True)


def preprocess_mun_df(df: pd.DataFrame, dataframes: dict[str, pd.DataFrame]) -> None:

    cols_to_drop = [
        "indicator_section_code", "indicator_section", "indicator_period",
        "oktmo_stable", "oktmo_history", "oktmo_year_from", "oktmo_year_to",
        "mun_type", "mun_type_oktmo",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    unique_okved_values = None

    if "okved2" in df.columns:
        df_okved = dataframes["okveds_df"]
        unique_okved_values = df['okved2'].drop_duplicates().reset_index(drop=True)

        if df_okved.empty:
            df_okved = unique_okved_values.to_frame(name="name")
            df_okved["id"] = df_okved.index + 1
        else:
            add_new_values(
                df_from=df,
                df_to=df_okved,
                col_names=["okved2"],
                cols_to_rename={"okved2": "name"}
            )

    regions_df = dataframes["regions_df"]
    if regions_df.empty:
        regions_df = df[["region_id", "region_name"]].drop_duplicates().reset_index(drop=True)
        regions_df.rename(columns={"region_id": "id", "region_name": "name"}, inplace=True)
    else:
        add_new_values(
            df_from=df,
            df_to=regions_df,
            col_names=["region_id", "region_name"],
            cols_to_rename={"region_id": "id", "region_name": "name"}
        )

    mun_districts_df = dataframes["mun_districts_df"]
    unique_mun_district_values = df["mun_district"].drop_duplicates().reset_index(drop=True)
    if mun_districts_df.empty:
        mun_districts_df = unique_mun_district_values.to_frame(name="name")
        mun_districts_df["id"] = mun_districts_df.index + 1
    else:
        add_new_values(
            df_from=df,
            df_to=df_okved,
            col_names=["mun_district"],
            cols_to_rename={"mun_district": "name"}
        )
        

    indicators_df = dataframes["indicators_df"]
    debug = 1
        



def parse_mun_data(root_dir):

    dataframes = {
        "indicators_df": pd.DataFrame(),
        "indicators_values_df": pd.DataFrame(),
        "regions_df": pd.DataFrame(),
        "municipalities_df": pd.DataFrame(),
        "mun_districts_df": pd.DataFrame(),
        "okveds_df": pd.DataFrame(),
        "units_df": pd.DataFrame(),
    }

    folder_path = os.path.join(root_dir, "datasets", "mun_data")
    if not os.listdir(folder_path):
        download_mun_data(folder_path)

    for zipfile in tqdm(os.listdir(folder_path)):
        full_file_path = os.path.join(folder_path, zipfile)

        with ZipFile(full_file_path) as archive:
            archive.extractall(folder_path)

        file_name = f"data_Y4{zipfile.replace(".zip", "")}_112_v20250918.xlsx"
        try:
            df = pd.read_excel(os.path.join(folder_path, file_name))
        except FileNotFoundError:
            file_name = file_name.replace(".xlsx", ".csv")
            df = pd.read_csv(os.path.join(folder_path, file_name), sep=';')

        preprocess_mun_df(df, dataframes)


if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parents[2]
    parse_mun_data(root_dir)
