import asyncio
import aiohttp
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from zipfile import ZipFile
from tqdm import tqdm
import shutil


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
        "mun_type", "mun_type_oktmo", "comment"
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

    municipalities_df = dataframes["municipalities_df"]
    cols = ["region_name", "municipality", "mun_district"]
    municipality_values = df[cols].drop_duplicates().reset_index(drop=True)
    municipality_values = municipality_values[
        municipality_values["municipality"] != municipality_values["mun_district"]
    ]

    municipality_values = municipality_values.merge(
        regions_df, how='left', left_on='region_name', right_on='name'
    )\
        .rename(columns={'id': 'region_id'})\
        .drop(columns=["name", "region_name", "mun_district"])\
        .rename(columns={"municipality": "name"})

    if municipalities_df.empty:
        municipalities_df = municipality_values.copy()
        municipalities_df["id"] = municipalities_df.index + 1
    else:
        add_new_values(
            df_from=municipality_values,
            df_to=municipalities_df,
            col_names=["name"]
        )

    units_df = dataframes["units_df"]
    units_values = df[["indicator_unit"]].drop_duplicates().reset_index(drop=True)
    units_values = units_values.rename(columns={"indicator_unit": "name"})
    if units_df.empty:
        units_df = units_values.copy()
        units_df["id"] = units_df.index + 1
    else:
        add_new_values(
            df_from=units_values,
            df_to=units_df,
            col_names=["name"]
        )

    okveds_df = dataframes["okveds_df"]
    if 'okved2' in df.columns:
        okveds_values = df[['okved2']].drop_duplicates().reset_index(drop=True)
        okveds_values = okveds_values.rename(columns={"okved2": "name"})
        if okveds_df.empty:
            okveds_df = okveds_values.copy()
            okveds_df["id"] = okveds_df.index + 1
        else:
            add_new_values(
                df_from=okveds_values,
                df_to=okveds_df,
                col_names=["name"]
            )

    indicators_df = dataframes["indicators_df"]
    cols = ['indicator_code', 'indicator_name', 'indicator_unit']
    if 'okved2' in df.columns:
        cols.append('okved2')
    indicators_values = df[cols].drop_duplicates().reset_index(drop=True)

    indicators_values = indicators_values.merge(
        units_df, how='left', left_on='indicator_unit', right_on='name'
    ).drop(columns=['indicator_unit', 'name']).rename(columns={"id": "unit_id"})

    if 'okved2' in df.columns:
        indicators_values = indicators_values.merge(
            okveds_df, how='left', left_on='okved2', right_on="name"
        ).drop(columns=['okved2']).rename(columns={"id": "okved2_id", "name": "okved2_name"})

        indicators_values['indicator_code'] = indicators_values['indicator_code'] + '_' + indicators_values['okved2_id'].astype('str')
        indicators_values['indicator_name'] = indicators_values['indicator_name'] + '(' + indicators_values['okved2_name'] + ')'

    indicators_values = indicators_values.drop(columns=["okved2_name", "okved2_id"], errors="ignore").rename(
        columns={"indicator_code": "code", "indicator_name": "name"}
    )

    if indicators_df.empty:
        indicators_df = indicators_values.copy()
        indicators_df["id"] = indicators_df.index + 1
    else:
        add_new_values(
            df_from=indicators_values,
            df_to=indicators_df,
            col_names=["name"]
        )

    indicators_values_df = dataframes["indicators_values_df"]
    cols = ['indicator_code', 'region_id', 'municipality', 'year', 'indicator_value']
    if "okved2" in df.columns:
        cols.append("okved2")

    indicators_data = df[cols]
    indicators_data = indicators_data.rename(columns={'municipality': 'name'})\
        .merge(municipalities_df, how='right').drop(columns=['name'])

    if "okved2" in df.columns:
        cols.append("okved2")
        indicators_data = indicators_data.merge(okveds_df, how='left', left_on='okved2', right_on='name')\
            .rename(columns={'id_x': 'id', 'id_y': 'okved_id'})

        indicators_data['indicator_code'] = indicators_data['indicator_code'] + '_' + indicators_data['okved_id'].astype(str)

    indicators_data = indicators_data.drop(columns=['okved2', 'region_id', 'name', 'okved_id'], errors="ignore")

    year_idx = indicators_data.groupby(['indicator_code', 'id'])['year'].idxmax()
    indicators_data = indicators_data.loc[year_idx]\
        .drop(columns=['year']).rename(columns={'id': 'municipality_id'})

    indicators_data = indicators_data.pivot(
        index='municipality_id', columns='indicator_code', values='indicator_value'
    ).reset_index()

    indicators_data.columns.name = None

    if indicators_values_df.empty:
        indicators_values_df = indicators_data.copy()
    else:
        indicators_values_df = indicators_values_df.merge(
            indicators_data,
            on='municipality_id', 
            how='left',
        )


def parse_mun_data(root_dir):

    dataframes = {
        "indicators_df": pd.DataFrame(),
        "indicators_values_df": pd.DataFrame(),
        "regions_df": pd.DataFrame(),
        "municipalities_df": pd.DataFrame(),
        "okveds_df": pd.DataFrame(),
        "units_df": pd.DataFrame(),
    }

    folder_path = os.path.join(root_dir, "datasets", "mun_data")
    if not os.listdir(folder_path):
        download_mun_data(folder_path)

    for zipfile in tqdm(os.listdir(folder_path)):
        if zipfile.endswith(".zip"):
            extracted_files = []
            full_file_path = os.path.join(folder_path, zipfile)

            with ZipFile(full_file_path) as archive:
                for file_info in archive.infolist():
                    target_path = os.path.join(folder_path, file_info.filename)
                    extracted_files.append(target_path)

                archive.extractall(folder_path)

            file_name = f"data_Y4{zipfile.replace(".zip", "")}_112_v20250918.xlsx"
            try:
                df = pd.read_excel(os.path.join(folder_path, file_name))
            except FileNotFoundError:
                file_name = file_name.replace(".xlsx", ".csv")
                df = pd.read_csv(os.path.join(folder_path, file_name), sep=';')

            preprocess_mun_df(df, dataframes)

            try:
                for file in extracted_files:
                    if os.path.isdir(file):
                        shutil.rmtree(file)
                    else:
                        os.remove(file)

                parts_folder = f"data_Y4{zipfile.replace(".zip", "")}_parts"
                if parts_folder in os.listdir(folder_path):
                    os.rmdir(os.path.join(folder_path, parts_folder))
            except FileNotFoundError:
                continue

    for k, v in dataframes.items():
        if not v.empty:
            v.to_csv(
                os.path.join(folder_path, f'{k}.csv'),
                sep=';',
                index=False,
                encoding='utf-8-sig'
            )


if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parents[2]
    parse_mun_data(root_dir)
