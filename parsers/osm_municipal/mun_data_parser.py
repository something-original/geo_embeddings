import asyncio
import aiohttp
import os
import glob
import shutil
import polars as pl

import tempfile
import logging
from pathlib import Path
from dotenv import load_dotenv
import fastzipfile  # noqa: F401
from zipfile import ZipFile
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def download_all_files(base_download_url, mun_datasets_df):
    tasks = []
    timeout = aiohttp.ClientTimeout(total=600)
    semaphore = asyncio.Semaphore(10)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for section_id, dataset_code in tqdm(
            zip(
                mun_datasets_df['Код раздела'],
                mun_datasets_df['Код показателя']
            )
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
    mun_datasets_df = pl.read_csv(mun_datasets_df_path, separator=";")

    asyncio.run(download_all_files(base_download_url, mun_datasets_df))


class FeatureStore:
    def __init__(self, name, columns, unique_column_name, use_tempfile=False):
        self.name = name
        self.columns = columns
        self.unique_column_name = unique_column_name
        self.unique_column_values = set()
        self.dfs = []
        self.use_tempfile = use_tempfile
        self.temp_dir = tempfile.TemporaryDirectory(prefix=f"store_{name}_") if use_tempfile else None
        self.next_id = 1

    def preprocess_feature(
        self,
        df: pl.DataFrame,
        merge_dfs: list[pl.DataFrame] | None = None,
        merge_specs: list[dict] | None = None
    ):
        if df.is_empty():
            return
        uc = self.unique_column_name
        uc_list = [uc] if isinstance(uc, str) else uc
        keys = df.select(uc_list).unique().iter_rows()
        new_keys = set()
        for k in keys:
            t = tuple(k)
            if t not in self.unique_column_values:
                new_keys.add(t)
        if not new_keys:
            return

        mask = None
        for i, c in enumerate(uc_list):
            vals = [k[i] for k in new_keys]
            m = pl.col(c).is_in(vals)
            mask = m if mask is None else mask & m
        new_df = df.filter(mask).select(self.columns)

        if merge_dfs and merge_specs:
            for i, mdf in enumerate(merge_dfs):
                if mdf.is_empty():
                    continue
                spec = merge_specs[i]
                left_on = spec.get("left_on")
                right_on = spec.get("right_on")
                if left_on and right_on:
                    suffix = f"_m{i}"
                    new_df = new_df.join(
                        mdf,
                        left_on=left_on,
                        right_on=right_on,
                        how="left",
                        suffix=suffix
                    )

        n = len(new_df)
        if n == 0:
            return
        ids = pl.int_range(self.next_id, self.next_id + n, dtype=pl.UInt32)
        new_df = new_df.with_columns(ids.alias("id"))
        self.next_id += n
        self.unique_column_values.update(new_keys)

        if self.use_tempfile:
            p = os.path.join(self.temp_dir.name, f"{len(self.dfs)}.parquet")
            new_df.write_parquet(p)
            self.dfs.append(p)
        else:
            self.dfs.append(new_df)

    def _get_current_df(self) -> pl.DataFrame:
        if self.use_tempfile:
            if not self.dfs:
                return pl.DataFrame()
            return pl.scan_parquet(self.dfs).collect()
        if not self.dfs:
            return pl.DataFrame()
        return pl.concat(self.dfs, how="diagonal")

    def finalize(self) -> pl.DataFrame:
        return self._get_current_df()


def parse_mun_data(root_dir):
    logger.info("Start")

    folder_path = os.path.join(root_dir, "datasets", "mun_data")
    if not os.listdir(folder_path):
        logger.info("Downloading files")
        download_mun_data(root_dir)
    else:
        logger.info("Not downloading files")

    regions_store = FeatureStore("regions", ["id", "name"], "name")
    municipalities_store = FeatureStore("municipalities", ["id", "region_id", "name"], "name")
    units_store = FeatureStore("units", ["id", "name"], "name")
    base_indicators_store = FeatureStore("base_indicators", ["id", "name", "unit_id"], ["name", "unit_id"])
    indicators_store = FeatureStore("indicators", ["id", "code", "name"], "code")
    indicator_values_store = FeatureStore(
        "indicator_values",
        ["municipality_id", "code", "value"],
        ["municipality_id", "code"],
        use_tempfile=True
    )

    special_stores = {}

    zip_files = [f for f in os.listdir(folder_path) if f.endswith(".zip")]

    for zf_name in tqdm(zip_files, desc="Processing files"):

        full_path = os.path.join(folder_path, zf_name)
        base_name = os.path.splitext(zf_name)[0]
        file_name_xlsx = f"data_Y4{base_name}_112_v20250918.xlsx"
        parts_folder = f"data_Y4{base_name}_parts"
        extracted_files = []

        try:
            with ZipFile(full_path) as archive:
                for info in archive.infolist():
                    extracted_files.append(os.path.join(folder_path, info.filename))
                archive.extractall(folder_path)
        except Exception as e:
            logger.error(f"Failed to extract {zf_name}: {e}")
            continue

        df = None
        try:
            df = pl.read_excel(os.path.join(folder_path, file_name_xlsx))
        except Exception:
            try:
                df = pl.scan_csv(os.path.join(folder_path, file_name_xlsx.replace(".xlsx", ".csv")), separator=";").collect()
            except Exception:
                dfs_folder = os.path.join(folder_path, parts_folder)
                if os.path.isdir(dfs_folder):
                    files = glob.glob(os.path.join(dfs_folder, "*.parquet"))
                    if files:
                        df = pl.scan_parquet(files).collect()

        if df is None or df.is_empty():
            continue

        cols_to_drop = [
            "indicator_section_code", "indicator_section", "indicator_period",
            "oktmo_stable", "oktmo_history", "oktmo_year_from", "oktmo_year_to",
            "mun_type", "mun_type_oktmo", "comment", "mun_level", "oktmo"
        ]
        df = df.drop([c for c in cols_to_drop if c in df.columns])

        region_df = df.select(pl.col("region_name").alias("name")).unique()
        regions_store.preprocess_feature(region_df)

        unit_df = df.select(pl.col("indicator_unit").alias("name")).unique()
        units_store.preprocess_feature(unit_df)

        mun_df = df.filter(pl.col("municipality") != pl.col("mun_district")) \
                   .select(pl.col("municipality").alias("name")) \
                   .unique()
        municipalities_store.preprocess_feature(mun_df)

        base_cols = [
            "region_id", "region_name", "municipality", "mun_district",
            "indicator_code", "indicator_name", "indicator_unit", "indicator_value",
            "year"
        ]
        special_cols = [c for c in df.columns if c not in base_cols]

        for sc in special_cols:
            if sc not in special_stores:
                special_stores[sc] = FeatureStore(sc, ["id", "name"], "name")
            special_stores[sc].preprocess_feature(df.select(pl.col(sc).alias("name")).unique())

        base_ind_df = df.select("indicator_name", "indicator_unit") \
                        .rename({"indicator_unit": "unit_id_placeholder"})
        base_ind_df = base_ind_df.join(units_store.finalize(), left_on="unit_id_placeholder", right_on="name", how="left") \
                                 .select(pl.col("indicator_name").alias("name"), pl.col("id").alias("unit_id")) \
                                 .unique()
        base_indicators_store.preprocess_feature(base_ind_df)

        ind_construct_df = df.select("indicator_name", "indicator_unit", *special_cols) \
                             .unique()
        ind_construct_df = ind_construct_df.join(base_indicators_store.finalize(), left_on="indicator_name", right_on="name", how="left")
        ind_construct_df = ind_construct_df.drop("name")

        merge_list = []
        merge_specs = []
        for sc in special_cols:
            store = special_stores[sc]
            final = store.finalize()
            if not final.is_empty():
                merge_list.append(final)
                merge_specs.append({"left_on": sc, "right_on": "name"})

        if merge_list:
            ind_construct_df = ind_construct_df.join(
                pl.concat(merge_list, how="diagonal"),
                left_on=[m["left_on"] for m in merge_specs],
                right_on=[m["right_on"] for m in merge_specs],
                how="left"
            )

        for sc in special_cols:
            if f"{sc}_right" in ind_construct_df.columns:
                ind_construct_df = ind_construct_df.drop([sc, f"{sc}_right"])
            elif sc in ind_construct_df.columns:
                ind_construct_df = ind_construct_df.drop(sc)

        base_final = base_indicators_store.finalize()
        if "indicator_name" in ind_construct_df.columns and "id" in base_final.columns:
            ind_construct_df = ind_construct_df.join(
                base_final.select("id", "name"),
                left_on="indicator_name",
                right_on="name",
                how="left",
                suffix="_base"
            )

        if special_cols:
            code_expr = pl.lit("ind") + pl.col("id").cast(pl.String)
            name_expr = pl.col("indicator_name").fill_null(pl.col("name_base")).cast(pl.String)
            for sc in special_cols:
                id_col = f"{sc}_id_right" if f"{sc}_id_right" in ind_construct_df.columns else f"id_{sc}"
                if id_col in ind_construct_df.columns:
                    code_expr = code_expr + pl.lit(f"_{sc}_") + pl.col(id_col).cast(pl.String)
                    name_expr = name_expr + pl.lit(" (") + pl.col(
                        f"{sc}_right" if f"{sc}_right" in ind_construct_df.columns else sc
                    ).cast(pl.String) + pl.lit(")")

            ind_construct_df = ind_construct_df.with_columns(code_expr.alias("code"), name_expr.alias("final_name"))
        else:
            ind_construct_df = ind_construct_df.with_columns(
                pl.lit("ind") + pl.col("id").cast(pl.String).alias("code"),
                pl.col("indicator_name").fill_null(pl.col("name_base")).alias("final_name")
            )

        ind_construct_df = ind_construct_df.select(pl.col("id").alias("base_id"), "code", "final_name").unique()
        indicators_store.preprocess_feature(ind_construct_df.select("code", pl.col("final_name").alias("name"), pl.col("base_id")))

        values_df = df.select("municipality", "indicator_name", "indicator_unit", "indicator_value", *special_cols)
        values_df = values_df.join(
            municipalities_store.finalize(),
            left_on="municipality",
            right_on="name",
            how="left"
        )
        values_df = values_df.drop(["municipality", "name"])
        values_df = values_df.join(
            base_final,
            left_on=["indicator_name", "indicator_unit"],
            right_on=["name", "unit_id"],
            how="left",
            suffix="_base"
        )
        values_df = values_df.drop(["indicator_name", "indicator_unit", "name", "unit_id_base"])

        if special_cols:
            merge_list_v = []
            merge_specs_v = []
            for sc in special_cols:
                s = special_stores[sc].finalize()
                if not s.is_empty():
                    merge_list_v.append(s)
                    merge_specs_v.append({"left_on": sc, "right_on": "name"})
            if merge_list_v:
                values_df = values_df.join(
                    pl.concat(merge_list_v, how="diagonal"),
                    left_on=[m["left_on"] for m in merge_specs_v],
                    right_on=[m["right_on"] for m in merge_specs_v],
                    how="left"
                )

            for sc in special_cols:
                id_c = f"{sc}_id_right" if f"{sc}_id_right" in values_df.columns else f"id_{sc}"
                if id_c in values_df.columns and sc in values_df.columns:
                    values_df = values_df.drop([sc, id_c, f"{sc}_right" if f"{sc}_right" in values_df.columns else sc])

            if "id_base" in values_df.columns:
                code_expr = pl.lit("ind") + pl.col("id_base").cast(pl.String)
                for sc in special_cols:
                    id_c = f"{sc}_id_right" if f"{sc}_id_right" in values_df.columns else f"id_{sc}"
                    if id_c in values_df.columns:
                        code_expr = code_expr + pl.lit(f"_{sc}_") + pl.col(id_c).cast(pl.String)
                values_df = values_df.with_columns(code_expr.alias("indicator_code"))
        else:
            values_df = values_df.with_columns(pl.lit("ind") + pl.col("id_base").cast(pl.String).alias("indicator_code"))

        values_df = values_df.join(
            indicators_store.finalize().select("code", "id"),
            left_on="indicator_code",
            right_on="code",
            how="left",
            suffix="_ind"
        )

        values_df = values_df.select("id_right", "indicator_code", "indicator_value").rename({"id_right": "municipality_id"})
        values_df = values_df.drop_nulls(subset=["indicator_value"])
        indicator_values_store.preprocess_feature(values_df)

        for ef in extracted_files:
            if os.path.isdir(ef):
                try:
                    shutil.rmtree(ef)
                except FileNotFoundError:
                    pass
            else:
                try:
                    os.remove(ef)
                except FileNotFoundError:
                    pass

        if os.path.isdir(os.path.join(folder_path, parts_folder)):
            try:
                shutil.rmtree(os.path.join(folder_path, parts_folder))
            except FileNotFoundError:
                pass
        mac_path = os.path.join(folder_path, "__MACOSX")
        if os.path.exists(mac_path):
            try:
                shutil.rmtree(mac_path)
            except FileNotFoundError:
                pass

    for name, store in {
        "regions": regions_store,
        "municipalities": municipalities_store,
        "units": units_store,
        "base_indicators": base_indicators_store,
        "indicators": indicators_store
    }.items():
        final = store.finalize()
        if not final.is_empty():
            final.write_csv(os.path.join(folder_path, f"{name}.csv"), separator=";", include_header=True)

    final_vals = indicator_values_store.finalize()
    if not final_vals.is_empty():
        wide = final_vals.pivot(index="municipality_id", columns="code", values="value", aggregate_function="first")
        wide.write_csv(os.path.join(folder_path, "indicator_values.csv"), separator=";", include_header=True)

    logger.info("Processing complete.")


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[2]
    parse_mun_data(root_dir)
