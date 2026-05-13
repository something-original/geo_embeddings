from abc import ABC
from ast import literal_eval
import logging
from typing import Any

import geopandas as gpd
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

from pathlib import Path
import os
from shapely import Polygon, wkt

from utils import PathBuilder

logger = logging.getLogger(__name__)


class BaseTask(ABC):
    task_name: str = "BaseTask"

    def __init__(
        self,
        dataset_path: str,
        features: list[str],
        target_col: str,
        geom_col: str,
        val_ratio: float,
        model: Any,
        dataset_link: str | None = None,
        dataset_crs: str = "EPSG:4326",
        cat_features: list[str] = [],
        features_to_drop: list[str] = [],
    ):
        super().__init__()
        self.dataset_link = dataset_link
        self.dataset_path = dataset_path
        self.model = model
        self.features = features
        self.target_col = target_col
        self.geom_col = geom_col
        self.val_ratio = val_ratio
        self.x_train: gpd.GeoDataFrame = None
        self.y_train: gpd.pd.Series = None
        self.x_val: gpd.GeoDataFrame = None
        self.y_val: gpd.pd.Series = None
        self.x_test: gpd.GeoDataFrame = None
        self.y_test: gpd.pd.Series = None

        self.x_train_geom: gpd.GeoSeries = None
        self.x_val_geom: gpd.GeoSeries = None
        self.x_test_geom: gpd.GeoSeries = None

        self.dataset_crs = dataset_crs
        self.crs = "EPSG:4326"
        self.cat_features = cat_features

        self._X_full: gpd.GeoDataFrame | None = None
        self._y_full: pd.Series | None = None

        self._initial_split_index: dict[str, pd.Index] | None = None
        self.features_to_drop = features_to_drop

    def __str__(self) -> str:
        return self.task_name

    def _drop_cols(self, columns: list[str]) -> None:
        self.x_train = self.x_train.drop(columns=columns)
        self.x_val = self.x_val.drop(columns=columns)
        self.x_test = self.x_test.drop(columns=columns)

    def prepare_dataset(self) -> None:
        X, y = self._load_dataset()
        
        not_nan_index = y[~y.isna()].index
        X = X[X.index.isin(not_nan_index)]
        y = y[y.index.isin(not_nan_index)]
        
        self._X_full = X.copy()
        self._y_full = y.copy()

        self._set_splits_from_xy(X, y)
        self._initial_split_index = {
            "train": self.x_train.index.copy(),
            "val": self.x_val.index.copy(),
            "test": self.x_test.index.copy(),
        }

        self.x_train_geom = self.x_train[self.geom_col]
        self.x_val_geom = self.x_val[self.geom_col]
        self.x_test_geom = self.x_test[self.geom_col]

        self._drop_cols([self.geom_col])

    def _set_splits_from_xy(self, X: gpd.GeoDataFrame, y: pd.Series) -> None:
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=self.val_ratio, random_state=42
        )
        self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(
            self.x_val, self.y_val, test_size=0.5, random_state=42
        )

    def reset_splits(self) -> None:
        if self._X_full is None or self._y_full is None:
            raise RuntimeError("Dataset is not prepared. Call prepare_dataset() first.")
        if not self._initial_split_index:
            raise RuntimeError("Initial split cache is missing.")

        X = self._X_full
        y = self._y_full
        self.x_train = X.loc[self._initial_split_index["train"]]
        self.y_train = y.loc[self._initial_split_index["train"]]
        self.x_val = X.loc[self._initial_split_index["val"]]
        self.y_val = y.loc[self._initial_split_index["val"]]
        self.x_test = X.loc[self._initial_split_index["test"]]
        self.y_test = y.loc[self._initial_split_index["test"]]

        self.x_train_geom = self.x_train[self.geom_col]
        self.x_val_geom = self.x_val[self.geom_col]
        self.x_test_geom = self.x_test[self.geom_col]
        self._drop_cols([self.geom_col])

    def resplit_on_index(self, row_index: pd.Index | list) -> None:
        """
        Rebuild train/val/test on a subset of the full dataset (used to align
        baseline splits with rows that have embeddings).
        """
        if self._X_full is None or self._y_full is None:
            raise RuntimeError("Dataset is not prepared. Call prepare_dataset() first.")

        X = self._X_full.loc[row_index]
        y = self._y_full.loc[row_index]
        self._set_splits_from_xy(X, y)

        self.x_train_geom = self.x_train[self.geom_col]
        self.x_val_geom = self.x_val[self.geom_col]
        self.x_test_geom = self.x_test[self.geom_col]
        self._drop_cols([self.geom_col])

    def get_index_with_embeddings(
        self,
        embeddings: gpd.GeoDataFrame,
        emb_geom_col: str,
    ) -> pd.Index:
        if self._X_full is None:
            raise RuntimeError("Dataset is not prepared. Call prepare_dataset() first.")

        Xg = (
            gpd.GeoDataFrame(self._X_full.copy())
            .set_geometry(self.geom_col)
            .set_crs(self.crs)
        )

        joined = Xg.sjoin(embeddings, how="left")
        joined = joined[~joined.index.duplicated(keep='first')]
        joined = joined.drop(columns=["index_right"], errors="ignore")

        emb_cols = [c for c in embeddings.columns if c != emb_geom_col]
        if not emb_cols:
            raise ValueError("Embeddings dataframe has no feature columns")

        valid_mask = joined[emb_cols].notna().any(axis=1)
        return joined.index[valid_mask]

    def add_embeddings(self, embeddings: gpd.GeoDataFrame, emb_geom_col: str) -> None:

        self.x_train[self.geom_col] = self.x_train_geom
        self.x_val[self.geom_col] = self.x_val_geom
        self.x_test[self.geom_col] = self.x_test_geom

        self.x_train = (
            gpd.GeoDataFrame(self.x_train).set_geometry(self.geom_col).set_crs(self.crs)
        )
        self.x_val = (
            gpd.GeoDataFrame(self.x_val).set_geometry(self.geom_col).set_crs(self.crs)
        )
        self.x_test = (
            gpd.GeoDataFrame(self.x_test).set_geometry(self.geom_col).set_crs(self.crs)
        )

        self.x_train = self.x_train.sjoin(embeddings, how='left')
        self.x_val = self.x_val.sjoin(embeddings, how='left')
        self.x_test = self.x_test.sjoin(embeddings, how='left')
        
        self.x_train = self.x_train[~self.x_train.index.duplicated(keep='first')]
        self.x_val = self.x_val[~self.x_val.index.duplicated(keep='first')]
        self.x_test = self.x_test[~self.x_test.index.duplicated(keep='first')]
        
        cols_to_drop = ['index_right', 'municipality_id_left', 'municipality_id_right']
        self.x_train.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        self.x_val.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        self.x_test.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        emb_col_set = set(embeddings.columns)
        emb_col_set.remove(emb_geom_col)
        self.features.extend(list(emb_col_set))

        self._drop_cols([self.geom_col])
        logger.info(f'Features after adding embddings: {self.features}')

    def clear_embeddings(self, embeddings: gpd.GeoDataFrame):
        emb_cols = embeddings.columns
        
        self.x_train.drop(columns=emb_cols, inplace=True, errors='ignore')
        self.x_val.drop(columns=emb_cols, inplace=True, errors='ignore')
        self.x_test.drop(columns=emb_cols, inplace=True, errors='ignore')
        
        self.features = [f for f in self.features if f not in emb_cols]
        self.cat_features = [f for f in self.cat_features if f not in emb_cols]
        logger.info(f'Features after cleaning: {self.features}')

    def _load_dataset(self):
        dataset = gpd.read_file(self.dataset_path)
        if isinstance(dataset[self.geom_col].iloc[0], str):
            dataset[self.geom_col] = dataset[self.geom_col].apply(wkt.loads)

        dataset = gpd.GeoDataFrame(dataset).set_geometry(self.geom_col)
        dataset[self.geom_col] = (
            dataset[self.geom_col].set_crs(self.dataset_crs).to_crs(self.crs)
        )

        for col in dataset.columns:
            if dataset[col].nunique() == 1:
                dataset.drop(columns=[col], inplace=True)
                if col in self.features:
                    self.features.remove(col)
                if col in self.cat_features:
                    self.cat_features.remove(col)
        
        if isinstance(dataset[self.target_col].iloc[0], str):
            dataset[self.target_col] = dataset[self.target_col].apply(float)

        X = dataset[self.features + [self.geom_col]]
        y = dataset[self.target_col]

        return X, y


class FNSTask(BaseTask):
    task_name = "FNS Task"

    def _load_dataset(self):
        cwd = os.getcwd()
        root_dir = Path(__file__).resolve().parent

        os.chdir(os.path.join(root_dir, "parsers", "fns_parser"))
        from parsers.fns_parser.main import start

        save_path_level_10 = os.path.join(self.dataset_path, "fns_level_10.csv")
        os.makedirs(self.dataset_path, exist_ok=True)

        if not os.path.exists(save_path_level_10):
            start(save_path_level_10, resolution=10)

        os.chdir(cwd)

        df_fns = pd.read_csv(save_path_level_10, encoding="utf-8")
        df_fns = df_fns[self.features + [self.geom_col, self.target_col]]

        df_fns[self.geom_col] = df_fns[self.geom_col].apply(
            lambda x: Polygon(literal_eval(x)[0])
        )
        for col in df_fns.columns:
            if col != self.geom_col and col not in self.cat_features and df_fns[col].dtype == 'object':
                df_fns[col] = df_fns[col].apply(
                    lambda x: x if x != x else literal_eval(x)
                )
            if col != self.geom_col and col not in self.cat_features:
                df_fns[col] = df_fns[col].apply(
                    lambda x: x[0] if isinstance(x, list) else x
                )

        X = df_fns[self.features + [self.geom_col]]
        y = df_fns[self.target_col]

        return X, y


class FlatsTask(BaseTask):
    task_name = "Flats price prediciton"


class PeopleHousesTask(BaseTask):
    task_name = "Population prediction"


class WorkplacesDistrictsTask(BaseTask):
    task_name = "Workplaces prediction"


class MunDataTask(BaseTask):
    task_name = "MunDataTask"
    
    def _load_dataset(self):
        mun_df = pd.read_csv(self.dataset_path, sep=';', encoding='utf-8')

        X = mun_df[self.features].copy()
        X = X.drop(columns=self.features_to_drop, errors='ignore')
        y = mun_df[self.target_col].copy()

        return X, y

    def get_index_with_embeddings(
        self,
        embeddings: gpd.GeoDataFrame,
        emb_geom_col: str,
    ) -> pd.Index:

        if self._X_full is None:
            raise RuntimeError("Dataset is not prepared. Call prepare_dataset() first.")

        return self._X_full.index.intersection(embeddings.index)

    def add_embeddings(self, embeddings: gpd.GeoDataFrame, emb_geom_col: str) -> None:

        if not hasattr(self, "_baseline_splits_cache") or self._baseline_splits_cache is None:
            self._baseline_splits_cache = {
                "x_train": self.x_train.copy(),
                "x_val": self.x_val.copy(),
                "x_test": self.x_test.copy(),
                "features": list(self.features),
                "cat_features": list(self.cat_features),
            }

        emb_df = embeddings.drop(columns=[emb_geom_col], errors="ignore")
        if emb_df.index.has_duplicates:
            emb_df = emb_df[~emb_df.index.duplicated(keep="first")]

        self.x_train = emb_df.loc[self.x_train.index].copy()
        self.x_val = emb_df.loc[self.x_val.index].copy()
        self.x_test = emb_df.loc[self.x_test.index].copy()

        self.features = list(emb_df.columns)
        self.cat_features = []

    def clear_embeddings(self, embeddings: gpd.GeoDataFrame):

        cache = getattr(self, "_baseline_splits_cache", None)
        if not cache:
            return
        self.x_train = cache["x_train"]
        self.x_val = cache["x_val"]
        self.x_test = cache["x_test"]
        self.features = cache["features"]
        self.cat_features = cache["cat_features"]
        self._baseline_splits_cache = None

    def prepare_dataset(self) -> None:
        X, y = self._load_dataset()
        
        not_nan_index = y[~y.isna()].index
        X = X[X.index.isin(not_nan_index)]
        y = y[y.index.isin(not_nan_index)]
        
        self._X_full = X.copy()
        self._y_full = y.copy()

        self._set_splits_from_xy(X, y)
        self._initial_split_index = {
            "train": self.x_train.index.copy(),
            "val": self.x_val.index.copy(),
            "test": self.x_test.index.copy(),
        }
        
        self.features = [f for f in self.features if f not in self.features_to_drop]

    def resplit_on_index(self, row_index: pd.Index | list) -> None:
        """
        Rebuild train/val/test on a subset of the full dataset (used to align
        baseline splits with rows that have embeddings).
        """
        if self._X_full is None or self._y_full is None:
            raise RuntimeError("Dataset is not prepared. Call prepare_dataset() first.")

        X = self._X_full.loc[row_index]
        y = self._y_full.loc[row_index]
        self._set_splits_from_xy(X, y)

    def reset_splits(self) -> None:
        if self._X_full is None or self._y_full is None:
            raise RuntimeError("Dataset is not prepared. Call prepare_dataset() first.")
        if not self._initial_split_index:
            raise RuntimeError("Initial split cache is missing.")

        X = self._X_full
        y = self._y_full
        self.x_train = X.loc[self._initial_split_index["train"]]
        self.y_train = y.loc[self._initial_split_index["train"]]
        self.x_val = X.loc[self._initial_split_index["val"]]
        self.y_val = y.loc[self._initial_split_index["val"]]
        self.x_test = X.loc[self._initial_split_index["test"]]
        self.y_test = y.loc[self._initial_split_index["test"]]
