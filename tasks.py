from abc import ABC
from ast import literal_eval
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


    def __str__(self) -> str:
        return self.task_name

    def _drop_cols(self, columns: list[str]) -> None:
        self.x_train = self.x_train.drop(columns=columns)
        self.x_val = self.x_val.drop(columns=columns)
        self.x_test = self.x_test.drop(columns=columns)

    def prepare_dataset(self) -> None:
        X, y = self._load_dataset()
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=self.val_ratio, random_state=42
        )
        self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(
            self.x_val, self.y_val, test_size=0.5, random_state=42
        )

        self.x_train_geom = self.x_train[self.geom_col]
        self.x_val_geom = self.x_val[self.geom_col]
        self.x_test_geom = self.x_test[self.geom_col]

        self._drop_cols([self.geom_col])

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

        self.x_train = self.x_train.sjoin(embeddings)
        self.x_val = self.x_val.sjoin(embeddings)
        self.x_test = self.x_test.sjoin(embeddings)

        emb_col_set = set(embeddings.columns)
        emb_col_set.remove(emb_geom_col)
        self.features.extend(list(emb_col_set))

        self._drop_cols([self.geom_col])

    def train_and_eval_model(
        self, param_distributions: dict, n_trials: int = 100
    ) -> dict:
        def objective(trial):
            params = {}
            for param_name, param_range in param_distributions.items():
                if isinstance(param_range, (list, tuple)) and len(param_range) == 2:
                    low, high = param_range
                    if isinstance(low, int) and isinstance(high, int):
                        params[param_name] = trial.suggest_int(param_name, low, high)
                    else:
                        params[param_name] = trial.suggest_float(param_name, low, high)
                elif isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_range
                    )
                else:
                    raise ValueError(
                        f"Unsupported parameter range format for {param_name}"
                    )

            model_instance = self.model.__class__(
                **{**self.model.get_params(), **params}
            )
            model_instance.fit(
                self.x_train, self.y_train, cat_features=self.cat_features
            )

            y_pred_val = model_instance.predict(self.x_val)
            val_rmse = root_mean_squared_error(self.y_val, y_pred_val)
            return val_rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_model = self.model.__class__(**{**self.model.get_params(), **best_params})
        best_model.fit(self.x_train, self.y_train, cat_features=self.cat_features)

        y_pred_test = best_model.predict(self.x_test)

        rmse = root_mean_squared_error(self.y_test, y_pred_test)
        mae = mean_absolute_error(self.y_test, y_pred_test)
        r2 = r2_score(self.y_test, y_pred_test)
        pearson_corr, _ = pearsonr(self.y_test, y_pred_test)

        return {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Pearson Correlation": pearson_corr,
            "Best Params": best_params,
        }

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
        from parsers import start

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
            if isinstance(df_fns[col].iloc[0], str):
                df_fns[col] = df_fns[col].apply(literal_eval)
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
