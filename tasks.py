from abc import ABC, abstractmethod
from typing import Any

import geopandas as gpd
from sklearn.model_selection import train_test_split
import optuna
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


class BaseTask(ABC):
    def __init__(
        self,
        dataset_link: str,
        dataset_path: str,
        features: list[str],
        target_col: str,
        geom_col: str,
        val_ratio: float,
        model: Any,
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
        self.crs = 'EPSG:4326'

    @abstractmethod
    def _load_dataset(self) -> tuple[Any, Any]:
        pass

    def prepare_dataset(self) -> None:
        X, y = self._load_dataset()
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=self.val_ratio, random_state=42
        )
        self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(
            self.x_val, self.y_val, test_size=0.5, random_state=42
        )
        self.x_train = gpd.GeoDataFrame(self.x_train).set_geometry(self.geom_col).set_crs(self.crs)
        self.x_val = gpd.GeoDataFrame(self.x_val).set_geometry(self.geom_col).set_crs(self.crs)
        self.x_test = gpd.GeoDataFrame(self.x_test).set_geometry(self.geom_col).set_crs(self.crs)

    def add_embeddings(
        self,
        embeddings: gpd.GeoDataFrame,
        emb_geom_col: str
    ) -> None:
        self.x_train = self.x_train.sjoin(embeddings)
        self.x_val = self.x_val.sjoin(embeddings)
        self.x_test = self.x_test.sjoin(embeddings)
        emb_col_set = set(embeddings.columns)
        emb_col_set.remove(emb_geom_col)
        self.features.extend(list(emb_col_set))

    def train_and_eval_model(self, param_distributions: dict, n_trials: int = 100) -> dict:
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
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
                else:
                    raise ValueError(f"Unsupported parameter range format for {param_name}")

            model_instance = self.model.__class__(**{**self.model.get_params(), **params})
            model_instance.fit(self.x_train, self.y_train)

            y_pred_val = model_instance.predict(self.x_val)
            val_rmse = root_mean_squared_error(self.y_val, y_pred_val, squared=False)
            return val_rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_model = self.model.__class__(**{**self.model.get_params(), **best_params})
        best_model.fit(self.x_train, self.y_train)

        y_pred_test = best_model.predict(self.x_test)

        rmse = root_mean_squared_error(self.y_test, y_pred_test, squared=False)
        mae = mean_absolute_error(self.y_test, y_pred_test)
        r2 = r2_score(self.y_test, y_pred_test)
        pearson_corr, _ = pearsonr(self.y_test, y_pred_test)

        return {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Pearson Correlation": pearson_corr,
            "Best Params": best_params
        }


class FlatsTask(BaseTask):
    def _load_dataset(self):
        pass


if __name__ == '__main__':
    from kaggle.api.kaggle_api_extended import KaggleApi

    kaggle.api.dataset_download_files(
        "egorkainov/moscow-housing-price-dataset",
        path="data/",
        unzip=True
    )
    