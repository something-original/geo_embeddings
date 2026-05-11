import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger(__name__)


class TabularImageDataset(Dataset):
    """
    Датасет для преобразования табличных данных в 'изображения' для S2Vec.
    """
    def __init__(
        self,
        csv_path: str,
        img_size: int = 32,
        fill_value: float = -1.0,
        sep: str = ';',
        cols_to_drop: list[str] | None = None,
    ):
        """
        Args:
            csv_path: Путь к CSV файлу.
            img_size: Размер стороны квадратного 'изображения' (H и W).
                      Общее кол-во признаков должно быть <= img_size**2.
            fill_value: Значение, которым заполняются NaN и недостающие признаки.
        """
        self.img_size = img_size
        self.fill_value = fill_value
        self.total_pixels = img_size * img_size
        self.sep = sep
        self.cols_to_drop = cols_to_drop

        self.df = pd.read_csv(csv_path, sep=self.sep)
        if self.cols_to_drop:
            self.df = self.df.drop(columns=self.cols_to_drop, errors='ignore')
        self.features = self.df.values.astype(np.float32)

        n_features = self.features.shape[1]
        if n_features > self.total_pixels:
            raise ValueError(f"Too many features ({n_features}) for image {img_size}x{img_size}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        row = self.features[idx]

        row = np.nan_to_num(
            row,
            nan=self.fill_value,
            posinf=self.fill_value,
            neginf=self.fill_value
        )

        n_features = len(row)
        if n_features < self.total_pixels:
            padding = np.full((self.total_pixels - n_features,), self.fill_value, dtype=np.float32)
            row = np.concatenate([row, padding])

        image = row.reshape(1, self.img_size, self.img_size)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(0)


def get_dataloader(
    csv_path: str,
    img_size: int = 32,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = False,
    cols_to_drop: list[str] | None = None
):
    dataset = TabularImageDataset(
        csv_path,
        img_size=img_size,
        cols_to_drop=cols_to_drop,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )


def prepare_and_save_dataset(
    dataset_path: str,
    indicators_path: str,
    features_to_drop: list[str],
    index_feature: str,
    experiment_target_features: list[str],
    train_path: str,
    train_full_path: str,
    val_path: str,
    csv_sep: str = ';',
    id_col: str | None = None,
    use_scaler: bool = False,
    scaler: StandardScaler | None = None,
    test_size: float = 0.25,
    nan_fill_threshold: float = 0.9,
) -> list[str]:

    df_full = pd.read_csv(dataset_path, sep=csv_sep)
    df_base_indicators = pd.read_csv(indicators_path, sep=csv_sep)

    logger.info(f'Full dataset shape: {df_full.shape}')
    mask_all_nan = df_full.isna().all(axis=1)

    if id_col:
        dropped_ids = df_full.loc[mask_all_nan, id_col].tolist()
        logger.info(f'Dropped ids: {dropped_ids}')

    df_full.drop(columns=features_to_drop, inplace=True, errors='ignore')
    df_full = df_full[~mask_all_nan].reset_index(drop=True)
    df_full = df_full.select_dtypes(include=[np.number])
    logger.info(f'Shape after dropping: {df_full.shape}')

    df_full = df_full.replace([np.inf, -np.inf], np.nan)

    logger.info(f'Nan threshold: {nan_fill_threshold}')
    nan_ratio = df_full.isnull().mean()

    cols_to_drop = nan_ratio[nan_ratio > nan_fill_threshold].index
    df_full = df_full.drop(columns=cols_to_drop)
    if cols_to_drop.any():
        print(f"Columns dropped with nan ratios: {len(list(cols_to_drop))}")

    target_feature_col_names = []

    if experiment_target_features:
        logger.info(f'Target features: {experiment_target_features}')

        target_feature_ids = df_base_indicators[
            df_base_indicators['name'].isin(experiment_target_features)
        ]['id']

        for id in target_feature_ids:
            target_feature_col_name = f'target_{id}'
            target_feature_col_names.append(target_feature_col_name)
            id_cols = [col for col in df_full.columns if f'ind{str(id)}' in col]
            df_full[target_feature_col_name] = np.nansum(df_full[id_cols].values, axis=1)
            df_full.drop(columns=id_cols, inplace=True)
    df_full = df_full.set_index(index_feature)

    df_train, df_val = train_test_split(df_full, test_size=test_size, random_state=42)
    logger.info(f'Train shape: {df_train.shape}')
    logger.info(f'Val shape: {df_val.shape}')

    if use_scaler and scaler is not None:
        # IMPORTANT:
        # - scaler is fit ONLY on train split
        # - NaN filling uses ONLY train medians (so val/full get consistent preprocessing)
        # - NaN positions are restored as sentinel -1.0 after scaling

        original_targets_train = df_train[target_feature_col_names].copy()
        original_targets_val = df_val[target_feature_col_names].copy()
        original_targets_full = df_full[target_feature_col_names].copy()

        df_train_features = df_train.drop(columns=target_feature_col_names)
        df_val_features = df_val.drop(columns=target_feature_col_names)
        df_full_features = df_full.drop(columns=target_feature_col_names)

        # Use medians computed on train ONLY for filling missing features
        feature_medians = df_train_features.median(numeric_only=True)

        train_mask = df_train_features.isnull()
        val_mask = df_val_features.isnull()
        full_mask = df_full_features.isnull()

        df_train_for_scaler = df_train_features.fillna(feature_medians)
        df_val_for_scaler = df_val_features.fillna(feature_medians)
        df_full_for_scaler = df_full_features.fillna(feature_medians)

        train_scaled = pd.DataFrame(
            scaler.fit_transform(df_train_for_scaler),
            columns=df_train_features.columns,
            index=df_train_features.index
        )
        val_scaled = pd.DataFrame(
            scaler.transform(df_val_for_scaler),
            columns=df_val_features.columns,
            index=df_val_features.index
        )
        full_scaled = pd.DataFrame(
            scaler.transform(df_full_for_scaler),
            columns=df_full_features.columns,
            index=df_full_features.index
        )

        train_scaled_final = np.where(train_mask, -1.0, train_scaled.values)
        val_scaled_final = np.where(val_mask, -1.0, val_scaled.values)
        full_scaled_final = np.where(full_mask, -1.0, full_scaled.values)

        df_train = pd.DataFrame(
            train_scaled_final,
            columns=df_train_features.columns,
            index=df_train_features.index,
        )
        df_val = pd.DataFrame(
            val_scaled_final,
            columns=df_val_features.columns,
            index=df_val_features.index,
        )
        df_full = pd.DataFrame(
            full_scaled_final,
            columns=df_full_features.columns,
            index=df_full_features.index,
        )

        # Reattach targets (without scaling)
        df_train[target_feature_col_names] = original_targets_train
        df_val[target_feature_col_names] = original_targets_val
        df_full[target_feature_col_names] = original_targets_full

    kwargs = {'sep': csv_sep, 'index': True}
    df_train.to_csv(train_path, **kwargs)
    df_val.to_csv(val_path, **kwargs)
    df_full.to_csv(train_full_path, **kwargs)

    return target_feature_col_names
