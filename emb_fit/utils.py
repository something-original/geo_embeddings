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
    emb_dataset_paths: dict[str, str],
    separate_inference: bool,
    features_to_drop: list[str],
    index_feature: str,
    experiment_target_features: list[str],
    csv_sep: str = ';',
    use_scaler: bool = False,
    scaler: StandardScaler | None = None,
    test_size: float = 0.25,
    nan_fill_threshold: float = 0.9,
    use_modifications: bool = False,
    modification_paths: list[str] = None,
    modification_values: list[str] = None,
    modification_names: list[str] = None,
) -> list[str]:

    train_path = emb_dataset_paths['train_path']
    val_path = emb_dataset_paths['val_path']
    inference_full_path = emb_dataset_paths['inference_full_path']
    dataset_path_old = emb_dataset_paths['dataset_path_old']
    dataset_path_new = emb_dataset_paths['dataset_path']
    indicators_path = emb_dataset_paths['indicators_path']

    df_new = pd.read_csv(dataset_path_new, sep=csv_sep)

    if separate_inference:
        df_old = pd.read_csv(dataset_path_old, sep=csv_sep)
        df_old.drop(columns=features_to_drop, inplace=True, errors='ignore')
        df_old = df_old[df_old[index_feature].isin(df_new[index_feature])]
        logger.info(f'OLD shape: {df_old.shape}, NEW shape: {df_new.shape}')
        mask_all_nan_old = df_old.isna().all(axis=1)
        df_old = df_old[~mask_all_nan_old].reset_index(drop=True)
        df_old = df_old.select_dtypes(include=[np.number])
        df_old = df_old.replace([np.inf, -np.inf], np.nan)

        if index_feature not in df_old.columns:
            raise ValueError(f'{index_feature} must be a column in old indicator CSV')

    df_new.drop(columns=features_to_drop, inplace=True, errors='ignore')

    mask_all_nan_new = df_new.isna().all(axis=1)

    df_new = df_new[~mask_all_nan_new].reset_index(drop=True)
    df_new = df_new.select_dtypes(include=[np.number])
    df_new = df_new.replace([np.inf, -np.inf], np.nan)

    if index_feature not in df_new.columns:
        raise ValueError(f'{index_feature} must be a column in both old and new indicator CSVs')

    if separate_inference:
        common_ids = sorted(set(df_old[index_feature]) & set(df_new[index_feature]))
        df_old = df_old[df_old[index_feature].isin(common_ids)].sort_values(index_feature).reset_index(drop=True)
        df_new = df_new[df_new[index_feature].isin(common_ids)].sort_values(index_feature).reset_index(drop=True)

        all_cols = sorted(set(df_old.columns) | set(df_new.columns))
        for c in all_cols:
            if c not in df_old.columns:
                df_old[c] = np.nan
            if c not in df_new.columns:
                df_new[c] = np.nan
        df_old = df_old[all_cols]
        df_new = df_new[all_cols]

    logger.info(f'Nan threshold: {nan_fill_threshold}')
    if separate_inference:
        nan_ratio_old = df_old.isnull().mean()
        cols_to_drop = nan_ratio_old[nan_ratio_old >= nan_fill_threshold].index
    else:
        nan_ratio_new = df_new.isnull().mean()
        cols_to_drop = nan_ratio_new[nan_ratio_new >= nan_fill_threshold].index
    cols_to_drop = [c for c in cols_to_drop if c != index_feature]

    if len(cols_to_drop):
        logger.info(f'Columns dropped (>= {nan_fill_threshold:.0%} NaN in OLD): {len(cols_to_drop)}')
        if separate_inference:
            df_old = df_old.drop(columns=cols_to_drop, errors='ignore')
        df_new = df_new.drop(columns=cols_to_drop, errors='ignore')

    df_base_indicators = pd.read_csv(indicators_path, sep=csv_sep)
    if use_modifications:
        mod_dfs = [pd.read_csv(mod_path, sep=csv_sep) for mod_path in modification_paths]

    target_feature_col_names: list[str] = []

    if experiment_target_features:
        logger.info(f'Target features: {experiment_target_features}')
        target_feature_ids = df_base_indicators[
            df_base_indicators['name'].isin(experiment_target_features)
        ]['id'].values

        if use_modifications:
            mod_ids = [
                mod_df[mod_df['name'] == mod_value]['id'].iloc[0]
                for mod_df, mod_value in zip(mod_dfs, modification_values)
            ]

        for i in range(0, len(target_feature_ids)):

            base_col_name = f'ind{str(target_feature_ids[i])}'
            if use_modifications:
                target_col_name = base_col_name + f'_{modification_names[i]}{mod_ids[i]}'
            else:
                target_col_name = base_col_name

            target_feature_col_names.append(target_col_name)

            if separate_inference:
                id_cols_old = [col for col in df_old.columns if base_col_name in col and target_col_name not in col]
                if df_old[target_col_name].isna().any():
                    mask = df_old[target_col_name].isna()
                    values_to_set = np.nansum(df_old.loc[mask, id_cols_new].values, axis=1)
                    df_old.loc[mask, target_col_name] = values_to_set
                df_old.drop(columns=id_cols_old, inplace=True)

            id_cols_new = [col for col in df_new.columns if base_col_name in col and target_col_name not in col]
            if df_new[target_col_name].isna().any():
                mask = df_new[target_col_name].isna()
                values_to_set = np.nansum(df_new.loc[mask, id_cols_new].values, axis=1)
                df_new.loc[mask, target_col_name] = values_to_set

            df_new.drop(columns=id_cols_new, inplace=True)
    else:
        target_feature_col_names = [
            c for c in df_new.columns if str(c).startswith('target_')
        ]
        if separate_inference:
            target_feature_col_names = sorted(
                set(target_feature_col_names)
                | {c for c in df_old.columns if str(c).startswith('target_')}
            )

    if separate_inference:
        df_old = df_old.set_index(index_feature)

    df_new = df_new.set_index(index_feature)

    df_train, df_val = train_test_split(
        df_old if separate_inference else df_new, test_size=test_size, random_state=42
    )
    logger.info(f'OLD train shape: {df_train.shape}, OLD val shape: {df_val.shape}')

    if not use_scaler or scaler is None:
        raise ValueError('prepare_and_save_dataset_old_new requires use_scaler=True and a StandardScaler instance')

    original_targets_train = df_train[target_feature_col_names].copy()
    original_targets_val = df_val[target_feature_col_names].copy()
    original_targets_inference = df_new[target_feature_col_names].copy()

    df_train_features = df_train.drop(columns=target_feature_col_names)
    df_val_features = df_val.drop(columns=target_feature_col_names)
    df_new_features = df_new.drop(columns=target_feature_col_names)

    feature_medians = df_train_features.median(numeric_only=True)

    train_mask = df_train_features.isnull()
    val_mask = df_val_features.isnull()
    new_mask = df_new_features.isnull()

    df_train_for_scaler = df_train_features.fillna(feature_medians)
    df_val_for_scaler = df_val_features.fillna(feature_medians)
    df_new_for_scaler = df_new_features.fillna(feature_medians)

    train_scaled = pd.DataFrame(
        scaler.fit_transform(df_train_for_scaler),
        columns=df_train_features.columns,
        index=df_train_features.index,
    )
    val_scaled = pd.DataFrame(
        scaler.transform(df_val_for_scaler),
        columns=df_val_features.columns,
        index=df_val_features.index,
    )
    inference_scaled = pd.DataFrame(
        scaler.transform(df_new_for_scaler),
        columns=df_new_features.columns,
        index=df_new_features.index,
    )

    train_scaled_final = np.where(train_mask, -1.0, train_scaled.values)
    val_scaled_final = np.where(val_mask, -1.0, val_scaled.values)
    inference_scaled_final = np.where(new_mask, -1.0, inference_scaled.values)

    df_train_out = pd.DataFrame(
        train_scaled_final,
        columns=df_train_features.columns,
        index=df_train_features.index,
    )
    df_val_out = pd.DataFrame(
        val_scaled_final,
        columns=df_val_features.columns,
        index=df_val_features.index,
    )
    df_new_features_out = pd.DataFrame(
        inference_scaled_final,
        columns=df_new_features.columns,
        index=df_new_features.index,
    )

    df_train_out[target_feature_col_names] = original_targets_train
    df_val_out[target_feature_col_names] = original_targets_val

    df_new_inference_only = df_new_features_out.copy()
    df_new_inference_only[target_feature_col_names] = original_targets_inference

    kwargs = {'sep': csv_sep, 'index': True}
    df_train_out.to_csv(train_path, **kwargs)
    df_val_out.to_csv(val_path, **kwargs)
    df_new_inference_only.to_csv(inference_full_path, **kwargs)

    return target_feature_col_names
