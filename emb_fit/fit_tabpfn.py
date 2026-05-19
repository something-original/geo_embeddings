import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tabpfn import TabPFNRegressor
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_tabpfn(
    X_train,
    y_train,
    X_val,
    y_val,
    output_path: str = "emb_fit/tabpfn_model.pkl",
    max_train_samples: int = 10000,
    ignore_pretraining_limits: bool = True,
    columns_to_drop: list = None,
    device: str = "cuda",
    random_state: int = 42,
    embed_dims: list[int] | None = None,
    model_name: str = "tabpfn",
):
    """
    Обучает TabPFN модель и сохраняет её.

    Args:
        X_train: pandas DataFrame или numpy array с обучающими признаками
        y_train: pandas Series или numpy array с целевыми значениями
        output_path: Путь для сохранения модели (по умолчанию в emb_fit/)
        max_train_samples: Максимальное количество образцов для обучения (TabPFN ограничен ~10k)
        ignore_pretraining_limits: Игнорировать ограничения предобучения
        columns_to_drop: Список колонок для удаления (например, ['level_0', 'index'])
        random_state: Random state для воспроизводимости

    Returns:
        model: Обученная TabPFN модель
        scaler: MinMaxScaler для масштабирования целевой переменной
    """
    # Конвертируем в DataFrame если нужно
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)
    if isinstance(X_val, np.ndarray):
        X_val = pd.DataFrame(X_val)
    if isinstance(y_val, np.ndarray):
        y_val = np.asarray(y_val)
    else:
        y_val = y_val.values

    # Удаляем указанные колонки
    if columns_to_drop:
        X_train = X_train.drop(columns=[col for col in columns_to_drop if col in X_train.columns], errors='ignore')
        X_val = X_val.drop(columns=[col for col in columns_to_drop if col in X_val.columns], errors='ignore')

    # Ограничиваем размер обучающей выборки
    if len(X_train) > max_train_samples:
        print(f"Ограничиваем размер обучающей выборки до {max_train_samples} образцов")
        X_train = X_train.iloc[:max_train_samples]
        y_train = y_train.iloc[:max_train_samples] if isinstance(y_train, pd.Series) else y_train[:max_train_samples]

    print(f"Обучаем TabPFN на {len(X_train)} образцах с {X_train.shape[1]} признаками")

    # Масштабирование целевой переменной (логарифмическое + MinMax)
    y_train_values = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_train_log = np.log10(y_train_values + 1)
    target_scaler = MinMaxScaler(feature_range=(0, 10))
    target_scaler.fit(y_train_log.reshape(-1, 1))
    y_train_scaled = target_scaler.transform(y_train_log.reshape(-1, 1)).flatten()

    # Создаём и обучаем модель
    model = TabPFNRegressor(
        ignore_pretraining_limits=ignore_pretraining_limits,
        random_state=42,
        device=device
    )
    print("Начинаем обучение TabPFN...")
    # Train on transformed target to match the fitted target scaler.
    model.fit(X_train, y_train_scaled)
    print("Обучение завершено")

    y_pred_s = model.predict(X_val)
    y_pred_log = target_scaler.inverse_transform(np.asarray(y_pred_s).reshape(-1, 1)).flatten()
    y_pred = np.power(10.0, y_pred_log) - 1.0
    rmse = root_mean_squared_error(y_val, y_pred)
    logger.info(f"Metric for model {model_name}, RMSE: {rmse}")

    save_dict = {
        'model': model,
        'scaler': target_scaler,
        'columns_to_drop': columns_to_drop,
        'max_train_samples': max_train_samples
    }

    dims = embed_dims if embed_dims else [128]
    root = Path(__file__).resolve().parent / "checkpoints"
    for d in dims:
        stem = f"{model_name}_{d}"
        ck_dir = root / model_name / stem
        ck_dir.mkdir(parents=True, exist_ok=True)
        ck_file = ck_dir / f"{stem}.pkl"
        with open(ck_file, 'wb') as f:
            pickle.dump(save_dict, f)

    print(f"Модель сохранена: checkpoints/{model_name}/...")

    return model, target_scaler


def get_tabpfn_embeddings(
    model,
    X,
    embs_save_path: str,
    batch_size: int = 10000,
    average_embeddings: bool = True,
    feature_scaler: Any | None = None,
    train_medians: Any | None = None,
    output_dim: int | None = None,
):
    """
    Генерирует эмбеддинги с помощью обученной TabPFN модели.

    Args:
        model: Обученная TabPFN модель
        X: pandas DataFrame или numpy array с данными
        batch_size: Размер батча для обработки
        average_embeddings: Если True, усредняет эмбеддинги по батчам

    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if feature_scaler is not None:
        # Mirror preprocessing from `prepare_and_save_dataset`:
        # - fill NaNs for scaler
        # - transform
        # - put sentinel -1.0 back where NaNs were originally
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        nan_mask = X_clean.isnull()
        if train_medians is None:
            # Fallback: dataset-wide median (may slightly leak feature distribution).
            fill_values = X_clean.median(numeric_only=True)
        else:
            fill_values = train_medians
        X_filled = X_clean.fillna(fill_values)
        X_scaled = feature_scaler.transform(X_filled)
        X_scaled = np.where(nan_mask.to_numpy(), -1.0, X_scaled)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    embeddings = []

    for i in tqdm(range(0, len(X), batch_size), desc="Генерация эмбеддингов"):
        batch = X.iloc[i:i + batch_size]
        batch_embeds = model.get_embeddings(batch)

        if average_embeddings:
            # Усредняем эмбеддинги по батчам (если модель возвращает несколько эмбеддингов)
            if len(batch_embeds.shape) > 2:
                batch_embeds = batch_embeds.mean(axis=0)
            embeddings.append(batch_embeds)
        else:
            embeddings.append(batch_embeds)

    if embeddings:
        tabpfn_embs = np.vstack(embeddings)
        if output_dim is not None and tabpfn_embs.shape[1] != output_dim:
            if tabpfn_embs.shape[1] < output_dim:
                print(
                    f"Cannot increase embedding dim from {tabpfn_embs.shape[1]} to {output_dim}"
                )
                return
            pca = PCA(n_components=output_dim, random_state=42)
            tabpfn_embs = pca.fit_transform(tabpfn_embs)
        np.save(embs_save_path, tabpfn_embs)
