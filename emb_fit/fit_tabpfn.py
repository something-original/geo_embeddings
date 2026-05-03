import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tabpfn import TabPFNRegressor
from tqdm import tqdm
import pickle
from typing import Any


def train_tabpfn(
    X_train,
    y_train,
    output_path: str = "emb_fit/tabpfn_model.pkl",
    max_train_samples: int = 10000,
    ignore_pretraining_limits: bool = True,
    columns_to_drop: list = None,
    random_state: int = 42
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

    # Удаляем указанные колонки
    if columns_to_drop:
        X_train = X_train.drop(columns=[col for col in columns_to_drop if col in X_train.columns], errors='ignore')

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
    y_train_scaled = target_scaler.fit_transform(y_train_log.reshape(-1, 1)).flatten()

    # Создаём и обучаем модель
    model = TabPFNRegressor(
        ignore_pretraining_limits=ignore_pretraining_limits,
        random_state=42,
        device='cuda'
    )
    print("Начинаем обучение TabPFN...")
    model.fit(X_train, y_train_scaled)
    print("Обучение завершено")

    # Сохраняем модель и scaler
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'model': model,
        'scaler': target_scaler,
        'columns_to_drop': columns_to_drop,
        'max_train_samples': max_train_samples
    }

    output_path = os.path.join(output_path, 'tabpfn.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f"Модель сохранена: {output_path}")

    return model, target_scaler


def get_tabpfn_embeddings(
    model,
    X,
    embs_save_path: str,
    batch_size: int = 10000,
    average_embeddings: bool = True,
    feature_scaler: Any | None = None,
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
        nan_mask = X.isnull()
        X_filled = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
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
        np.save(embs_save_path, tabpfn_embs)
