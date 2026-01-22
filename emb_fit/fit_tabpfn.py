import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
from tqdm import tqdm
import pickle

from utils import load_dataset


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

    with open(output_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f"Модель сохранена: {output_path}")

    return model, target_scaler


def get_tabpfn_embeddings(
    model,
    X,
    batch_size: int = 10000,
    average_embeddings: bool = True
):
    """
    Генерирует эмбеддинги с помощью обученной TabPFN модели.

    Args:
        model: Обученная TabPFN модель
        X: pandas DataFrame или numpy array с данными
        batch_size: Размер батча для обработки
        average_embeddings: Если True, усредняет эмбеддинги по батчам

    Returns:
        embeddings: numpy array с эмбеддингами формы (n_samples, embedding_dim)
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

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
        return np.vstack(embeddings)
    else:
        return np.array([])


if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parent.parent
    df_spb = load_dataset(Path(os.path.join(root_dir, 'datasets/spb_merged.csv')))
    df_msk = load_dataset(Path(os.path.join(root_dir, 'datasets/moscow_merged.csv')))
    df_ekb = load_dataset(Path(os.path.join(root_dir, 'datasets/ekb_merged.csv')))

    cols = df_spb.columns
    df_ekb = df_ekb[[col for col in cols if col in df_ekb.columns]]
    df_msk = df_msk[[col for col in cols if col in df_msk.columns]]
    df = pd.concat([df_spb, df_ekb, df_msk], axis=0)

    feature_start_index = df.columns.get_loc('mun_district')

    X = df.iloc[:, feature_start_index + 1:]
    y = df['price']

    X = X.reset_index().drop(columns=['index'])
    y = y.reset_index().drop(columns=['index'])

    X = X.dropna()
    y = y[y.index.isin(X.index)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, random_state=42, train_size=0.7
    )

    out_path = os.path.join(root_dir, "emb_fit/tab_pfn/tabpfn_model.pkl")
    model, scaler = train_tabpfn(X_train, y_train['price'], output_path=out_path)
    tabpfn_embs = get_tabpfn_embeddings(model, X_test)

    np.save(os.path.join(root_dir, 'emb_fit/tab_pfn/tab_pfn_embs.npy'), tabpfn_embs)
    np.save(os.path.join(root_dir, 'emb_fit/x_test_index.npy'), X_test.index.to_numpy())
