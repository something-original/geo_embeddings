import pandas as pd
import numpy as np
from pathlib import Path


def load_dataset(data_path: Path, target_col: str = None):
    """
    Загружает датасет из файла.

    Args:
        data_path: Путь к файлу данных
        target_col: Название колонки с таргетом (если None, возвращает только X)

    Returns:
        X, y (если target_col указан) или только X
    """
    if data_path.suffix == '.pkl':
        data = pd.read_pickle(data_path)
    elif data_path.suffix == '.csv':
        data = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    if target_col:
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        X = data.drop(columns=[target_col])
        y = data[target_col]
        return X, y
    return data


def load_embeddings(embeddings_dir: Path, id_col: str = 'level_0', prefix: str = 'emb_'):
    """
    Загружает эмбеддинги из директории с .npy файлами.

    Args:
        embeddings_dir: Директория с .npy файлами эмбеддингов
        id_col: Название колонки с ID для сопоставления
        prefix: Префикс для названий колонок эмбеддингов

    Returns:
        DataFrame с эмбеддингами
    """
    embeddings_dict = {}
    for file_path in embeddings_dir.glob('*.npy'):
        file_key = file_path.stem
        embeddings_dict[file_key] = np.load(file_path)

    if not embeddings_dict:
        raise ValueError(f"No .npy files found in {embeddings_dir}")

    emb_dim = next(iter(embeddings_dict.values())).shape[0]
    embeddings_df = pd.DataFrame.from_dict(
        embeddings_dict,
        orient='index',
        columns=[f'{prefix}{i}' for i in range(emb_dim)]
    )
    embeddings_df.index.name = id_col
    embeddings_df = embeddings_df.reset_index()

    return embeddings_df
