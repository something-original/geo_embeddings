"""
Базовый скрипт для бенчмарка моделей эмбеддингов.

Сравнивает производительность различных моделей эмбеддингов
на различных downstream задачах.

Запуск из корня проекта:
    python benchmarks/benchmark.py
    или
    python -m benchmarks.benchmark
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from models import (
    DeepGNN,
    RasterEmbedder,
    GeoCLIP,
    SatCLIP,
    S2VecModel
)
from utils import load_dataset, load_embeddings


PROJECT_ROOT = Path(__file__).parent.parent


def evaluate_model(X_train, X_test, y_train, y_test, model_name: str = "CatBoost"):
    """
    Обучает модель и оценивает её производительность.

    Args:
        X_train: Обучающие фичи
        X_test: Тестовые фичи
        y_train: Обучающий таргет
        y_test: Тестовый таргет
        model_name: Название модели

    Returns:
        Словарь с метриками
    """

    ignore_cols = ['index', 'level_0', 'Unnamed: 0']
    ignore_cols = [col for col in ignore_cols if col in X_train.columns]

    model = CatBoostRegressor(
        ignored_features=ignore_cols if ignore_cols else None,
        verbose=False,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    mae = np.mean(np.abs(y_test - predictions))

    return {
        'RMSE': rmse,
        'R2': r2,
        'MAE': mae,
        'MSE': mse
    }


def run_benchmark(
    dataset_path: Path,
    target_col: str,
    embeddings_dir: Path = None,
    embeddings_prefix: str = 'emb_',
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Запускает бенчмарк для одной конфигурации.

    Args:
        dataset_path: Путь к датасету
        target_col: Название колонки с таргетом
        embeddings_dir: Директория с эмбеддингами (опционально)
        embeddings_prefix: Префикс для колонок эмбеддингов
        test_size: Размер тестовой выборки
        random_state: Random state для воспроизводимости

    Returns:
        Словарь с результатами
    """
    # Загружаем данные
    X, y = load_dataset(dataset_path, target_col=target_col)

    # Подготовка данных
    if 'level_0' in X.columns:
        X['level_0'] = X['level_0'].astype(str)

    # Загружаем эмбеддинги, если указаны
    if embeddings_dir and embeddings_dir.exists():
        embeddings_df = load_embeddings(embeddings_dir, prefix=embeddings_prefix)

        # Объединяем с основными данными
        merge_col = 'level_0' if 'level_0' in X.columns else embeddings_df.columns[0]
        X = X.merge(embeddings_df, on=merge_col, how='left')

        # Удаляем строки с пропущенными эмбеддингами
        emb_cols = [col for col in X.columns if col.startswith(embeddings_prefix)]
        missing_mask = X[emb_cols].isnull().any(axis=1)
        if missing_mask.any():
            print(f"Warning: {missing_mask.sum()} rows with missing embeddings will be dropped")
            X = X[~missing_mask]
            y = y[~missing_mask]

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Оценка модели
    results = evaluate_model(X_train, X_test, y_train, y_test)

    return results


def main():
    """
    Основная функция для запуска бенчмарка.

    Пример использования:
        python benchmarks/benchmark.py
    """
    # Пути к данным (относительно корня проекта)
    datasets_dir = PROJECT_ROOT / 'datasets' / 'pd_splits'

    print("=" * 80)
    print("Geo-Embeddings Benchmark")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Datasets dir: {datasets_dir}")
    print()

    # Пример 1: Базовый бенчмарк без эмбеддингов
    print("-" * 80)
    print("Benchmark 1: Baseline (no embeddings)")
    print("-" * 80)

    # Загружаем данные (пример с SPB)
    x_train_path = datasets_dir / 'x_train_spb.pkl'
    x_test_path = datasets_dir / 'x_test_spb.pkl'
    y_train_path = datasets_dir / 'y_train_spb.pkl'
    y_test_path = datasets_dir / 'y_test_spb.pkl'

    if all(p.exists() for p in [x_train_path, x_test_path, y_train_path, y_test_path]):
        X_train = pd.read_pickle(x_train_path)
        X_test = pd.read_pickle(x_test_path)
        y_train = pd.read_pickle(y_train_path)
        y_test = pd.read_pickle(y_test_path)

        # Конвертируем y в numpy array если нужно
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        if hasattr(y_test, 'values'):
            y_test = y_test.values

        # Оценка модели
        results = evaluate_model(X_train, X_test, y_train, y_test)

        print(f"RMSE: {results['RMSE']:.4f}")
        print(f"R²:   {results['R2']:.4f}")
        print(f"MAE:  {results['MAE']:.4f}")
    else:
        print(f"Dataset files not found in {datasets_dir}")
        print("Available files:", list(datasets_dir.glob('*.pkl')))

    # Пример 2: Бенчмарк с эмбеддингами EfficientNet
    print("\n" + "-" * 80)
    print("Benchmark 2: With EfficientNet embeddings")
    print("-" * 80)

    # Путь к эмбеддингам (пример)
    embeddings_dir = PROJECT_ROOT / 'map_embeds' / 'maps_emb_msc'

    if embeddings_dir.exists():
        print(f"Embeddings directory found: {embeddings_dir}")
        print(f"Number of embedding files: {len(list(embeddings_dir.glob('*.npy')))}")
        print("\nTo use embeddings in benchmark:")
        print("1. Ensure your dataset has 'level_0' column with IDs matching .npy filenames")
        print("2. Call run_benchmark() with embeddings_dir parameter")
    else:
        print(f"Embeddings directory not found: {embeddings_dir}")
        print("Skipping embeddings benchmark")

    print("\n" + "=" * 80)
    print("Benchmark completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
