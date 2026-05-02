import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

from config import (
    CHECK_PEOPLE_WORKPLACES_TASKS,
    EXPERIMENT_TARGET_FEATURES,
    PEOPLE_FEATURES,
    PEOPLE_CAT_FEATURES,
    WORKPLACES_FEATURES,
    WORKPLACES_CAT_FEATURES
)

from emb_fit import (
    get_dataloader,
    get_gnn_embeddings,
    get_tabpfn_embeddings,
    get_s2vec_embeddings,
    get_satclip_embeddings,
    prepare_and_save_dataset,
    train_gnn,
    train_tabpfn,
    train_s2vec,
)

from tasks import (
    BaseTask,
    FlatsTask,
    FNSTask,
    PeopleHousesTask,
    WorkplacesDistrictsTask
)

from utils import get_geometry_points

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

root_dir = Path(__file__).resolve().parent

path_parts = [root_dir, 'datasets']

flats_dataset_path = os.path.join(*path_parts, 'flats', 'flats_merged.csv')
fns_dataset_path = os.path.join(*path_parts, 'fns')
people_dataset_path = os.path.join(*path_parts, 'people', 'building_people_dataset.csv')
workplaces_dataset_path = os.path.join(*path_parts, 'workplaces', 'workplaces_districts.csv')
municiplaities_path = os.path.join(*path_parts, 'mun_data', 'municipalities_csv')

model = CatBoostRegressor()

flats_task = FlatsTask(
    dataset_path=flats_dataset_path,
    features=[
        'dealType', 'roomsCount', 'repairType', 'hasFurniture', 'isApartments',
        'floorNumber', 'flatType', 'livingArea', 'windowsViewType', 'balconiesCount',
        'kitchenArea', 'isRecidivist', 'totalArea', 'hasLift', 'buildYear',
        'materialType' ,'distance_to_center', 'highways_count', 'undergrounds_count',
        'railways_count', 'time_to_metro', 'price'
    ],
    geom_col='geometry',
    val_ratio=0.2,
    model=model.copy(),
    target_col='price',
    dataset_crs='EPSG:3857',
    cat_features=[
        'hasFurniture', 'isApartments', 'flatType',
        'isRecidivist', 'hasLift', 'materialType',
        'windowsViewType'
    ]
)

fns_task_kkt = FNSTask(
    dataset_path=fns_dataset_path,
    features=[
        'CacheBillPercent', 'CachePayPercent', 'IntensityOfNumberBills', 'RevenueIntensity',
        'IsMall', 'IsRare', 'IsEcommerce', 'TopCategories',
        'ReceiptTotalCount'
    ],
    cat_features=['IsMall', 'IsRare', 'IsEcommerce'],
    geom_col='coordinates',
    val_ratio=0.2,
    model=model.copy(),
    target_col='KktCount'
)

fns_task_avg_bill = FNSTask(
    dataset_path=fns_dataset_path,
    features=[
        'CacheBillPercent', 'CachePayPercent', 'IntensityOfNumberBills', 'RevenueIntensity',
        'IsMall', 'IsRare', 'IsEcommerce', 'TopCategories',
        'ReceiptTotalCount'
    ],
    cat_features=['IsMall', 'IsRare', 'IsEcommerce'],
    geom_col='coordinates',
    val_ratio=0.2,
    model=model.copy(),
    target_col='AverageBill'
)

tasks: list[BaseTask] = [flats_task, fns_task_kkt, fns_task_avg_bill]

if CHECK_PEOPLE_WORKPLACES_TASKS:
    people_task = PeopleHousesTask(
        dataset_path=people_dataset_path,
        features=PEOPLE_FEATURES,
        target_col='people',
        geom_col='geometry',
        val_ratio=0.2,
        model=model.copy(),
        cat_features=PEOPLE_CAT_FEATURES
    )

    workplaces_task = WorkplacesDistrictsTask(
        dataset_path=workplaces_dataset_path,
        features=WORKPLACES_FEATURES,
        target_col='workplaces',
        geom_col='geometry',
        val_ratio=0.2,
        model=model.copy(),
        cat_features=WORKPLACES_CAT_FEATURES
    )
    
    tasks.extend([people_task, workplaces_task])
    

for task in tasks:
    logger.info(f'Preparing dataset for task: {str(task)}, target: {task.target_col}')
    task.prepare_dataset()


features_to_drop = []
index_feature = 'municipality_id'
scaler = StandardScaler()

path_parts = [root_dir, 'datasets', 'mun_data']
dataset_path = os.path.join(*path_parts, 'indicator_values.csv')
train_path = os.path.join(*path_parts, 'indicator_values_train.csv')
val_path = os.path.join(*path_parts, 'indicator_values_val.csv')
train_full_path = os.path.join(*path_parts, 'indicator_values_full.csv')
indicators_path = os.path.join(*path_parts, 'base_indicators.csv')

target_col_names, train_full_path = prepare_and_save_dataset(
    dataset_path=dataset_path,
    indicators_path=indicators_path,
    features_to_drop=features_to_drop,
    index_feature=index_feature,
    experiment_target_features=EXPERIMENT_TARGET_FEATURES,
    train_path=train_path,
    train_full_path=train_full_path,
    val_path=val_path,
    csv_sep=';',
    use_scaler=True,
    scaler=scaler,
    test_size=0.25,
)


X_train = pd.read_csv(train_path, sep=';')
X_test = pd.read_csv(val_path, sep=';')
X = pd.concat([X_train, X_test], ignore_index=False, axis=0)

target_cols = {}
y_train = X_train[target_col_names[0]].copy()
y_test = X_test[target_col_names[0]].copy()

if len(target_col_names) > 1:
    for col in target_col_names[1:]:
        target_cols[f'{col}_train'] = X_train[col].copy()
        target_cols[f'{col}_test'] = X_test[col].copy()
    
X_train.drop(columns=target_col_names, inplace=True)
X_test.drop(columns=target_col_names, inplace=True)


save_path_parts = [root_dir, 'emb_fit']
deep_gnn_output_path = os.path.join(*save_path_parts, 'gnn')
tabpfn_output_path = os.path.join(*save_path_parts, 'tab_pfn')
s2vec_output_path = os.path.join(*save_path_parts, 's2vec')
satclip_output_path = os.path.join(*save_path_parts, 'satclip')

deep_gnn_model, deep_gnn_scaler = train_gnn(
    X_train=X_train,
    y_train=y_train,
    device=DEVICE,
    X_test=X_test,
    y_test=y_test,
    output_path=deep_gnn_output_path,    
)
tabpfn_model, tabpfn_scaler = train_tabpfn(
    X_train=X_train,
    y_train=y_train,
    output_path=tabpfn_output_path,
)
s2vec_model = train_s2vec(
    train_path=train_path,
    val_path=val_path,
    checkpoint_path=s2vec_output_path,
    device=DEVICE,
)

emb_save_paths = {
    'gnn': os.path.join(deep_gnn_output_path, 'gnn_embs.npy'),
    'tabpfn': os.path.join(tabpfn_output_path, 'tab_pfn_embs.npy'),
    's2vec': os.path.join(s2vec_output_path, 's2vec_embs_npy'),
    'satclip':os.path.join(satclip_output_path, 'satclip_embs.npy')
}

s2vec_checkpoint_save_path = os.path.join(s2vec_output_path, 'satclip-resnet18-l40.ckpt')

s2vec_data_loader = get_dataloader(
    csv_path=train_full_path
)

mun_geometry_points = get_geometry_points(municiplaities_path)

get_gnn_embeddings(
    model=deep_gnn_model,
    X=X,
    edge_index=None,
    scaler=tabpfn_scaler,
    device=DEVICE,
    embs_save_path=emb_save_paths['gnn'],
)

get_tabpfn_embeddings(
    model=tabpfn_model,
    X=X,
    embs_save_path=emb_save_paths['tabpfn'],
    scaler=tabpfn_scaler,
)

get_s2vec_embeddings(
    model=s2vec_model,
    loader=s2vec_data_loader,
    embs_save_path=emb_save_paths['s2vec'],
    device=DEVICE,
)

get_satclip_embeddings(
    coordinates=mun_geometry_points,
    device=DEVICE,
    checkpoint_filename=s2vec_checkpoint_save_path,
    output_path=emb_save_paths['satclip']
)

param_distributions = {
    "learning_rate": (0.01, 0.1),
    "depth": (4, 8),
    "l2_leaf_reg": (1, 10),
    "bagging_temperature": (0, 1),
    "random_strength": (0, 5),
    "min_data_in_leaf": (10, 50),
    "subsample": (0.6, 1.0)
}

for task in tasks:
    logger.info(f'Task: {str(task)}, target: {task.target_col}')
    logger.info('Solving without embeddings')
    
    basic_results = task.train_and_eval_model(
        param_distributions=param_distributions,
        n_trials=50
    )
    
    logger.info(f'Basic results:\n {basic_results} \n')

    for model, path in emb_save_paths.items():
        logger.info(f'Solving with embeddings from model {model}')
        



models_list = []

PROJECT_ROOT = Path(__file__).parent.parent
checkpoint_path = [Path(__file__).resolve().parent, 'models', 's2vec', 'checkpoints']



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
        # коэффициент корреляции 
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
