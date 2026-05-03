import logging
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

from config import (
    BENCHMARK_LOG_PATH,
    CHECK_PEOPLE_WORKPLACES_TASKS,
    DEVICE,
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

from utils import (
    get_geometry_points,
    load_embeddings,
    PathBuilder
)

os.makedirs("logs", exist_ok=True)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(BENCHMARK_LOG_PATH, mode='w', encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

model = CatBoostRegressor()


def create_and_prepare_tasks() -> list[BaseTask]:
    logger.info('Start!')
    logger.info('Creating tasks')

    task_datasets_paths = PathBuilder.build_tasks_dataset_paths()

    flats_task = FlatsTask(
        dataset_path=task_datasets_paths['flats_dataset_path'],
        features=[
            'dealType', 'roomsCount', 'repairType', 'hasFurniture', 'isApartments',
            'floorNumber', 'flatType', 'livingArea', 'windowsViewType', 'balconiesCount',
            'kitchenArea', 'isRecidivist', 'totalArea', 'hasLift', 'buildYear',
            'materialType' ,'distance_to_center', 'highways_count', 'undergrounds_count',
            'railways_count', 'time_to_metro'
        ],
        geom_col='geometry',
        val_ratio=0.2,
        model=model.copy(),
        target_col='price',
        dataset_crs='EPSG:3857',
        cat_features=[
            'repairType', 'hasFurniture', 'isApartments', 'flatType',
            'isRecidivist', 'hasLift', 'materialType',
            'windowsViewType'
        ]
    )

    fns_task_kkt = FNSTask(
        dataset_path=task_datasets_paths['fns_dataset_path'],
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
        dataset_path=task_datasets_paths['fns_dataset_path'],
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
            dataset_path=task_datasets_paths['people_dataset_path'],
            features=PEOPLE_FEATURES,
            target_col='people',
            geom_col='geometry',
            val_ratio=0.2,
            model=model.copy(),
            cat_features=PEOPLE_CAT_FEATURES
        )

        workplaces_task = WorkplacesDistrictsTask(
            dataset_path=task_datasets_paths['workplaces_dataset_path'],
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

    return tasks


def prepare_emb_dataset(
    features_to_drop: list[str],
    index_feature: str
) -> tuple[dict, dict, StandardScaler]:

    scaler = StandardScaler()
    emb_dataset_paths = PathBuilder.build_emb_datasets_paths()

    target_col_names = prepare_and_save_dataset(
        dataset_path=emb_dataset_paths['dataset_path'],
        indicators_path=emb_dataset_paths['indicators_path'],
        features_to_drop=features_to_drop,
        index_feature=index_feature,
        experiment_target_features=EXPERIMENT_TARGET_FEATURES,
        train_path=emb_dataset_paths['train_path'],
        train_full_path=emb_dataset_paths['train_full_path'],
        val_path=emb_dataset_paths['val_path'],
        csv_sep=';',
        use_scaler=True,
        scaler=scaler,
        test_size=0.25,
    )


    X_train = pd.read_csv(emb_dataset_paths['train_path'], sep=';')
    X_test = pd.read_csv(emb_dataset_paths['val_path'], sep=';')
    X = pd.read_csv(emb_dataset_paths['train_full_path'], sep=';')

    target_cols = {}
    y_train = X_train[target_col_names[0]].copy()
    y_test = X_test[target_col_names[0]].copy()

    if len(target_col_names) > 1:
        for col in target_col_names[1:]:
            target_cols[f'{col}_train'] = X_train[col].copy()
            target_cols[f'{col}_test'] = X_test[col].copy()

    train_index = X_train[index_feature].copy()
    test_index = X_test[index_feature].copy()
    full_index = X[index_feature].copy()

    X_train.drop(columns=target_col_names + [index_feature], inplace=True)
    X_test.drop(columns=target_col_names + [index_feature], inplace=True)
    X.drop(columns=target_col_names + [index_feature], inplace=True)
    
    dataset_dict = {
        'X': X,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    index_dict = {
        'train_index': train_index,
        'test_index': test_index,
        'full_index': full_index
    }
    
    return dataset_dict, index_dict, scaler


def train_embedding_models(
    dataset_dict: dict,
    index_feature: str
) -> dict:

    X_train = dataset_dict['X_train']
    y_train = dataset_dict['y_train']
    X_test = dataset_dict['X_test']
    y_test = dataset_dict['y_test']

    models_save_paths = PathBuilder.build_models_save_paths()
    emb_dataset_paths = PathBuilder.build_emb_datasets_paths()

    model_dict = {}

    logger.info('-----------')
    logger.info('Training GNN')
    deep_gnn_model, deep_gnn_scaler = train_gnn(
        X_train=X_train,
        y_train=y_train,
        device=DEVICE,
        X_test=X_test,
        y_test=y_test,
        output_path=models_save_paths['deep_gnn_output_path'],    
    )
    model_dict['deep_gnn_model'] = deep_gnn_model
    model_dict['deep_gnn_scaler'] = deep_gnn_scaler

    logger.info('----------')
    logger.info('Training TabPFN')
    tabpfn_model, tabpfn_target_scaler = train_tabpfn(
        X_train=X_train,
        y_train=y_train,
        output_path=models_save_paths['tabpfn_output_path'],
    )
    model_dict['tabpfn_model'] = tabpfn_model
    model_dict['tabpfn_target_scaler'] = tabpfn_target_scaler

    logger.info('----------')
    logger.info('Training s2vec')
    s2vec_model = train_s2vec(
        train_path=emb_dataset_paths['train_path'],
        val_path=emb_dataset_paths['val_path'],
        checkpoint_path=models_save_paths['s2vec_output_path'],
        device=DEVICE,
        cols_to_drop=[index_feature],
    )

    model_dict['s2vec_model'] = s2vec_model

    return model_dict


def generate_embeddings(
    model_dict: dict,
    index_feature: str,
    index_dict: dict,
    dataset_dict: dict,
) -> None:

    emb_save_paths = PathBuilder.build_embs_save_paths()
    emb_dataset_paths = PathBuilder.build_emb_datasets_paths()
    municiplaities_path = emb_dataset_paths['municiplaities_path']
    models_save_paths = PathBuilder.build_models_save_paths()

    logger.info('----------')
    logger.info('Getting GNN embeddings')
    get_gnn_embeddings(
        model=model_dict['deep_gnn_model'],
        X=dataset_dict['X'],
        edge_index=None,
        scaler=model_dict['deep_gnn_scaler'],
        device=DEVICE,
        embs_save_path=emb_save_paths['gnn'],
    )

    logger.info('----------')
    logger.info('Getting TabPFN embeddings')
    get_tabpfn_embeddings(
        model=model_dict['tabpfn_model'],
        X=dataset_dict['X'],
        embs_save_path=emb_save_paths['tabpfn'],
        feature_scaler=model_dict.get('feature_scaler'),
    )

    logger.info('----------')
    logger.info('Getting s2vec embeddings')
    get_s2vec_embeddings(
        model=model_dict['s2vec_model'],
        loader=get_dataloader(
            csv_path=emb_dataset_paths['train_path'],
            img_size=128,
            batch_size=128,
            shuffle=True,
            cols_to_drop=[index_feature],
        ),
        embs_save_path=emb_save_paths['s2vec'],
        device=DEVICE,
    )

    logger.info('----------')
    logger.info('Getting Satclip embeddings')
    get_satclip_embeddings(
        coordinates=get_geometry_points(
            index_col=index_dict['full_index'],
            dataset_path=municiplaities_path,
        ),
        device=DEVICE,
        checkpoint_filename='satclip-resnet18-l40.ckpt',
        output_path=emb_save_paths['satclip']
    )


def check_tasks_performance(
    tasks: list[BaseTask],
    index_dict: dict
) -> None:
    param_distributions = {
        "learning_rate": (0.01, 0.1),
        "depth": (4, 8),
        "l2_leaf_reg": (1, 10),
        "bagging_temperature": (0, 1),
        "random_strength": (0, 5),
        "min_data_in_leaf": (10, 50),
        "subsample": (0.6, 1.0)
    }

    emb_save_paths = PathBuilder.build_embs_save_paths()
    emb_dataset_paths = PathBuilder.build_emb_datasets_paths()
    municiplaities_path = emb_dataset_paths['municiplaities_path']

    logger.info('----------')
    for task in tasks:
        logger.info(f'Task: {str(task)}, target: {task.target_col}')
        logger.info('Solving without embeddings')
        
        basic_results = task.train_and_eval_model(
            param_distributions=param_distributions,
            n_trials=10
        )
        
        logger.info(f'Basic results:\n {basic_results} \n')

        for model, path in emb_save_paths.items():
            logger.info(f'Solving with embeddings from model {model}')

            embeddings = load_embeddings(
                emb_path=path,
                municipality_path=municiplaities_path,
                index_col=index_dict['full_index'],
            )
            task.add_embeddings(embeddings, 'geometry')
            
            emb_results = task.train_and_eval_model(
                param_distributions=param_distributions,
                n_trials=10            
            )
            logger.info(f'Results with embs from model {model}:\n {emb_results} \n')
            
        logger.info('-------------------')


if __name__ == '__main__':
    index_feature = 'municipality_id'
    tasks = create_and_prepare_tasks()

    dataset_dict, index_dict, feature_scaler = prepare_emb_dataset(
        features_to_drop=[],
        index_feature=index_feature
    )

    model_dict = train_embedding_models(
        dataset_dict=dataset_dict,
        index_feature=index_feature
    )
    model_dict['feature_scaler'] = feature_scaler
    
    generate_embeddings(
        model_dict=model_dict,
        index_feature=index_feature,
        index_dict=index_dict,
        dataset_dict=dataset_dict
    )
    
    check_tasks_performance(
        tasks=tasks,
        index_dict=index_dict
    )
