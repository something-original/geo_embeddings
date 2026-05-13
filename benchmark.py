import logging
import os
from typing import Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from collections import defaultdict
import json
from pathlib import Path

from config import (
    CHECK_PEOPLE_WORKPLACES_TASKS,
    DEVICE,
    EXPERIMENT_TARGET_FEATURES,
    PEOPLE_FEATURES,
    PEOPLE_CAT_FEATURES,
    WORKPLACES_FEATURES,
    WORKPLACES_CAT_FEATURES,
    HF_TOKEN,
    HF_BEST_REPO_ID,
    HF_BEST_REPO_PRIVATE,
    HF_BEST_REPO_REVISION,
    HF_BEST_REPO_TYPE,
    HF_BEST_PATH_IN_REPO,
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
    MunDataTask,
    PeopleHousesTask,
    WorkplacesDistrictsTask
)

from utils import (
    get_geometry_points,
    load_embeddings,
    PathBuilder,
    setup_logging
)

os.makedirs("logs", exist_ok=True)
setup_logging()
logger = logging.getLogger(__name__)


def create_and_prepare_tasks(
    model: Any,
    target_cols: list[str],
) -> list[BaseTask]:

    logger.info('Creating tasks')

    task_datasets_paths = PathBuilder.build_tasks_dataset_paths()
    emb_dataset_paths = PathBuilder.build_emb_datasets_paths()

    flats_task = FlatsTask(
        dataset_path=task_datasets_paths['flats_dataset_path'],
        features=[
            'dealType', 'roomsCount', 'repairType', 'hasFurniture', 'isApartments',
            'floorNumber', 'flatType', 'livingArea', 'windowsViewType', 'balconiesCount',
            'kitchenArea', 'isRecidivist', 'totalArea', 'hasLift', 'buildYear',
            'materialType','distance_to_center', 'highways_count', 'undergrounds_count',
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
            'IsMall', 'IsRare', 'IsEcommerce', 'ReceiptTotalCount'
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
            'IsMall', 'IsRare', 'IsEcommerce', 'ReceiptTotalCount'
        ],
        cat_features=['IsMall', 'IsRare', 'IsEcommerce'],
        geom_col='coordinates',
        val_ratio=0.2,
        model=model.copy(),
        target_col='AverageBill'
    )

    tasks: list[BaseTask] = [fns_task_avg_bill, fns_task_kkt, flats_task]

    municipalities_path = emb_dataset_paths['inference_full_path']
    mun_cols = pd.read_csv(municipalities_path, sep=';', nrows=1).columns.tolist()
    mun_features = [c for c in mun_cols if c not in ['id', 'geometry']]

    for target_col in target_cols[1:]:
        tasks.append(
            MunDataTask(
                dataset_path=municipalities_path,
                features=mun_features,
                target_col=target_col,
                geom_col='geometry',
                val_ratio=0.2,
                model=model.copy(),
                cat_features=[],
                features_to_drop=target_cols,
            )
        )

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
    index_feature: str,
    separate_inference: bool,
) -> tuple[dict, dict, StandardScaler, list[str]]:

    scaler = StandardScaler()
    emb_dataset_paths = PathBuilder.build_emb_datasets_paths()

    target_col_names = prepare_and_save_dataset(
        separate_inference=separate_inference,
        dataset_path_old=emb_dataset_paths['dataset_path_old'],
        dataset_path_new=emb_dataset_paths['dataset_path'],
        indicators_path=emb_dataset_paths['indicators_path'],
        features_to_drop=features_to_drop,
        index_feature=index_feature,
        experiment_target_features=EXPERIMENT_TARGET_FEATURES,
        train_path=emb_dataset_paths['train_path'],
        val_path=emb_dataset_paths['val_path'],
        inference_full_path=emb_dataset_paths['inference_full_path'],
        csv_sep=';',
        use_scaler=True,
        scaler=scaler,
        test_size=0.25,
    )

    X_train = pd.read_csv(emb_dataset_paths['train_path'], sep=';')
    X_test = pd.read_csv(emb_dataset_paths['val_path'], sep=';')
    X = pd.read_csv(emb_dataset_paths['inference_full_path'], sep=';')

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
    
    return dataset_dict, index_dict, scaler, target_col_names


def train_embedding_models(
    dataset_dict: dict,
    index_feature: str,
    emb_dims: list[int],
    target_col_names: list[str],
) -> dict:

    X_train = dataset_dict['X_train']
    y_train = dataset_dict['y_train']
    X_test = dataset_dict['X_test']
    y_test = dataset_dict['y_test']

    models_save_paths = PathBuilder.build_models_save_paths()
    emb_dataset_paths = PathBuilder.build_emb_datasets_paths()

    model_dict: dict = {}

    model_dict['deep_gnn_model'] = {}
    model_dict['deep_gnn_scaler'] = {}
    model_dict['deep_gnn_imputer'] = {}

    for d in emb_dims:
        logger.info('-----------')
        logger.info(f'Training GNN (emb_dim={d})')
        deep_gnn_model, deep_gnn_scaler, deep_gnn_imputer = train_gnn(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            device=DEVICE,
            output_path=os.path.join(models_save_paths['deep_gnn_output_path']),
            hidden_channels=d,
            model_name="gnn",
        )
        model_dict['deep_gnn_model'][d] = deep_gnn_model
        model_dict['deep_gnn_scaler'][d] = deep_gnn_scaler
        model_dict['deep_gnn_imputer'][d] = deep_gnn_imputer

    logger.info('----------')
    logger.info('Training TabPFN')
    tabpfn_model, tabpfn_target_scaler = train_tabpfn(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        output_path=models_save_paths['tabpfn_output_path'],
        device=DEVICE,
        embed_dims=emb_dims,
        model_name="tabpfn",
    )
    model_dict['tabpfn_model'] = tabpfn_model
    model_dict['tabpfn_target_scaler'] = tabpfn_target_scaler
    model_dict['tabpfn_train_medians'] = X_train.median(numeric_only=True)

    model_dict['s2vec_model'] = {}
    for d in emb_dims:
        logger.info('----------')
        logger.info(f'Training s2vec (emb_dim={d})')
        s2vec_model = train_s2vec(
            train_path=emb_dataset_paths['train_path'],
            val_path=emb_dataset_paths['val_path'],
            checkpoint_path=os.path.join(models_save_paths['s2vec_output_path']),
            device=DEVICE,
            embed_dim=d,
            cols_to_drop=[index_feature] + list(target_col_names),
            model_name="s2vec",
        )
        model_dict['s2vec_model'][d] = s2vec_model

    return model_dict


def generate_embeddings(
    model_dict: dict,
    index_feature: str,
    index_dict: dict,
    dataset_dict: dict,
    emb_dims: list[int],
    target_col_names: list[str],
) -> None:

    emb_save_paths = PathBuilder.build_embs_save_paths_by_dim(emb_dims)
    emb_dataset_paths = PathBuilder.build_emb_datasets_paths()

    for d in emb_dims:
        logger.info('----------')
        logger.info(f'Getting GNN embeddings (emb_dim={d})')
        get_gnn_embeddings(
            model=model_dict['deep_gnn_model'][d],
            X=dataset_dict['X'],
            X_train_reference=dataset_dict['X_train'],
            edge_index=None,
            scaler=model_dict['deep_gnn_scaler'][d],
            imputer=model_dict['deep_gnn_imputer'][d],
            device=DEVICE,
            embs_save_path=emb_save_paths['gnn'][d],
        )

    for d in emb_dims:
        logger.info('----------')
        logger.info(f'Getting TabPFN embeddings (emb_dim={d})')
        get_tabpfn_embeddings(
            model=model_dict['tabpfn_model'],
            X=dataset_dict['X'],
            embs_save_path=emb_save_paths['tabpfn'][d],
            feature_scaler=None,
            train_medians=None,
            output_dim=d,
        )

    for d in emb_dims:
        logger.info('----------')
        logger.info(f'Getting s2vec embeddings (emb_dim={d})')
        get_s2vec_embeddings(
            model=model_dict['s2vec_model'][d],
            loader=get_dataloader(
                csv_path=emb_dataset_paths['inference_full_path'],
                img_size=128,
                batch_size=128,
                shuffle=False,
                cols_to_drop=[index_feature] + list(target_col_names),
            ),
            embs_save_path=emb_save_paths['s2vec'][d],
            device=DEVICE,
        )

    coords = get_geometry_points(
        index_col=index_dict['full_index'],
        dataset_path=emb_dataset_paths['municiplaities_path']
    )

    for d in emb_dims:
        logger.info('----------')
        logger.info(f'Getting Satclip embeddings (emb_dim={d})')
        get_satclip_embeddings(
            coordinates=coords,
            device=DEVICE,
            checkpoint_filename='satclip-resnet18-l40.ckpt',
            output_path=emb_save_paths['satclip'][d],
            output_dim=d,
        )


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float('nan')
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def bootstrap_significance(
    y_true: pd.Series,
    y_pred_base: np.ndarray,
    y_pred_new: np.ndarray,
    n_boot: int = 300,
    random_state: int = 42,
) -> dict:
    rng = np.random.default_rng(random_state)
    y = y_true.to_numpy()
    base = np.asarray(y_pred_base)
    new = np.asarray(y_pred_new)
    n = len(y)
    idx = rng.integers(0, n, size=(n_boot, n))

    deltas_rmse = []
    deltas_corr = []
    for b in range(n_boot):
        ii = idx[b]
        y_b = y[ii]
        base_b = base[ii]
        new_b = new[ii]
        deltas_rmse.append(_rmse(y_b, new_b) - _rmse(y_b, base_b))
        deltas_corr.append(_pearson(y_b, new_b) - _pearson(y_b, base_b))

    deltas_rmse = np.asarray(deltas_rmse)
    deltas_corr = np.asarray(deltas_corr)

    return {
        "delta_rmse_mean": float(np.nanmean(deltas_rmse)),
        "delta_corr_mean": float(np.nanmean(deltas_corr)),
        "p_rmse": float(np.mean(deltas_rmse >= 0)),
        "p_corr": float(np.mean(deltas_corr <= 0)),
        "rmse_base": _rmse(y, base),
        "rmse_new": _rmse(y, new),
        "corr_base": _pearson(y, base),
        "corr_new": _pearson(y, new),
    }


def fit_best_model_and_predict(task: BaseTask, param_distributions: dict, n_trials: int = 1) -> np.ndarray:
    import optuna
    from sklearn.metrics import root_mean_squared_error

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

        params['random_state'] = 42 
        model_instance = task.model.__class__(**{**task.model.get_params(), **params})
        if model_instance.__class__.__name__.lower().startswith("catboost"):
            model_instance.fit(
                task.x_train,
                task.y_train,
                cat_features=task.cat_features,
                eval_set=(task.x_val, task.y_val),
                use_best_model=True,
                early_stopping_rounds=100,
                verbose=False,
            )
        else:
            model_instance.fit(task.x_train, task.y_train)

        y_pred_val = model_instance.predict(task.x_val)
        return root_mean_squared_error(task.y_val, y_pred_val)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    study.best_params['random_state'] = 42
    best_model = task.model.__class__(**{**task.model.get_params(), **study.best_params})
    if best_model.__class__.__name__.lower().startswith("catboost"):
        best_model.fit(
            task.x_train,
            task.y_train,
            cat_features=task.cat_features,
            eval_set=(task.x_val, task.y_val),
            use_best_model=True,
            early_stopping_rounds=100,
            verbose=False,
        )
    else:
        best_model.fit(task.x_train, task.y_train)
    return best_model.predict(task.x_test)


def check_tasks_performance(
    tasks: list[BaseTask],
    index_dict: dict,
    emb_dims: list[int],
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

    emb_save_paths = PathBuilder.build_embs_save_paths_by_dim(emb_dims)
    emb_dataset_paths = PathBuilder.build_emb_datasets_paths()
    municiplaities_path = emb_dataset_paths['municiplaities_path']

    logger.info('----------')
    wins: dict[tuple[str, int], int] = defaultdict(int)

    for task in tasks:
        logger.info(f'Task: {str(task)}, target: {task.target_col}')

        ref_model = next(iter(emb_save_paths.keys()))
        ref_dim = emb_dims[0]
        ref_path = emb_save_paths[ref_model][ref_dim]
        ref_embeddings = load_embeddings(
            emb_path=ref_path,
            municipality_path=municiplaities_path,
            index_col=index_dict['full_index'],
        )
        valid_index = task.get_index_with_embeddings(ref_embeddings, emb_geom_col='geometry')
        logger.info(f'Rows with embeddings for task (ref={ref_model}, dim={ref_dim}): {len(valid_index)}')

        task.resplit_on_index(valid_index)

        logger.info('Solving baseline')
        logger.info(f'Task features: {task.features}')
        y_pred_base = fit_best_model_and_predict(task, param_distributions, n_trials=10)
        logger.info(
            f'Baseline RMSE={_rmse(task.y_test.to_numpy(), y_pred_base):.6f}, '
            f'Corr={_pearson(task.y_test.to_numpy(), y_pred_base):.6f}'
        )

        for model_name, dim_map in emb_save_paths.items():
            for d, path in dim_map.items():
                logger.info(f'\nSolving with embeddings from model {model_name}, dim={d}')

                embeddings = None
                try:
                    embeddings = load_embeddings(
                        emb_path=path,
                        municipality_path=municiplaities_path,
                        index_col=index_dict['full_index'],
                    )
                except FileNotFoundError:
                    logger.info(f'No combination for {model_name}, dim={d}')
                
                if embeddings is None:
                    continue
                    
                emb_shape = embeddings.drop(columns=['geometry'], errors='ignore').shape
                logger.info(f'Embeddings dataframe shape (no geom): {emb_shape}')

                task.add_embeddings(embeddings, 'geometry')
                y_pred_new = fit_best_model_and_predict(task, param_distributions, n_trials=10)

                stats = bootstrap_significance(
                    y_true=task.y_test,
                    y_pred_base=y_pred_base,
                    y_pred_new=y_pred_new,
                    n_boot=300,
                )

                logger.info(
                    "Bootstrap: "
                    f"RMSE {stats['rmse_base']:.6f}->{stats['rmse_new']:.6f} "
                    f"(mean Δ={stats['delta_rmse_mean']:.6f}, p={stats['p_rmse']:.4f}); "
                    f"Corr {stats['corr_base']:.6f}->{stats['corr_new']:.6f} "
                    f"(mean Δ={stats['delta_corr_mean']:.6f}, p={stats['p_corr']:.4f})"
                )

                significant = (
                    stats["p_rmse"] < 0.05
                    and stats["p_corr"] < 0.05
                    and stats["delta_rmse_mean"] < 0
                    and stats["delta_corr_mean"] > 0
                )
                if significant:
                    wins[(model_name, d)] += 1

                task.clear_embeddings(embeddings)

        task.reset_splits()
        logger.info('-------------------')

    if wins:
        best_pair = max(wins.items(), key=lambda kv: kv[1])[0]
        logger.info("==========")
        logger.info("BEST (model, dim) by #tasks with significant improvement (RMSE↓ & Corr↑):")
        logger.info(f"Best model: {best_pair[0]}, emb_dim: {best_pair[1]}, wins: {wins[best_pair]}")
        logger.info("==========")
        upload_best_to_hf(
            best_pair=best_pair,
            wins=wins,
            emb_save_paths=emb_save_paths,
        )


def checkpoint_model_path(model_name: str, emb_dim: int) -> Path | None:
    root = Path(__file__).resolve().parent / "emb_fit" / "checkpoints"
    stem = f"{model_name}_{emb_dim}"
    sub = root / model_name / stem
    for fn in (f"{stem}.pt", f"{stem}.pkl", f"{stem}.ckpt"):
        p = sub / fn
        if p.is_file():
            return p
    return None


def upload_best_to_hf(
    best_pair: tuple[str, int],
    wins: dict[tuple[str, int], int],
    emb_save_paths: dict[str, dict[int, str]],
) -> None:

    if not HF_TOKEN:
        logger.warning("HF_TOKEN is not set; skipping HuggingFace upload.")
        return
    if not HF_BEST_REPO_ID:
        logger.warning("HF_BEST_REPO_ID is not set; skipping HuggingFace upload.")
        return

    model_name, emb_dim = best_pair
    if model_name not in emb_save_paths or emb_dim not in emb_save_paths[model_name]:
        logger.warning(f"Cannot find embedding path for {best_pair}; skipping HuggingFace upload.")
        return

    emb_path = emb_save_paths[model_name][emb_dim]
    if not os.path.exists(emb_path):
        logger.warning(f"Embedding file does not exist: {emb_path}; skipping HuggingFace upload.")
        return

    try:
        from huggingface_hub import HfApi, create_repo
    except Exception as e:
        logger.warning(f"huggingface_hub import failed ({e}); skipping HuggingFace upload.")
        return

    api = HfApi(token=HF_TOKEN)

    try:
        create_repo(
            repo_id=HF_BEST_REPO_ID,
            token=HF_TOKEN,
            repo_type=HF_BEST_REPO_TYPE,
            private=HF_BEST_REPO_PRIVATE,
            exist_ok=True,
        )
    except Exception as e:
        logger.warning(f"Failed to create/resolve HF repo {HF_BEST_REPO_ID}: {e}")
        return

    summary = {
        "best_model": model_name,
        "emb_dim": emb_dim,
        "wins": {f"{k[0]}_{k[1]}": int(v) for k, v in wins.items()},
        "selected_by": "max #tasks with significant improvement (RMSE↓ & Corr↑)",
    }

    out_dir = Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "best_embedding_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    path_in_repo_prefix = HF_BEST_PATH_IN_REPO.strip("/")
    if path_in_repo_prefix:
        path_in_repo_prefix += "/"

    try:
        api.upload_file(
            path_or_fileobj=emb_path,
            path_in_repo=f"{path_in_repo_prefix}{Path(emb_path).name}",
            repo_id=HF_BEST_REPO_ID,
            repo_type=HF_BEST_REPO_TYPE,
            revision=HF_BEST_REPO_REVISION or None,
            commit_message=f"Upload best embedding: {model_name}, dim={emb_dim}",
        )
        api.upload_file(
            path_or_fileobj=str(summary_path),
            path_in_repo=f"{path_in_repo_prefix}{summary_path.name}",
            repo_id=HF_BEST_REPO_ID,
            repo_type=HF_BEST_REPO_TYPE,
            revision=HF_BEST_REPO_REVISION or None,
            commit_message=f"Upload summary for best embedding: {model_name}, dim={emb_dim}",
        )
        logger.info(f"Uploaded best embedding to HF: repo={HF_BEST_REPO_ID}, file={Path(emb_path).name}")
        mp = checkpoint_model_path(model_name, emb_dim)
        if mp is not None:
            api.upload_file(
                path_or_fileobj=str(mp),
                path_in_repo=f"{path_in_repo_prefix}{mp.name}",
                repo_id=HF_BEST_REPO_ID,
                repo_type=HF_BEST_REPO_TYPE,
                revision=HF_BEST_REPO_REVISION or None,
                commit_message=f"Upload best model: {model_name}, dim={emb_dim}",
            )
            logger.info(f"Uploaded best model to HF: repo={HF_BEST_REPO_ID}, file={mp.name}")
        else:
            logger.warning(f"No checkpoint file for {best_pair}; skipping model upload.")
    except Exception as e:
        logger.warning(f"Failed to upload to HF ({HF_BEST_REPO_ID}): {e}")


def run_experiments(
    model: Any,
    train_and_generate_embs: bool,
    index_feature: str,
    features_to_drop: list[str],
    emb_dims: list[int],
    separate_inference: bool,
):

    logger.info('Start!')
    logger.info(f'Using device: {DEVICE}')

    word = "different" if separate_inference else "similar"
    logger.info(f"Training and inference on {word} datasets")

    dataset_dict, index_dict, feature_scaler, target_cols = prepare_emb_dataset(
        features_to_drop=features_to_drop,
        index_feature=index_feature,
        separate_inference=separate_inference,
    )
    tasks = create_and_prepare_tasks(model, target_cols)
    tasks = [tasks[1]]

    if train_and_generate_embs:

        model_dict = train_embedding_models(
            dataset_dict=dataset_dict,
            index_feature=index_feature,
            emb_dims=emb_dims,
            target_col_names=target_cols,
        )
        
        model_dict = {}
        model_dict['feature_scaler'] = feature_scaler

        generate_embeddings(
            model_dict=model_dict,
            index_feature=index_feature,
            index_dict=index_dict,
            dataset_dict=dataset_dict,
            emb_dims=emb_dims,
            target_col_names=target_cols,
        )

    check_tasks_performance(
        tasks=tasks,
        index_dict=index_dict,
        emb_dims=emb_dims,
    )


if __name__ == '__main__':

    run_experiments(
        model=CatBoostRegressor(),
        train_and_generate_embs=True,
        index_feature='municipality_id',
        features_to_drop=[],
        emb_dims=[128, 192, 256],
        separate_inference=True,
    )
