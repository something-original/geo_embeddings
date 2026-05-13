import logging
import os
from pathlib import Path

from config import BENCHMARK_LOG_PATH

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import make_valid
from shapely.wkt import loads


class PathBuilder:
    root_dir = Path(__file__).resolve().parent
    
    @classmethod
    def build_tasks_dataset_paths(cls) -> dict:
        path_parts = [cls.root_dir, 'datasets']

        return {
            'flats_dataset_path': os.path.join(*path_parts, 'flats', 'flats_merged.csv'),
            'fns_dataset_path': os.path.join(*path_parts, 'fns'),
            'people_dataset_path': os.path.join(*path_parts, 'people', 'building_people_dataset.csv'),
            'workplaces_dataset_path': os.path.join(*path_parts, 'workplaces', 'workplaces_districts.csv'),
            'municiplaities_path': os.path.join(*path_parts, 'mun_data', 'municipalities_csv'),
        }
    
    @classmethod
    def build_emb_datasets_paths(cls) -> dict:
        path_parts = [cls.root_dir, 'datasets', 'mun_data']
        
        return {
            'dataset_path': os.path.join(*path_parts, 'indicator_values.csv'),
            'dataset_path_old': os.path.join(*path_parts, 'indicator_values_old.csv'),
            'train_path': os.path.join(*path_parts, 'indicator_values_train.csv'),
            'val_path': os.path.join(*path_parts, 'indicator_values_val.csv'),
            'indicators_path': os.path.join(*path_parts, 'base_indicators.csv'),
            'municiplaities_path': os.path.join(*path_parts, 'municipalities.csv'),
            'inference_full_path': os.path.join(*path_parts, 'indicator_values_inference.csv')
        }
 
    @classmethod
    def build_models_save_paths(cls) -> dict:
        path_parts = [cls.root_dir, 'emb_fit']
        return {
            'deep_gnn_output_path': os.path.join(*path_parts, 'gnn'),
            'tabpfn_output_path': os.path.join(*path_parts, 'tab_pfn'),
            's2vec_output_path': os.path.join(*path_parts, 's2vec'),
            'satclip_output_path': os.path.join(*path_parts, 'satclip'),
        }

    @classmethod
    def build_embs_save_paths(cls) -> dict:
        models_save_paths = cls.build_models_save_paths()
        return {
            'gnn': os.path.join(models_save_paths['deep_gnn_output_path'], 'gnn_embs.npy'),
            'tabpfn': os.path.join(models_save_paths['tabpfn_output_path'], 'tab_pfn_embs.npy'),
            's2vec': os.path.join(models_save_paths['s2vec_output_path'], 's2vec_embs.npy'),
            'satclip': os.path.join(models_save_paths['satclip_output_path'], 'satclip_embs.npy'),
        }

    @classmethod
    def build_embs_save_paths_by_dim(cls, emb_dims: list[int]) -> dict[str, dict[int, str]]:
        """
        Build per-model embedding output paths for multiple embedding dimensions.

        File naming convention: *_embs_{dim}.npy
        """
        models_save_paths = cls.build_models_save_paths()
        base = {
            'satclip': os.path.join(models_save_paths['satclip_output_path'], 'satclip_embs'),
            'gnn': os.path.join(models_save_paths['deep_gnn_output_path'], 'gnn_embs'),
            'tabpfn': os.path.join(models_save_paths['tabpfn_output_path'], 'tab_pfn_embs'),
            's2vec': os.path.join(models_save_paths['s2vec_output_path'], 's2vec_embs'),
        }

        out: dict[str, dict[int, str]] = {}
        for model_name, prefix in base.items():
            out[model_name] = {d: f"{prefix}_{d}.npy" for d in emb_dims}
        return out


def prepare_mun_df(
    mun_dataset_path: str,
    geom_col: str = 'geometry'
) -> gpd.GeoDataFrame:
    df = gpd.read_file(mun_dataset_path, encoding='utf-8')
    df[geom_col] = df[geom_col].apply(loads)
    df[geom_col] = df[geom_col].apply(make_valid)

    df = gpd.GeoDataFrame(df).set_geometry(geom_col).set_crs('EPSG:4326')
    df['id'] = df['id'].apply(int)

    return df 


def get_geometry_points(
    index_col: pd.Series,
    dataset_path: str,
    geom_col: str = 'geometry'
) -> list[tuple[float, float]]:

    df = prepare_mun_df(dataset_path, geom_col)
    df['id'] = df['id'].apply(int)
    df = df[df['id'].isin(index_col.values)]
    df['centroids'] = df[geom_col].centroid
    
    return df['centroids'].apply(lambda p: (p.y, p.x)).tolist()


def load_embeddings(
    emb_path: str,
    municipality_path: str,
    index_col: pd.Series,
    geom_col: str = 'geometry'
) -> gpd.GeoDataFrame:

    index_col_name = index_col.name
    arr = np.load(emb_path)
    embeddings = pd.DataFrame(arr)
    if len(embeddings) != len(index_col):
        print(
            f"Embeddings length mismatch for {emb_path}: "
            f"embeddings={len(embeddings)} vs index_col={len(index_col)} "
            f"(index_col.name={index_col_name}). "
            f"Make sure you generate embeddings on the same CSV/order as index_col and with shuffle=False."
        )
        return
    embeddings[index_col_name] = index_col.to_numpy()
    
    mun_df = prepare_mun_df(municipality_path, geom_col)
    mun_df = mun_df[['id', geom_col]]
    
    embeddings = embeddings.merge(mun_df, how='left', left_on=index_col_name, right_on='id')
    embeddings = embeddings[~embeddings['geometry'].isna()]
    embeddings = gpd.GeoDataFrame(embeddings).set_geometry(geom_col).set_crs('EPSG:4326')
    
    embeddings = embeddings.drop_duplicates(subset=[index_col_name])
    embeddings = embeddings.set_index(index_col_name, drop=True)
    embeddings.drop(columns=['id'], inplace=True, errors='ignore')

    return embeddings


def setup_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_path = Path(BENCHMARK_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_path)
        for h in root_logger.handlers
    ):
        return

    for h in list(root_logger.handlers):
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_path):
            root_logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
