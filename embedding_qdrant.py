from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from benchmark import (
    checkpoint_model_path,
    prepare_emb_dataset,
    run_experiments,
)
from config import (
    DEVICE,
    EXPERIMENT_TARGET_FEATURES,
    HF_BEST_PATH_IN_REPO,
    HF_BEST_REPO_ID,
    HF_BEST_REPO_REVISION,
    HF_BEST_REPO_TYPE,
    HF_TOKEN,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_HTTPS,
    QDRANT_PORT,
)
from emb_fit import (
    fit_pca,
    get_dataloader,
    get_gnn_embeddings,
    get_pca_embeddings,
    get_s2vec_embeddings,
    get_satclip_embeddings,
    get_tabpfn_embeddings,
)
from emb_fit.models import DeepGNN, S2VecModel
from utils import PathBuilder, get_geometry_points

logger = logging.getLogger(__name__)


def make_qdrant_client() -> QdrantClient:
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY or None,
        https=QDRANT_HTTPS,
        prefer_grpc=False,
    )


INDEX_FEATURE = "municipality_id"
EMB_DIMS_DEFAULT = [128, 192, 256]

EMBEDDING_GENERATORS: dict[str, Callable[..., Any]] = {
    "gnn": get_gnn_embeddings,
    "tabpfn": get_tabpfn_embeddings,
    "s2vec": get_s2vec_embeddings,
    "satclip": get_satclip_embeddings,
    "pca": get_pca_embeddings,
}


def _repo_relative_path(filename: str) -> str:
    prefix = HF_BEST_PATH_IN_REPO.strip("/")
    return f"{prefix}/{filename}" if prefix else filename


def _try_hf_download(filename: str, local_dir: Path) -> Path | None:
    if not HF_TOKEN or not HF_BEST_REPO_ID:
        return None
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        logger.warning("huggingface_hub unavailable (%s)", e)
        return None
    local_dir.mkdir(parents=True, exist_ok=True)
    rel = _repo_relative_path(filename)
    try:
        p = hf_hub_download(
            repo_id=HF_BEST_REPO_ID,
            filename=rel,
            repo_type=HF_BEST_REPO_TYPE,
            revision=HF_BEST_REPO_REVISION or None,
            token=HF_TOKEN,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        return Path(p)
    except Exception as e:
        logger.info("HF download failed for %s: %s", rel, e)
        return None


def _load_summary_local() -> dict | None:
    p = Path("logs") / "best_embedding_summary.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_summary_hf_or_local() -> dict | None:
    s = _load_summary_local()
    if s:
        return s
    cache = Path("logs") / "hf_cache"
    p = _try_hf_download("best_embedding_summary.json", cache)
    if p and p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _emb_filename_for_summary(summary: dict) -> str:
    best = summary["best_model"]
    dim = int(summary["emb_dim"])
    paths = PathBuilder.build_embs_save_paths_by_dim([dim])
    return Path(paths[best][dim]).name


def _resolve_emb_npy_path(summary: dict) -> Path | None:
    dim = int(summary["emb_dim"])
    best = summary["best_model"]
    expected = Path(PathBuilder.build_embs_save_paths_by_dim([dim])[best][dim])
    if expected.is_file():
        return expected
    cache = Path("logs") / "hf_cache"
    downloaded = _try_hf_download(_emb_filename_for_summary(summary), cache)
    if downloaded and downloaded.is_file():
        expected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, expected)
        return expected
    return None


def _inference_municipality_ids() -> np.ndarray:
    emb_paths = PathBuilder.build_emb_datasets_paths()
    inf = Path(emb_paths["inference_full_path"])
    if not inf.is_file():
        raise FileNotFoundError(f"Inference dataset missing: {inf}")
    X = pd.read_csv(inf, sep=";")
    if INDEX_FEATURE not in X.columns:
        raise ValueError(f"Column {INDEX_FEATURE!r} not in {inf}")
    return X[INDEX_FEATURE].to_numpy(dtype=np.int64)


def _load_gnn_from_checkpoint(path: Path, device: str) -> tuple[Any, Any, Any, int]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["model_config"]
    model = DeepGNN(
        in_channels=cfg["in_channels"],
        hidden_channels=cfg["hidden_channels"],
        out_channels=cfg["out_channels"],
        dropout=cfg.get("dropout", 0.3),
    )
    model.load_state_dict(ck["model_state_dict"])
    model = model.to(device)
    imputer = ck["imputer"]
    scaler = ck.get("scaler")
    n_neighbors = int(ck.get("n_neighbors", 10))
    return model, scaler, imputer, n_neighbors


def _load_s2vec_from_checkpoint(path: Path, device: str, embed_dim: int) -> S2VecModel:
    ck = torch.load(path, map_location=device, weights_only=False)
    ed = int(ck.get("embed_dim", embed_dim))
    model = S2VecModel(
        img_size=128,
        patch_size=16,
        in_ch=1,
        num_heads=8,
        encoder_layers=6,
        decoder_layers=2,
        embed_dim=ed,
        decoder_dim=128,
        mask_ratio=0.75,
        lr=1e-4,
    )
    model.load_state_dict(ck["state_dict"])
    return model


def _ensure_hf_checkpoint_local(model_name: str, emb_dim: int) -> Path | None:
    local = checkpoint_model_path(model_name, emb_dim)
    if local is not None and local.is_file():
        return local
    stem = f"{model_name}_{emb_dim}"
    for ext in (".pt", ".pkl", ".ckpt"):
        name = f"{stem}{ext}"
        cache = Path("logs") / "hf_cache"
        got = _try_hf_download(name, cache)
        if got and got.is_file():
            dest_dir = local.parent if local is not None else (
                Path(__file__).resolve().parent / "emb_fit" / "checkpoints" / model_name / stem
            )
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / name
            shutil.copy2(got, dest)
            return dest
    return None


def _generate_best_embeddings_npy(summary: dict) -> Path:
    """Считает .npy для (best_model, emb_dim) через EMBEDDING_GENERATORS."""
    best = summary["best_model"]
    if best not in EMBEDDING_GENERATORS:
        raise ValueError(f"Unknown best_model {best!r}; expected one of {list(EMBEDDING_GENERATORS)}")

    emb_dim = int(summary["emb_dim"])
    dataset_dict, index_dict, feature_scaler, target_col_names = prepare_emb_dataset(
        features_to_drop=[],
        index_feature=INDEX_FEATURE,
        separate_inference=True,
    )
    emb_save_paths = PathBuilder.build_embs_save_paths_by_dim([emb_dim])
    out_path = Path(emb_save_paths[best][emb_dim])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if best == "gnn":
        ck = _ensure_hf_checkpoint_local(best, emb_dim)
        if ck is None:
            raise FileNotFoundError("GNN checkpoint not found locally or on HF")
        model, scaler, imputer, n_neighbors = _load_gnn_from_checkpoint(ck, DEVICE)
        get_gnn_embeddings(
            model=model,
            X=dataset_dict["X"],
            X_train_reference=dataset_dict["X_train"],
            edge_index=None,
            scaler=scaler,
            imputer=imputer,
            device=DEVICE,
            embs_save_path=str(out_path),
            n_neighbors=n_neighbors,
        )
    elif best == "tabpfn":
        ck = _ensure_hf_checkpoint_local(best, emb_dim)
        if ck is None:
            raise FileNotFoundError("TabPFN checkpoint not found locally or on HF")
        with open(ck, "rb") as f:
            blob = pickle.load(f)
        model = blob["model"]
        get_tabpfn_embeddings(
            model=model,
            X=dataset_dict["X"],
            embs_save_path=str(out_path),
            feature_scaler=feature_scaler,
            train_medians=dataset_dict["X_train"].median(numeric_only=True),
            output_dim=emb_dim,
        )
    elif best == "s2vec":
        ck = _ensure_hf_checkpoint_local(best, emb_dim)
        if ck is None:
            raise FileNotFoundError("s2vec checkpoint not found locally or on HF")
        model = _load_s2vec_from_checkpoint(ck, DEVICE, emb_dim)
        emb_paths = PathBuilder.build_emb_datasets_paths()
        loader = get_dataloader(
            csv_path=emb_paths["inference_full_path"],
            img_size=128,
            batch_size=128,
            shuffle=False,
            cols_to_drop=[INDEX_FEATURE] + list(target_col_names),
        )
        get_s2vec_embeddings(
            model=model,
            loader=loader,
            embs_save_path=str(out_path),
            device=DEVICE,
        )
    elif best == "pca":
        pca_model, pca_imputer = fit_pca(
            X_train=dataset_dict["X_train"],
            n_components=emb_dim,
        )
        get_pca_embeddings(
            pca=pca_model,
            imputer=pca_imputer,
            X=dataset_dict["X"],
            embs_save_path=str(out_path),
            output_dim=emb_dim,
        )
    else:  # satclip
        mun_path = PathBuilder.build_emb_datasets_paths()["municiplaities_path"]
        coords = get_geometry_points(index_dict["full_index"], str(mun_path))
        get_satclip_embeddings(
            coordinates=coords,
            device=DEVICE,
            checkpoint_filename="satclip-resnet18-l40.ckpt",
            output_path=str(out_path),
            output_dim=emb_dim,
        )
    if not out_path.is_file():
        raise RuntimeError(f"Embedding file was not created: {out_path}")
    return out_path


def _run_full_experiments_pipeline() -> None:
    logger.info("Running run_experiments (train + embed + benchmark + HF upload)")
    run_experiments(
        model=CatBoostRegressor(),
        train_and_generate_embs=True,
        index_feature=INDEX_FEATURE,
        features_to_drop=[],
        emb_dims=EMB_DIMS_DEFAULT,
        separate_inference=True,
    )


def _vectors_to_qdrant(municipality_ids: np.ndarray, vectors: np.ndarray) -> None:
    if len(municipality_ids) != len(vectors):
        raise ValueError(
            f"municipality_ids ({len(municipality_ids)}) vs vectors ({len(vectors)}) length mismatch"
        )
    vec_size = int(vectors.shape[1])
    client = make_qdrant_client()
    names = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION not in names:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
        )
    else:
        info = client.get_collection(QDRANT_COLLECTION)
        params = info.config.params
        vec_cfg = params.vectors
        if isinstance(vec_cfg, dict):
            existing = int(next(iter(vec_cfg.values())).size)
        else:
            existing = int(getattr(vec_cfg, "size"))
        if existing != vec_size:
            logger.warning(
                "Recreating Qdrant collection %s: vector size %s -> %s",
                QDRANT_COLLECTION,
                existing,
                vec_size,
            )
            client.delete_collection(QDRANT_COLLECTION)
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
            )

    batch: list[PointStruct] = []
    bs = 256
    for i in range(0, len(municipality_ids), bs):
        chunk_ids = municipality_ids[i : i + bs]
        chunk_vecs = vectors[i : i + bs]
        for mid, vec in zip(chunk_ids, chunk_vecs, strict=True):
            mid_i = int(mid)
            batch.append(
                PointStruct(
                    id=mid_i,
                    vector=np.asarray(vec, dtype=np.float32).tolist(),
                    payload={"municipality_id": mid_i},
                )
            )
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
        batch.clear()
    logger.info(
        "Qdrant upsert complete: collection=%s, points=%s, dim=%s",
        QDRANT_COLLECTION,
        len(municipality_ids),
        vec_size,
    )


def ensure_inference_embeddings_qdrant() -> None:
    """
    1) Пытается взять best_embedding_summary.json (локально или с HF).
    2) Пытается взять .npy эмбеддингов для best_model / emb_dim.
    3) Если .npy нет — тянет чекпоинт с HF (если есть) и генерирует через соответствующую get_*.
    4) Если summary или чекпоинта нет — run_experiments (обучение, генерация, публикация на HF).
    5) Загружает векторы в Qdrant; id точки = municipality_id (порядок строк inference CSV).
    """
    if not EXPERIMENT_TARGET_FEATURES:
        logger.warning("EXPERIMENT_TARGET_FEATURES empty; skipping embedding / Qdrant sync.")
        return

    summary = _load_summary_hf_or_local()

    if summary is None or "best_model" not in summary or "emb_dim" not in summary:
        logger.info("No best_embedding_summary; running full experiments pipeline.")
        _run_full_experiments_pipeline()
        summary = _load_summary_local()

        if summary is None:
            summary = _load_summary_hf_or_local()
        if summary is None:
            logger.error("Still no best_embedding_summary after run_experiments.")
            return

    best = summary["best_model"]
    if best not in EMBEDDING_GENERATORS:
        logger.error("Unsupported best_model %r in summary.", best)
        return

    emb_path = _resolve_emb_npy_path(summary)
    if emb_path is None:
        logger.info("Embedding npy missing; trying regeneration from HF/local checkpoints.")
    
        try:
            emb_path = _generate_best_embeddings_npy(summary)
        except Exception as e:
            logger.warning("Regeneration failed (%s); running full experiments.", e)
            _run_full_experiments_pipeline()
            summary = _load_summary_local()
            if summary is None:
                summary = _load_summary_hf_or_local()
            if summary is None:
                logger.error("No summary after run_experiments.")
                return
            emb_path = _resolve_emb_npy_path(summary)
            if emb_path is None:
                try:
                    emb_path = _generate_best_embeddings_npy(summary)
                except Exception as e2:
                    logger.error("Could not build embeddings: %s", e2)
                    return

    arr = np.load(str(emb_path))
    mids = _inference_municipality_ids()
    if len(mids) != len(arr):
        logger.error(
            "Shape mismatch: embeddings %s vs inference rows %s. Regenerating once.",
            len(arr),
            len(mids),
        )
        emb_path = _generate_best_embeddings_npy(summary)
        arr = np.load(str(emb_path))
        mids = _inference_municipality_ids()
        if len(mids) != len(arr):
            raise RuntimeError(f"Persistent mismatch: emb {len(arr)} vs ids {len(mids)}")

    _vectors_to_qdrant(mids, arr)


async def ensure_inference_embeddings_qdrant_async() -> None:
    await asyncio.to_thread(ensure_inference_embeddings_qdrant)
