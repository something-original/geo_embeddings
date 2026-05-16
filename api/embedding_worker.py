"""Background worker: train embedding models and run inference."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from catboost import CatBoostRegressor

from api.state import get_state
from config import DEVICE
from embedding_qdrant import EMB_DIMS_DEFAULT, ensure_inference_embeddings_qdrant
from benchmark import run_experiments
from utils import PathBuilder, setup_logging

logger = logging.getLogger(__name__)

_LOCAL_ARTIFACTS = (
    "indicator_values_train.csv",
    "indicator_values_val.csv",
    "indicator_values_inference.csv",
    "indicator_values_old.csv",
    "indicator_values.csv",
)


def _cleanup_local_artifacts() -> None:
    mun_dir = Path(PathBuilder.build_emb_datasets_paths()["dataset_path"]).parent
    for name in _LOCAL_ARTIFACTS:
        p = mun_dir / name
        if p.is_file():
            try:
                p.unlink()
            except OSError as e:
                logger.warning("Could not remove %s: %s", p, e)

    for model_dir in ("gnn", "tab_pfn", "s2vec", "satclip"):
        root = Path(__file__).resolve().parents[1] / "emb_fit" / model_dir
        if not root.is_dir():
            continue
        for f in root.glob("*.npy"):
            try:
                f.unlink()
            except OSError:
                pass


def run_train_and_inference(emb_dims: list[int] | None = None) -> None:
    setup_logging()
    st = get_state()
    separate_inference = not st.same_file_inference
    dims = emb_dims or EMB_DIMS_DEFAULT

    logger.info(
        "train_and_inference: separate_inference=%s, emb_dims=%s, device=%s",
        separate_inference,
        dims,
        DEVICE,
    )

    try:
        run_experiments(
            model=CatBoostRegressor(),
            train_and_generate_embs=True,
            index_feature="municipality_id",
            features_to_drop=[],
            emb_dims=dims,
            separate_inference=separate_inference,
        )
        ensure_inference_embeddings_qdrant()
    finally:
        _cleanup_local_artifacts()
        logger.info("train_and_inference finished; local CSV artifacts removed")
