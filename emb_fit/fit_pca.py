from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


def _matrix_with_nan_sentinel(X: pd.DataFrame) -> np.ndarray:
    arr = X.to_numpy(dtype=np.float64)
    arr[arr == -1.0] = np.nan
    return arr


def fit_pca(
    X_train: pd.DataFrame,
    n_components: int,
    random_state: int = 42,
) -> tuple[PCA, SimpleImputer]:
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(_matrix_with_nan_sentinel(X_train))
    n_comp = min(n_components, X_imputed.shape[0], X_imputed.shape[1])
    if n_comp < 1:
        raise ValueError(
            f"Cannot fit PCA with n_components={n_components} "
            f"for train shape {X_train.shape}"
        )
    pca = PCA(n_components=n_comp, random_state=random_state)
    pca.fit(X_imputed)
    return pca, imputer


def _align_embedding_dim(embeddings: np.ndarray, output_dim: int | None) -> np.ndarray:
    if output_dim is None:
        return embeddings
    n = embeddings.shape[1]
    if n == output_dim:
        return embeddings
    if n > output_dim:
        return embeddings[:, :output_dim]
    pad = np.zeros((embeddings.shape[0], output_dim - n), dtype=embeddings.dtype)
    return np.hstack([embeddings, pad])


def get_pca_embeddings(
    pca: PCA,
    imputer: SimpleImputer,
    X: pd.DataFrame,
    embs_save_path: str,
    output_dim: int | None = None,
) -> None:
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X_imputed = imputer.transform(_matrix_with_nan_sentinel(X))
    embeddings = pca.transform(X_imputed)
    embeddings = _align_embedding_dim(embeddings, output_dim)

    out = Path(embs_save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, embeddings)
