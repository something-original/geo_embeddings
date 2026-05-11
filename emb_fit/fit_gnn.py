import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer

from torch_geometric.data import Data

from emb_fit.models import DeepGNN
from config import DEVICE


def train_gnn(
    X_train,
    y_train,
    device: str,
    output_path: str = "emb_fit/gnn_model.pt",
    hidden_channels: int = 128,
    out_channels: int = 1,
    dropout: float = 0.3,
    n_neighbors: int = 10,
    n_epochs: int = 100,
    learning_rate: float = 0.001,
    columns_to_drop: list = None,
    random_state: int = 42
):
    """
    Обучает GNN модель и сохраняет её.

    Args:
        X_train: pandas DataFrame или numpy array с обучающими признаками
        y_train: pandas Series или numpy array с целевыми значениями
        X_test: Опциональные тестовые данные для валидации
        y_test: Опциональные тестовые целевые значения
        output_path: Путь для сохранения модели
        hidden_channels: Размерность скрытых слоёв
        out_channels: Размерность выходного слоя (1 для регрессии)
        dropout: Коэффициент dropout
        n_neighbors: Количество соседей для построения графа
        n_epochs: Количество эпох обучения
        learning_rate: Скорость обучения
        columns_to_drop: Список колонок для удаления
        device: Устройство для обучения ('cuda' или 'cpu')
        random_state: Random state для воспроизводимости

    Returns:
        model: Обученная GNN модель
        scaler: StandardScaler для масштабирования признаков
    """

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)

    if columns_to_drop:
        X_train = X_train.drop(columns=[col for col in columns_to_drop if col in X_train.columns], errors='ignore')

    # Inductive training: build the graph ONLY on train split.
    X_full = X_train
    y_full = y_train.values if isinstance(y_train, pd.Series) else np.asarray(y_train)
    train_mask = np.ones(len(X_train), dtype=bool)

    print(f"Обучаем GNN на {len(X_train)} образцах с {X_full.shape[1]} признаками")

    # Inputs are expected to be pre-scaled by `prepare_and_save_dataset(..., use_scaler=True)`.
    # We do not re-fit/transform scalers here to avoid train/inference mismatch.
    scaler = None
    X_full_scaled = X_full.values

    # Масштабирование целевой переменной: fit on y_train only
    target_scaler = StandardScaler()
    y_train_arr = y_train.values if isinstance(y_train, pd.Series) else np.asarray(y_train)
    target_scaler.fit(y_train_arr.reshape(-1, 1))
    y_full_scaled = target_scaler.transform(y_full.reshape(-1, 1)).flatten()

    # Impute after scaling; fit on train only and reuse for inference
    imputer = SimpleImputer(strategy='median')
    X_full_imputed = imputer.fit_transform(X_full_scaled)

    # Построение графа (k-NN граф)
    print(f"Строим k-NN граф с k={n_neighbors}...")
    A = kneighbors_graph(
        X_full_imputed,
        n_neighbors=n_neighbors,
        mode='connectivity',
        include_self=False,
    )
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)

    # Создание PyTorch Geometric Data объекта
    data = Data(
        x=torch.tensor(X_full_imputed, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y_full_scaled, dtype=torch.float32),
        train_mask=torch.tensor(train_mask, dtype=torch.bool)
    )

    # Создание модели
    model = DeepGNN(
        in_channels=X_full_scaled.shape[1],
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout
    ).to(device)

    data = data.to(device)

    # Оптимизатор и функция потерь
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Обучение
    print(f"Начинаем обучение на {device}...")
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index).squeeze()
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")

    print("Обучение завершено")

    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'in_channels': X_full_imputed.shape[1],
            'hidden_channels': hidden_channels,
            'out_channels': out_channels,
            'dropout': dropout
        },
        'scaler': scaler,
        'imputer': imputer,
        'target_scaler': target_scaler,
        'columns_to_drop': columns_to_drop,
        'n_neighbors': n_neighbors
    }

    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, f'gnn_{hidden_channels}.pt')
    torch.save(save_dict, output_path)
    print(f"Модель сохранена: {output_path}")

    return model, scaler, imputer


def get_gnn_embeddings(
    model,
    X,
    embs_save_path: str,
    X_train_reference=None,
    edge_index=None,
    scaler=None,
    imputer=None,
    n_neighbors: int = 10,
    device: str = None,
):
    """
    Генерирует эмбеддинги с помощью обученной GNN модели.

    Args:
        model: Обученная GNN модель
        X: pandas DataFrame или numpy array с данными
        X_train_reference: train-признаки, на которых обучался граф; если переданы,
            базовый граф строится только на них, а для X добавляются только связи к train.
        edge_index: Опциональный edge_index для графа (если None, строится k-NN граф)
        scaler: StandardScaler для масштабирования признаков
        n_neighbors: Количество соседей для построения графа (если edge_index не предоставлен)
        device: Устройство для вычислений

    Returns:
        embeddings: numpy array с эмбеддингами формы (n_samples, hidden_channels)
    """
    if device is None:
        device = DEVICE

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(X_train_reference, np.ndarray):
        X_train_reference = pd.DataFrame(X_train_reference)

    # Масштабирование (use train-fitted scaler if provided)
    X_scaled = scaler.transform(X.values) if scaler is not None else X.values
    X_train_scaled = None
    if X_train_reference is not None:
        X_train_scaled = (
            scaler.transform(X_train_reference.values)
            if scaler is not None
            else X_train_reference.values
        )

    # Imputation:
    # - if train reference is provided and imputer is missing, fit on train reference
    # - otherwise fallback to fitting on current X
    if imputer is None:
        imputer = SimpleImputer(strategy='median')
        if X_train_scaled is not None:
            imputer.fit(X_train_scaled)
            X_train_imputed = imputer.transform(X_train_scaled)
            X_imputed = imputer.transform(X_scaled)
        else:
            X_imputed = imputer.fit_transform(X_scaled)
            X_train_imputed = None
    else:
        X_imputed = imputer.transform(X_scaled)
        X_train_imputed = (
            imputer.transform(X_train_scaled)
            if X_train_scaled is not None
            else None
        )

    # Построение графа если не предоставлен
    if edge_index is None:
        if X_train_imputed is not None and len(X_train_imputed) > 0:
            # Inductive mode:
            # 1) Keep base graph on train anchors only.
            # 2) Add edges from each current node to nearest train anchors (+ reverse).
            n_full = X_imputed.shape[0]
            n_train = X_train_imputed.shape[0]
            k = max(1, min(n_neighbors, n_train))

            # Base graph on train anchors with node ids shifted by n_full
            A_train = kneighbors_graph(
                X_train_imputed,
                n_neighbors=k,
                mode='connectivity',
                include_self=False,
            ).tocoo()
            train_src = A_train.row + n_full
            train_dst = A_train.col + n_full

            # New edges: full nodes <-> train anchors
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(X_train_imputed)
            neigh_idx = nn.kneighbors(X_imputed, return_distance=False)

            full_src = np.repeat(np.arange(n_full), k)
            full_to_train_dst = neigh_idx.reshape(-1) + n_full

            # Bidirectional message passing between current nodes and anchors
            src = np.concatenate([train_src, full_src, full_to_train_dst])
            dst = np.concatenate([train_dst, full_to_train_dst, full_src])

            edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
        else:
            A = kneighbors_graph(
                X_imputed,
                n_neighbors=n_neighbors,
                mode='connectivity',
                include_self=False,
            )
            edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)

    # Конвертация в тензоры
    if X_train_imputed is not None and edge_index is not None:
        x_for_graph = np.vstack([X_imputed, X_train_imputed])
        n_output = X_imputed.shape[0]
    else:
        x_for_graph = X_imputed
        n_output = X_imputed.shape[0]

    x_tensor = torch.tensor(x_for_graph, dtype=torch.float32).to(device)
    edge_index = edge_index.to(device)

    # Генерация эмбеддингов
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(x_tensor, edge_index)

    # Save only embeddings for requested X (exclude train anchor nodes if used)
    np.save(embs_save_path, embeddings[:n_output].cpu().numpy())
