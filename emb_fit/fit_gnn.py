import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.impute import SimpleImputer

from torch_geometric.data import Data

from emb_fit.models import DeepGNN
from config import DEVICE


def train_gnn(
    X_train,
    y_train,
    device: str,
    X_test=None,
    y_test=None,
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
        if X_test is not None:
            X_test = X_test.drop(columns=[col for col in columns_to_drop if col in X_test.columns], errors='ignore')

    if X_test is not None:
        X_full = pd.concat([X_train, X_test], ignore_index=True)
        y_full = np.concatenate([
            y_train.values if isinstance(y_train, pd.Series) else y_train,
            y_test.values if isinstance(y_test, pd.Series) else y_test
        ])
        train_mask = np.concatenate([
            np.ones(len(X_train), dtype=bool),
            np.zeros(len(X_test), dtype=bool)
        ])
    else:
        X_full = X_train
        y_full = y_train.values if isinstance(y_train, pd.Series) else y_train
        train_mask = np.ones(len(X_train), dtype=bool)

    print(f"Обучаем GNN на {len(X_train)} образцах с {X_full.shape[1]} признаками")
    if X_test is not None:
        print(f"Тестовая выборка: {len(X_test)} образцов")

    # Масштабирование признаков
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full.values)

    # Масштабирование целевой переменной
    target_scaler = StandardScaler()
    y_full_scaled = target_scaler.fit_transform(y_full.reshape(-1, 1)).flatten()

    # Построение графа (k-NN граф)
    print(f"Строим k-NN граф с k={n_neighbors}...")
    A = kneighbors_graph(X_full_scaled, n_neighbors=n_neighbors, mode='connectivity', include_self=False)
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)

    # Создание PyTorch Geometric Data объекта
    data = Data(
        x=torch.tensor(X_full_scaled, dtype=torch.float32),
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
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

    print("Обучение завершено")

    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'in_channels': X_full_scaled.shape[1],
            'hidden_channels': hidden_channels,
            'out_channels': out_channels,
            'dropout': dropout
        },
        'scaler': scaler,
        'target_scaler': target_scaler,
        'columns_to_drop': columns_to_drop,
        'n_neighbors': n_neighbors
    }

    output_path = os.path.join(output_path, 'gnn.pt')
    torch.save(save_dict, output_path)
    print(f"Модель сохранена: {output_path}")

    return model, scaler


def get_gnn_embeddings(
    model,
    X,
    embs_save_path: str,
    edge_index=None,
    scaler=None,
    n_neighbors: int = 10,
    device: str = None,
):
    """
    Генерирует эмбеддинги с помощью обученной GNN модели.

    Args:
        model: Обученная GNN модель
        X: pandas DataFrame или numpy array с данными
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

    # Масштабирование
    if scaler is not None:
        X_scaled = scaler.transform(X.values)
    else:
        X_scaled = X.values

    # Построение графа если не предоставлен
    if edge_index is None:
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_scaled)
        
        A = kneighbors_graph(X_imputed, n_neighbors=n_neighbors, mode='connectivity', include_self=False)
        edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)

    # Конвертация в тензоры
    x_tensor = torch.tensor(X_imputed, dtype=torch.float32).to(device)
    edge_index = edge_index.to(device)

    # Генерация эмбеддингов
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(x_tensor, edge_index)

    np.save(embs_save_path, embeddings.cpu().numpy())
