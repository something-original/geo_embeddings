import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import sys

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from emb_fit.models import DeepGNN
from emb_fit.utils import load_dataset


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

    # Сохранение модели
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    torch.save(save_dict, output_path)
    print(f"Модель сохранена: {output_path}")

    return model, scaler


def get_gnn_embeddings(
    model,
    X,
    edge_index=None,
    scaler=None,
    n_neighbors: int = 10,
    device: str = None
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    # Масштабирование
    if scaler is not None:
        X_scaled = scaler.transform(X.values)
    else:
        X_scaled = X.values

    # Построение графа если не предоставлен
    if edge_index is None:
        A = kneighbors_graph(X_scaled, n_neighbors=n_neighbors, mode='connectivity', include_self=False)
        edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)

    # Конвертация в тензоры
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    edge_index = edge_index.to(device)

    # Генерация эмбеддингов
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(x_tensor, edge_index)

    return embeddings.cpu().numpy()


if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parent.parent
    df_spb = load_dataset(Path(os.path.join(root_dir, 'datasets/spb_merged.csv')))
    df_msk = load_dataset(Path(os.path.join(root_dir, 'datasets/moscow_merged.csv')))
    df_ekb = load_dataset(Path(os.path.join(root_dir, 'datasets/ekb_merged.csv')))

    cols = df_spb.columns
    df_ekb = df_ekb[[col for col in cols if col in df_ekb.columns]]
    df_msk = df_msk[[col for col in cols if col in df_msk.columns]]
    df = pd.concat([df_spb, df_ekb, df_msk], axis=0)

    feature_start_index = df.columns.get_loc('mun_district')

    X = df.iloc[:, feature_start_index + 1:]
    y = df['price']

    X = X.reset_index().drop(columns=['index'])
    y = y.reset_index().drop(columns=['index'])

    X = X.dropna()
    y = y[y.index.isin(X.index)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, random_state=42, train_size=0.7
    )

    model, scaler = train_gnn(X_train, y_train['price'], X_test, y_test['price'])
    gnn_embs = get_gnn_embeddings(model, X_test)

    np.save(os.path.join(root_dir, 'emb_fit/gnn/gnn_embs.npy'), gnn_embs)
    np.save(os.path.join(root_dir, 'emb_fit/x_test_index.npy'), X_test.index.to_numpy())
