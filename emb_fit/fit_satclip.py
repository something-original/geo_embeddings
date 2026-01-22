"""
Генерация эмбеддингов SatCLIP для координат с использованием предобученной модели.

SatCLIP - это модель, которая обучается сопоставлять спутниковые изображения
и GPS координаты. Location encoder может генерировать эмбеддинги для координат
без необходимости в изображениях.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Union, List, Tuple
import sys

# Добавляем путь к satclip модулю
project_root = Path(__file__).resolve().parent.parent
satclip_path = project_root / 'models' / 'satclip' / 'satclip'
if str(satclip_path) not in sys.path:
    sys.path.insert(0, str(satclip_path))

try:
    from huggingface_hub import hf_hub_download
    from load import get_satclip
except ImportError as e:
    raise ImportError(
        f"Не удалось импортировать необходимые модули для SatCLIP: {e}\n"
        "Установите зависимости: pip install huggingface_hub"
    )


def get_satclip_embeddings(
    coordinates: Union[List[Tuple[float, float]], np.ndarray, torch.Tensor],
    model_name: str = "microsoft/SatCLIP-ResNet18-L40",
    checkpoint_filename: str = "satclip-resnet18-l40.ckpt",
    output_path: str = None,
    device: str = None,
    batch_size: int = 32
) -> np.ndarray:
    """
    Генерирует эмбеддинги SatCLIP для списка координат используя предобученную модель.

    Args:
        coordinates: Список координат в формате [(lat, lon), ...] или [[lat, lon], ...]
                     или numpy array формы (n_samples, 2) или torch.Tensor
                     Может быть в формате [lat, lon] или [lon, lat] - функция автоматически определит
        model_name: Название модели на Hugging Face (по умолчанию: microsoft/SatCLIP-ResNet18-L40)
                    Доступные модели:
                    - microsoft/SatCLIP-ViT16-L40
                    - microsoft/SatCLIP-ResNet18-L40
                    - microsoft/SatCLIP-ResNet50-L40
        checkpoint_filename: Имя файла чекпоинта в репозитории
        output_path: Опциональный путь для сохранения эмбеддингов (.npy файл)
        device: Устройство для вычислений ('cuda' или 'cpu'). Если None, определяется автоматически
        batch_size: Размер батча для обработки координат

    Returns:
        embeddings: numpy array с эмбеддингами формы (n_samples, embed_dim)
                    где embed_dim обычно 512

    Example:
        >>> coords = [(55.7558, 37.6173), (59.9343, 30.3351)]  # Москва, СПб
        >>> embeddings = get_satclip_embeddings(coords)
        >>> print(embeddings.shape)  # (2, 512)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(device)

    # Конвертируем координаты в правильный формат
    if isinstance(coordinates, list):
        coords_array = np.array(coordinates, dtype=np.float64)
    elif isinstance(coordinates, np.ndarray):
        coords_array = coordinates.astype(np.float64)
    elif isinstance(coordinates, torch.Tensor):
        coords_array = coordinates.detach().cpu().numpy().astype(np.float64)
    else:
        raise ValueError(f"Неподдерживаемый тип координат: {type(coordinates)}")

    # Проверяем формат координат
    if coords_array.shape[1] != 2:
        raise ValueError(f"Координаты должны быть формы (n_samples, 2), получено {coords_array.shape}")

    n_samples = coords_array.shape[0]
    print(f"Генерируем эмбеддинги SatCLIP для {n_samples} координат")


    # SatCLIP ожидает формат [lon, lat]
    if np.abs(coords_array[:, 0]).max() > 90:
        # Первая колонка - это lon, вторая - lat (уже правильный формат)
        coords_tensor = torch.tensor(coords_array, dtype=torch.float64)
    else:
        # Первая колонка - это lat, вторая - lon, нужно переставить
        print("Обнаружен формат [lat, lon], переставляем в [lon, lat]")
        coords_tensor = torch.tensor(coords_array[:, [1, 0]], dtype=torch.float64)

    # Загружаем предобученную модель из Hugging Face
    print(f"Загружаем модель {model_name} из Hugging Face...")
    try:
        checkpoint_path = hf_hub_download(
            repo_id=model_name,
            filename=checkpoint_filename,
            cache_dir=None  # Использует кеш по умолчанию
        )
        print(f"Модель загружена: {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(
            f"Не удалось загрузить модель {model_name} из Hugging Face: {e}\n"
            "Проверьте подключение к интернету и правильность названия модели."
        )

    # Загружаем location encoder из чекпоинта
    print("Загружаем location encoder...")
    try:
        location_encoder = get_satclip(
            ckpt_path=checkpoint_path,
            device=device,
            return_all=False  # Возвращает только location encoder
        )
        location_encoder.eval()
        location_encoder = location_encoder.to(device)
        print("Location encoder загружен успешно")
    except Exception as e:
        raise RuntimeError(f"Не удалось загрузить location encoder: {e}")

    # Генерируем эмбеддинги батчами
    embeddings_list = []
    coords_tensor = coords_tensor.to(device)

    print(f"Генерируем эмбеддинги батчами по {batch_size} координат...")
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_coords = coords_tensor[i:i + batch_size]
            batch_embeddings = location_encoder(batch_coords)
            embeddings_list.append(batch_embeddings.cpu().numpy())

    # Объединяем все батчи
    embeddings = np.vstack(embeddings_list)

    print(f"Эмбеддинги сгенерированы: форма {embeddings.shape}")
    print(f"Размерность эмбеддинга: {embeddings.shape[1]}")

    # Сохраняем если указан путь
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embeddings)
        print(f"Эмбеддинги сохранены: {output_path}")

    return embeddings

if __name__ == '__main__':
    print('ok')
