# EfficientNet Module

Модуль для генерации эмбеддингов из растровых изображений (карты, спутниковые снимки) с использованием EfficientNet B7.

## Структура модуля

- `model.py` - Класс `EfficientNetEmbedder` - модель EfficientNet B7 для генерации эмбеддингов
- `transforms.py` - Функции предобработки изображений для EfficientNet
- `embedder.py` - Высокоуровневый интерфейс `RasterEmbedder` для генерации эмбеддингов
- `__init__.py` - Экспорт основных классов и функций

## Использование

### Базовое использование

```python
from models.EfficientNet import get_embedding

# Получить эмбеддинг для одного изображения
embedding = get_embedding("path/to/image.png")
print(embedding.shape)  # (2560,)
```

### Использование RasterEmbedder

```python
from models.EfficientNet import RasterEmbedder

# Создать embedder
embedder = RasterEmbedder()

# Обработать одно изображение
embedding = embedder.embed_image("path/to/image.png")

# Обработать директорию с изображениями
embeddings_dict = embedder.embed_directory(
    directory="path/to/images",
    output_dir="path/to/output",
    pattern="*.png",
    batch_size=32
)
```

### Использование модели напрямую

```python
from models.EfficientNet import EfficientNetEmbedder
from models.EfficientNet.transforms import preprocess_image
import torch

# Создать модель
model = EfficientNetEmbedder(pretrained=True)
model.eval()

# Предобработать изображение
img_tensor = preprocess_image("path/to/image.png")

# Получить эмбеддинг
with torch.no_grad():
    embedding = model(img_tensor)
```

## Параметры

- **Размерность эмбеддинга**: 2560 (EfficientNet B7 features)
- **Размер входного изображения**: 224x224
- **Количество каналов**: 3 (RGB, даже если исходное изображение grayscale)

## Зависимости

- torch
- torchvision
- PIL (Pillow)
- numpy
- tqdm
