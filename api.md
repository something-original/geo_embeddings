# Geo Embeddings API

Базовый URL: `http://{HOST}:{PORT}` (по умолчанию `http://0.0.0.0:1414`).

Переменная окружения **`INIT_EMBEDDINGS`** (по умолчанию `true`): при `true` при старте вызываются `parse_mun_data` и `form_mun_geometry`; при `false` сервис ждёт загрузку данных через API (`/datasets/*`).

Все эндпоинты загрузки и обучения возвращают **200 OK** сразу после приёма запроса; тяжёлая обработка выполняется в **отдельном процессе**.

---

## Datasets (`/datasets`)

### `POST /datasets/upload_territories`

Загрузка территорий в таблицу `municipalities`. Геометрия приводится к **EPSG:4326** и сохраняется в колонку `geometry`. Схема таблицы синхронизируется со столбцами CSV (лишние атрибуты удаляются, недостающие добавляются).

**Content-Type:** `multipart/form-data`

| Поле | Тип | Описание |
|------|-----|----------|
| `file` | file | CSV с геометрией и атрибутами |
| `params` | string (JSON) | Параметры загрузки |

**`params` (JSON):**

```json
{
  "crs": "EPSG:4326",
  "geom_col": "geometry",
  "index_col": "territory_id"
}
```

| Ключ | Обязательный | Описание |
|------|--------------|----------|
| `crs` | нет | Исходная проекция (по умолчанию `EPSG:4326`) |
| `geom_col` | да | Имя столбца с геометрией (WKT) в файле |
| `index_col` | нет | Столбец для `id`; если не указан — порядковый индекс + 1 |

**Ответ:**

```json
{
  "status": "accepted",
  "message": "Territories file accepted; processing in background"
}
```

---

### `POST /datasets/upload_feature_values`

Загрузка признаков в БД и подготовка локальных CSV для пайплайна эмбеддингов.

**Content-Type:** `multipart/form-data`

| Поле | Тип | Описание |
|------|-----|----------|
| `files` | file[] | Один или два CSV |
| `params` | string (JSON) | Параметры |

**`params` (JSON):**

```json
{
  "index_col": "municipality_id",
  "target_col": "income",
  "geom_col": null,
  "crs": "EPSG:4326",
  "same_file_inference": true,
  "train_file_name": "train.csv",
  "inference_file_name": "inference.csv"
}
```

| Ключ | Обязательный | Описание |
|------|--------------|----------|
| `index_col` | нет | Столбец ID территории → `municipality_id`; иначе индекс строки |
| `target_col` | нет | Имя столбца-таргета → `target_{id}` в БД |
| `geom_col` | нет | Если задан — геометрия берётся из инференс-файла (нужен `crs`), территории пишутся в `municipalities`. Если `null` — считается, что `upload_territories` уже вызван |
| `crs` | нет | CRS геометрии в файле (при `geom_col`) |
| `same_file_inference` | нет | `true` — один файл для train и inference; `train_file_name` / `inference_file_name` игнорируются |
| `train_file_name` | при `same_file_inference=false` | Имя загруженного train-файла |
| `inference_file_name` | при `same_file_inference=false` | Имя загруженного inference-файла |

**Поведение:**

1. Столбцы признаков (кроме ID, геометрии, таргета) записываются в `indicators` с `id` по порядку и `name` = имя столбца.
2. В таблицах значений столбцы переименовываются: `ind_{id}`, таргет — `target_{id}`.
3. `same_file_inference=true` → данные в `indicator_values_inference`; локально — `indicator_values_inference.csv`.
4. `same_file_inference=false` → train в `indicator_values_train`, inference в `indicator_values_inference`; из inference удаляется таргет; в `municipalities` остаются только ID из inference; локально — `indicator_values_old.csv` (train) и `indicator_values.csv` (inference).
5. После записи столбец геометрии из датафреймов удаляется.
6. Вызывается `prepare_and_save_dataset` для `indicator_values_train/val/inference.csv`.

**Ответ:** как у `upload_territories`.

---

## Embeddings (`/embeddings`)

### `POST /embeddings/from_polygon`

(существующий) Эмбеддинги по пересечению с GeoJSON-полигоном.

### `POST /embeddings/train_and_inference`

Обучение моделей эмбеддингов на train-выборке и инференс на inference-выборке (как `run_experiments` в `benchmark.py`), загрузка лучших векторов в Qdrant. Локальные промежуточные CSV удаляются по завершении.

**Body (JSON, опционально):**

```json
{
  "emb_dims": [128, 192, 256]
}
```

**Ответ:**

```json
{
  "status": "accepted",
  "message": "Training and inference started in background"
}
```

---

## Stats (`/stats`)

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/stats/` | Сводка: модель, размерность, размеры выборок, индикаторы, число точек в Qdrant |
| GET | `/stats/model` | `{"model": "s2vec", "embedding_dim": 128}` |
| GET | `/stats/embedding-dimension` | `{"embedding_dim": 128}` |
| GET | `/stats/dataset-sizes` | Размеры train/inference (БД и локальные CSV) |
| GET | `/stats/indicators` | Список `[{"id": 1, "name": "..."}, ...]` |

---

## Рекомендуемый порядок вызовов

1. **Вариант A:** `upload_territories` → `upload_feature_values` (`geom_col: null`)
2. **Вариант B:** один вызов `upload_feature_values` с `geom_col` и `crs`
3. `train_and_inference`
4. `GET /stats/` — проверка результата

---

## Примеры (curl)

```bash
# Территории
curl -X POST "http://localhost:1414/datasets/upload_territories" \
  -F 'file=@territories.csv' \
  -F 'params={"crs":"EPSG:3857","geom_col":"wkt_geom","index_col":"id"}'

# Признаки (один файл)
curl -X POST "http://localhost:1414/datasets/upload_feature_values" \
  -F 'files=@features.csv' \
  -F 'params={"index_col":"id","target_col":"budget","same_file_inference":true}'

# Обучение
curl -X POST "http://localhost:1414/embeddings/train_and_inference" \
  -H "Content-Type: application/json" \
  -d '{"emb_dims":[128,256]}'

# Статистика
curl "http://localhost:1414/stats/"
```
