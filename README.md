# Гео-эмбеддинги

Мультимодальные векторные представления географических локаций для многопрофильных downstream задач.

## 📊 Структура таблицы результатов бенчмарка

| Задача (таргет) | Модель эмбеддингов | Фичи | ML-модель | Метрика | Улучшение |
|-----------------|-------------------|------|-----------|---------|-----------|
| Price           | None (baseline)  | Original | CatBoost | X.XXX | - |
| Price           | EfficientNet     | Original + Emb | CatBoost | X.XXX | +Y% |
| Price           | TabPFN           | Original + Emb | CatBoost | X.XXX | +Y% |
| Price           | GNN              | Original + Emb | CatBoost | X.XXX | +Y% |
| Price           | GeoCLIP          | Original + Emb | CatBoost | X.XXX | +Y% |
| Price           | S2Vec            | Original + Emb | CatBoost | X.XXX | +Y% |
| Price           | SatCLIP          | Original + Emb | CatBoost | X.XXX | +Y% |
| AverageBill     | None (baseline)  | Original | CatBoost | X.XXX | - |
| AverageBill     | EfficientNet     | Original + Emb | CatBoost | X.XXX | +Y% |
| ...             | ...              | ... | ... | ... | ... |

**Примечание:** 
- **Original** - оригинальные табличные фичи
- **Original + Emb** - оригинальные фичи + эмбеддинги модели
- Метрики: RMSE, R², MAPE
- Улучшение рассчитывается относительно baseline

## 🔍 Кратко о проекте

Проект нацелен на создание универсальных эмбеддингов географических точек путем объединения:
- **~80 табличных признаков** (социально-экономические показатели, инфраструктура)
- **Растровых данных** (дорожная инфраструктура, тип застройки)
- **Графовых структур** (взаимосвязи точек на карте)

Источники данных: ФНС, муниципальная статистика, ЦИАН, OpenStreetMap.

## 🛠️ Методы

| Модальность       | Метод обработки         | Выходная размерность |
|-------------------|-------------------------|----------------------|
| Табличные данные  | TabPFN                  | 192                  |
| Графовые данные   | GNN                     | 192                  |
| Растровые данные  | EfficientNet (B7)       | 2600                 |

## 🌟 Результаты моделирования

| Регион                | Catboost | TabPFN (10k) |
|-----------------------|----------|--------------|
| Moscow (50k)          | 7.386M   | 7.891M       |
| Saint-Petersburg (30k)| 2.588M   | 2.552M       |
| Ekaterinburg (10k)    | 1.323M   | 1.218M       |
| Moscow + municipal    | **5.210M***   | 5.781M       |
| Saint-Petersburg + mun.| 2.054M   | 2.151M       |
| Ekaterinburg + mun.   | 0.937M   | 0.931M       |

## 🌟 Валидация эмбеддингов

'*' means cross-validation with p-test (95%) applied

| Moscow+mun. (Catboost)| Catboost  |
|-----------------------|-----------|
| no_embs               | 5.210M    |
| + tabpfn_emb          | 5.215M    |
| + rastrs_emb          | **4.967M*** |

| Msc_invoices. (Catboost)| Catboost  |
|-----------------------|-----------|
| no_embs               | **37.46*** |
| + tabpfn_emb          | 61.94* |
| + rastrs_emb          | 63.37* |

