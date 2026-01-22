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

