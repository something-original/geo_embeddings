# Гео-эмбеддинги

Мультимодальные векторные представления географических локаций для многопрофильных downstream задач.

Переменные окружения:
- HF_TOKEN=токен HF для выгрузки моделей
- TOCHNO_ST_BASE_LINK=базовая ссылка для загрузки датасетов по муниципальным образованиям

    Значение по умолчанию:\
    https://storage.yandexcloud.net/tochno-st-catalog/Rosstat/data_bdmo_118_v20250918/indicators/section<section_id>/data_Y4<dataset_code>_112_v20250918.zip

- OVERPASS_API_LINK=ссылка на интерпретатор Overpass для загрузки границ муниципальных образований

    Значение по умолчанию:\
    overpass.openstreetmap.fr/api/