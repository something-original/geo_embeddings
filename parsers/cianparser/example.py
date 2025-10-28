import json
import logging
import os.path

import cian.parser


def main():
    logging.getLogger().setLevel(logging.INFO)

    # Чтение GeoJSON файла в СТРОКУ (координаты должны быть в проекции EPSG:4326)
    geojson = None
    with open("moscow_less.geojson", encoding='utf-8') as f:
        geojson = f.read()

    # Объект фильтров, аналогичен тому, который отправляет фронтенд ЦИАН
    jsonQuery = {
        "_type": "flatsale",
        "offer_type": "flat",
        "deal_type": "sale",
        "minprice": {"type": "term", "value": 500000},
        "currency": {"type": "term", "value": 2},
        "engine_version": {"type": "term", "value": 2},
        "room": {"type": "terms", "value": [1,2,3,4,5]},
        #"with_newobject": {"type": "term", "value": True},  # Только новостройки
    }

    headers = {}

    # Подстановка Cookie (необязательны, только если потребовалось ввести капчу, копируются из браузера)
    cookie_filename = "cookie.txt"
    if os.path.isfile(cookie_filename):
        with open(cookie_filename, encoding='utf-8') as f:
            headers["Cookie"] = f.read().strip()

    parser = cian.parser.Parser(
        geojson=geojson,
        query=jsonQuery,
        max_tile_size=5000, # Максимальный размер границы тайла в метрах
        max_workers_collect_ids=1, # Максимальное количество воркеров, собирающих идентификаторы предложений на карте
        max_workers_collect_offers=4, # Максимальное количество воркеров, собирающих информацию о предложениях
        headers=headers,
    )

    offers = parser.parse()

    logging.info("Write offers to file")
    with open("offers_moscow_wprices.json", "w", encoding='utf-8') as f:
        for i in range(0, len(offers), len(offers)//17000):
            json.dump(offers[i], f, ensure_ascii=False)


if __name__ == "__main__":
    main()
