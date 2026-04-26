import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
TOCHNO_ST_BASE_LINK = os.getenv('TOCHNO_ST_BASE_LINK') or 'https://storage.yandexcloud.net/tochno-st-catalog/Rosstat/data_bdmo_118_v20250918/indicators/section<section_id>/data_Y4<dataset_code>_112_v20250918.zip'
OVERPASS_API_LINK = os.getenv('OVERPASS_API_LINK') or 'https://overpass.openstreetmap.fr/api/'
MUN_GEOMS_LINK = os.getenv('MUN_GEOMS_LINK') or 'https://s.sber.ru/GthXk7'
FLATS_TASK_LINK = os.getenv('FLATS_TASK_LINK') or 'https://raw.githubusercontent.com/mishannn/cianparser/main/offers.json'
