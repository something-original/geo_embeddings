import os
from ast import literal_eval
from dotenv import load_dotenv
import torch

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
HF_BEST_REPO_ID = os.getenv('HF_BEST_REPO_ID')
HF_BEST_REPO_TYPE = os.getenv('HF_BEST_REPO_TYPE') or 'model'
HF_BEST_REPO_PRIVATE = (os.getenv('HF_BEST_REPO_PRIVATE') or 'false') == 'true'
HF_BEST_REPO_REVISION = os.getenv('HF_BEST_REPO_REVISION')
HF_BEST_PATH_IN_REPO = os.getenv('HF_BEST_PATH_IN_REPO') or 'main'


os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

TOCHNO_ST_BASE_LINK = os.getenv('TOCHNO_ST_BASE_LINK') or 'https://storage.yandexcloud.net/tochno-st-catalog/Rosstat/data_bdmo_118_v20250918/indicators/section<section_id>/data_Y4<dataset_code>_112_v20250918.zip'
OVERPASS_API_LINK = os.getenv('OVERPASS_API_LINK') or 'https://overpass.openstreetmap.fr/api/'
MUN_GEOMS_LINK = os.getenv('MUN_GEOMS_LINK') or 'https://s.sber.ru/GthXk7'
FLATS_TASK_LINK = os.getenv('FLATS_TASK_LINK') or 'https://raw.githubusercontent.com/mishannn/cianparser/main/offers.json'

CHECK_PEOPLE_WORKPLACES_TASKS = (os.getenv('CHECK_PEOPLE_WORKPLACES_TASKS') or 'false') == 'true'
PEOPLE_FEATURES = literal_eval(os.getenv('PEOPLE_FEATURES') or '[]')
PEOPLE_CAT_FEATURES = literal_eval(os.getenv('PEOPLE_CAT_FEATURES') or '[]')
WORKPLACES_FEATURES = literal_eval(os.getenv('WORKPLACES_FEATURES') or '[]')
WORKPLACES_CAT_FEATURES = literal_eval(os.getenv('WORKPLACES_CAT_FEATURES') or '[]')

EXPERIMENT_TARGET_FEATURES = [
    'Доходы местного бюджета, фактически исполненные',
    'Расходы местного бюджета, фактически исполненные'
]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BENCHMARK_LOG_PATH = "logs/models_tasks_performance.log"

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
