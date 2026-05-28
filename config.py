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
MUN_GEOMS_LINK = os.getenv('MUN_GEOMS_LINK') or 'https://downloader.disk.yandex.ru/disk/f1759b4f680533ade9936b664b33d774141e57625b4e593c467e6e6fc3ac274c/6a18efd4/DMMYV1dENCl7Exq5oXEdXagvA-PdsaDUsC4u4UZXxH0vghTsly1iSWxGvKvo_h8oQ87c418STMLuXTtg_RxwVw%3D%3D?uid=0&filename=t_dict_municipal.zip&disposition=attachment&hash=NUETVyj7iPjeWEcS5/MtcNeP1mI0F%2BlVG45MxwcZaDFnOjswUZ5vmQptq9OF4QJQq/J6bpmRyOJonT3VoXnDag%3D%3D&limit=0&content_type=application%2Fzip&owner_uid=421343728&fsize=52943185&hid=5a9e0314a1c94873538ef06c26acb57c&media_type=compressed&tknv=v3&is_direct_zip_experiment=1'
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

_device_env = (os.getenv("DEVICE") or "").strip().lower()
if _device_env in ("cuda", "cpu"):
    DEVICE = _device_env
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BENCHMARK_LOG_PATH = "logs/models_tasks_performance.log"

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '1414'))

DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
DB_PORT = os.getenv('DB_PORT', '1212')
DB_NAME = os.getenv('DB_NAME', 'geo_embeddings')
DB_USER = os.getenv('DB_USER', 'georgiykiselev')
DB_PWD = os.getenv('DB_PWD', 'georgiykiselev')

DB_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PWD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "municipality_embeddings")
QDRANT_HTTPS = (os.getenv("QDRANT_HTTPS") or "false").lower() == "true"

INIT_EMBEDDINGS = (os.getenv("INIT_EMBEDDINGS") or "true").lower() == "true"
SEVEN_ZIP_BIN = os.getenv("SEVEN_ZIP_BIN") or os.getenv("SEVENZIP_BIN") or ""
