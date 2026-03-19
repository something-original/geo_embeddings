from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models import S2VecModel


def train_s2vec(*args, **kwargs):
    pass
