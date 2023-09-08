from pathlib import Path
import shutil
import requests
from loguru import logger


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def attempt_download_onnx(path: str, url: str):
    if not Path(path).is_file():
        logger.warning(f"Downloading weights from {url} to {path}")
        with requests.get(url, stream=True) as r:
            with open(path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
