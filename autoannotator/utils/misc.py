from pathlib import Path
import shutil
import requests
from loguru import logger
import functools
from tqdm.auto import tqdm


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def download(url: str, path: Path, desc: str = ""):
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get("Content-Length", 0))

    desc = "(Unknown total file size)" if file_size == 0 else desc
    r.raw.read = functools.partial(
        r.raw.read, decode_content=True
    )  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with open(path, "wb") as f:
            shutil.copyfileobj(r_raw, f)
    return path


def attempt_download_onnx(path: str, url: str):
    if not Path(path).is_file():
        logger.warning(f"Downloading weights from {url} to {path}")
        try:
            download(url, path, f"Downloading {Path(path).name}")
        except Exception as ex:
            logger.error(f"Failed to download ONNX weights from {url}. Error: {ex}")
        logger.info(f"Weights saved to {path}")


def attempt_download_custom_op(path: str, url: str):
    if not Path(path).is_file():
        logger.warning(f"Downloading custom operations from {url} to {path}")
        try:
            download(url, path, f"Downloading {Path(path).name}")
        except Exception as ex:
            logger.error(
                f"Failed to download custom operations from {url}. Error: {ex}"
            )
        logger.info(f"Custom operations saved to {path}")
