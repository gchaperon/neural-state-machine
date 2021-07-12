import logging
import requests
import tqdm
import typing as tp
from pathlib import Path

logger = logging.getLogger(__name__)


def download(url: str, out_path: tp.Union[str, Path], progress: bool = True) -> None:
    """Download file from url to out_path. Noop if out_path exists"""
    out_path = Path(out_path)
    if out_path.exists():
        return

    try:
        response = requests.get(url, stream=True)
        total = response.headers.get("Content-Length")
        logger.info(f"Downloading {url} to {out_path}.")
        with open(out_path, "wb") as out_file, tqdm.tqdm(
            desc="Progress",
            total=total and int(total),
            disable=not progress,
            unit="b",
            unit_scale=True,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=2 ** 18):
                out_file.write(chunk)
                progress_bar.update(len(chunk))
        logger.info("Done!")
    except BaseException as e:
        logger.error(f"Something went wrong, deleting {out_path}.")
        if out_path.exists():
            out_path.unlink()
        raise e from None
