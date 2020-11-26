from functools import wraps
from pathlib import Path
from typing import Any, Literal

from pydantic.dataclasses import dataclass


@dataclass
class Config:
    data_dir: Path
    embedding_size: Literal[50, 100, 200, 300]


# TODO: terminar esta funcioncita
def disk_cache(file_name: str) -> Any:
    def deco(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            pass
