from pathlib import Path
from pydantic.dataclasses import dataclass
from typing import Any
from functools import wraps


@dataclass
class Config:
    data_dir: Path


# TODO: terminar esta funcioncita
def disk_cache(file_name: str) -> Any:
    def deco(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            pass
