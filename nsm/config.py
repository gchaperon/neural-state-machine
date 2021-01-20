from pydantic import BaseModel
from typing import Literal, Union
from pathlib import Path
import toml


class Config(BaseModel):
    learn_rate: float
    batch_size: int
    # Possibly remove `epochs` since in the paper they use early stopping
    epochs: int
    computation_steps: int
    dropout: float
    glove_dim: Literal[50, 100, 200, 300]
    data_path: Path


def get_config(config_path: Union[str, Path]) -> Config:
    with open(config_path) as f:
        d = toml.loads(f.read())
    return Config(**d)
