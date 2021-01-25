import argparse
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
    subset_size: float


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", dest="train", action="store_false")
    parser.add_argument("--config-path", type=Path, default=Path("model_config.toml"))
    for field in Config.__fields__.keys():
        parser.add_argument(f"--{field}".replace("_", "-"))
    args = parser.parse_args()
    return args


def get_config(args: argparse.Namespace) -> Config:
    with open(args.config_path) as f:
        d = toml.loads(f.read())

    return Config(
        **{
            **d,
            **{k: v for k, v in vars(args).items() if v and k in Config.__fields__},
        }
    )
