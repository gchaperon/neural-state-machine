#!/usr/bin/env python
import argparse
import json
import sys
from functools import cached_property
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from pydantic import ValidationError

from nsm.utils import Config

def parse_args(args: Optional[List[str]] = None) -> Config:
    parser = argparse.ArgumentParser("Train some models.")
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--embedding-size", default=300, type=int)
    try:
        config = Config(**vars(parser.parse_args(args)))  # type: ignore
    except ValidationError as e:
        print(e)
        sys.exit(1)

    return config


def vocab_sandbox(config: Config) -> None:
    pass


def get_objects():
    # possibly should have more logic, using synsets n' stuff, for now
    # objects with more than one token should be removed, i think glove
    # has single token concepts only
    f = open("data/GQA/sceneGraphs/train_sceneGraphs.json")
    objects = (
        obj
        for scene_graph in json.load(f).values()
        for obj in scene_graph["objects"].values()
    )
    return {obj["name"] for obj in objects if len(obj["name"].split()) == 1}


def main(args: Optional[List[str]] = None) -> None:
    config = parse_args(args)
    # objects = get_objects()
    breakpoint()


if __name__ == "__main__":
    main()
