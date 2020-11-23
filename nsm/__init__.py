#!/usr/bin/env python
import argparse
from pathlib import Path
from functools import cached_property
import json
from nsm.utils import Config
from typing import List, Optional
import ijson


class GQAVocab:
    def __init__(self, args):
        self.gqa_scene_graphs_dir = args.data_dir / "GQA" / "sceneGraphs"
        self.vg_dir = args.data_dir / "VG_v1.v"

    @cached_property
    def _all_object_names(self):
        scene_graphs_train_file = self.gqa_scene_graphs_dir / "train_sceneGraphs.json"
        with scene_graphs_train_file.open() as f:
            return {
                obj["name"]
                for scene_graph in json.load(f).values()
                for obj in scene_graph["objects"].values()
            }

    @cached_property
    def object_names(self):
        pass


def parse_args(args: Optional[List[str]] = None) -> Config:
    parser = argparse.ArgumentParser("Train some models.")
    parser.add_argument("--data-dir", default="data", type=Path)
    config = Config(**vars(parser.parse_args(args)))  # type: ignore
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
    objects = get_objects()
    breakpoint()


if __name__ == "__main__":
    main()
