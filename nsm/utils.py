import dataclasses
import itertools
import json
import math
import random
from collections import defaultdict
from functools import wraps
from operator import eq
from pathlib import Path
from typing import (
    Any,
    Hashable,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Set,
    Tuple,
)

import pydantic
import torch


@pydantic.dataclasses.dataclass
class Config:
    data_dir: Path
    embedding_size: Literal[50, 100, 200, 300]


# @dataclasses.dataclass
class Graph(NamedTuple):
    node_attrs: torch.Tensor
    edge_index: torch.Tensor
    edge_attrs: torch.Tensor


# TODO: terminar esta funcioncita
def disk_cache(file_name: str) -> Any:
    def deco(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            pass


def check_conectivity_temp(fp):
    imgs = json.load(fp)


def is_connected(
    edges: Iterable[Tuple[Hashable, Hashable]], n_nodes: int
) -> bool:
    """Check for (weak) connectivity"""
    adj_set: Mapping[Hashable, set] = defaultdict(set)
    for fr, to in edges:
        adj_set[fr].add(to)
        adj_set[to].add(fr)

    if len(adj_set) > n_nodes:
        raise ValueError("to many nodes in edge list")

    try:
        stack = [next(iter(adj_set.keys()))]
    except StopIteration:
        return n_nodes == 1

    seen = set()
    while stack:
        val = stack.pop()
        if val in seen:
            continue
        seen.add(val)
        for neigh in adj_set[val]:
            stack.append(neigh)

    return len(seen) == n_nodes


def infinite_graphs(
    hidden_size: int,
    n_properties: int,
    node_distribution: Tuple[float, float],
    density_distribution: Tuple[float, float],
) -> Iterator[Graph]:
    while True:
        n_nodes = abs(round(random.gauss(*node_distribution)))
        if n_nodes == 0:
            # In the GQA dataset there are images that have no objects
            # in it. I think we should filter those, so i'm doing
            # the same here
            continue
        node_attrs = torch.rand(n_nodes, n_properties, hidden_size)

        n_edges = abs(
            round(random.gauss(*density_distribution) * n_nodes ** 2)
        )
        edge_index = torch.randint(n_nodes, (2, n_edges))
        edge_attrs = torch.rand(n_edges, hidden_size)

        yield Graph(node_attrs, edge_index, edge_attrs)
