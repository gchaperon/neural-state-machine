import dataclasses
from itertools import zip_longest
import json
import math
import random
from collections import defaultdict
from functools import wraps, cached_property
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
    Union,
    TypeVar,
    Sequence,
    Callable,
)

import pydantic
import torch


@pydantic.dataclasses.dataclass
class Config:
    data_dir: Path
    embedding_size: Literal[50, 100, 200, 300]


@dataclasses.dataclass(frozen=True)
class Graph:
    node_attrs: torch.Tensor
    edge_indices: torch.Tensor
    edge_attrs: torch.Tensor


@dataclasses.dataclass(frozen=True)
class Batch(Graph):
    nodes_per_graph: torch.Tensor
    edges_per_graph: torch.Tensor

    @property
    def node_indices(self) -> torch.Tensor:
        return self._get_batch_indices_from_items_per_graph(
            self.nodes_per_graph
        )

    @property
    def edge_batch_indices(self) -> torch.Tensor:
        return self._get_batch_indices_from_items_per_graph(
            self.edges_per_graph
        )

    def _get_batch_indices_from_items_per_graph(self, tensor):
        return torch.arange(
            tensor.size(0), device=tensor.device
        ).repeat_interleave(tensor)

    @property
    def sparse_coo_indices(self) -> torch.Tensor:
        device = self.nodes_per_graph.device
        catenated_ranges = torch.cat(
            [torch.arange(n, device=device) for n in self.nodes_per_graph]
        )
        indices = torch.vstack((self.node_indices, catenated_ranges))
        return indices

    def to(self, *args, **kwargs) -> "Batch":
        return Batch(*[T.to(*args, **kwargs) for T in vars(self).values()])

    def __repr__(self) -> str:
        """ This is really informal, just for debuggin purposes """
        return "Batch:\n" + "\n".join(
            f"    {name}: {'x'.join(map(str, T.size()))}"
            for name, T in vars(self).items()
        )


# TODO: terminar esta funcioncita
# def disk_cache(file_name: str) -> Any:
#    def deco(f):
#        @wraps(f)
#        def wrapper(*args, **kwargs):
#            pass


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


def collate_graphs(
    batch: List[Graph], device: Union[str, torch.device] = "cpu"
) -> Batch:
    nodes_per_graph = torch.tensor(
        [graph.node_attrs.size(0) for graph in batch]
    )
    edges_per_graph = torch.tensor(
        [graph.edge_attrs.size(0) for graph in batch]
    )
    node_attrs = torch.cat([graph.node_attrs for graph in batch])
    edge_attrs = torch.cat([graph.edge_attrs for graph in batch])
    edge_indices = torch.cat(
        [
            graph.edge_indices + shift
            for graph, shift in zip(
                batch, [0, *nodes_per_graph.cumsum(0).tolist()]
            )
        ],
        dim=1,
    )
    return Batch(
        node_attrs.to(device),
        edge_indices.to(device),
        edge_attrs.to(device),
        nodes_per_graph.to(device),
        edges_per_graph.to(device),
    )


def broadcast_size(s1: Sequence[int], s2: Sequence[int]) -> Sequence[int]:
    assert (
        len(s1) > 0 and len(s2) > 0
    ), "Each tensor must have at least one dimension"

    out: List[int] = []
    for d1, d2 in zip_longest(reversed(s1), reversed(s2), fillvalue=1):
        assert d1 == d2 or any(
            d == 1 for d in (d1, d2)
        ), "Dimensions must be either equal or one of them is one"
        out.append(max(d1, d2))

    # fuckit
    return type(s1)(reversed(out))  # type:ignore


def matmul_split_size(
    t1: torch.Tensor, t2: torch.Tensor, max_memory: int = 10
):
    pass
