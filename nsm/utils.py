import dataclasses
import json
import math
import random
from collections import defaultdict, namedtuple
from functools import cached_property, wraps, partial
from itertools import zip_longest, chain
from operator import eq
from pathlib import Path
from typing import (
    Any,
    Protocol,
    Callable,
    Hashable,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    NamedTuple,
    Optional,
)
import re
import pydantic
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from torch_scatter import scatter_max, scatter_sum


@pydantic.dataclasses.dataclass
class Config:
    data_dir: Path
    embedding_size: Literal[50, 100, 200, 300]


# @dataclasses.dataclass
class Graph(NamedTuple):
    node_attrs: torch.Tensor
    edge_indices: torch.Tensor
    edge_attrs: torch.Tensor


# @dataclasses.dataclass
# class Batch(Graph):
class Batch(NamedTuple):
    node_attrs: torch.Tensor
    edge_indices: torch.Tensor
    edge_attrs: torch.Tensor
    nodes_per_graph: torch.Tensor
    edges_per_graph: torch.Tensor

    @property
    def node_indices(self) -> torch.Tensor:
        return self._get_batch_indices_from_items_per_graph(self.nodes_per_graph)

    @property
    def edge_batch_indices(self) -> torch.Tensor:
        return self._get_batch_indices_from_items_per_graph(self.edges_per_graph)

    def _get_batch_indices_from_items_per_graph(self, tensor):
        return torch.arange(tensor.size(0), device=tensor.device).repeat_interleave(
            tensor
        )

    @property
    def sparse_coo_indices(self) -> torch.Tensor:
        device = self.nodes_per_graph.device
        catenated_ranges = torch.cat(
            [torch.arange(n, device=device) for n in self.nodes_per_graph]
        )
        indices = torch.vstack((self.node_indices, catenated_ranges))
        return indices

    def to(self, *args, **kwargs) -> "Batch":
        return Batch(*[T.to(*args, **kwargs) for T in self])

    def __repr__(self) -> str:
        """ This is really informal, just for debuggin purposes """
        return "Batch:\n" + "\n".join(
            f"    {name}: {'x'.join(map(str, T.size()))}"
            for name, T in self._asdict().items()
        )


# Graph, defined above. Tensor of dimension LxH, where L is the number of tokens
# and H is the hidden size of the network (glove size). int corresponds to the index of
# the answer token (coming from the vocab of the dataset)
NSMItem = Tuple[Graph, Tensor, int]
# Batch, as defined above. PackedSequence is a nice way of managing variable-length
# sequences and the last item (Tensor) is the batch of targets, should be a LongTensor
# (or equivalent)
NSMBatch = Tuple[Batch, PackedSequence, Tensor]



def is_connected(edges: Iterable[Tuple[Hashable, Hashable]], n_nodes: int) -> bool:
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
    hidden_size: int = 300,
    n_properties: int = 78,
    node_distribution: Tuple[float, float] = (16.4, 8.2),
    density_distribution: Tuple[float, float] = (0.2, 0.4),
) -> Iterator[Graph]:
    while True:
        n_nodes = abs(round(random.gauss(*node_distribution)))
        node_attrs = torch.rand(n_nodes, n_properties, hidden_size)

        n_edges = abs(round(random.gauss(*density_distribution) * n_nodes ** 2))
        edge_index = torch.randint(n_nodes or 1, (2, n_edges))
        edge_attrs = torch.rand(n_edges, hidden_size)

        yield Graph(node_attrs, edge_index, edge_attrs)


def scatter_softmax(
    src: Tensor, index: Tensor, dim: int, dim_size: Optional[int] = None
) -> Tensor:
    if src.numel() == 0:
        return src
    slice_tuple = (slice(None),) * dim + (index,)
    expand_args = src.size()[:dim] + (-1,)
    src = src - scatter_max(src, index, dim, dim_size=dim_size)[0][slice_tuple]
    exp = torch.exp(src)
    return exp / scatter_sum(exp, index, dim, dim_size=dim_size)[slice_tuple]


def collate_graphs(batch: Sequence[Graph]) -> Batch:
    if len(batch) == 0:
        raise ValueError("Batch cannot be an empty list")
    device = batch[0].node_attrs.device
    nodes_per_graph = torch.tensor(
        [graph.node_attrs.size(0) for graph in batch], device=device
    )
    edges_per_graph = torch.tensor(
        [graph.edge_attrs.size(0) for graph in batch], device=device
    )
    node_attrs = torch.cat([graph.node_attrs for graph in batch])
    edge_attrs = torch.cat([graph.edge_attrs for graph in batch])
    edge_indices = torch.cat(
        [
            graph.edge_indices + shift
            for graph, shift in zip(batch, [0, *nodes_per_graph.cumsum(0).tolist()])
        ],
        dim=1,
    )
    return Batch(
        node_attrs,
        edge_indices,
        edge_attrs,
        nodes_per_graph,
        edges_per_graph,
    )


def collate_nsmitems(batch: List[NSMItem]) -> NSMBatch:
    graphs, questions, targets = zip(*batch)
    return (
        collate_graphs(graphs),
        pack_sequence(questions, enforce_sorted=False),
        torch.tensor(targets),
    )


def split_batch(batch: Batch) -> List[Graph]:
    node_attrs_list = batch.node_attrs.split(batch.nodes_per_graph.tolist())
    edge_attrs_list = batch.edge_attrs.split(batch.edges_per_graph.tolist())
    edge_indices_list = [
        edge_indices - shift
        for edge_indices, shift in zip(
            batch.edge_indices.split(batch.edges_per_graph.tolist(), dim=1),
            [0, *torch.cumsum(batch.nodes_per_graph, dim=0).tolist()],
        )
    ]
    return list(map(Graph, node_attrs_list, edge_indices_list, edge_attrs_list))


def to_snake(name: str):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def to_camel(name: str):
    return "".join(
        [
            word if i == 0 else word.capitalize()
            for i, word in enumerate(name.split("_"))
        ]
    )


class forwardingpartial(partial):
    def __getattr__(self, attr):
        return getattr(self.func, attr)


class partial_module(nn.Module):
    def __init__(self, module, /, *args, **kwargs):
        super(partial_module, self).__init__()
        self.module = module
        self.args = args
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        device = next(self.module.parameters()).device
        merged_args = [arg.to(device) for arg in chain(self.args, args)]
        merged_kwargs = {
            key: kwarg.to(device)
            for key, kwarg in chain(self.kwargs.items(), kwargs.items())
        }
        return self.module(*merged_args, **merged_kwargs)
