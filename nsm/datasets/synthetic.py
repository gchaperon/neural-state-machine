import os
from contextlib import contextmanager
import pytorch_lightning as pl
from itertools import product
import torch
from torch import Tensor
import torch.utils.data as data
from dataclasses import dataclass, field
import typing as tp
import collections.abc as abc
from functools import singledispatchmethod, partial
import random

from nsm.utils import NSMItem, Graph, collate_nsmitems


@dataclass
class Concepts:
    n_objects: int
    n_relations: int

    embedded_concepts: Tensor = field(init=False, repr=False)
    property_embeddings: Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.embedded_concepts = self.embed(self.concepts)

        property_embeddings = torch.zeros(2, len(self))
        property_embeddings[0, self.objects] = 1e6
        property_embeddings[1, self.relations] = 1e6
        self.property_embeddings = property_embeddings

    @property
    def concepts(self) -> tp.Sequence[int]:
        return range(len(self))

    @property
    def objects(self) -> tp.Sequence[int]:
        return range(self.n_objects)

    @property
    def relations(self) -> tp.Sequence[int]:
        return range(self.n_objects, len(self))

    def __len__(self) -> int:
        return self.n_objects + self.n_relations

    @singledispatchmethod
    def embed(self, arg):
        raise NotImplementedError(f"Invalid argument type {type(arg)}")

    @embed.register
    def _(self, concept: int) -> Tensor:
        out = torch.zeros(len(self))
        out[concept] = 1
        return out

    @embed.register
    def _(self, concepts: abc.Sequence) -> Tensor:
        return torch.stack([self.embed(concept) for concept in concepts])

    @embed.register
    def _(self, graph: Graph) -> Graph:
        fields = ("node_attrs", "edge_attrs")
        return graph._replace(
            **{field: self.embed(getattr(graph, field)) for field in fields},
            edge_indices=torch.tensor(graph.edge_indices).T.contiguous(),
        )


@contextmanager
def seeded_context(seed: tp.Optional[int] = None):
    if seed is None:
        yield
    else:
        old_state = random.getstate()
        random.seed(seed)
        yield
        random.setstate(old_state)


def random_scenegraph(vocab: Concepts, n_nodes: int, n_edges: int) -> Graph:
    nodes = [
        [el]
        for el in random.sample(
            vocab.objects,
            k=max(
                1,
                min(
                    len(vocab.objects),
                    int(random.gauss(n_nodes, 0.2 * n_nodes)),
                ),
            ),
        )
    ]
    edge_indices = random.choices(
        list(product(range(len(nodes)), repeat=2)),
        k=max(1, int(random.gauss(n_edges, 0.2 * n_edges))),
    )
    edge_attrs = random.choices(vocab.relations, k=len(edge_indices))
    return Graph(nodes, edge_indices, edge_attrs)


def random_question_and_target(
    sg: Graph, question_length: int, jump_prob: float
) -> tp.Tuple[tp.List[int], int]:
    assert question_length > 1
    question = []
    current = -1
    for _ in range(question_length - 1):
        adjacent = [
            (m, attr)
            for (n, m), attr in zip(sg.edge_indices, sg.edge_attrs)
            if n == current
        ]
        should_jump = random.random() < jump_prob
        if len(adjacent) == 0 or should_jump:
            next_node = random.choice(range(len(sg.node_attrs)))
            question.append(sg.node_attrs[next_node][0])
        else:
            next_node, attr = random.choice(adjacent)
            question.append(attr)

        current = next_node
    question.append(question[0])
    # current != sg.node_attrs[current]
    return question, sg.node_attrs[current][0]
    # return question, current


@dataclass
class SyntheticDataset(data.Dataset):
    vocab: Concepts
    n_nodes: float
    n_edges: float
    size: int
    jump_prob: float
    question_length: int

    scenegraphs: tp.List[Graph] = field(init=False, repr=False)
    questions: tp.List[tp.List[int]] = field(init=False, repr=False)
    targets: tp.List[int] = field(init=False, repr=False)

    def __post_init__(self):
        self.scenegraphs = [
            random_scenegraph(self.vocab, self.n_nodes, self.n_edges)
            for _ in range(self.size)
        ]
        self.questions = []
        self.targets = []
        for sg in self.scenegraphs:
            q, t = random_question_and_target(sg, self.question_length, self.jump_prob)
            self.questions.append(q)
            self.targets.append(t)

    def __getitem__(self, key: int) -> tuple:
        return (
            self.vocab.embed(self.scenegraphs[key]),
            self.vocab.embed(self.questions[key]),
            self.targets[key],
        )

    def get(self, key: int) -> NSMItem:
        return self.scenegraphs[key], self.questions[key], self.targets[key]

    def __len__(self):
        return self.size


@dataclass
class DumbestDataset(data.Dataset):
    vocab: Concepts
    n_unique: int
    n_nodes: float
    n_edges: float
    size: int
    question_length: int
    jump_prob: float = 0.0
    seed: tp.Optional = None

    uniques: tp.List[tp.Tuple[Graph, tp.List[int], int]] = field(init=False, repr=False)
    indices: tp.List[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        assert 0 < self.n_unique <= self.size
        self.uniques = []
        with seeded_context(self.seed):
            for _ in range(self.n_unique):
                graph = random_scenegraph(self.vocab, self.n_nodes, self.n_edges)
                question, target = random_question_and_target(
                    graph, self.question_length, self.jump_prob
                )
                self.uniques.append((graph, question, target))
            self.indices = random.choices(range(self.n_unique), k=self.size)

    def __getitem__(self, key: int) -> NSMItem:
        graph, question, target = self.uniques[self.indices[key]]
        embed = self.vocab.embed
        return embed(graph), embed(question), target

    def get(self, key: int) -> tuple:
        return self.uniques[self.indices[key]]

    def __len__(self) -> int:
        return self.size


class SyntheticDataModule(pl.LightningDataModule):
    def __init__(
        self, dataset: SyntheticDataset, batch_size: int, split_ratio: float = 0.9
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.split_ratio = split_ratio

        train_len = int(self.split_ratio * len(self.dataset))
        val_len = len(self.dataset) - train_len
        self.syn_train, self.syn_val = data.random_split(
            self.dataset, [train_len, val_len]
        )

    @classmethod
    def from_splits(cls, train_split, val_split, batch_size):
        datamodule = cls(train_split, batch_size, split_ratio=1.0)
        datamodule.syn_train = train_split
        datamodule.syn_val = val_split
        return datamodule

    def _get_dataloader(self, split):
        vocab = self.dataset.vocab

        def collate(batch):
            *first, targets = collate_nsmitems(batch)
            return (
                *first,
                vocab.embedded_concepts,
                vocab.property_embeddings,
                targets,
            )

        return data.DataLoader(
            split,
            batch_size=self.batch_size,
            collate_fn=collate,
            num_workers=os.cpu_count(),
        )

    def train_dataloader(self):
        return self._get_dataloader(self.syn_train)

    def val_dataloader(self):
        return self._get_dataloader(self.syn_val)
