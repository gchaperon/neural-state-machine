from itertools import islice
import torch
import random
from torch import Tensor
import torch.utils.data as data
from nsm.utils import NSMItem, infinite_graphs, Graph
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class RandomDataset(data.Dataset[NSMItem]):
    graphs: List[Graph]
    questions: List[Tensor]
    answers: List[int]

    def __init__(
        self,
        size: int,
        hidden_size: int,
        n_token_distribution: Tuple[float, float],
        n_properties: int,
        node_distribution: Tuple[float, float],
        density_distribution: Tuple[float, float],
        answer_vocab_size: int,
    ) -> None:

        self.graphs = list(
            islice(
                infinite_graphs(
                    hidden_size, n_properties, node_distribution, density_distribution
                ),
                size,
            )
        )
        self.questions = [
            torch.rand(max(1, int(random.gauss(*n_token_distribution))), hidden_size)
            for _ in range(size)
        ]
        self.answers = random.choices(range(answer_vocab_size), k=size)

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, key: int) -> NSMItem:
        return self.graphs[key], self.questions[key], self.answers[key]

class RandomConceptVocab:
    def __init__(self) ->None:
        pass
