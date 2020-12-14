import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from nsm.utils import Batch, matmul_memcapped

from typing import Optional, List, Union, Tuple


class Tagger(nn.Module):
    def __init__(self, embedding_size) -> None:
        super(Tagger, self).__init__()

        self.default_embedding = nn.Parameter(torch.rand(embedding_size))
        self.weight = nn.Parameter(torch.eye(embedding_size))

    def forward(
        self, vocab: Tensor, question_batch: PackedSequence
    ) -> PackedSequence:
        tokens, *rest = question_batch
        similarity = F.softmax(
            tokens
            @ self.weight
            @ torch.vstack((vocab, self.default_embedding)).T,
            dim=1,
        )
        concept_based = (
            similarity[:, -1:] * tokens + similarity[:, :-1] @ vocab
        )
        return PackedSequence(concept_based, *rest)


class InstructionDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_instructions: int,
        bias: bool = True,
        nonlinearity: str = "relu",
    ) -> None:
        super(InstructionDecoder, self).__init__()
        self.n_instructions = n_instructions
        self.rnn_cell = nn.RNNCell(input_size, hidden_size, bias, nonlinearity)

    def forward(self, input: Tensor) -> Tensor:
        # Explicit better than implicit, amarite (?)
        hx: torch.Tensor = torch.zeros_like(input)
        hiddens: List[Tensor] = []
        for _ in range(self.n_instructions):
            hx = self.rnn_cell(input, hx)
            hiddens.append(hx)
        return torch.stack(hiddens, dim=1)


class InstructionsModel(nn.Module):
    def __init__(self, embedding_size: int, n_instructions: int) -> None:
        super(InstructionsModel, self).__init__()

        self.tagger = Tagger(embedding_size)
        self.encoder = nn.LSTM(
            input_size=embedding_size, hidden_size=embedding_size, dropout=0.0
        )
        self.decoder = InstructionDecoder(
            input_size=embedding_size,
            hidden_size=embedding_size,
            n_instructions=n_instructions,
        )

    def forward(
        self, vocab: Tensor, question_batch: PackedSequence
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input should be a PackedSequence
        tagged = self.tagger(vocab, question_batch)
        encoded = self.encoder(tagged)[1][0].squeeze()  # get last hidden
        hidden = self.decoder(encoded)
        # Unpack sequences
        tagged_unpacked, lens_unpacked = pad_packed_sequence(
            tagged, batch_first=True
        )
        # Prepare mask for attention
        max_seq_len = tagged_unpacked.size(1)
        batch_size, n_instructions = hidden.size()[:2]
        # Intermediate multiplication
        tmp = hidden @ tagged_unpacked.transpose(1, 2)
        # Mask values for softmax
        tmp[
            torch.arange(batch_size).repeat_interleave(
                max_seq_len - lens_unpacked
            ),
            :,
            torch.cat([torch.arange(l, max_seq_len) for l in lens_unpacked]),
        ] = float("-inf")
        # Instructions
        instructions = F.softmax(tmp, dim=-1) @ tagged_unpacked
        return instructions, encoded


class NSMCell(nn.Module):
    def __init__(self, prop_embeds: torch.Tensor) -> None:
        super(NSMCell, self).__init__()

        n_properties, hidden_size = prop_embeds.size()
        self.prop_embeds = nn.Parameter(prop_embeds, requires_grad=False)
        self.Ws_property = nn.Parameter(
            torch.rand(n_properties, hidden_size, hidden_size)
        )
        # self.W_edge = nn.Parameter(torch.rand(*[hidden_size] * 2))
        self.W_state = nn.Parameter(torch.rand(hidden_size))
        self.W_relation = nn.Parameter(torch.rand(hidden_size))

    def forward(
        self,
        batch: Batch,
        instruction: torch.Tensor,
        distribution: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute similarity between instruction and attribute
        # categories, aka properties
        prop_similarities = F.softmax(instruction @ self.prop_embeds.T, dim=1)
        # breakpoint()
        # Compute node and edge score
        node_scores = F.elu(
            torch.sum(
                prop_similarities[batch.node_indices, :-1, None]
                * instruction[batch.node_indices, None]
                * matmul_memcapped(
                    self.Ws_property[:-1],
                    batch.node_attrs.unsqueeze(-1),
                    # 8 gigs memory cap
                    memory_cap=5 * 10 ** 9,
                ).squeeze(),
                dim=1,
            )
        )
        edge_scores = F.elu(
            instruction[batch.edge_batch_indices]
            * self.Ws_property[-1]
            .matmul(batch.edge_attrs.unsqueeze(-1))
            .squeeze()
        )
        # Compute state component for next distribution
        next_distribution_states = (
            torch.sparse.softmax(
                torch.sparse_coo_tensor(
                    batch.sparse_coo_indices, self.W_state @ node_scores.T
                ),
                dim=1,
            )
            .coalesce()
            .values()
        )
        # Compute neighbour component for next distribution
        next_distribution_relations = (
            torch.sparse.softmax(
                torch.sparse_coo_tensor(
                    batch.sparse_coo_indices,
                    self.W_relation
                    @ torch.zeros_like(node_scores)
                    .index_add_(
                        0,
                        batch.edge_indices[1],
                        distribution[batch.edge_indices[0], None]
                        * edge_scores,
                    )
                    .T,
                ),
                dim=1,
            )
            .coalesce()
            .values()
        )
        # Compute next distribution
        # breakpoint()
        next_distribution = (
            prop_similarities[batch.node_indices, -1]
            * next_distribution_relations
            + (1 - prop_similarities[batch.node_indices, -1])
            * next_distribution_states
        )
        return next_distribution, prop_similarities


class NSM(nn.Module):
    def __init__(
        self,
        vocab: torch.Tensor,
        prop_embeds: torch.Tensor,
        computation_steps: int,
        out_size: int,
    ) -> None:
        super(NSM, self).__init__()
        self.instructions_model = InstructionsModel(
            vocab, n_instructions=computation_steps
        )
        self.nsm_cell = NSMCell(prop_embeds)
        vocab_size, hidden_size = vocab.size()
        self.linear = nn.Linear(2 * hidden_size, out_size)

    def forward(
        self, graph_batch: Batch, question_batch: PackedSequence
    ) -> torch.Tensor:
        instructions, encoded_questions = self.instructions_model(
            question_batch
        )
        # Initialize distribution
        distribution = (1 / graph_batch.nodes_per_graph)[
            graph_batch.node_indices
        ]
        # Just so that mypy doesn't complain
        prop_similarities = torch.empty([])
        # Simulate execution of finite automaton
        for instruction in instructions.transpose(0, 1):
            distribution, prop_similarities = self.nsm_cell(
                graph_batch, instruction, distribution
            )

        # breakpoint()
        aggregated: torch.Tensor = torch.zeros_like(
            encoded_questions
        ).index_add_(
            0,
            graph_batch.node_indices,
            distribution[:, None]
            * torch.sum(
                prop_similarities[graph_batch.node_indices, :-1, None]
                * graph_batch.node_attrs,
                dim=1,
            ),
        )
        return self.linear(torch.hstack((encoded_questions, aggregated)))
