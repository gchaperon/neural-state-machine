import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from nsm.utils import Batch, matmul_memcapped

from typing import Optional, List, Union, Tuple


class NormalizeWordsModel(nn.Module):
    def __init__(self, vocab: torch.Tensor) -> None:
        super(NormalizeWordsModel, self).__init__()

        hidden_size = vocab.size(1)
        self.vocab = nn.Parameter(vocab, requires_grad=False)
        self.default_embed = nn.Parameter(torch.rand(hidden_size))
        self.W = nn.Parameter(torch.eye(hidden_size))

    def forward(self, input: PackedSequence) -> PackedSequence:
        # input should be PackedSequence
        words, *rest = input
        C = torch.vstack((self.vocab, self.default_embed)).T
        P = F.softmax(words @ self.W @ C, dim=1)
        V = P[:, -1:] * words + P[:, :-1] @ self.vocab
        return PackedSequence(V, *rest)


class InstructionsDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_instructions: int,
        bias: bool = True,
        nonlinearity: str = "relu",
    ) -> None:
        super(InstructionsDecoder, self).__init__()
        self.n_instructions = n_instructions
        self.rnn_cell = nn.RNNCell(
            hidden_size, hidden_size, bias, nonlinearity
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hx: torch.Tensor = torch.zeros_like(input)
        hiddens: List[torch.Tensor] = []
        for _ in range(self.n_instructions):
            hx = self.rnn_cell(input, hx)
            hiddens.append(hx)
        return torch.cat([t.unsqueeze(1) for t in hiddens], dim=1)


class InstructionsModel(nn.Module):
    def __init__(self, vocab: torch.Tensor, n_instructions: int) -> None:
        super(InstructionsModel, self).__init__()

        hidden_size = vocab.size(1)

        self.n_instructions = n_instructions
        self.tagger = NormalizeWordsModel(vocab)
        self.encoder = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, dropout=0.0
        )
        self.decoder = InstructionsDecoder(
            hidden_size=hidden_size, n_instructions=n_instructions
        )

    def forward(
        self, input: PackedSequence
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input should be a PackedSequence
        V = self.tagger(input)
        Q = self.encoder(V)[1][0].squeeze()  # get last hidden
        H = self.decoder(Q)
        # Unpack sequences
        V, lens_unpacked = pad_packed_sequence(V, batch_first=True)
        # Prepare mask for attention
        seq_len = V.size(1)
        mask = (
            torch.cat(
                [
                    torch.ones(seq_len - l, dtype=torch.long) * i
                    for i, l in enumerate(lens_unpacked)
                ]
            ),
            torch.arange(self.n_instructions)[:, None],
            torch.cat([torch.arange(l, seq_len) for l in lens_unpacked]),
        )
        # Intermediate multiplication
        tmp = H @ V.transpose(1, 2)
        # Mask values for softmax
        tmp[mask] = float("-inf")
        # Instructions
        R = F.softmax(tmp, dim=-1) @ V
        return R, Q


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
