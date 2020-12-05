import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from nsm.utils import Batch

from typing import Optional, List, Union


class NormalizeWordsModel(nn.Module):
    def __init__(self, vocab: torch.Tensor) -> None:
        super(NormalizeWordsModel, self).__init__()

        self.vocab = nn.Parameter(vocab, requires_grad=False)
        self.default_embed = nn.Parameter(torch.rand(300))
        self.W = nn.Parameter(torch.eye(300))

    def forward(self, input: PackedSequence) -> PackedSequence:
        # input should be PackedSequence
        words, *rest = input
        C = torch.cat((self.vocab.T, self.default_embed.view(-1, 1)), dim=1)
        P = F.softmax(words @ self.W @ C, dim=1)
        V = P[:, -1:] * words + P[:, :-1] @ self.vocab
        return PackedSequence(V, *rest)


class InstructionsDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_instructions: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
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
        self.n_instructions = n_instructions

        self.tagger = NormalizeWordsModel(vocab)
        self.encoder = nn.LSTM(input_size=300, hidden_size=300, dropout=0.0)
        self.decoder = InstructionsDecoder(
            hidden_size=300, n_instructions=n_instructions
        )

    def forward(self, input: PackedSequence) -> torch.Tensor:
        # input should be a PackedSequence
        V = self.tagger(input)
        _, (q, _) = self.encoder(V)
        H = self.decoder(q.squeeze())
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
        return R


class NSMCell(nn.Module):
    def __init__(self, hidden_size: int, n_properties: int) -> None:
        super(NSMCell, self).__init__()

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
        prop_embeds: torch.Tensor,
        distribution: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Initialize distribution to uniform if not present
        distribution = (
            distribution or (1 / batch.nodes_per_graph)[batch.node_indices]
        )
        # Compute similarity between instruction and attribute
        # categories, aka properties
        prop_similarities = F.softmax(instruction @ prop_embeds.T, dim=0)
        # Compute node and edge score
        node_scores = F.elu(
            torch.sum(
                prop_similarities[:-1, None]
                * instruction
                * self.Ws_property[:-1]
                .matmul(batch.node_attrs.unsqueeze(-1))
                .squeeze(),
                dim=1,
            )
        )
        edge_scores = F.elu(
            instruction
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
        next_distribution = (
            prop_similarities[-1] * next_distribution_relations
            + (1 - prop_similarities[-1]) * next_distribution_states
        )
        return next_distribution
