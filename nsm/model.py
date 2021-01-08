from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from torch_scatter import scatter_sum, segment_sum_coo
from nsm.utils import Batch, segment_softmax_coo


class Tagger(nn.Module):
    def __init__(self, embedding_size) -> None:
        super(Tagger, self).__init__()

        self.default_embedding = nn.Parameter(torch.rand(embedding_size))
        self.weight = nn.Parameter(torch.eye(embedding_size))

    def forward(self, vocab: Tensor, question_batch: PackedSequence) -> PackedSequence:
        tokens, *rest = question_batch
        similarity = F.softmax(
            tokens @ self.weight @ torch.vstack((vocab, self.default_embedding)).T,
            dim=1,
        )
        concept_based = similarity[:, -1:] * tokens + similarity[:, :-1] @ vocab
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
        tagged_unpacked, lens_unpacked = pad_packed_sequence(tagged, batch_first=True)
        # Intermediate multiplication
        tmp = hidden @ tagged_unpacked.transpose(1, 2)
        # Mask values for softmax
        max_seq_len = tagged_unpacked.size(1)
        batch_size, n_instructions = hidden.size()[:2]
        tmp[
            torch.arange(batch_size).repeat_interleave(max_seq_len - lens_unpacked),
            :,
            torch.cat([torch.arange(l, max_seq_len) for l in lens_unpacked]),
        ] = float("-inf")
        # Instructions
        instructions = F.softmax(tmp, dim=-1) @ tagged_unpacked
        return instructions, encoded


class NSMCell(nn.Module):
    def __init__(self, input_size: int, n_node_properties: int) -> None:
        super(NSMCell, self).__init__()

        self.weight_node_properties = nn.Parameter(
            torch.rand(n_node_properties, input_size, input_size)
        )
        self.weight_edge = nn.Parameter(torch.rand(input_size, input_size))
        self.weight_node_score = nn.Parameter(torch.rand(input_size))
        self.weight_relation_score = nn.Parameter(torch.rand(input_size))

    def forward(
        self,
        graph_batch: Batch,
        instruction_batch: Tensor,
        distribution: Tensor,
        node_prop_similarities: Tensor,
        relation_similarity: Tensor,
    ) -> Tensor:
        """
        Dimensions:
            graph_batch:
                node_attrs:         N x P x H
                edge_indices:       2 x E
                edge_attrs:         E x H
            instruction_batch:      B x H
            distribution:           N
            node_prop_similarities: B x P
            relation_similarity:    B
        Legend:
            N: Total number of nodes
            P: Number of node properties
            H: Hidde/input size (glove size)
            E: Total number of edges
            B: Batch size
        """
        # Compute node and edge score
        # N x H
        node_scores = F.elu(
            torch.sum(
                # P x N x 1
                node_prop_similarities.T[:, graph_batch.node_indices, None]
                # N x H
                * instruction_batch[graph_batch.node_indices]
                # P x N x H
                * graph_batch.node_attrs.transpose(0, 1)
                # P x H x H
                @ self.weight_node_properties,
                dim=0,
            )
        )
        # E x H
        edge_scores = F.elu(
            # E x H
            instruction_batch[graph_batch.edge_batch_indices]
            # E x H
            * graph_batch.edge_attrs
            # H x H
            @ self.weight_edge
        )
        # Compute state component for next distribution
        # N
        next_distribution_states = segment_softmax_coo(
            # N x H       H
            node_scores @ self.weight_node_score,
            graph_batch.node_indices,
            dim=0,
        )
        # Compute neighbour component for next distribution
        # N
        next_distribution_relations = segment_softmax_coo(
            # N x H
            scatter_sum(
                # E x 1                                           E x H
                distribution[graph_batch.edge_indices[0], None] * edge_scores,
                graph_batch.edge_indices[1],
                dim=0,
                dim_size=graph_batch.node_indices.size(0),
            )
            # H
            @ self.weight_relation_score,
            graph_batch.node_indices,
            dim=0,
        )

        # Compute next distribution
        # N
        next_distribution = (
            relation_similarity[graph_batch.node_indices] * next_distribution_relations
            + (1 - relation_similarity[graph_batch.node_indices])
            * next_distribution_states
        )
        return next_distribution


class NSM(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_node_properties: int,
        computation_steps: int,
        output_size: int,
    ) -> None:
        super(NSM, self).__init__()
        self.instructions_model = InstructionsModel(
            input_size, n_instructions=computation_steps
        )
        self.nsm_cell = NSMCell(input_size, n_node_properties)
        self.linear = nn.Linear(2 * input_size, output_size)

    def forward(
        self,
        graph_batch: Batch,
        question_batch: PackedSequence,
        concept_vocabulary: Tensor,
        # Last property embedding must be the relation
        property_embeddings: Tensor,
    ) -> torch.Tensor:
        """
        Dimensions:
            graph_batch:
                node_attrs: N x P x H
                edge_indices: 2 x E
                edge_attrs: E x H
            question_batch: B x L* x H
            concept_vocab: C x H
            property_embeddings: D x H
        Legend:
            N: Total number of nodes
            P: Number of node properties
            H: Hidden size (glove size)
            E: Total number of edges
            B: Batch size
            C: Number of concepts
            D = P + 1: Number of properties (concept categories)
            L*: Question length, variable, see PackedSequence docs for more info
            I: Computation steps
        """
        # B x I x H,  B x H
        instructions, encoded_questions = self.instructions_model(
            concept_vocabulary, question_batch
        )
        # Initialize distribution
        # N
        distribution = (1 / graph_batch.nodes_per_graph)[graph_batch.node_indices]
        # Simulate execution of finite automaton
        # B x H
        for instruction_batch in instructions.unbind(1):
            # Property similarities, denoted by a capital R in the paper
            # B x P,                B
            node_prop_similarities, relation_similarity = (
                foo := F.softmax(instruction_batch @ property_embeddings.T, dim=1)
            )[:, :-1], foo[:, -1]
            # N
            distribution = self.nsm_cell(
                graph_batch,
                instruction_batch,
                distribution,
                node_prop_similarities,
                relation_similarity,
            )

        # B x H
        aggregated = segment_sum_coo(
            distribution[:, None]
            * torch.sum(
                node_prop_similarities[graph_batch.node_indices, :, None]
                * graph_batch.node_attrs,
                dim=1,
            ),
            graph_batch.node_indices,
            dim_size = encoded_questions.size(0)

        )
        # B x 2H
        return self.linear(torch.hstack((encoded_questions, aggregated)))
