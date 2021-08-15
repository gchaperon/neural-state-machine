from typing import List, Optional, Tuple, Union, Callable, Literal
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from torch_scatter import scatter_sum
from nsm.utils import Batch, scatter_softmax


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

    def extra_repr(self):
        return f"embedding_size={self.weight.shape[0]}"


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
        """
        Dimensions:
            B: Batch size
            H: Hidden size (glove size)
            I: Number of instructions
        """
        # Explicit better than implicit, amarite (?)
        hx: torch.Tensor = torch.zeros_like(input)
        # [B x H, ...]
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
        # Use softmax as nn.Module to allow extracting attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, vocab: Tensor, question_batch: PackedSequence
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input should be a PackedSequence
        tagged = self.tagger(vocab, question_batch)
        encoded = self.encoder(tagged)[1][0].squeeze(dim=0)  # get last hidden
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
        # breakpoint()
        instructions = self.softmax(tmp) @ tagged_unpacked
        return instructions, encoded


class DummyInstructionsModel(nn.Module):
    """
    Instruction model that only unpacks the questions, and expects all questions to be the same length.
    The encoded representation of a question is just a vector of zeros
    """

    def __init__(self, embedding_size: int, n_instructions: int):
        super(DummyInstructionsModel, self).__init__()
        self.embedding_size = embedding_size
        self.n_instructions = n_instructions

    def forward(self, vocab: Tensor, questions: PackedSequence, encoded: Tensor = None):
        instructions, lens_unpacked = pad_packed_sequence(questions, batch_first=True)
        assert all(
            l == lens_unpacked[0] for l in lens_unpacked
        ), "all question lengths must be the same for DummyInstructionsModel"
        batch_size, n_instructions, embedding_size = instructions.shape

        # This check is not necessary for this dummy model
        # assert self.n_instructions == n_instructions
        return instructions, encoded or torch.zeros(
            batch_size, embedding_size, device=vocab.device
        )


class NSMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_node_properties: int,
        nonlinearity: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        super(NSMCell, self).__init__()
        self.nonlinearity = nonlinearity or F.elu

        self.weight_node_properties = nn.Parameter(
            torch.rand(n_node_properties, input_size, input_size)
        )
        self.weight_edge = nn.Parameter(torch.rand(input_size, input_size))
        self.weight_node_score = nn.Parameter(torch.rand(input_size))
        self.weight_relation_score = nn.Parameter(torch.rand(input_size))

    def extra_repr(self):
        n_node_properties, input_size, _ = self.weight_node_properties.size()
        return f"{input_size=}, {n_node_properties=}"

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
        # breakpoint()
        node_scores = self.nonlinearity(
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
        edge_scores = self.nonlinearity(
            # E x H
            instruction_batch[graph_batch.edge_batch_indices]
            # E x H
            * graph_batch.edge_attrs
            # H x H
            @ self.weight_edge
        )
        # Compute state component for next distribution
        # N
        next_distribution_states = scatter_softmax(
            # N x H       H
            node_scores @ self.weight_node_score,
            graph_batch.node_indices,
            dim=0,
        )
        # Compute neighbour component for next distribution
        # N
        next_distribution_relations = scatter_softmax(
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


class AnswerClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        nonlinearity: Optional[Callable[[Tensor], Tensor]] = None,
        dropout: float = 0.0,
    ):
        super(AnswerClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.nonlinearity = nonlinearity or F.elu
        self.fc_layers = nn.ModuleList(
            [nn.Linear(input_size, input_size), nn.Linear(input_size, output_size)]
        )

    def forward(self, input: Tensor) -> Tensor:
        z = self.fc_layers[0](input)
        z = self.nonlinearity(z)
        z = self.dropout(z)
        z = self.fc_layers[1](z)
        return z


_instruction_model_types = {
    "normal": InstructionsModel,
    "dummy": DummyInstructionsModel,
}


class NSM(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_node_properties: int,
        computation_steps: int,
        output_size: int,
        dropout: float = 0.0,
        instruction_model: Literal["normal", "dummy"] = "normal",
    ) -> None:
        super(NSM, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.instructions_model = _instruction_model_types[instruction_model](
            input_size, n_instructions=computation_steps + 1
        )
        self.nsm_cell = NSMCell(input_size, n_node_properties)
        self.classifier = AnswerClassifier(2 * input_size, output_size, dropout=dropout)

    def forward(
        self,
        graph_batch: Batch,
        question_batch: PackedSequence,
        concept_vocabulary: Tensor,
        # Last property embedding must be the relation
        property_embeddings: Tensor,
    ) -> Tensor:
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
        # breakpoint()
        # B x (I + 1) x H,  B x H
        instructions, encoded_questions = self.instructions_model(
            concept_vocabulary, question_batch
        )
        # Apply dropout to state and edge representations
        graph_batch = graph_batch._replace(
            node_attrs=self.dropout(graph_batch.node_attrs),
            edge_attrs=self.dropout(graph_batch.edge_attrs),
        )
        # Initialize distribution
        # N
        distribution = (1 / graph_batch.nodes_per_graph)[graph_batch.node_indices]
        # Simulate execution of finite automaton
        # B x H
        for instruction_batch in instructions[:, :-1].unbind(1):
            # TODO: maybe do these all at once?
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

        # B x P
        node_prop_similarities = F.softmax(
            instructions[:, -1] @ property_embeddings.T, dim=1
        )[:, :-1]
        # B x H
        aggregated = scatter_sum(
            distribution[:, None]
            * torch.sum(
                node_prop_similarities[graph_batch.node_indices, :, None]
                * graph_batch.node_attrs,
                dim=1,
            ),
            graph_batch.node_indices,
            dim=0,
            dim_size=encoded_questions.size(0),
        )
        # B x 2H
        return self.classifier(torch.hstack((encoded_questions, aggregated)))


class NSMLightningModule(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        n_node_properties: int,
        computation_steps: int,
        output_size: int,
        dropout: float = 0.0,
        instruction_model: Literal["normal", "dummy"] = "normal",
        learn_rate: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.nsm = NSM(
            input_size,
            n_node_properties,
            computation_steps,
            output_size,
            dropout,
            instruction_model,
        )
        self.learn_rate = learn_rate

    def forward(self, *args):
        return self.nsm(*args)

    def training_step(self, batch, batch_idx):
        graphs, questions, concepts, properties, targets = batch
        out = self(graphs, questions, concepts, properties)
        loss = F.cross_entropy(out, targets)
        running_acc = torch.sum(out.argmax(dim=1) == targets) / targets.size(0)
        self.log("train_loss", loss)
        self.log("running_train_acc", running_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        graphs, questions, concepts, properties, targets = batch
        out = self(graphs, questions, concepts, properties)
        return out, targets

    def validation_epoch_end(self, validation_step_outputs):
        outs, targets = zip(*validation_step_outputs)
        outs = torch.vstack(outs)
        targets = torch.cat(targets)
        loss = F.cross_entropy(outs, targets)
        acc = torch.sum(outs.argmax(dim=1) == targets) / outs.size(0)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return outs, targets

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learn_rate)
