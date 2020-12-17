import os
import random
import unittest
import warnings
from itertools import islice

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import (
    PackedSequence,
    pack_sequence,
    pad_packed_sequence,
)

from nsm.model import (
    NSM,
    InstructionDecoder,
    InstructionsModel,
    NSMCell,
    Tagger,
)
from nsm.utils import collate_graphs, infinite_graphs

warnings.simplefilter("ignore", category=UserWarning)


class TaggerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding_size = 300
        vocab_len = 1335
        self.batch_size = 64
        self.vocab = torch.rand(vocab_len, self.embedding_size)
        self.model = Tagger(self.embedding_size)
        # batch of questions, lengths between 10 and 20
        self.question_batch = pack_sequence(
            sorted(
                [
                    torch.rand(random.randint(10, 20), self.embedding_size)
                    for i in range(self.batch_size)
                ],
                key=len,
                reverse=True,
            )
        )

    def test_output_type(self) -> None:
        output = self.model(self.vocab, self.question_batch)
        self.assertIsInstance(output, PackedSequence)

    def test_output_shape(self) -> None:
        output = self.model(self.vocab, self.question_batch)
        self.assertEqual(output.data.shape, self.question_batch.data.shape)
        self.assertEqual(
            pad_packed_sequence(output, batch_first=True)[0].size(0),
            self.batch_size,
        )

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_output_cuda(self) -> None:
        model = self.model.cuda()
        output = model(self.vocab.to("cuda"), self.question_batch.to("cuda"))
        self.assertTrue(output.is_cuda)


class InstructionDecoderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.n_instructions = 8
        self.batch_size = 64
        self.input_size = 300
        # for each question in the batch there is a vector representing it
        self.encoded = torch.rand(self.batch_size, self.input_size)
        self.model = InstructionDecoder(
            input_size=self.input_size,
            hidden_size=self.input_size,
            n_instructions=self.n_instructions,
        )

    def test_output_type(self) -> None:
        output = self.model(self.encoded)
        self.assertIsInstance(output, torch.Tensor)

    def test_hidden_states(self) -> None:
        output = self.model(self.encoded)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.n_instructions, self.input_size),
        )

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_output_cuda(self) -> None:
        model = self.model.cuda()
        output = model(self.encoded.cuda())
        self.assertTrue(output.is_cuda)


class InstructionsModelTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.n_instructions = 8
        self.embedding_size = 300
        self.batch_size = 64
        vocab_len = 1335
        self.vocab = torch.rand(vocab_len, self.embedding_size)
        self.model = InstructionsModel(
            self.embedding_size, self.n_instructions
        )
        self.question_batch = pack_sequence(
            sorted(
                [
                    torch.rand(random.randint(10, 20), self.embedding_size)
                    for _ in range(self.batch_size)
                ],
                key=len,
                reverse=True,
            )
        )

    def test_output_type(self) -> None:
        instructions, encoded = self.model(self.vocab, self.question_batch)
        self.assertIsInstance(instructions, torch.Tensor)
        self.assertIsInstance(encoded, torch.Tensor)

    def test_output_shape(self) -> None:
        instructions, encoded = self.model(self.vocab, self.question_batch)
        self.assertEqual(
            instructions.size(),
            (self.batch_size, self.n_instructions, self.embedding_size),
        )
        self.assertFalse(instructions.isnan().any())
        self.assertEqual(
            encoded.size(), (self.batch_size, self.embedding_size)
        )

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_output_cuda(self) -> None:
        model = self.model.cuda()
        instructions, encoded = model(
            self.vocab.to("cuda"), self.question_batch.to("cuda")
        )
        self.assertTrue(instructions.is_cuda)
        self.assertTrue(encoded.is_cuda)


class NSMCellTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 8
        self.input_size = 100
        self.n_node_properties = 20
        self.model = NSMCell(self.input_size, self.n_node_properties)
        self.graph_batch = collate_graphs(
            list(
                islice(
                    infinite_graphs(
                        self.input_size,
                        self.n_node_properties,
                        node_distribution=(16.4, 8.2),
                        density_distribution=(0.2, 0.4),
                    ),
                    self.batch_size,
                )
            )
        )
        self.instruction_batch = torch.rand(self.batch_size, self.input_size)
        self.distribution = (1.0 / self.graph_batch.nodes_per_graph)[
            self.graph_batch.node_indices
        ]
        all_prop_similarities = F.softmax(
            self.instruction_batch
            @ torch.rand(self.input_size, self.n_node_properties + 1),
            dim=1,
        )
        self.node_prop_similarities = all_prop_similarities[:, :-1]
        self.relation_similarity = all_prop_similarities[:, -1]

    def test_output_shape(self) -> None:
        out_distribution = self.model(
            self.graph_batch,
            self.instruction_batch,
            self.distribution,
            self.node_prop_similarities,
            self.relation_similarity,
        )
        self.assertEqual(out_distribution.ndim, 1)
        self.assertEqual(
            out_distribution.size(0), self.graph_batch.node_attrs.size(0)
        )

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_output_shape_cuda(self) -> None:
        device = torch.device("cuda")
        model = self.model.to(device)
        graph_batch = self.graph_batch.to(device)
        output = self.model(
            graph_batch,
            self.instruction_batch.to(device),
            self.distribution.to(device),
            self.node_prop_similarities.to(device),
            self.relation_similarity.to(device),
        )
        self.assertEqual(output.device.type, device.type)

    def test_output_sum_one_per_graph(self) -> None:
        output = self.model(
            self.graph_batch,
            self.instruction_batch,
            self.distribution,
            self.node_prop_similarities,
            self.relation_similarity,
        )
        self.assertTrue(
            torch.zeros(self.batch_size)
            .index_add_(0, self.graph_batch.node_indices, output)
            .allclose(torch.ones(self.batch_size))
        )

    def test_grad_init_is_none(self) -> None:
        # Maybe add subtests
        for param in self.model.parameters():
            self.assertIsNone(param.grad)

    def test_backward_simple(self) -> None:
        warnings.simplefilter("ignore", category=UserWarning)
        # Detect anomaly should raise an error whenever a backward op
        # produces a nan value
        with torch.autograd.detect_anomaly():
            output = self.model(
                self.graph_batch,
                self.instruction_batch,
                self.distribution,
                self.node_prop_similarities,
                self.relation_similarity,
            )
            output.sum().backward()
            for param in self.model.parameters():
                self.assertIsNotNone(param.grad)

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_backward_simple_cuda(self) -> None:
        warnings.simplefilter("ignore", category=UserWarning)
        with torch.autograd.detect_anomaly():
            dev = torch.device("cuda")
            model = self.model.to(dev)
            output = model(
                self.graph_batch.to(dev),
                self.instruction_batch.to(dev),
                self.distribution.to(dev),
                self.node_prop_similarities.to(dev),
                self.relation_similarity.to(dev),
            )
            output.sum().backward()
            for param in self.model.parameters():
                self.assertIsNotNone(param.grad)


class NSMTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # These tests should be unsing expected dimensions
        self.batch_size = 64
        self.output_size = 2000
        n_node_properties = 78
        input_size = 300
        computation_steps = 8

        self.vocab = torch.rand(1335, input_size)
        self.prop_embeds = torch.rand(n_node_properties + 1, input_size)
        self.model = NSM(
            input_size,
            n_node_properties,
            computation_steps,
            self.output_size,
        )
        self.graph_batch = collate_graphs(
            list(
                islice(
                    infinite_graphs(
                        input_size,
                        n_node_properties,
                        node_distribution=(16.4, 8.2),
                        density_distribution=(0.2, 0.4),
                    ),
                    self.batch_size,
                )
            )
        )
        self.question_batch = pack_sequence(
            sorted(
                [
                    torch.rand(random.randint(10, 20), input_size)
                    for _ in range(self.batch_size)
                ],
                key=len,
                reverse=True,
            )
        )

    @unittest.skipUnless("RUN_LONGASS_TESTS" in os.environ, "I warned you")
    def test_output_shape(self) -> None:
        output = self.model(
            self.graph_batch, self.question_batch, self.vocab, self.prop_embeds
        )
        self.assertEqual(output.shape, (self.batch_size, self.output_size))

    @unittest.skipIf(not torch.cuda.is_available(), "cuda not available")
    def test_output_shape_cuda(self) -> None:
        device = torch.device("cuda")
        model = self.model.to(device)
        output = model(
            self.graph_batch.to(device),
            self.question_batch.to(device),
            self.vocab.to(device),
            self.prop_embeds.to(device),
        )
        self.assertEqual(output.device.type, device.type)
        self.assertEqual(output.shape, (self.batch_size, self.output_size))

    @unittest.skipIf(not torch.cuda.is_available(), "cuda not available")
    def test_backward_cuda(self) -> None:
        warnings.simplefilter("ignore", category=UserWarning)
        with torch.autograd.detect_anomaly():
            device = torch.device("cuda")
            model = self.model.to(device)
            output = model(
                self.graph_batch.to(device),
                self.question_batch.to(device),
                self.vocab.to(device),
                self.prop_embeds.to(device),
            )
            output.sum().backward()
            for param in model.parameters():
                self.assertIsNotNone(param.grad)
