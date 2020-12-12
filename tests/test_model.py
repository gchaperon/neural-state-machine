import os 
import random
import unittest
import warnings

from itertools import islice
import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence

from nsm.model import (
    InstructionsDecoder,
    InstructionsModel,
    NormalizeWordsModel,
    NSMCell,
    NSM,
)

from nsm.utils import collate_graphs, infinite_graphs

warnings.simplefilter("ignore", category=UserWarning)


class TestQuestionNormalization(unittest.TestCase):
    def setUp(self) -> None:
        self.hidden_size = 300
        self.vocab_size = 1335
        self.vocab = torch.rand(self.vocab_size, self.hidden_size)
        self.model = NormalizeWordsModel(self.vocab)
        self.batch_size = 64
        # batch of questions, lengths between 10 and 20
        self.input = pack_sequence(
            sorted(
                [
                    torch.rand(random.randint(10, 20), self.hidden_size)
                    for i in range(self.batch_size)
                ],
                key=len,
                reverse=True,
            )
        )

    def test_output_type(self) -> None:
        output = self.model(self.input)
        self.assertIsInstance(output, PackedSequence)

    def test_question_normalization(self) -> None:
        output = self.model(self.input)
        self.assertEqual(output.data.shape, self.input.data.shape)

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_output_cuda(self) -> None:
        model = self.model.cuda()
        output = model(self.input.to("cuda"))
        self.assertTrue(output.is_cuda)


class TestInstructionsDecoder(unittest.TestCase):
    def setUp(self) -> None:
        self.n_instructions = 8
        self.batch_size = 64
        self.hidden_size = 300
        # for each question in the batch there is a vector representing it
        self.input = torch.rand(self.batch_size, self.hidden_size)
        self.model = InstructionsDecoder(
            hidden_size=self.hidden_size,
            n_instructions=self.n_instructions,
            nonlinearity="relu",
        )

    def test_output_type(self) -> None:
        output = self.model(self.input)
        self.assertIsInstance(output, torch.Tensor)

    def test_hidden_states(self) -> None:
        output = self.model(self.input)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.n_instructions, self.hidden_size),
        )

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_output_cuda(self) -> None:
        model = self.model.cuda()
        output = model(self.input.cuda())
        self.assertTrue(output.is_cuda)


class TestInstructionsModel(unittest.TestCase):
    def setUp(self) -> None:
        self.n_instructions = 8
        self.hidden_size = 300
        self.batch_size = 64
        self.vocab = torch.rand(1335, self.hidden_size)
        self.model = InstructionsModel(self.vocab, self.n_instructions)
        self.questions = [
            torch.rand(random.randint(10, 20), self.hidden_size)
            for _ in range(self.batch_size)
        ]
        self.input = pack_sequence(
            sorted(self.questions, key=len, reverse=True)
        )

    def test_output_type(self) -> None:
        output = self.model(self.input)
        self.assertIsInstance(output, tuple)
        self.assertIsInstance(output[0], torch.Tensor)
        self.assertIsInstance(output[1], torch.Tensor)

    def test_output_shape(self) -> None:
        instructions, encoded = self.model(self.input)
        self.assertEqual(
            instructions.size(),
            (self.batch_size, self.n_instructions, self.hidden_size),
        )
        self.assertFalse(instructions.isnan().any())
        self.assertEqual(encoded.size(), (self.batch_size, self.hidden_size))

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_output_cuda(self) -> None:
        model = self.model.cuda()
        instructions, encoded = model(self.input.to("cuda"))
        self.assertTrue(instructions.is_cuda)
        self.assertTrue(encoded.is_cuda)


class TestNSMCell(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 8
        self.hidden_size = 100
        self.n_properties = 20
        # NOTE: plus one to account for the relations
        self.prop_embeds = torch.rand(self.n_properties + 1, self.hidden_size)
        self.model = NSMCell(self.prop_embeds)
        self.graph_list = list(
            islice(
                infinite_graphs(
                    self.hidden_size,
                    self.n_properties,
                    node_distribution=(16.4, 8.2),
                    density_distribution=(0.2, 0.4),
                ),
                self.batch_size,
            )
        )
        self.input = collate_graphs(self.graph_list)
        self.instruction = torch.rand(self.batch_size, self.hidden_size)
        # for testing purposes it doesn't matter that the probs sum to
        # one per graph
        self.distribution = torch.rand(self.input.node_attrs.size(0))

    def test_output_shape(self) -> None:
        distribution, prop_similarities = self.model(
            self.input, self.instruction, self.distribution
        )
        self.assertEqual(distribution.ndim, 1)
        self.assertEqual(prop_similarities.ndim, 2)
        self.assertEqual(distribution.size(0), self.input.node_attrs.size(0))
        self.assertEqual(
            prop_similarities.size(),
            (self.batch_size, self.prop_embeds.size(0)),
        )

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_output_shape_cuda(self) -> None:
        device = torch.device("cuda")
        model = self.model.to(device)
        input = self.input.to(device)
        distribution, prop_similarities = self.model(
            input, self.instruction.to(device), self.distribution.to(device)
        )
        self.assertEqual(distribution.device.type, device.type)
        self.assertEqual(prop_similarities.device.type, device.type)

    def test_output_sum_one_per_graph(self) -> None:
        output, _ = self.model(self.input, self.instruction, self.distribution)
        self.assertTrue(
            torch.zeros(self.batch_size)
            .index_add_(0, self.input.node_indices, output)
            .allclose(torch.ones(self.batch_size))
        )

    def test_grad_init_is_none(self) -> None:
        # Maybe add subtests
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNone(param.grad)

    def test_backward_simple(self) -> None:
        warnings.simplefilter("ignore", category=UserWarning)
        # Detect anomaly should raise an error whenever a backward op
        # produces a nan value
        with torch.autograd.detect_anomaly():
            output, _ = self.model(
                self.input, self.instruction, self.distribution
            )
            output.sum().backward()
            for param in self.model.parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad)

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_backward_simple_cuda(self) -> None:
        warnings.simplefilter("ignore", category=UserWarning)
        with torch.autograd.detect_anomaly():
            dev = torch.device("cuda")
            input = self.input.to(dev)
            model = self.model.to(dev)
            output, _ = model(
                input, self.instruction.to(dev), self.distribution.to(dev)
            )
            output.sum().backward()
            for param in self.model.parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad)


# @unittest.skip("wip")
class TestNSM(unittest.TestCase):
    def setUp(self) -> None:
        # These tests should be unsing expected dimensions
        self.n_properties = 78
        self.hidden_size = 300
        self.computation_steps = 8
        self.batch_size = 64
        self.vocab = torch.rand(1335, self.hidden_size)
        self.prop_embeds = torch.rand(self.n_properties, self.hidden_size)
        self.output_size = 2000
        self.model = NSM(
            self.vocab,
            self.prop_embeds,
            self.computation_steps,
            self.output_size,
        )
        self.graph_batch = collate_graphs(
            list(
                islice(
                    infinite_graphs(
                        self.hidden_size,
                        self.n_properties - 1,
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
                    torch.rand(random.randint(10, 20), self.hidden_size)
                    for _ in range(self.batch_size)
                ],
                key=len,
                reverse=True,
            )
        )
    @unittest.skipIf(os.environ.get("SKIP_LONGASS_CPU_TESTS"), "skipping long cpu tests")
    def test_output_shape(self) -> None:
        output = self.model(self.graph_batch, self.question_batch)
        self.assertEqual(output.shape, (self.batch_size, self.output_size))

    @unittest.skipIf(not torch.cuda.is_available(), "cuda not available")
    def test_output_shape_cuda(self) -> None:
        device = torch.device("cuda")
        model = self.model.to(device)
        output = model(
            self.graph_batch.to(device), self.question_batch.to(device)
        )
        self.assertEqual(output.device.type, device.type)
        self.assertEqual(output.shape, (self.batch_size, self.output_size))

    @unittest.skipIf(not torch.cuda.is_available(), "cuda not available")
    def test_backward_Cuda(self) ->None:
        warnings.simplefilter("ignore", category=UserWarning)
        with torch.autograd.detect_anomaly():
            device = torch.device("cuda")
            model = self.model.to(device)
            output = model(self.graph_batch.to(device), self.question_batch.to(device))
            output.sum().backward()
            for param in model.parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad)
