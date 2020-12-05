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
)

from nsm.utils import collate_graphs, infinite_graphs

warnings.simplefilter("ignore", category=UserWarning)


class TestQuestionNormalization(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab = torch.rand(1335, 300)
        self.input = pack_sequence(
            sorted(
                [torch.rand(random.randint(10, 20), 300) for i in range(64)],
                key=len,
                reverse=True,
            )
        )
        self.model = NormalizeWordsModel(self.vocab)

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
        self.vocab = torch.rand(1335, 300)
        self.model = InstructionsModel(self.vocab, self.n_instructions)
        self.questions = [
            torch.rand(random.randint(10, 20), 300) for _ in range(64)
        ]
        self.input = pack_sequence(
            sorted(self.questions, key=len, reverse=True)
        )

    def test_output_type(self) -> None:
        output = self.model(self.input)
        self.assertIsInstance(output, torch.Tensor)

    def test_output_shape(self) -> None:
        output = self.model(self.input)
        self.assertEqual(
            output.shape, (len(self.questions), self.n_instructions, 300)
        )
        self.assertFalse(output.isnan().any())

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_output_cuda(self) -> None:
        model = self.model.cuda()
        output = model(self.input.to("cuda"))
        self.assertTrue(output.is_cuda)


class TestNSMCell(unittest.TestCase):
    def setUp(self) -> None:
        self.hidden_size = 300
        self.n_properties = 77
        # NOTE: Plus one to account for the relations
        self.model = NSMCell(self.hidden_size, self.n_properties + 1)
        self.graph_list = list(
            islice(
                infinite_graphs(
                    self.hidden_size,
                    self.n_properties,
                    node_distribution=(16.4, 8.2),
                    density_distribution=(0.2, 0.4),
                ),
                10,
            )
        )
        self.instruction = torch.rand(self.hidden_size)
        # NOTE: plus one to account for the relations
        self.prop_embeds = torch.rand(self.n_properties + 1, self.hidden_size)

    def test_output_shape(self) -> None:
        input = collate_graphs(self.graph_list)
        output = self.model(input, self.instruction, self.prop_embeds)
        self.assertEqual(output.ndim, 1)
        self.assertEqual(output.size(0), input.node_attrs.size(0))

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_output_shape_cuda(self) -> None:
        device = torch.device("cuda")
        model = self.model.to(device)
        input = collate_graphs(self.graph_list, device=device)
        output = self.model(
            input, self.instruction.to(device), self.prop_embeds.to(device)
        )
        self.assertEqual(output.device.type, device.type)

    def test_grad_init_is_none(self) -> None:
        # Maybe add subtests
        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)
            self.assertIsNone(param.grad)

    def test_backward_simple(self) -> None:
        warnings.simplefilter("ignore", category=UserWarning)
        # Detect anomaly should raise an error whenever a backward op
        # produces a nan value
        with torch.autograd.detect_anomaly():
            input = collate_graphs(self.graph_list)
            output = self.model(input, self.instruction, self.prop_embeds)
            output.sum().backward()
            for param in self.model.parameters():
                self.assertIsNotNone(param.grad)

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_backward_simple_cuda(self) -> None:
        warnings.simplefilter("ignore", category=UserWarning)
        with torch.autograd.detect_anomaly():
            dev = torch.device("cuda")
            input = collate_graphs(self.graph_list, device=dev)
            model = self.model.to(dev)
            output = model(
                input, self.instruction.to(dev), self.prop_embeds.to(dev)
            )
            output.sum().backward()
            for param in self.model.parameters():
                self.assertIsNotNone(param.grad)
