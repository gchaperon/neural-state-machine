import random
import unittest
import warnings

import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence

from nsm.model import (
    InstructionsDecoder,
    InstructionsModel,
    NormalizeWordsModel,
    NSMCell,
)

warnings.filterwarnings("ignore", category=UserWarning)


class TestQuestionNormalization(unittest.TestCase):
    def setUp(self):
        self.vocab = torch.rand(1335, 300)
        self.input = pack_sequence(
            sorted(
                [torch.rand(random.randint(10, 20), 300) for i in range(64)],
                key=len,
                reverse=True,
            )
        )
        self.model = NormalizeWordsModel(self.vocab)

    def test_output_type(self):
        output = self.model(self.input)
        self.assertIsInstance(output, PackedSequence)

    def test_question_normalization(self):
        output = self.model(self.input)
        self.assertEqual(output.data.shape, self.input.data.shape)

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_output_cuda(self):
        model = self.model.cuda()
        output = model(self.input.to("cuda"))
        self.assertTrue(output.is_cuda)


class TestInstructionsDecoder(unittest.TestCase):
    def setUp(self):
        self.n_instructions = 8
        self.batch_size = 64
        self.hidden_size = 300
        # for each question in the batch there is a vector representing it
        self.input = torch.rand(self.batch_size, self.hidden_size)
        self.model = InstructionsDecoder(
            n_instructions=self.n_instructions,
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            nonlinearity="relu",
        )

    def test_output_type(self):
        output = self.model(self.input)
        self.assertIsInstance(output, torch.Tensor)

    def test_hidden_states(self):
        output = self.model(self.input)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.n_instructions, self.hidden_size),
        )

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_output_cuda(self):
        model = self.model.cuda()
        output = model(self.input.cuda())
        self.assertTrue(output.is_cuda)


class TestInstructionsModel(unittest.TestCase):
    def setUp(self):
        self.n_instructions = 8
        self.vocab = torch.rand(1335, 300)
        self.model = InstructionsModel(self.vocab, self.n_instructions)
        self.questions = [
            torch.rand(random.randint(10, 20), 300) for _ in range(64)
        ]
        self.input = pack_sequence(
            sorted(self.questions, key=len, reverse=True)
        )

    def test_output_type(self):
        output = self.model(self.input)
        self.assertIsInstance(output, torch.Tensor)

    def test_output_shape(self):
        output = self.model(self.input)
        self.assertEqual(
            output.shape, (len(self.questions), self.n_instructions, 300)
        )
        self.assertFalse(output.isnan().any())

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_output_cuda(self):
        model = self.model.cuda()
        output = model(self.input.to("cuda"))
        self.assertTrue(output.is_cuda)


@unittest.skip("wip")
class TestNSMCell(unittest.TestCase):
    def setUp(self):
        self.model = NSMCell()
        # my batch structure is just every prob distribution concatenated

    def test_output_shape(self):
        pass
