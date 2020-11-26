import random
import unittest

import torch
from torch.nn.utils.rnn import pack_sequence, PackedSequence

from nsm.model import (
    NormalizeWordsModel,
    InstructionsDecoder,
    InstructionsModel,
)


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


class TestInstructionsDecoder(unittest.TestCase):
    def setUp(self):
        self.n_instructions = 8
        self.hidden_size = 300
        self.model = InstructionsDecoder(
            n_instructions=self.n_instructions,
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            nonlinearity="relu",
        )

    def test_hidden_states(self):
        # for each question in the batch there is a vector representing it
        batch_size = 64
        input = torch.rand(batch_size, self.hidden_size)
        output = self.model(input)
        self.assertEqual(
            output.shape, (batch_size, self.n_instructions, self.hidden_size)
        )


class TestInstructionsModel(unittest.TestCase):
    def setUp(self):
        self.n_instructions = 8
        self.vocab = torch.rand(1335, 300)
        self.model = InstructionsModel(self.vocab, self.n_instructions)
        self.questions = [
            torch.rand(random.randint(10, 20), 300) for _ in range(64)
        ]

    def test_output_shape(self):
        input = pack_sequence(sorted(self.questions, key=len, reverse=True))
        output = self.model(input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(
            output.shape, (len(self.questions), self.n_instructions, 300)
        )
        self.assertFalse(output.isnan().any())
