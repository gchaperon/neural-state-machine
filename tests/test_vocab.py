from nsm.vocab.experimental import Vocab, EmbeddedVocab, GloVe
import random
import string
from typing import Tuple, Hashable
import torch

import unittest


class VocabTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.token_source = string.ascii_lowercase
        self.vocab = Vocab(self.token_source)
    
    @staticmethod
    def is_consistent(vocab:Vocab) ->bool:
        return all(vocab.itoe[vocab.etoi[e]] == e for e in vocab)


    def test_len(self) -> None:
        self.assertEqual(len(self.vocab), len(self.token_source))

    def test_contains(self) -> None:
        s = random.choice(self.token_source)
        self.assertTrue(s in self.vocab)
        self.assertFalse("A" in self.vocab)

    def test_iter(self) -> None:
        vocab_tokens = set(iter(self.vocab))
        source_tokens = set(self.token_source)
        self.assertEqual(vocab_tokens, source_tokens)

    def test_etoi(self) -> None:
        s = random.choice(self.token_source)
        self.assertEqual(self.vocab.etoi[s], self.vocab.itoe.index(s))
        with self.assertRaises(KeyError):
            self.vocab.etoi["A"]

    def test_itoe(self) -> None:
        i = random.choice(range(len(self.vocab)))
        out = self.vocab.itoe[i]
        expected = [k for k, v in self.vocab.etoi.items() if v == i][0]
        self.assertEqual(out, expected)

    def test_consistency(self) -> None:
        self.assertTrue(self.is_consistent(self.vocab))

    def test_unique_indices(self) -> None:
        all_indices = list(self.vocab.etoi.values())
        self.assertEqual(len(all_indices), len(set(all_indices)))

    def test_unk_token(self) -> None:
        unk_tok: Tuple[Hashable, ...] = ("oaksdjoaj", 1, tuple())
        vocab = Vocab(tuple(), unk_token=unk_tok)
        self.assertEqual(vocab.unk_token, unk_tok)
        with self.assertRaises(AttributeError):
            self.vocab.unk_token

    def test_len_w_unk(self) -> None:
        vocab = Vocab(string.ascii_lowercase, unk_token="<unk>")
        self.assertEqual(len(vocab), len(string.ascii_lowercase) + 1)

    def test_not_in_vocab(self) -> None:
        unk_tok = "<unk>"
        vocab = Vocab(string.ascii_lowercase, unk_token=unk_tok)
        unk_i = vocab.etoi[unk_tok]
        self.assertEqual(vocab.etoi["A"], unk_i)

    def test_unk_index(self) -> None:
        vocab = Vocab(string.ascii_lowercase, unk_token="oaskdjoaskjd")
        self.assertEqual(vocab.etoi["A"], vocab.unk_index)
        with self.assertRaises(AttributeError):
            self.vocab.unk_index

    def test_unk_in_elements(self) -> None:
        unk_tok = "z"
        vocab = Vocab(string.ascii_lowercase, unk_token=unk_tok)

        self.assertEqual(vocab.etoi[unk_tok], vocab.unk_index)
        self.assertEqual(vocab.etoi["asd"], vocab.unk_index)
        self.assertEqual(len(vocab), len(string.ascii_lowercase))
        self.assertTrue(self.is_consistent(vocab))

    def test_consistency_w_repeated(self) ->None:
        vocab = Vocab(string.ascii_lowercase * 2)
        self.assertTrue(self.is_consistent(vocab))


class EmbeddedVocabTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.source_elements = string.ascii_lowercase
        self.embeddings = torch.rand(len(self.source_elements), 10)
        self.vocab = EmbeddedVocab(self.source_elements, self.embeddings)

    def test_getitem_single(self) -> None:
        i = random.randrange(len(self.source_elements))
        key = self.source_elements[i]
        correct_embedding = self.embeddings[i]
        self.assertTrue(self.vocab[key].eq(correct_embedding).all())

    def test_getitem_multiple(self) -> None:
        indices = random.sample(range(len(self.source_elements)), k=10)
        keys = [self.source_elements[i] for i in indices]
        correct = self.embeddings[indices]
        self.assertTrue(self.vocab[keys].eq(correct).all())

    def test_getitem_dimensions(self) -> None:
        _, *last_embed_dims = self.embeddings.size()
        self.assertEqual(self.vocab["a"].size(), last_embed_dims)
        self.assertEqual(self.vocab[["a"]].size(), (1,) + last_embed_dims)
        self.assertEqual(self.vocab[["a", "d", "q"]].size(), (3,) + last_embed_dims)

    def test_unk_embedding(self) -> None:
        
        pass


class GloVeTestCase(unittest.TestCase):
    pass
