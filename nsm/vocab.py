from __future__ import annotations

import json
import operator
import pickle
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property, lru_cache, reduce
from itertools import chain
from operator import add
from pathlib import Path
from typing import ClassVar, Dict, Iterable, List, Literal, Optional, cast, Any
import random
import torch
from torch import Tensor
from torchtext.vocab import Vocab as TTVocab, GloVe as TTGloVe
import copy
from contextlib import redirect_stderr
import os
from collections import Counter
import logging

PUNC_POSTAGS = (".", ",", ":", "-RRB-", "-LRB-", "''", "``", "HYPH", "NFP")
logger = logging.getLogger(__name__)


class Vocab(TTVocab):
    def __getitem__(self, token: str) -> int:
        return self.stoi[token]

    def __contains__(self, value: Any) -> bool:
        return value in self.stoi

    def extend(self, other: Vocab) -> None:
        assert (self.vectors is None) == (  # type:ignore[has-type]
            other.vectors is None
        ), "one has tensors and the other does not"
        words = other.itos
        new_vectors = []
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
                if other.vectors is not None:
                    new_vectors.append(other.vectors[other.stoi[w]])
        self.vectors: Tensor = torch.vstack([self.vectors, *new_vectors])


class GloVe(TTGloVe):
    def __init__(self, *, name="6B", dim=300, cache="data/glove.6B", **kwargs):
        super(GloVe, self).__init__(name, dim, cache=cache, **kwargs)
        del self.unk_init

    def unk_init(self, _):
        """ Return the average of all vectors"""
        return self.vectors.mean(dim=0)

    def __contains__(self, value: Any) -> bool:
        return value in self.stoi


@dataclass
class ConceptVocab:
    grouped_attrs: Dict[str, Vocab]
    attr_types: Vocab
    objects: Vocab
    relations: Vocab

    @cached_property
    def concepts(self) -> Vocab:
        out = copy.deepcopy(self.objects)
        out.extend(self.attributes)
        out.extend(self.relations)
        return out

    @cached_property
    def attributes(self) -> Vocab:
        vocabs = list(self.grouped_attrs.values())
        out = copy.deepcopy(vocabs[0])
        for vocab in vocabs[1:]:
            out.extend(vocab)
        return out

    def to(self, *args):
        for vocab in self.grouped_attrs.values():
            vocab.vectors = vocab.vectors.to(*args)

        for vocab in (v for k, v in vars(self).items() if k != "grouped_attrs"):
            vocab.vectors = vocab.vectors.to(*args)

        return self


@lru_cache
def get_concept_vocab(
    data_path: Path, glove_size: Literal[50, 100, 200, 300] = 300
) -> ConceptVocab:
    logger.info("loading concept vocab")
    cache_path = data_path / "vgLists" / "cache" / f"concept_vocab_{glove_size}.pt"
    if cache_path.exists():
        logger.info(f"found cache at {cache_path}, loading...")
        return torch.load(cache_path.open(mode="rb"))

    concept_vocab = _get_concept_vocab(data_path, glove_size)

    logger.info(f"saving cached file to {cache_path}")
    cache_path.parent.mkdir(exist_ok=True)
    torch.save(concept_vocab, cache_path.open(mode="wb"))
    return concept_vocab


def _get_concept_vocab(
    data_path: Path, glove_size: Literal[50, 100, 200, 300] = 300
) -> ConceptVocab:
    # I don't like tqdm here, so redirect stderr to hide the porgress bar
    with redirect_stderr(open(os.devnull, "w")):
        logger.info("loading glove")
        glove = GloVe(name="6B", dim=300, cache=data_path / "glove.6B")
    # Get grouped_attrs
    grouped_attrs: Dict[str, Vocab] = {}
    for key, tokens in json.load(open(data_path / "vgLists" / "attrMap.json")).items():
        vocab = Vocab(Counter(tokens), specials=[])
        vocab.vectors = _get_concept_vectors(vocab.itos, glove)
        grouped_attrs[key] = vocab
    # Get attr_types
    attr_types = Vocab(
        Counter(list(json.load(open(data_path / "vgLists" / "typeInfo.json")))),
        specials=[],
    )
    attr_types.vectors = _get_attr_type_vectors(attr_types.itos, grouped_attrs, glove)

    def simple_load(path: Path) -> Vocab:
        itos = list(json.load(path.open()))
        vocab = Vocab(Counter(itos), specials=[])
        vocab.vectors = _get_concept_vectors(vocab.itos, glove)
        return vocab

    # Objects and relations are straightforward
    objects = simple_load(data_path / "vgLists" / "hObjInfo.json")
    relations = simple_load(data_path / "vgLists" / "relInfo.json")
    return ConceptVocab(grouped_attrs, attr_types, objects, relations)


def _get_concept_vectors(concepts: List[str], glove: GloVe) -> Tensor:
    """
    What happens if some concepts don't appear in glove? And if some
    contain two or more tokens?

    Well, this functions decides what to do
    """
    split_concepts = [c.split() for c in concepts]
    # If some concepts don't appear in glove use default unk embedding
    all_vectors = dict(
        zip(
            flat := list(chain.from_iterable(split_concepts)),
            glove.get_vecs_by_tokens(flat),
        )
    )
    # If some concepts contain many tokens, average them
    concept_vectors = [
        torch.mean(torch.vstack([all_vectors[s] for s in split]), dim=0)
        for split in split_concepts
    ]
    return torch.vstack(concept_vectors)


def _get_attr_type_vectors(
    attr_types: List[str], grouped_attrs: Dict[str, Vocab], glove: GloVe
) -> Tensor:
    """How to get glove embeddings for the attribute types, when
    some fo them are "42", "sportActivity" or "texture2"?

    Well, here are the arbitrary decision I made.
    """
    assert set(attr_types) == set(
        grouped_attrs
    ), "shit, attr_types and grouped_attrs keys don't match"
    assert all(
        vocab.vectors is not None for vocab in grouped_attrs.values()
    ), "shit, some vocabs in grouped vocabs don't have tensor, i don't know what to do"

    # Get glove vectors for attr types that don't contain number
    # and that are fount in Glove
    def condition(attr_type: str) -> bool:
        return attr_type.isalpha() and attr_type in glove

    glove_attr_types = [t for t in attr_types if condition(t)]
    glove_vectors: Dict[str, Tensor] = dict(
        zip(glove_attr_types, glove.get_vecs_by_tokens(glove_attr_types))
    )

    # If attr type matches our condition use both the glove embedding plus
    # the average of all its attrs as the final embedding
    # If not, use only the average of all its attrs
    vectors = [
        torch.mean(
            torch.vstack(
                (  # type: ignore[arg-type]
                    glove_vectors[attr_type],
                    grouped_attrs[attr_type].vectors,
                )
            ),
            dim=0,
        )
        if condition(attr_type)
        else torch.mean(
            grouped_attrs[attr_type].vectors, dim=0  # type:ignore[arg-type]
        )
        for attr_type in attr_types
    ]
    return torch.vstack(vectors)
