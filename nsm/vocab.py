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
from typing import ClassVar, Dict, Iterable, List, Literal, Optional, cast

import torch
from torch import Tensor


@dataclass(frozen=True)
class Vocab:
    itos: List[str]
    vectors: Optional[Tensor] = None

    def __post_init__(self):
        if len(self.itos) != len(set(self.itos)):
            raise ValueError("itos contains duplicates")
        if self.vectors and len(self.itos) != len(self.vectors):
            raise ValueError("itos and vectors sizes don't match")

    @classmethod
    def from_iterable(
        cls, tokens: Iterable[str], vectors: Optional[Iterable[Tensor]]
    ) -> Vocab:
        if vectors:
            d = dict(zip(tokens, vectors))
            return cls(list(d), torch.vstack(list(d.values())))
        else:
            return cls(list(set(tokens)))

    @cached_property
    def stoi(self) -> Dict[str, int]:
        return {s: i for i, s in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def __or__(self, other: Vocab) -> Vocab:
        if self.vectors and other.vectors:
            d = dict(
                chain(
                    zip(self.itos, self.vectors),
                    zip(other.itos, other.vectors),
                )
            )
            return Vocab(list(d), torch.vstack(list(d.values())))
        elif not self.vectors and not other.vectors:
            return Vocab(list(set(chain(self.itos, other.itos))))
        else:
            raise ValueError("one vocab has tensors and the other doesn't")


class Glove:
    filename_template: ClassVar[str] = "glove.6B.{}d.txt"
    _words: List[str]
    _file_path: Path

    def __init__(self, root_dir: Path, d_size: Literal[50, 100, 200, 300]) -> None:
        self._file_path = root_dir / self.filename_template.format(d_size)
        with self._file_path.open() as f:
            self._words = [line.split()[0] for line in f]

    def __contains__(self, el: str) -> bool:
        return el in self._words

    def get_vectors(self, words: List[str], allow_unk: bool = False) -> Tensor:
        if not allow_unk and (not_found := [w for w in words if w not in self._words]):
            raise ValueError(
                "The following words don't appear in Glove: " f"{', '.join(not_found)}"
            )
        vectors: Dict[str, Tensor] = defaultdict(lambda: self.unk_embedding)
        with self._file_path.open() as f:
            for line in f:
                glove_word, *rest = line.split()
                if glove_word in words:
                    vectors[glove_word] = torch.tensor([float(s) for s in rest])
        return torch.vstack([vectors[word] for word in words])

    @cached_property
    def unk_embedding(self) -> Tensor:
        with self._file_path.open() as f:
            return reduce(
                add,
                (torch.tensor([float(s) for s in line.split()[1:]]) for line in f),
            )


@dataclass(frozen=True)
class ConceptVocab:
    grouped_attrs: Dict[str, Vocab]
    attr_types: Vocab
    objects: Vocab
    relations: Vocab

    @cached_property
    def concepts(self):
        return reduce(
            operator.or_,
            [self.objects, self.relations, *self.grouped_attrs.values()],
        )

    @cached_property
    def attributes(self):
        return reduce(operator.or_, self.grouped_attrs.values())


@lru_cache
def get_concept_vocab(
    data_path: Path, glove_size: Literal[50, 100, 200, 300] = 300
) -> ConceptVocab:
    print("loading concept vocab")
    cache_path = data_path / "vgLists" / "cache" / f"concept_vocab_{glove_size}.pkl"
    if cache_path.exists():
        print("found cache at {cache_path}, loading...")
        return pickle.load(cache_path.open(mode="rb"))

    concept_vocab = _get_concept_vocab(data_path, glove_size)

    print(f"saving cached file to {cache_path}")
    cache_path.parent.mkdir(exist_ok=True)
    pickle.dump(concept_vocab, cache_path.open(mode="wb"))
    return concept_vocab


def _get_concept_vocab(
    data_path: Path, glove_size: Literal[50, 100, 200, 300] = 300
) -> ConceptVocab:
    glove = Glove(data_path / "glove.6B", d_size=300)
    # Get grouped_attrs
    raw_grouped_attrs = json.load(open(data_path / "vgLists" / "attrMap.json"))
    all_attrs = [attr for group in raw_grouped_attrs.values() for attr in group]
    attr_vectors = dict(zip(all_attrs, _get_concept_vectors(all_attrs, glove)))
    grouped_attrs = {
        key: Vocab(itos, torch.vstack([attr_vectors[w] for w in itos]))
        for key, itos in raw_grouped_attrs.items()
    }
    # Get attr_types
    attr_types = Vocab(
        itos := list(json.load(open(data_path / "vgLists" / "typeInfo.json"))),
        vectors=_get_attr_type_vectors(itos, grouped_attrs, glove),
    )

    def simple_load(path: Path) -> Vocab:
        itos = list(json.load(path.open()))
        return Vocab(itos, vectors=_get_concept_vectors(itos, glove))

    # Objects and relations are straightforward
    objects = simple_load(data_path / "vgLists" / "hObjInfo.json")
    relations = simple_load(data_path / "vgLists" / "relInfo.json")
    return ConceptVocab(grouped_attrs, attr_types, objects, relations)


def _get_concept_vectors(concepts: List[str], glove: Glove) -> Tensor:
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
            glove.get_vectors(flat, allow_unk=True),
        )
    )
    # If some concepts contain many tokens, average them
    concept_vectors = [
        torch.mean(torch.vstack([all_vectors[s] for s in split]), dim=0)
        for split in split_concepts
    ]
    return torch.vstack(concept_vectors)


def _get_attr_type_vectors(
    attr_types: List[str], grouped_attrs: Dict[str, Vocab], glove: Glove
) -> Tensor:
    """How to get glove embeddings for the attribute types, when
    some fo them are "42", "sportActivity" or "texture2"?

    Well, here are the arbitrary decision I made.
    """
    assert set(attr_types) == set(
        grouped_attrs
    ), "shit, attr_types and grouped_attrs keys don't match"
    assert all(
        vocab.vectors for vocab in grouped_attrs.values()
    ), "shit, some vocabs in grouped vocabs don't have tensor, i don't know what to do"

    # Get glove vectors for attr types that don't contain number
    # and that are fount in Glove
    def condition(attr_type: str) -> bool:
        return attr_type.isalpha() and attr_type in glove

    glove_attr_types = [t for t in attr_types if condition(t)]
    glove_vectors: Dict[str, Tensor] = dict(
        zip(glove_attr_types, glove.get_vectors(glove_attr_types))
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
