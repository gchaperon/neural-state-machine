import os
import h5py
import orjson
import inspect
from operator import itemgetter, attrgetter
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch import Tensor
from nsm.vocab import Vocab, GloVe, PUNC_POSTAGS, ConceptVocab
from nsm.utils import collate_graphs, split_batch, scatter_softmax, Batch, NSMItem
from torch_scatter import scatter_sum
from pathlib import Path
import json
import time
from typing import (
    FrozenSet,
    overload,
    get_type_hints,
    TypeVar,
    Callable,
    KeysView,
    ClassVar,
    Dict,
    List,
    Any,
    Union,
    Optional,
    Tuple,
    NamedTuple,
    Literal,
    Iterator,
    Iterable,
    TypedDict,
    Container,
    Set,
)
from collections import Counter
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from nsm.utils import to_snake, to_camel, Graph
from functools import cached_property, lru_cache
from itertools import chain, tee
import keyword
import ijson
import string
import re
from stanza.server import CoreNLPClient
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import logging
from mypy_extensions import VarArg

logger = logging.getLogger(__name__)

# NOTE: Possibly add more fields
class Question(BaseModel):
    image_id: str
    question: str
    answer: str

    class Config:
        alias_generator = to_camel
        allow_population_by_field_name = True


class ProcessedQuestion(NamedTuple):
    image_id: str
    question: List[str]
    answer: str


# NOTE: Possibly add more fields in the future
class SceneGraph(BaseModel):
    class Object(BaseModel):

        name: str
        attributes: List[str]

        class Relation(BaseModel):
            name: str
            object: str

        relations: List[Relation]

    objects: Dict[str, Object]


class GQAItem(NamedTuple):
    graph: Graph
    question: Tensor
    target: int


class BatchedGQAItem(NamedTuple):
    graph_batch: Batch
    question_batch: nn.utils.rnn.PackedSequence
    target: Tensor


class TDQuestion(TypedDict):
    imageId: str
    question: str
    answer: str


class TDTaggerProcessed(TypedDict):
    imageId: str
    question: List[str]
    answer: str


class TDRelation(TypedDict):
    name: str
    object: str


class TDObject(TypedDict):
    name: str
    attributes: List[str]
    relations: List[TDRelation]


class TDSceneGraph(TypedDict):
    objects: Dict[str, TDObject]


class SceneGraphProcessor:
    objects: Dict[str, Tensor]
    attributes: Dict[frozenset, Tensor]
    relations: Dict[str, Tensor]
    concept_vocab: ConceptVocab
    glove_dim: int

    def __init__(
        self,
        scene_graphs: Iterable[TDSceneGraph],
        glove: GloVe,
        concept_vocab: ConceptVocab,
    ):
        obj_strs: Set[str] = set()
        attr_sets: Set[FrozenSet[str]] = set()
        rel_strs: Set[str] = set()
        for sg in scene_graphs:
            for obj in sg["objects"].values():
                obj_strs.add(obj["name"])
                attr_sets.add(frozenset(obj["attributes"]))
                for rel in obj["relations"]:
                    rel_strs.add(rel["name"])
        self.objects = self.get_object_embeddings(
            obj_strs, glove, concept_vocab.objects.vectors
        )
        self.attributes = self.get_attr_embeddings(
            attr_sets, glove, [v.vectors for v in concept_vocab.grouped_attrs.values()]
        )
        self.relations = self.get_rel_embeddings(
            rel_strs, glove, concept_vocab.relations.vectors
        )
        # Mainly for verification purposes
        self.concept_vocab = concept_vocab
        self.glove_dim = glove.dim

    def process(self, sg: TDSceneGraph) -> Graph:
        oidtoi = {key: i for i, key in enumerate(sg["objects"].keys())}

        obj_names: List[str] = []
        attr_sets: List[FrozenSet[str]] = []
        rel_indices: List[Tuple[int, int]] = []
        rel_names: List[str] = []

        for from_id, obj in sg["objects"].items():
            from_index = oidtoi[from_id]
            obj_names.append(obj["name"])
            attr_sets.append(frozenset(obj["attributes"]))

            for rel in obj["relations"]:
                to_index = oidtoi[rel["object"]]
                rel_indices.append((from_index, to_index))
                rel_names.append(rel["name"])

        node_attrs = (
            torch.stack(
                [
                    torch.vstack((self.objects[o_name], self.attributes[a_set]))
                    for o_name, a_set in zip(obj_names, attr_sets)
                ]
            )
            if obj_names
            else torch.empty(
                0, len(self.concept_vocab.grouped_attrs) + 1, self.glove_dim
            )
        )

        edge_indices = (
            torch.tensor(rel_indices).T.contiguous()
            if rel_indices
            else torch.randint(1, (2, 0))
        )
        edge_attrs = (
            torch.stack([self.relations[r_name] for r_name in rel_names])
            if rel_names
            else torch.empty(0, self.glove_dim)
        )

        assert node_attrs.size(0) == len(sg["objects"])
        assert node_attrs.size(1) == len(self.concept_vocab.grouped_attrs) + 1
        assert edge_indices.size(1) == edge_attrs.size(0)

        return Graph(node_attrs, edge_indices, edge_attrs)

    @staticmethod
    def get_object_embeddings(
        obj_strs: Set[str], glove: GloVe, concept_obj_vectors: Tensor
    ):
        """
        Dims:
            O: number of different object names, =len(obj_strs)
            H: glove dim
            C: number of objects in the concept vocab, =len(concept_obj_vectors)
        """
        logger.info("Computing object embeddings")
        fixed_order = list(obj_strs)
        # O x H
        obj_glove = torch.stack(
            [glove.get_vecs_by_tokens(s.split()).mean(0) for s in fixed_order]
        )
        embedded = (
            F.softmax(obj_glove @ concept_obj_vectors.T, dim=1) @ concept_obj_vectors
        )
        assert len(fixed_order) == embedded.size(0)
        return dict(zip(fixed_order, embedded.unbind(0)))

    @staticmethod
    def get_attr_embeddings(
        attr_sets: Set[FrozenSet[str]],
        glove: GloVe,
        concept_attrs_vectors: List[Tensor],
    ):
        """
        Dims:
            A: total number fo attribute sets, =len(attr_sets)
            H: glove size
            Ci: size of each set of attrs in concept vocab, =len(concept_attrs_vectors[i])
            P: number of attr categories, =len(concept_attrs_vectors)
        """
        logger.info("Computing attribute embeddings")
        fixed_order = list(attr_sets)
        # A x H
        glove_attrs = torch.stack(
            [
                glove.get_vecs_by_tokens(
                    [tok for attr in attrs for tok in attr.split()]
                ).mean(0)
                if attrs
                else torch.zeros(glove.dim)
                for attrs in attr_sets
            ]
        )
        # List[A x H]
        per_concept_attr_embed = [
            F.softmax(glove_attrs @ Ci.T, dim=1) @ Ci for Ci in concept_attrs_vectors
        ]
        # P x A x H
        embedded = torch.stack(per_concept_attr_embed)
        assert len(fixed_order) == embedded.size(1)
        return dict(zip(fixed_order, embedded.unbind(1)))

    @staticmethod
    def get_rel_embeddings(
        rel_strs: Set[str], glove: GloVe, concept_rel_vectors: Tensor
    ):
        logger.info("Computing relation embeddings")
        fixed_order = list(rel_strs)
        glove_rels = torch.stack(
            [glove.get_vecs_by_tokens(s.split()).mean(0) for s in fixed_order]
        )
        embedded = (
            F.softmax(glove_rels @ concept_rel_vectors.T, dim=1) @ concept_rel_vectors
        )
        assert len(fixed_order) == embedded.size(0)
        return dict(zip(fixed_order, embedded.unbind(0)))


def process_tagged(sent: Any) -> List[str]:
    return [t.word.lower() for t in sent.token if t.pos not in PUNC_POSTAGS]


def process_q_batch(batch: List[str], tagger_client: CoreNLPClient) -> List[List[str]]:
    n_questions = len(batch)
    assert n_questions > 0
    text = " ".join(batch)
    assert len(text) <= tagger_client.DEFAULT_MAX_CHAR_LENGTH
    ann = tagger_client.annotate(text)
    assert len(ann.sentence) == n_questions
    return [process_tagged(s) for s in ann.sentence]


def process_questions(
    key_questions: List[Tuple[str, TDQuestion]], tagger_root: Path
) -> List[Tuple[str, TDTaggerProcessed]]:
    with CoreNLPClient(
        annotators=["tokenize", "ssplit", "pos"],
        be_quiet=True,
        classpath=str(tagger_root / "*"),
        memory="12G",
        threads=os.cpu_count(),
    ) as tagger_client:
        batch_slices: List[int] = []
        curr_chars = 0
        curr_n_questions = 0
        for key, question in key_questions:
            # Try to maximize number of questions sent to tagger client
            if (
                curr_chars + len(question["question"])
                > tagger_client.DEFAULT_MAX_CHAR_LENGTH
            ):
                batch_slices.append(curr_n_questions)
                curr_n_questions = 1
                curr_chars = len(question["question"]) + 1
            else:
                curr_chars += len(question["question"]) + 1
                curr_n_questions += 1
        # Last slice should make sure that all questions are processed
        batch_slices.append(len(key_questions))

        tagger_processed: List[Tuple[str, TDTaggerProcessed]] = []
        q_it = iter(tqdm(key_questions, desc="Processing questions", smoothing=0))
        for slice_ in batch_slices:
            batch = list(islice(q_it, slice_))
            q_strs = [q["question"] for _, q in batch]
            processed = process_q_batch(q_strs, tagger_client)
            for (key, q), pq in zip(batch, processed):
                tagger_processed.append(
                    (
                        key,
                        {
                            "imageId": q["imageId"],
                            "question": pq,
                            "answer": q["answer"],
                        },
                    )
                )
    return tagger_processed


def load_processed_questions(
    q_path: Path, cache_path: Path, corenlp_root: Path
) -> List[Tuple[str, TDTaggerProcessed]]:
    logger.info(f"Processing questions from {q_path}")
    if cache_path.exists():
        logger.info(f"Found cache at {cache_path}, loading...")
        with cache_path.open("rb") as f:
            return orjson.loads(f.read())

    logger.info("Tokenizing and removin punctuation")
    with q_path.open("rb") as f:
        questions = orjson.loads(f.read())
    processed = process_questions(list(questions.items()), corenlp_root)
    cache_path.parent.mkdir(parents=False, exist_ok=True)
    logger.info(f"Saving to {cache_path}")
    with cache_path.open("wb") as f:
        f.write(orjson.dumps(processed))
    return processed


# Turns out data.Dataset is the most useless class ever,
# but I like to inherit anyways just cause it looks pretty
class GQASceneGraphsOnlyDataset(data.Dataset[NSMItem]):

    resources: ClassVar[Dict[str, Dict[str, Path]]] = {
        "questions": {
            "train": Path("questions/train_balanced_questions.json"),
            "val": Path("questions/val_balanced_questions.json"),
        },
        "scene_graphs": {
            "train": Path("sceneGraphs/train_sceneGraphs.json"),
            "val": Path("sceneGraphs/val_sceneGraphs.json"),
        },
    }

    root_path: Path
    split: Literal["train", "val"]
    tagger_processed_qs: Dict[str, TDTaggerProcessed]
    scene_graphs: Dict[str, TDSceneGraph]
    preprocessing_vocab: Vocab
    answer_vocab: Vocab

    def __init__(
        self,
        gqa_root: Path,
        split: Literal["train", "val"],
        glove: GloVe,
        concept_vocab: ConceptVocab,
        corenlp_root: Path,
    ) -> None:
        self.check_root_dir(gqa_root)
        # Load questions
        self.root_path = gqa_root
        self.split = split

        self.scene_graphs = self._get_scene_graphs()
        self.processor = SceneGraphProcessor(
            self.scene_graphs.values(), glove, concept_vocab
        )
        self.tagger_processed_qs = self._get_tagger_processed(tagger_root=corenlp_root)
        self.preprocessing_vocab = self.get_preprocessing_vocab(
            gqa_root, glove, corenlp_root
        )
        self.answer_vocab = self.get_answer_vocab(gqa_root)

    def _get_scene_graphs(self) -> Dict[str, TDSceneGraph]:
        # Caching is useful here only because the raw scene graphs contain
        # a lot more information. We only need some dict values, so caching should
        # make loading faster
        cache_path = self.root_path / "cache" / f"scene_graphs_{self.split}.json"
        if cache_path.exists():
            logger.info(f"Loading {cache_path}")
            with cache_path.open("rb") as f:
                return orjson.loads(f.read())

        # Load all
        sgs_path = self.root_path / self.resources["scene_graphs"][self.split]
        logger.info(f"Loading scene graphs from {sgs_path}")
        with sgs_path.open("rb") as f:
            raw_scene_graphs = orjson.loads(f.read())

        filtered: Dict[str, TDSceneGraph] = {}

        for sg_key, sg in tqdm(
            raw_scene_graphs.items(), desc="Filtering irrelevant keys"
        ):
            filtered[sg_key] = {
                "objects": {
                    o_key: {
                        "name": o["name"],
                        "attributes": o["attributes"],
                        "relations": o["relations"],
                    }
                    for o_key, o in sg["objects"].items()
                }
            }
        logger.info(f"Saving cached file to {cache_path}")
        cache_path.parent.mkdir(parents=False, exist_ok=True)
        with cache_path.open("wb") as f:
            f.write(orjson.dumps(filtered))

        return filtered

    def _get_tagger_processed(self, tagger_root: Path) -> Dict[str, TDTaggerProcessed]:
        cache_path = (
            self.root_path / "cache" / f"tagger_processed_balanced_{self.split}.json"
        )
        questions_path = self.root_path / self.resources["questions"][self.split]
        processed = load_processed_questions(questions_path, cache_path, tagger_root)
        return dict(processed)

    def __getitem__(self, key: str) -> NSMItem:
        question = self.tagger_processed_qs[key]
        graph = self.processor.process(self.scene_graphs[question["imageId"]])
        embedded_q = torch.stack(
            [
                self.preprocessing_vocab.vectors[self.preprocessing_vocab.stoi[tok]]
                for tok in question["question"]
                if tok in self.preprocessing_vocab
            ]
        )
        target = self.answer_vocab.stoi[question["answer"]]
        return graph, embedded_q, target

    def __len__(self) -> int:
        return len(self.tagger_processed_qs)

    def __contains__(self, key) -> bool:
        return key in self.tagger_processed_qs

    def keys(self) -> KeysView[str]:
        return self.tagger_processed_qs.keys()

    @classmethod
    def splits(
        cls,
        gqa_root: Path,
        glove: GloVe,
        concept_vocab: ConceptVocab,
        corenlp_root: Path,
    ) -> Tuple[GQASceneGraphsOnlyDataset, GQASceneGraphsOnlyDataset]:
        first, *rest = [gqa_root, glove, concept_vocab, corenlp_root]
        return cls(first, "train", *rest), cls(first, "val", *rest)  # type:ignore

    @classmethod
    @lru_cache
    def get_answer_vocab(cls, gqa_root: Path) -> Vocab:
        cls.check_root_dir(gqa_root)
        cache_path = gqa_root / "cache" / "answer_vocab_balanced.pt"
        if cache_path.exists():
            logger.info(f"Found cache at {cache_path}, loading...")
            return torch.load(cache_path.open("rb"))

        logger.info("Creating answer vocab")
        q_train_path = gqa_root / cls.resources["questions"]["train"]
        with open(q_train_path, "rb") as f:
            questions: Dict[str, TDQuestion] = orjson.loads(f.read())

        answers = (q["answer"] for q in questions.values())
        vocab = Vocab(Counter(answers), max_size=2000, specials=("<unk>",))
        logger.info(f"Saving cached file to {cache_path}")
        cache_path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(vocab, cache_path.open("wb"))
        return vocab

    @classmethod
    @lru_cache
    def get_preprocessing_vocab(
        cls,
        gqa_root: Path,
        glove: GloVe,
        corenlp_root: Path,
    ) -> Vocab:
        cls.check_root_dir(gqa_root)
        # Caching stuff
        cache_path = (
            gqa_root / "cache" / f"preprocessing_vocab_balanced_{glove.dim}d.pt"
        )
        if cache_path.exists():
            logger.info(f"Found cache at {cache_path}, loading...")
            return torch.load(cache_path)

        # If not found in cache, create de vocab
        logger.info("Creating preprocessing vocab")
        q_train_path = gqa_root / cls.resources["questions"]["train"]
        # Use the same cache that is used in processing question
        processed_questions_cache_path = (
            gqa_root / "cache" / f"tagger_processed_balanced_train.json"
        )
        processed = load_processed_questions(
            q_train_path, processed_questions_cache_path, corenlp_root
        )
        tokens = (tok for _, q in processed for tok in q["question"])
        vocab = Vocab(Counter(tokens), max_size=5000, specials=[])
        vectors = glove.get_vecs_by_tokens(vocab.itos)
        vocab.set_vectors(
            {s: i for i, s in enumerate(vocab.itos)},
            vectors,
            glove.dim,
            unk_init=lambda *args: 1 / 0,  # This should never be called
        )
        logger.info(f"Saving preprocessing vocab to {cache_path}")
        cache_path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(vocab, cache_path)
        return vocab

    @classmethod
    def check_root_dir(cls, path: Union[str, Path]) -> None:
        if nonexistant := [
            res
            for res_type in cls.resources.values()
            for res in res_type.values()
            if not (path / res).exists()
        ]:
            raise ValueError(f"Resource(s) not found: {list(map(str, nonexistant))}")
