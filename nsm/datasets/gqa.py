import h5py
from operator import itemgetter, attrgetter
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch import Tensor
from nsm.vocab import Vocab, GloVe, PUNC_POSTAGS, ConceptVocab
from nsm.utils import collate_graphs, split_batch, segment_softmax_coo, Batch
from torch_scatter.segment_coo import segment_sum_coo
from pathlib import Path
import json
from typing import (
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


def collate_gqa(batch):
    graphs, questions, targets = zip(*batch)

    return BatchedGQAItem(
        collate_graphs(graphs),
        nn.utils.rnn.pack_sequence(questions, enforce_sorted=False),
        torch.tensor(targets),
    )


def tag_questions(*questions: str, client: CoreNLPClient) -> Union[Any, List[Any]]:
    n_questions = len(questions)
    assert n_questions > 0
    text = " ".join(questions)
    assert len(text) < client.DEFAULT_MAX_CHAR_LENGTH
    ann = client.annotate(text)
    assert len(ann.sentence) == n_questions
    return ann.sentence[0] if n_questions == 1 else list(ann.sentence)


def batch_tag(
    questions: Iterator[str], client: CoreNLPClient, size: int = 1000
) -> Iterator[Any]:
    it = iter(questions)
    while qs := list(islice(it, size)):
        yield from tag_questions(*qs, client=client)


_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


def batch_process(
    iterable: Iterable[_T1],
    process_fn: Callable[[List[_T1]], List[_T2]],
    batch_size: int,
) -> Iterable[_T2]:
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield from process_fn(batch)


class SceneGraphProcessor:
    glove: GloVe
    concept_vocab: ConceptVocab

    def __init__(self, glove: GloVe, concept_vocab: ConceptVocab) -> None:
        self.glove = glove
        self.concept_vocab = concept_vocab

    def obj_glove_embed(self, obj: SceneGraph.Object) -> Tensor:
        name_embed = self.glove.get_vecs_by_tokens(obj.name.split()).mean(dim=0)
        attr_embed = (
            self.glove.get_vecs_by_tokens(
                [s for attr in obj.attributes for s in attr.split()]
            ).mean(0)
            if obj.attributes
            else torch.zeros_like(name_embed)
        )
        return torch.vstack([name_embed, attr_embed])

    def rel_glove_embed(self, rel: SceneGraph.Object.Relation) -> Tensor:
        return self.glove.get_vecs_by_tokens(rel.name.split()).mean(dim=0)

    def glove_node_attrs(self, scene_graph: SceneGraph) -> Tensor:
        return (
            torch.stack(
                [self.obj_glove_embed(obj) for obj in scene_graph.objects.values()]
            )
            if scene_graph.objects
            else torch.empty(0, 2, self.glove.dim, device=self.glove.vectors.device)
        )

    def indices_n_glove_edge_attrs(
        self, scene_graph: SceneGraph
    ) -> Tuple[Tensor, Tensor]:
        obj_index = {key: i for i, key in enumerate(scene_graph.objects.keys())}

        edge_indices_list: List[Tuple[int, int]] = []
        edge_attrs_list: List[Tensor] = []
        for obj_id, obj in scene_graph.objects.items():
            from_ndx = obj_index[obj_id]
            for rel in obj.relations:
                to_ndx = obj_index[rel.object]
                edge_attrs_list.append(self.rel_glove_embed(rel))
                edge_indices_list.append((from_ndx, to_ndx))
        edge_indices = (
            torch.tensor(list(zip(*edge_indices_list)))
            if edge_indices_list
            else torch.randint(1, (2, 0), device=self.glove.vectors.device)
        )
        assert edge_indices.dtype in (torch.short, torch.int, torch.long)
        edge_attrs = (
            torch.vstack(edge_attrs_list)
            if edge_attrs_list
            else torch.empty(0, self.glove.dim, device=self.glove.vectors.device)
        )
        assert edge_indices.size(1) == edge_attrs.size(0)
        return edge_indices, edge_attrs

    def glove_embed(self, scene_graph: SceneGraph) -> Graph:
        return Graph(
            self.glove_node_attrs(scene_graph),
            *self.indices_n_glove_edge_attrs(scene_graph),
        )

    def __call__(self, scene_graphs: List[SceneGraph]) -> List[Graph]:
        glove_embedded = [self.glove_embed(sg) for sg in scene_graphs]
        batch = collate_graphs(glove_embedded)

        node_props = [self.concept_vocab.objects.vectors] + [
            v.vectors for v in self.concept_vocab.grouped_attrs.values()
        ]
        prop_indices = torch.repeat_interleave(
            torch.tensor([prop.size(0) for prop in node_props])
        )
        extended = batch.node_attrs[
            :,
            torch.repeat_interleave(
                torch.tensor([len(node_props[0]), sum(map(len, node_props[1:]))])
            ),
        ]
        prop_probs = segment_softmax_coo(
            torch.sum(extended * torch.vstack(node_props), dim=2),
            prop_indices,
            dim=1,
        )
        processed_node_attrs = segment_sum_coo(
            prop_probs[..., None] * torch.vstack(node_props),
            prop_indices.expand(batch.node_attrs.size(0), -1),
        )
        processed_edge_attrs = (
            batch.edge_attrs.matmul(self.concept_vocab.relations.vectors.T)
            .softmax(1)
            .unsqueeze(2)
            .mul(self.concept_vocab.relations.vectors)
            .sum(1)
        )
        processed_batch = Batch(
            **{
                **batch._asdict(),
                "node_attrs": processed_node_attrs,
                "edge_attrs": processed_edge_attrs,
            }
        )

        processed_graphs = split_batch(processed_batch)
        return processed_graphs


def process_tagged(sentence: Any) -> List[str]:
    """ Lowercase question and remove punctuation tokens"""
    return [tok.word.lower() for tok in sentence.token if tok.pos not in PUNC_POSTAGS]


# Turns out data.Dataset is the most useless class ever,
# but I like to inherit anyways just cause it looks pretty
class GQASceneGraphsOnlyDataset(data.Dataset[GQAItem]):

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

    preprocessed_questions: Dict[str, ProcessedQuestion]
    graphs_path: Path
    answer_vocab: Vocab
    preprocessing_vocab: Vocab

    def __init__(
        self,
        gqa_root: Path,
        split: Literal["train", "val"],
        glove: GloVe,
        concept_vocab: ConceptVocab,
        corenlp_root: Path,
        graph_preprocessing_device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.check_root_dir(gqa_root)
        # Load questions

        self.preprocessed_questions = self._tokenize_remove_punc(
            gqa_root, split, corenlp_root
        )
        self.graphs_path = self._preprocess_graphs(
            gqa_root, split, glove, concept_vocab, device=graph_preprocessing_device
        )
        self.answer_vocab = self.get_answer_vocab(gqa_root)
        self.preprocessing_vocab = self.get_preprocessing_vocab(
            gqa_root, glove, corenlp_root
        )
        self.glove = glove

    def __getitem__(self, key: str) -> GQAItem:
        question = self.preprocessed_questions[key]
        with h5py.File(self.graphs_path, "r") as graphs_h5:
            item = GQAItem(
                Graph(
                    **{
                        key: torch.from_numpy(np.array(dset))
                        for key, dset in graphs_h5[question.image_id].items()
                    }
                ),
                self.glove.get_vecs_by_tokens(
                    [
                        tok
                        for tok in question.question
                        if tok in self.preprocessing_vocab
                    ]
                ),
                self.answer_vocab.stoi[question.answer],
            )
            return item

    def __len__(self):
        return len(self.preprocessed_questions)

    def __contains__(self, key):
        return key in self.preprocessed_questions

    def keys(self) -> KeysView[str]:
        return self.preprocessed_questions.keys()

    @classmethod
    def _tokenize_remove_punc(
        cls,
        gqa_root: Path,
        split: Literal["train", "val"],
        corenlp_root: Path,
    ) -> Dict[str, ProcessedQuestion]:
        cls.check_root_dir(gqa_root)
        cache_path = gqa_root / "cache" / f"nopunct_questions_{split}_balanced.pt"
        if cache_path.exists():
            logger.info(f"Loading {cache_path}")
            return torch.load(cache_path)

        # logger.info("Preprocessing questions")
        q_path = gqa_root / cls.resources["questions"][split]
        with q_path.open("rb") as f:
            for i, _ in enumerate(ijson.kvitems(f, "")):
                pass
            total_questions = i + 1
            f.seek(0)

            kv_iter = ijson.kvitems(f, "")
            keys, values = map(itemgetter(0), (teed := tee(kv_iter))[0]), map(
                itemgetter(1), teed[1]
            )
            questions = (Question(**raw) for raw in values)
            image_ids, q_strs, answers = [
                map(attrgetter(attr_name), it)
                for attr_name, it in zip(
                    Question.__fields__, tee(questions, len(Question.__fields__))
                )
            ]

            with CoreNLPClient(
                annotators=["tokenize", "ssplit", "pos"],
                be_quiet=True,
                classpath=str(corenlp_root / "*"),
                memory="12G",
                threads=8,
            ) as client:
                processed_strs = (
                    process_tagged(tagged) for tagged in batch_tag(q_strs, client)
                )
                processed_questions = (
                    ProcessedQuestion(*item)
                    for item in zip(image_ids, processed_strs, answers)
                )
                out = dict(
                    zip(
                        keys,
                        tqdm(
                            processed_questions,
                            total=total_questions,
                            smoothing=0,
                            desc="Preprocessing questions",
                        ),
                    )
                )
        logger.info(f"Saving to {cache_path}")
        cache_path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(out, cache_path)
        return out

    @classmethod
    def _preprocess_graphs(
        cls,
        gqa_root: Path,
        split: Literal["train", "val"],
        glove: GloVe,
        concept_vocab: ConceptVocab,
        device: Union[str, torch.device],
    ) -> Path:
        cls.check_root_dir(gqa_root)
        cache_path = gqa_root / "cache" / f"graph_features_{split}.h5"
        if cache_path.exists():
            logger.info(f"Found cached graph features at {cache_path}")
            return cache_path

        old_device = glove.vectors.device
        glove.vectors = glove.vectors.to(device)
        concept_vocab = concept_vocab.to(device)
        scene_graphs_path = gqa_root / cls.resources["scene_graphs"][split]
        try:
            with scene_graphs_path.open("rb") as sgs_file, h5py.File(
                cache_path, "w"
            ) as h5_file:
                for i, _ in enumerate(ijson.kvitems(sgs_file, "")):
                    pass
                total_sgs = i + 1
                sgs_file.seek(0)

                process_fn = SceneGraphProcessor(glove, concept_vocab)

                teed = tee(ijson.kvitems(sgs_file, ""))
                keys = (k for k, _ in teed[0])
                graphs = batch_process(
                    (SceneGraph(**v) for _, v in teed[1]), process_fn, 100
                )

                for graph_key, graph in tqdm(
                    zip(keys, graphs),
                    smoothing=0,
                    total=total_sgs,
                    desc="Creating graph features",
                ):
                    group = h5_file.create_group(graph_key)
                    for tensor_name, tensor in graph._asdict().items():
                        group[tensor_name] = tensor.cpu().numpy()
        except BaseException as e:
            cache_path.unlink(missing_ok=True)
            raise e

        glove.vectors = glove.vectors.to(old_device)
        concept_vocab = concept_vocab.to(old_device)
        return cache_path

    @classmethod
    def get_answer_vocab(cls, gqa_root: Path) -> Vocab:
        cls.check_root_dir(gqa_root)
        q_train_path = gqa_root / cls.resources["questions"]["train"]
        cache_path = gqa_root / "cache" / "answer_vocab_balanced.pt"
        if cache_path.exists():
            logger.info(f"found cache at {cache_path}, loading...")
            return torch.load(cache_path.open("rb"))
        logger.info("creating answer vocab")
        with open(q_train_path, "rb") as f:
            for i, _ in enumerate(ijson.kvitems(f, "")):
                pass
            total_questions = i + 1
            f.seek(0)

            q_iterator = ijson.kvitems(f, "")
            # Why Question? No reason.
            all_answers = (Question(**question).answer for _, question in q_iterator)
            it = tqdm(all_answers, total=total_questions, desc="Reading questions")
            vocab = Vocab(Counter(it), max_size=2000, specials=("<unk>",))
        logger.info(f"saving cached file to {cache_path}")
        cache_path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(vocab, cache_path.open("wb"))
        return vocab

    @classmethod
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
            logger.info(f"found cache at {cache_path}, loading...")
            return torch.load(cache_path)

        # If not found in cache, create de vocab
        logger.info("creating preprocessing vocab")
        q_train_path = gqa_root / cls.resources["questions"]["train"]
        with q_train_path.open("rb") as f, CoreNLPClient(
            annotators=["tokenize", "ssplit", "pos"],
            be_quiet=True,
            classpath=str(corenlp_root / "*"),
            memory="12G",
            threads=8,
        ) as tagger_client:
            q_iterator = (Question(**q).question for _, q in ijson.kvitems(f, ""))
            tagged_sentences = batch_tag(q_iterator, tagger_client)
            it = tqdm(tagged_sentences, desc="Reading questions")
            tokens = (tok for sent in it for tok in process_tagged(sent))
            vocab = Vocab(Counter(tokens), max_size=5000, specials=[])
        vectors = glove.get_vecs_by_tokens(vocab.itos)
        vocab.set_vectors(
            {s: i for i, s in enumerate(vocab.itos)},
            vectors,
            glove.dim,
            unk_init=lambda *args: 1 / 0,  # This should never be called
        )
        logger.info(f"saving preprocessing vocab to {cache_path}")
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
