import random
import os
import logging
import torch
import torch.nn.functional as F
import torch.utils.data as data
import pytorch_lightning as pl
from pathlib import Path
import typing as tp
import zipfile
import re
import json
from . import utils as data_utils
from operator import itemgetter, eq, ne
from functools import (
    partial,
    partialmethod,
    singledispatchmethod,
    cached_property,
    lru_cache,
)
import abc
from nsm.utils import Graph, NSMItem, collate_nsmitems, collate_graphs
import collections
import pydantic


logger = logging.getLogger(__name__)

# Determined via manual inspection of templates in
# https://github.com/facebookresearch/clevr-dataset-gen
# and using the info mentioned in this comment
# https://github.com/facebookresearch/clevr-dataset-gen/issues/14#issuecomment-482896597
# For reference, the loading order of templates in the question generation is the following
#
#    compare_integer.json
#    comparison.json
#    three_hop.json
#    single_and.json
#    same_relate.json
#    single_or.json
#    one_hop.json
#    two_hop.json
#    zero_hop.json
#
# Using this and the number of templates inside each file we can relate each
# "question_family_index" to each template
NHOPS_TO_CATS = {
    3: range(27, 31),
    1: range(74, 78),
    2: range(80, 84),
    0: range(86, 90),
}

ALL_EASY_CATS = [cat for range_ in NHOPS_TO_CATS.values() for cat in range_]
ALL_CATS = range(90)


class Glove:

    DATA_URL = "https://nlp.stanford.edu/data/glove.6B.zip"

    class Paths:
        NAME = "glove"
        ZIP_NAME = "glove.6B.zip"
        FILE_TEMPLATE = "glove.6B.{dim}d.txt"

        def __init__(self, datadir: str):
            self.datadir = Path(datadir)

        @property
        def root(self):
            return self.datadir / self.NAME

        @property
        def zip_path(self):
            return self.root / self.ZIP_NAME

        def file_path(self, dim):
            return self.root / self.FILE_TEMPLATE.format(dim=dim)

    tok2vec: tp.Dict[str, list]

    def __init__(self, datadir: str, dim: tp.Literal[50, 100, 200, 300]):
        self.paths = self.Paths(datadir)
        self.dim = dim

        if not self.paths.file_path(dim).exists():
            if not self.paths.zip_path.exists():
                logger.info("Downloading Glove")
                self.paths.root.mkdir(parents=True, exist_ok=True)
                data_utils.download(
                    self.DATA_URL,
                    self.paths.zip_path,
                    progress=logger.isEnabledFor(logging.INFO),
                )
            logger.info("Extracting Glove")
            with zipfile.ZipFile(self.paths.zip_path) as myzip:
                myzip.extractall(path=self.paths.root)
            logger.info("Done!")

        logger.info(f"Loading {self.paths.file_path(dim)}")
        with open(self.paths.file_path(dim)) as vec_file:
            self.tok2vec = {
                token: tuple(map(float, vector))
                for token, *vector in map(str.split, vec_file)
            }
        logger.info("Done!")

    def __repr__(self):
        return type(self).__name__ + f"(datadir={self.paths.datadir!r}, dim={self.dim})"

    @singledispatchmethod
    def embed(self, arg):
        raise NotImplementedError(f"Unable to embed type {type(arg)}")

    @embed.register
    @lru_cache
    def _(self, token: str) -> torch.Tensor:
        try:
            return torch.tensor(self.tok2vec[token])
        except KeyError:
            raise ValueError(f"Unable to embed {token=}") from None

    @embed.register
    def _(self, tokens: collections.abc.Sequence) -> torch.Tensor:
        return (
            torch.stack([self.embed(token) for token in tokens])
            if tokens
            else torch.empty(0, self.dim)
        )

    @embed.register
    def _(self, graph: Graph) -> Graph:
        node_attrs, edge_indices, edge_attrs = graph
        return Graph(
            self.embed(node_attrs), torch.tensor(edge_indices).T, self.embed(edge_attrs)
        )


class Vocab:
    METADATA_URL = (
        "https://raw.githubusercontent.com/facebookresearch"
        "/clevr-dataset-gen/master/question_generation/metadata.json"
    )
    # Some functions defined in metadata.json where used only for the generation process
    # and are not present in the final version
    SKIPPED_FNS = [
        "equal_object",
        "filter",
        "filter_count",
        "filter_exist",
        "filter_unique",
        "relate_filter",
        "relate_filter_count",
        "relate_filter_exist",
        "relate_filter_unique",
    ]

    metadata_path: Path
    metadata: dict
    prop_embed_const: float

    def __init__(self, metadata_path: tp.Union[str, Path]):
        self.metadata_path = Path(metadata_path)
        if not self.metadata_path.exists():
            data_utils.download(
                self.METADATA_URL,
                metadata_path,
                progress=logger.isEnabledFor(logging.INFO),
            )
        with open(self.metadata_path) as file:
            self.metadata = json.load(file)
        # self.properties = ["shape", "color", "size", "material"]

    @cached_property
    def properties(self):
        props = [
            type_.lower()
            for type_, value in self.metadata["types"].items()
            if value is not None
        ][::-1]
        # send property "relation" to last
        return sorted(props, key=partial(eq, "relation"))

    @cached_property
    def attributes(self):
        return [
            attr
            for prop in self.properties[:-1]
            for attr in self.grouped_attributes[prop]
        ]

    @cached_property
    def grouped_attributes(self):
        return {
            prop: self.metadata["types"][prop.capitalize()] for prop in self.properties
        }

    @cached_property
    def relations(self):
        return self.grouped_attributes[self.properties[-1]]

    @cached_property
    def concepts(self):
        # In this dataset I will handle all node attributes as "attributes", instead
        # of the first one beeing its "identity" and the rest the attributes
        return self.attributes + self.relations

    @cached_property
    def functions(self):
        return [
            fn["name"]
            for fn in self.metadata["functions"]
            if fn["name"] not in self.SKIPPED_FNS
        ]

    @cached_property
    def everything(self):
        return self.concepts + self.functions

    @cached_property
    def answers(self):
        # Answers include integers in [0..10] and bool (yes/no)
        # Answers dont' include relations
        return self.attributes + [str(i) for i in range(11)] + ["yes", "no"]

    @cached_property
    def concept_embeddings(self):
        return self.embed(self.concepts)

    @cached_property
    def property_embeddings(self):
        return torch.stack(
            [
                torch.sum(self.embed(self.grouped_attributes[prop]), dim=0)
                for prop in self.properties
            ]
        )

    @property
    def embed_size(self):
        return len(self.everything)

    @singledispatchmethod
    def embed(self, arg):
        raise NotImplementedError(f"Unable to embed type {type(arg)}")

    @embed.register
    @lru_cache
    def _(self, token: str) -> torch.Tensor:
        if token in self.everything:
            idx = self.everything.index(token)
            out = torch.zeros(self.embed_size)
            out[idx] = 1
            return out
        else:
            raise ValueError(f"Unable to embed {token=}")

    @embed.register
    def _(self, tokens: collections.abc.Sequence) -> torch.Tensor:
        return torch.stack([self.embed(token) for token in tokens])

    @embed.register
    def _(self, graph: Graph) -> Graph:
        node_attrs, edge_indices, edge_attrs = graph
        return Graph(
            self.embed(node_attrs), torch.tensor(edge_indices).T, self.embed(edge_attrs)
        )


class GloveVocab(Glove, Vocab):
    def __init__(
        self,
        datadir,
        dim,
        metadata_path,
        prop_embed_method: tp.Literal["embed", "mean", "sum"] = "embed",
        prop_embed_scale: float = 1.0,
    ):
        super(GloveVocab, self).__init__(datadir, dim)
        # prop_embed_const is unused here
        super(Glove, self).__init__(metadata_path, prop_embed_const=0.0)
        self.prop_embed_method = prop_embed_method
        self.prop_embed_scale = prop_embed_scale

    # first property embedding heuristic
    @cached_property
    def property_embeddings(self):
        if self.prop_embed_method == "embed":
            out = self.embed(list(self.grouped_attributes.keys()))
        elif self.prop_embed_method == "mean":
            out = torch.stack(
                [
                    self.embed(self.grouped_attributes[prop]).mean(dim=0)
                    for prop in self.properties
                ]
            )
        elif self.prop_embed_method == "sum":
            out = torch.stack(
                [
                    self.embed(self.grouped_attributes[prop]).sum(dim=0)
                    for prop in self.properties
                ]
            )
        else:
            raise ValueError(f"Invalid prop_embed_method={self.prop_embed_method}")
        return self.prop_embed_scale * out


def scene_to_graph(scene: dict, vocab: Vocab) -> Graph:
    node_attrs = list(map(itemgetter(*vocab.properties[:-1]), scene["objects"]))
    edge_indices = []
    edge_attrs = []
    for rel_attr, adjacencies in scene["relationships"].items():
        for u, adjacency in enumerate(adjacencies):
            for v in adjacency:
                edge_indices.append((u, v))
                edge_attrs.append(rel_attr)
    return Graph(node_attrs, edge_indices, edge_attrs)


def program_to_polish(program: tp.List[dict]) -> tp.List[str]:
    def convert_node(node: dict) -> tp.List[str]:
        return [
            node["function"],
            *node["value_inputs"],
            *[s for input in node["inputs"] for s in convert_node(program[input])],
        ]

    return convert_node(program[-1])


class ClevrNoImagesDataset(data.Dataset):
    DATA_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip"

    class Paths:
        NAME = "clevr"
        ZIP_NAME = "CLEVR_v1.0_no_images.zip"
        METADATA_NAME = "metadata.json"

        QUESTIONS_PATH_TEMPLATE = "CLEVR_v1.0/questions/CLEVR_{split}_questions.json"
        SCENES_PATH_TEMPLATE = "CLEVR_v1.0/scenes/CLEVR_{split}_scenes.json"

        datadir: Path

        def __init__(self, datadir: tp.Union[str, Path]):
            self.datadir = Path(datadir)
            self.dataset_root.mkdir(parents=True, exist_ok=True)

        @property
        def dataset_root(self):
            return self.datadir / self.NAME

        @property
        def zip_path(self):
            return self.dataset_root / self.ZIP_NAME

        @property
        def metadata_path(self):
            return self.dataset_root / self.METADATA_NAME

    paths: Paths
    split: tp.Literal["train", "val"]
    questions: list
    scenes: list
    vocab: Vocab

    def __init__(
        self,
        datadir: tp.Union[str, Path],
        split: tp.Literal["train", "val"],
        download: bool = False,
        filter_fn: tp.Optional[tp.Callable] = None,
    ):
        self.paths = self.Paths(datadir)
        self.split = split
        self.vocab = Vocab(self.paths.metadata_path)

        if download:
            self.download(datadir)

        logger.info(f"Loading {self.questions_path}")
        # if no filter function, filter empty dicts
        filter_fn = filter_fn or bool
        with open(self.questions_path) as questions_file:
            self.questions = sorted(
                filter(filter_fn, json.load(questions_file)["questions"]),
                key=itemgetter("question_index"),
            )
        logger.info(f"Loading {self.scenes_path}")
        with open(self.scenes_path) as scenes_file:
            self.scenes = {
                scene["image_index"]: scene
                for scene in json.load(scenes_file)["scenes"]
            }

    def __getitem__(self, key: int):
        graph, polish, answer = self.get_raw(key)
        return (
            self.vocab.embed(graph),
            self.vocab.embed(polish),
            self.vocab.answers.index(answer),
        )

    def get_raw(self, key: int):
        question = self.questions[key]
        scene = self.scenes[question["image_index"]]
        return (
            scene_to_graph(scene, self.vocab),
            # filter "scene", which is always the last word/token
            # EDIT: don't filter scene, when the program is more complex, scene
            # appears more than 1 time and it is tricky to remove everywhere
            program_to_polish(question["program"]),
            question["answer"],
        )

    def __len__(self):
        return len(self.questions)

    @classmethod
    def download(cls, datadir: tp.Union[str, Path]):
        paths = cls.Paths(datadir)

        data_utils.download(
            cls.DATA_URL, paths.zip_path, progress=logger.isEnabledFor(logging.INFO)
        )
        if len(list(paths.dataset_root.iterdir())) == 1:
            logger.info(f"Extracting {paths.zip_path} to {paths.dataset_root}")
            with zipfile.ZipFile(paths.zip_path) as myzip:
                myzip.extractall(path=paths.dataset_root)
            logger.info("Done!")

    @property
    def questions_path(self):
        return self.paths.dataset_root / self.paths.QUESTIONS_PATH_TEMPLATE.format(
            split=self.split
        )

    @property
    def scenes_path(self):
        return self.paths.dataset_root / self.paths.SCENES_PATH_TEMPLATE.format(
            split=self.split
        )


class ClevrGlove(ClevrNoImagesDataset):
    def __init__(
        self,
        datadir: str,
        split: tp.Literal["train", "val"],
        glove_dim: tp.Literal[50, 100, 200, 300],
        download: bool = False,
        nhops: tp.Optional[tp.List[int]] = None,
        question_type: tp.Literal["program", "question"] = "program",
        prop_embed_method: tp.Literal["embed", "mean"] = "embed",
        prop_embed_scale: float = 1.0,
    ):
        self.nhops = nhops or list(NHOPS_TO_CATS.keys())
        cats = [cat for hop in self.nhops for cat in NHOPS_TO_CATS[hop]]

        def filter_fn(question):
            return question["question_family_index"] in cats

        super().__init__(
            datadir,
            split,
            download,
            filter_fn=filter_fn,
        )
        assert question_type in ("program", "question")
        self.question_type = question_type
        self.vocab = GloveVocab(
            datadir,
            glove_dim,
            self.paths.metadata_path,
            prop_embed_method=prop_embed_method,
            prop_embed_scale=prop_embed_scale,
        )

    def get_raw(self, key):
        graph, raw_program, answer = super().get_raw(key)
        if self.question_type == "program":
            question = [sub for token in raw_program for sub in token.split("_")]
        elif self.question_type == "question":
            question = (
                re.sub(r"[;?]", " ", self.questions[key]["question"]).lower().split()
            )
        return graph, question, answer


class ClevrWInstructions(ClevrNoImagesDataset):
    """Change the way questions are embedded, generate instructions directly"""

    def __init__(
        self,
        datadir,
        split,
        download=False,
        nhops: tp.Optional[tp.List[int]] = None,
        prop_embeds_const: float = 5.0,
    ):
        self.nhops = nhops or list(NHOPS_TO_CATS.keys())
        cats = [cat for hop in self.nhops for cat in NHOPS_TO_CATS[hop]]

        def filter_fn(question):
            return question["question_family_index"] in cats

        super().__init__(
            datadir,
            split,
            download,
            filter_fn=filter_fn,
            prop_embeds_const=prop_embeds_const,
        )

    def __getitem__(self, key):
        graph, question, answer = self.get_raw(key)
        return (
            self.vocab.embed(graph),
            self.vocab.embed(question),
            self.vocab.answers.index(answer),
            self.instructions_from_question(question),
        )

    def instructions_from_question(self, question):
        # + 2 because of first instruction to focus on initial node and last instruction
        # to query for a prop
        n_ins = max(self.nhops) + 2
        vocab = self.vocab
        instructions = []
        group = []

        for word in reversed(question):
            if word in vocab.relations:
                instructions.append(vocab.embed(group).sum(0))
                group = [word]
            elif word in vocab.attributes:
                group.append(word)
        if group:
            instructions.append(vocab.embed(group).sum(0))
        # last instruction, check which property is beeing queried
        # "query_color" -> "color"
        # NOTE: this might not be necessary since I changed the magnitude of
        # the prop embeddings
        # normalize to keep l2 loss reasonable when using generated instructions
        # as supervision
        prop = question[0].split("_")[1]
        instructions.append(
            F.normalize(vocab.property_embeddings[vocab.properties.index(prop)], dim=0)
        )
        # pad
        instructions = [
            torch.zeros(vocab.embed_size) for _ in range(n_ins - len(instructions))
        ] + instructions

        return torch.stack(instructions)

    def random_adversarial(self, n_impostors=5):
        """
        The idea here is to create an impostor, repeat it N times, then create
        a node for which some property will actually be queried and change that
        specific property to a different value compared to all of the impostors. An adversarial
        example should include at leat one relation, so a completely different node
        must be created so that the NSM attends to that one, and the relate the target
        node to this starting node using whatever relation
        """

        vocab = self.vocab

        impostor = tuple(
            random.choice(vocab.grouped_attributes[prop])
            for prop in vocab.properties[:-1]
        )
        query_prop = random.choice(range(len(vocab.properties) - 1))
        node = list(impostor)
        node[query_prop] = random.choice(
            [
                attr
                for attr in vocab.grouped_attributes[vocab.properties[query_prop]]
                if attr != impostor[query_prop]
            ]
        )
        node = tuple(node)
        # this node should be completely different from impostor
        start = tuple(
            random.choice(
                [
                    attr
                    for attr in vocab.grouped_attributes[vocab.properties[prop]]
                    if attr != impostor[prop]
                ]
            )
            for prop in range(len(vocab.properties) - 1)
        )
        rel = random.choice(vocab.grouped_attributes["relation"])
        graph = Graph(
            [node, start] + [impostor] * n_impostors,
            edge_indices=[(1, 0)],
            edge_attrs=[rel],
        )

        def flatten(tups):
            return [el for tup in tups for el in tup]

        # make question
        question = [
            f"query_{vocab.properties[query_prop]}",
            "unique",
            *flatten(
                (f"filter_{vocab.properties[prop]}", node[prop])
                for prop in [
                    p for p in range(len(vocab.properties) - 1) if p != query_prop
                ]
            ),
            "relate",
            rel,
            *flatten(
                (f"filter_{vocab.properties[prop]}", start[prop])
                for prop in range(len(vocab.properties) - 1)
            ),
            "scene",
        ]
        return graph, question, node[query_prop]

    def old_random_adversarial(self, n_impostors=5):
        vocab = self.vocab
        # make random node, ignore the "relation" property
        node = tuple(
            random.choice(vocab.grouped_attributes[prop])
            for prop in vocab.properties[:-1]
        )
        query_prop = random.choice(range(len(vocab.properties[:-1])))

        # make a bunch of impostors that satisfy
        # * for the queried prop they have all the same value, different to the value from "node"
        # * among the other n-1 properties, each imposor should share n-2 properties with "node",
        #   making "node" unique, but at the same time all the impostors are as close as possible
        #   to "node".

        impostors = []
        # chose diferent target
        fake_target = random.choice(
            list(
                filter(
                    partial(ne, node[query_prop]),
                    vocab.grouped_attributes[vocab.properties[query_prop]],
                )
            )
        )
        for _ in range(n_impostors):
            impostor = list(node)
            # set fake target
            impostor[query_prop] = fake_target
            # change any other property to a value different from "node"
            other_prop = random.choice(
                [i for i in range(len(vocab.properties[:-1])) if i != query_prop]
            )
            impostor[other_prop] = random.choice(
                [
                    attr
                    for attr in vocab.grouped_attributes[vocab.properties[other_prop]]
                    if attr != node[other_prop]
                ]
            )
            impostors.append(tuple(impostor))

        # make a question that selects "node" and queries for the property, kinda hardcoded
        # since I know how the programs look like
        question = [
            f"query_{vocab.properties[query_prop]}",
            "unique",
            *[
                s
                for t in [
                    (f"filter_{vocab.properties[i]}", node[i])
                    for i in [
                        prop
                        for prop in range(len(vocab.properties) - 1)
                        if prop != query_prop
                    ]
                ]
                for s in t
            ],
            "scene",
        ]
        return (
            # single edge so embedding doesn't break
            Graph(
                node_attrs=[node] + impostors,
                edge_indices=[(0, 1)],
                edge_attrs=["left"],
            ),
            question,
            node[query_prop],
        )


def is_gud_for_nsm(question):
    return question["question_family_index"] in ALL_EASY_CATS


class ClevrGloveDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datadir: str,
        batch_size: int,
        glove_dim: int,
        question_type: tp.Literal["program", "question"],
        prop_embed_method: tp.Literal["embed", "mean", "sum"],
        prop_embed_scale: float,
        nhops: tp.Optional[tp.List[int]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))
        self.datadir = datadir
        self.batch_size = batch_size
        self.glove_dim = glove_dim
        self.nhops = nhops
        self.question_type = question_type
        self.prop_embed_method = prop_embed_method
        self.prop_embed_scale = prop_embed_scale

    def prepare_data(self):
        ClevrWInstructions.download(self.datadir)

    def setup(self, stage):
        if stage in ("fit", "validate", None):
            self.clevr_val = ClevrGlove(
                self.datadir,
                split="val",
                glove_dim=self.glove_dim,
                nhops=self.nhops,
                question_type=self.question_type,
                prop_embed_method=self.prop_embed_method,
                prop_embed_scale=self.prop_embed_scale,
            )
        if stage in ("fit", None):
            self.clevr_train = ClevrGlove(
                self.datadir,
                split="train",
                glove_dim=self.glove_dim,
                nhops=self.nhops,
                question_type=self.question_type,
                prop_embed_method=self.prop_embed_method,
                prop_embed_scale=self.prop_embed_scale,
            )

    def _get_dataloader(self, split: tp.Literal["train", "val"]):
        dataset = getattr(self, f"clevr_{split}")
        vocab = dataset.vocab

        def collate_fn(batch):
            graphs, questions, targets = zip(*batch)

            # return a 6-tuple, thats what the last iteration of
            # NSMLightningModule expects
            return (
                collate_graphs(graphs),
                torch.nn.utils.rnn.pack_sequence(questions, enforce_sorted=False),
                vocab.concept_embeddings,
                vocab.property_embeddings,
                torch.tensor(targets),
                # dummy tensor, gold_instructions not used
                torch.empty(1),
            )

        return data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=split == "train",
            collate_fn=collate_fn,
            num_workers=os.cpu_count(),
        )

    train_dataloader = partialmethod(_get_dataloader, "train")
    val_dataloader = partialmethod(_get_dataloader, "val")


class ClevrWInstructionsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datadir: str,
        batch_size: int,
        prop_embeds_const: float,
        nhops: tp.Optional[tp.List[int]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))
        self.datadir = datadir
        self.batch_size = batch_size
        self.nhops = nhops
        self.prop_embeds_const = prop_embeds_const

    def prepare_data(self):
        ClevrWInstructions.download(self.datadir)

    def setup(self, stage):
        if stage in ("fit", "validate", None):
            self.clevr_val = ClevrWInstructions(
                self.datadir,
                split="val",
                nhops=self.nhops,
                prop_embeds_const=self.prop_embeds_const,
            )
        if stage in ("fit", None):
            self.clevr_train = ClevrWInstructions(
                self.datadir,
                split="train",
                nhops=self.nhops,
                prop_embeds_const=self.prop_embeds_const,
            )

    def _get_dataloader(self, split: tp.Literal["train", "val"]):
        dataset = getattr(self, f"clevr_{split}")
        vocab = dataset.vocab

        def collate_fn(batch):
            graphs, questions, targets, instructionss = zip(*batch)
            return (
                collate_graphs(graphs),
                torch.nn.utils.rnn.pack_sequence(questions, enforce_sorted=False),
                vocab.concept_embeddings,
                vocab.property_embeddings,
                torch.tensor(targets),
                torch.stack(instructionss),
            )

        return data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=split == "train",
            collate_fn=collate_fn,
            num_workers=os.cpu_count(),
        )

    train_dataloader = partialmethod(_get_dataloader, "train")
    val_dataloader = partialmethod(_get_dataloader, "val")


class ClevrDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datadir: tp.Union[str, Path],
        batch_size: int,
        cats: tp.List[int],
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))
        self.datadir = datadir
        self.batch_size = batch_size
        assert all(c in ALL_CATS for c in cats)
        self.cats = cats

    def prepare_data(self):
        ClevrNoImagesDataset.download(self.datadir)

    def setup(self, stage: tp.Optional[str] = None):
        def filter_fn(question):
            return question["question_family_index"] in self.cats

        if stage in ("fit", "validate", None):
            self.clevr_val = ClevrNoImagesDataset(
                self.datadir,
                split="val",
                filter_fn=filter_fn,
            )
        if stage in ("fit", None):
            self.clevr_train = ClevrNoImagesDataset(
                self.datadir,
                split="train",
                filter_fn=filter_fn,
            )

    def _get_dataloader(self, split: str):
        dataset = getattr(self, f"clevr_{split}")
        vocab = dataset.vocab

        def collate(batch):
            graphs, questions, targets = collate_nsmitems(batch)
            return (
                graphs,
                questions,
                vocab.concept_embeddings,
                vocab.property_embeddings,
                targets,
            )

        return data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=split == "train",
            collate_fn=collate,
            num_workers=os.cpu_count() or 0,
        )

    def train_dataloader(self):
        return self._get_dataloader("train")

    def val_dataloader(self):
        return self._get_dataloader("val")
