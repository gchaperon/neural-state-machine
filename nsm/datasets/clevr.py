import random
import os
import logging
import torch
import torch.utils.data as data
import pytorch_lightning as pl
from pathlib import Path
import typing as tp
import zipfile
import json
from . import utils as data_utils
from operator import itemgetter, eq, ne
from functools import partial, singledispatchmethod, cached_property, lru_cache
from nsm.utils import Graph, NSMItem, collate_nsmitems
import collections.abc as abc


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


class Vocab:
    METADATA_URL = (
        "https://raw.githubusercontent.com/facebookresearch"
        "/clevr-dataset-gen/master/question_generation/metadata.json"
    )
    # Some functions defined in metdata.json where used only for the generation process
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
        out = torch.zeros(len(self.properties), self.embed_size)
        lens = [len(self.grouped_attributes[prop]) for prop in self.properties]
        start = 0
        for i, len_ in enumerate(lens):
            out[i, start : start + len_] = 10 ** 6
            start += len_
        return out

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
    def _(self, tokens: abc.Sequence) -> torch.Tensor:
        return torch.stack([self.embed(token) for token in tokens])

    @embed.register
    def _(self, graph: Graph) -> Graph:
        node_attrs, edge_indices, edge_attrs = graph
        return Graph(
            self.embed(node_attrs), torch.tensor(edge_indices).T, self.embed(edge_attrs)
        )


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
        question = self.questions[key]
        scene = self.scenes[question["image_index"]]
        return (
            self.vocab.embed(scene_to_graph(scene, self.vocab)),
            self.vocab.embed(program_to_polish(question["program"])),
            self.vocab.answers.index(question["answer"]),
        )

    def get_raw(self, key: int):
        question = self.questions[key]
        scene = self.scenes[question["image_index"]]
        return (
            scene_to_graph(scene, self.vocab),
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
            logger.info(f"Done!")

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


class ClevrNoImagesWInstructionsDataset(ClevrNoImagesDataset):
    """Change the way questions are embedded, generate instructions directly"""

    def __init__(self, datadir, split, download=False, filter_fn=None):
        # Ignore filter_fn argument to keep compat
        # This only makes sense for "hop" questions, hence the filter_fn
        super().__init__(datadir, split, download, filter_fn=is_gud_for_nsm)

    def __getitem__(self, key):
        graph, question, answer = self.get_raw(key)
        return (
            self.vocab.embed(graph),
            self.instructions_from_question(question),
            self.vocab.answers.index(answer),
        )

    def instructions_from_question(self, question):
        # set number of instructions to 5, pad the shorter (5 is the max in the dataset)
        # also pad first
        n_ins = 5
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
        prop = question[0].split("_")[1]
        instructions.append(vocab.property_embeddings[vocab.properties.index(prop)])
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
                random.choice(vocab.grouped_attributes[prop]) for prop in vocab.properties[:-1]
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


class ClevrNoImagesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: tp.Union[str, Path],
        batch_size: int,
        num_workers: int = 4,
        w_instructions: bool = False,
        nhops: tp.Optional[tp.List[int]] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.w_instructions = w_instructions
        if w_instructions:
            assert not nhops, "w_instructions is incompatible with nhops"
        self.nhops = nhops or []
        assert all(n in range(0, 4) for n in self.nhops), "nhops range is [0..3]"

    def prepare_data(self):
        ClevrNoImagesDataset.download(self.data_dir)

    def setup(self, stage: tp.Optional[str] = None):
        dataset_cls = (
            ClevrNoImagesWInstructionsDataset
            if self.w_instructions
            else ClevrNoImagesDataset
        )
        filter_cats = (
            [cat for nhops in self.nhops for cat in NHOPS_TO_CATS[nhops]]
            if self.nhops
            else ALL_EASY_CATS
        )

        def filter_fn(question):
            return question["question_family_index"] in filter_cats

        if stage in ("fit", "validate", None):
            self.clevr_val = dataset_cls(
                self.data_dir, split="val", filter_fn=filter_fn
            )
        if stage in ("fit", None):
            self.clevr_train = dataset_cls(
                self.data_dir, split="train", filter_fn=filter_fn
            )

    def _get_dataloader(self, split):
        vocab = getattr(self, "clevr_train", self.clevr_val).vocab

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
            split,
            batch_size=self.batch_size,
            collate_fn=collate,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self._get_dataloader(self.clevr_train)

    def val_dataloader(self):
        return self._get_dataloader(self.clevr_val)
