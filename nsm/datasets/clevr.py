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
from operator import itemgetter, eq
from functools import partial, singledispatchmethod, cached_property, lru_cache
from nsm.utils import Graph, NSMItem, collate_nsmitems
import collections.abc as abc


import pydantic


logger = logging.getLogger(__name__)

# Determined via manual inspection of templates in
# https://github.com/facebookresearch/clevr-dataset-gen
# and using the info mentioned in this comment
# https://github.com/facebookresearch/clevr-dataset-gen/issues/14#issuecomment-482896597
THESE_TEMPLATES_SHOULD_BE_EASY_FOR_THE_NSM = [
    *range(27, 31),
    *range(74, 78),
    *range(80, 84),
    *range(86, 90),
]


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
        ]
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
        # if no filter function, filter empty dicst
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

    def __getitem__(self, key: int) -> NSMItem:
        question = self.questions[key]
        scene = self.scenes[question["image_index"]]
        return (
            self.vocab.embed(scene_to_graph(scene, self.vocab)),
            self.vocab.embed(program_to_polish(question["program"])),
            self.vocab.answers.index(question["answer"]),
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


def is_gud_for_nsm(question):
    return (
        question["question_family_index"] in THESE_TEMPLATES_SHOULD_BE_EASY_FOR_THE_NSM
    )


class ClevrNoImagesDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: tp.Union[str, Path], batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        ClevrNoImagesDataset.download(self.data_dir)

    def setup(self, stage: tp.Optional[str] = None):
        self.clevr_train = ClevrNoImagesDataset(
            self.data_dir, split="train", filter_fn=is_gud_for_nsm
        )
        self.clevr_val = ClevrNoImagesDataset(
            self.data_dir, split="val", filter_fn=is_gud_for_nsm
        )

    def _get_dataloader(self, split):
        vocab = self.clevr_train.vocab

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
            num_workers=os.cpu_count(),
        )

    def train_dataloader(self):
        return self._get_dataloader(self.clevr_train)

    def val_dataloader(self):
        return self._get_dataloader(self.clevr_val)
