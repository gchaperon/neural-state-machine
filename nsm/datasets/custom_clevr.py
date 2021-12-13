import json
import torch
import typing as tp
import pathlib
import os
import itertools
import functools
import random
import nsm.datasets.clevr as og_clevr
import pytorch_lightning as pl

import nsm.utils as utils


PathLike = tp.Union[str, bytes, os.PathLike]


class Question(tp.TypedDict):
    class Node(tp.TypedDict):
        type: str
        inputs: tp.List[int]
        value_inputs: tp.List[str]

    image_index: int
    question: str
    program: tp.List[Node]
    answer: tp.Union[str, bool, int]
    template_filename: str
    question_family_index: int
    question_index: int


class Scene(tp.TypedDict):
    class Object(tp.TypedDict):
        color: str
        size: str
        shape: str
        material: str

    class Relationships(tp.TypedDict):
        right: tp.List[tp.List[int]]
        behind: tp.List[tp.List[int]]
        front: tp.List[tp.List[int]]
        left: tp.List[tp.List[int]]

    image_index: int
    objects: tp.List[Object]
    relationships: Relationships


def balance_counts(
    questions: tp.List[Question], scenes: tp.Dict[int, Scene]
) -> tp.Tuple[tp.List[Question], tp.Dict[int, Scene]]:
    MAX_COUNT = 4
    questions = (
        q for q in questions if type(q["answer"]) == int and q["answer"] <= MAX_COUNT
    )
    grouped = [[] for _ in range(MAX_COUNT + 1)]

    for q in questions:
        answer = q["answer"]
        grouped[answer].append(q)

    max_sample = min(len(g) for g in grouped)
    balanced_questions = list(
        itertools.chain.from_iterable(group[:max_sample] for group in grouped)
    )
    filtered_scenes_indices = {q["image_index"] for q in balanced_questions}
    filtered_scenes = {
        key: s for key, s in scenes.items() if key in filtered_scenes_indices
    }
    return balanced_questions, filtered_scenes


PostProcessFn = tp.Optional[
    tp.Callable[
        [tp.List[Question], tp.Dict[int, Scene]],
        tp.Tuple[tp.List[Question], tp.Dict[int, Scene]],
    ]
]


class CustomClevrBase(torch.utils.data.Dataset):
    questions = tp.List[Question]
    scenes = tp.Dict[int, Scene]
    vocab: og_clevr.Vocab

    def __init__(
        self,
        questions_path: PathLike,
        scenes_path: PathLike,
        metadata_path: PathLike,
        postprocess_fn: PostProcessFn = None,
    ):
        self.vocab = og_clevr.Vocab(metadata_path, prop_embed_const=1.0)
        self.questions_path = questions_path
        self.scenes_path = scenes_path
        with open(questions_path) as q_file:
            questions = json.load(q_file)["questions"]
            for q in questions:
                for n in q["program"]:
                    n["function"] = n["type"]
                    del n["type"]
        with open(scenes_path) as s_file:
            scenes = {s["image_index"]: s for s in json.load(s_file)["scenes"]}

        postprocess_fn = postprocess_fn or (
            lambda *args: args
        )  # identity, does nothing
        self.questions, self.scenes = postprocess_fn(questions, scenes)

    def __getitem__(self, key: int):
        question = self.questions[key]
        scene = self.scenes[question["image_index"]]
        return (
            og_clevr.scene_to_graph(scene, self.vocab),
            og_clevr.program_to_polish(question["program"]),
            # esto es medio hacky, quiza el str() deberia estar en otro lado, postprocess_fn?
            str(question["answer"]),
        )

    def __len__(self):
        return len(self.questions)


class CustomClevr(CustomClevrBase):
    def __getitem__(self, key):
        graph, polish, answer = super().__getitem__(key)
        vocab = self.vocab
        return vocab.embed(graph), vocab.embed(polish), vocab.answers.index(answer)


class CustomClevrDataModule(pl.LightningDataModule):
    def __init__(
        self,
        questions_template: str,
        scenes_template: str,
        metadata_path: PathLike,
        postprocess_fn_name: str,
        batch_size: int,
    ):
        super().__init__()
        self.save_hyperparameters(
            "questions_template", "postprocess_fn_name", "batch_size"
        )
        self.questions_template = questions_template
        self.scenes_template = scenes_template
        self.metadata_path = metadata_path
        self.postprocess_fn = globals()[postprocess_fn_name]
        self.batch_size = batch_size

    def setup(self, stage):
        if stage in ("fit", "validate", None):
            self.val_split = CustomClevr(
                self.questions_template.format("val"),
                self.scenes_template.format("val"),
                self.metadata_path,
                balance_counts,
            )
        if stage in ("fit", None):
            self.train_split = CustomClevr(
                self.questions_template.format("train"),
                self.scenes_template.format("train"),
                self.metadata_path,
                balance_counts,
            )

    def _get_dataloader(self, split):
        dataset = getattr(self, f"{split}_split")
        vocab = dataset.vocab

        def collate_fn(batch):
            graphs, questions, targets = utils.collate_nsmitems(batch)
            return (
                graphs,
                questions,
                vocab.concept_embeddings,
                vocab.property_embeddings,
                targets,
                torch.empty(1),
            )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=split == "train",
            collate_fn=collate_fn,
            num_workers=os.cpu_count(),
        )

    train_dataloader = functools.partialmethod(_get_dataloader, "train")
    val_dataloader = functools.partialmethod(_get_dataloader, "val")
