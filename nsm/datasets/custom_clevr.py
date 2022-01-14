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
        function: str
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
    questions: tp.List[Question],
    scenes: tp.Dict[int, Scene],
    allowed_counts: tp.List[int],
) -> tp.Tuple[tp.List[Question], tp.Dict[int, Scene]]:
    questions = (
        q
        for q in questions
        if type(q["answer"]) == int and q["answer"] in allowed_counts
    )
    grouped = [[] for _ in range(len(allowed_counts))]

    for q in questions:
        answer = q["answer"]
        grouped[allowed_counts.index(answer)].append(q)

    max_sample = min(len(g) for g in grouped)
    balanced_questions = list(
        itertools.chain.from_iterable(group[:max_sample] for group in grouped)
    )
    filtered_scenes_indices = {q["image_index"] for q in balanced_questions}
    filtered_scenes = {
        key: s for key, s in scenes.items() if key in filtered_scenes_indices
    }

    # convert answers to string instead of int
    for q in balanced_questions:
        q["answer"] = str(q["answer"])
    return balanced_questions, filtered_scenes


def process_exist_qs(
    questions: tp.List[Question], scenes: tp.Dict[int, Scene]
) -> tp.Tuple[tp.List[Question], tp.Dict[int, Scene]]:
    translate = {True: "yes", False: "no"}

    for q in questions:
        q["answer"] = translate[q["answer"]]

    return questions, scenes


AND_CATS = ["count", "query_size", "query_color", "query_material", "query_shape"]


def process_and_qs(
    questions: tp.List[Question],
    scenes: tp.Dict[int, Scene],
    cats: tp.List[str],
) -> tp.Tuple[tp.List[Question], tp.Dict[int, Scene]]:
    assert all(cat in AND_CATS for cat in cats), "you are mixing cats dude"
    cat_indices = [i for i, cat in enumerate(AND_CATS) if cat in cats]
    processed = []
    for q in questions:
        if q["question_family_index"] in cat_indices:
            q["answer"] = str(q["answer"])
            processed.append(q)

    return processed, scenes


PostProcessFn = tp.Callable[
    [tp.List[Question], tp.Dict[int, Scene]],
    tp.Tuple[tp.List[Question], tp.Dict[int, Scene]],
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
        postprocess_fn: tp.Optional[PostProcessFn] = None,
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
            question["answer"],
        )

    def __len__(self):
        return len(self.questions)


class CustomClevr(CustomClevrBase):
    def __getitem__(self, key):
        graph, polish, answer = super().__getitem__(key)
        vocab = self.vocab
        return vocab.embed(graph), vocab.embed(polish), vocab.answers.index(answer)


class SameRelateClevrDataModule(pl.LightningDataModule):
    def __init__(self, datadir: str, batch_size: int, subset_ratio: float = 1.0):
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))
        self.datadir = datadir
        self.batch_size = batch_size
        self.subset_ratio = subset_ratio

    def setup(self, stage=None):
        questions_base = pathlib.Path(self.datadir) / "custom-clevr" / "same-relate"
        scenes_base = pathlib.Path(self.datadir) / "clevr" / "CLEVR_v1.0" / "scenes"
        metadata_path = pathlib.Path(self.datadir) / "clevr" / "metadata.json"

        if stage in ("fit", "validate", None):
            dataset = CustomClevr(
                questions_base / "same_relate_query_val_questions.json",
                scenes_base / "CLEVR_val_scenes.json",
                metadata_path,
            )
            val_indices = range(int(self.subset_ratio * len(dataset)))
            self.val_split = torch.utils.data.Subset(dataset, val_indices)
        if stage in ("fit", None):
            dataset = CustomClevr(
                questions_base / "same_relate_query_train_questions.json",
                scenes_base / "CLEVR_train_scenes.json",
                metadata_path,
            )
            train_indices = range(int(self.subset_ratio * len(dataset)))
            self.train_split = torch.utils.data.Subset(dataset, train_indices)

    def _get_dataloader(self, split):
        dataset = getattr(self, f"{split}_split")
        # dataset is now a torch.utils.data.Subset
        vocab = dataset.dataset.vocab

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
            num_workers=os.cpu_count() if split == "train" else 0,
        )

    train_dataloader = functools.partialmethod(_get_dataloader, "train")
    val_dataloader = functools.partialmethod(_get_dataloader, "val")


class SingleAndClevrDataModule(pl.LightningDataModule):
    def __init__(
        self, datadir: str, batch_size: int, cats, subset_ratio: float = 0.5
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))
        self.datadir = datadir
        self.batch_size = batch_size
        self.cats = cats
        self.subset_ratio = subset_ratio

    def setup(self, stage=None):
        questions_base = pathlib.Path(self.datadir) / "custom-clevr" / "single-and"
        scenes_base = pathlib.Path(self.datadir) / "clevr" / "CLEVR_v1.0" / "scenes"
        metadata_path = pathlib.Path(self.datadir) / "clevr" / "metadata.json"
        postprocess_fn = functools.partial(process_and_qs, cats=self.cats)

        if stage in ("fit", "validate", None):
            print("Loading val dataset")
            dataset = CustomClevr(
                questions_base / "single_and_val_questions.json",
                scenes_base / "CLEVR_val_scenes.json",
                metadata_path,
                postprocess_fn=postprocess_fn,
            )

            val_indices = range(int(self.subset_ratio * len(dataset)))
            self.val_split = torch.utils.data.Subset(dataset, val_indices)
        if stage in ("fit", None):
            print("Loading train dataset")
            dataset = CustomClevr(
                questions_base / "single_and_train_questions.json",
                scenes_base / "CLEVR_train_scenes.json",
                metadata_path,
                postprocess_fn=postprocess_fn,
            )
            train_indices = range(int(self.subset_ratio * len(dataset)))
            self.train_split = torch.utils.data.Subset(dataset, train_indices)

    def _get_dataloader(self, split):
        dataset = getattr(self, f"{split}_split")
        # dataset is now a torch.utils.data.Subset
        vocab = dataset.dataset.vocab

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
            num_workers=os.cpu_count() if split == "train" else 0,
        )

    train_dataloader = functools.partialmethod(_get_dataloader, "train")
    val_dataloader = functools.partialmethod(_get_dataloader, "val")


class ExistClevrDataModule(pl.LightningDataModule):
    def __init__(self, datadir: str, batch_size: int) -> None:
        super().__init__()
        self.save_hyperparameters("batch_size")
        self.datadir = datadir
        self.batch_size = batch_size

    def setup(self, stage):
        questions_base = pathlib.Path(self.datadir) / "custom-clevr" / "exist-only"
        scenes_base = pathlib.Path(self.datadir) / "clevr" / "CLEVR_v1.0" / "scenes"
        metadata_path = pathlib.Path(self.datadir) / "clevr" / "metadata.json"

        if stage in ("fit", "validate", None):
            self.val_split = CustomClevr(
                questions_base / "exist_only_val_questions.json",
                scenes_base / "CLEVR_val_scenes.json",
                metadata_path,
                postprocess_fn=process_exist_qs,
            )
        if stage in ("fit", None):
            self.train_split = CustomClevr(
                questions_base / "exist_only_train_questions.json",
                scenes_base / "CLEVR_train_scenes.json",
                metadata_path,
                postprocess_fn=process_exist_qs,
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
            num_workers=os.cpu_count() if split == "train" else 0,
        )

    train_dataloader = functools.partialmethod(_get_dataloader, "train")
    val_dataloader = functools.partialmethod(_get_dataloader, "val")


class BalancedCountsClevrDataModule(pl.LightningDataModule):
    def __init__(self, datadir: str, batch_size: int) -> None:
        super().__init__()
        self.save_hyperparameters("batch_size")
        self.datadir = datadir
        self.batch_size = batch_size

    def setup(self, stage):
        questions_base = pathlib.Path(self.datadir) / "custom-clevr" / "count-only"
        scenes_base = pathlib.Path(self.datadir) / "clevr" / "CLEVR_v1.0" / "scenes"
        metadata_path = pathlib.Path(self.datadir) / "clevr" / "metadata.json"

        if stage in ("fit", "validate", None):
            self.val_split = CustomClevr(
                questions_base / "count_only_val_questions.json",
                scenes_base / "CLEVR_val_scenes.json",
                metadata_path,
                postprocess_fn=functools.partial(
                    balance_counts, allowed_counts=list(range(0, 6))
                ),
            )
        if stage in ("fit", None):
            self.train_split = CustomClevr(
                questions_base / "count_only_train_questions.json",
                scenes_base / "CLEVR_train_scenes.json",
                metadata_path,
                postprocess_fn=functools.partial(
                    balance_counts, allowed_counts=list(range(0, 6))
                ),
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


class GeneralizeCountsClevrDataModule(pl.LightningDataModule):
    def __init__(self, datadir: str, batch_size: int) -> None:
        super().__init__()
        self.save_hyperparameters("batch_size")
        self.datadir = datadir
        self.batch_size = batch_size

    def setup(self, stage):
        questions_path = (
            pathlib.Path(self.datadir)
            / "custom-clevr"
            / "count-only"
            / "count_only_train_questions.json"
        )
        scenes_path = (
            pathlib.Path(self.datadir)
            / "clevr"
            / "CLEVR_v1.0"
            / "scenes"
            / "CLEVR_train_scenes.json"
        )
        metadata_path = pathlib.Path(self.datadir) / "clevr" / "metadata.json"

        if stage in ("fit", "validate", None):
            self.val_split = CustomClevr(
                questions_path,
                scenes_path,
                metadata_path,
                postprocess_fn=functools.partial(
                    balance_counts, allowed_counts=list(range(5, 11))
                ),
            )
        if stage in ("fit", None):
            self.train_split = CustomClevr(
                questions_path,
                scenes_path,
                metadata_path,
                postprocess_fn=functools.partial(
                    balance_counts, allowed_counts=list(range(0, 5))
                ),
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
