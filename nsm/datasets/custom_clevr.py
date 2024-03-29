import json
import typing as tp
import pathlib
import os
import itertools
import functools
import collections

import torch
import pytorch_lightning as pl

import nsm.utils as utils
import nsm.datasets.clevr as og_clevr


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


class Subset(torch.utils.data.Subset):
    def __getattr__(self, name):
        return getattr(self.dataset, name)


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


def process_comparison_qs(
    questions: tp.List[Question],
    scenes: tp.Dict[int, Scene],
) -> tp.Tuple[tp.List[Question], tp.Dict[int, Scene]]:

    _answer_map = {True: "yes", False: "no"}

    for q in questions:
        q["answer"] = _answer_map[q["answer"]]
    return questions, scenes


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
        self.vocab = og_clevr.Vocab(metadata_path)
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


class ComparisonClevrDataModule(pl.LightningDataModule):
    def __init__(self, datadir: str, batch_size: int):
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))
        self.datadir = datadir
        self.batch_size = batch_size

    def setup(self, stage=None):
        datadir = pathlib.Path(self.datadir)
        questions_base = datadir / "custom-clevr" / "comparison"
        scenes_base = datadir / "clevr" / "CLEVR_v1.0" / "scenes"
        metadata_path = datadir / "clevr" / "metadata.json"

        if stage in ("fit", "validate", "test", None):
            print("Loading comparison val split")
            dataset = CustomClevr(
                questions_base / "comparison_val_questions.json",
                scenes_base / "CLEVR_val_scenes.json",
                metadata_path,
                postprocess_fn=process_comparison_qs,
            )
            self.val_split = dataset

        if stage in ("fit", None):
            print("Loading comparison train split")
            dataset = CustomClevr(
                questions_base / "comparison_train_questions.json",
                scenes_base / "CLEVR_train_scenes.json",
                metadata_path,
                postprocess_fn=process_comparison_qs,
            )
            self.train_split = dataset

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

    def test_dataloader(self):
        # mapping of question_family_index to (nhops, equal_*)
        # here nhops means the number of hops used in the question
        # these questions have 2 "branches", and the final operation requires
        # checking some property at the end of each branch.
        # Some questions require a hop within one of the branches, a hop in each branch
        # or the branches simply don't have hops
        # so 0 would be no hops, 1 means a hop in one of the two branches and 2 means
        # both branches have hops

        cat_to_key = {
            0: (0, "equal_size"),
            1: (0, "equal_color"),
            2: (0, "equal_material"),
            3: (0, "equal_shape"),
            4: (1, "equal_size"),
            5: (1, "equal_size"),
            6: (2, "equal_size"),
            7: (1, "equal_color"),
            8: (1, "equal_color"),
            9: (2, "equal_color"),
            10: (1, "equal_material"),
            11: (1, "equal_material"),
            12: (2, "equal_material"),
            13: (1, "equal_shape"),
            14: (1, "equal_shape"),
            15: (2, "equal_shape"),
        }
        # note that are only 12 unique tuples in the left side
        key_to_indices = {key: [] for key in cat_to_key.values()}

        dataset = self.val_split

        for i, question in enumerate(dataset.questions):
            key_to_indices[cat_to_key[question["question_family_index"]]].append(i)

        vocab = dataset.vocab

        def collate_fn(batch):
            graphs, questions, targets = utils.collate_nsmitems(batch)
            return (
                graphs,
                questions,
                vocab.concept_embeddings,
                vocab.property_embeddings,
                targets,
            )

        return [
            torch.utils.data.DataLoader(
                Subset(dataset, indices),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=os.cpu_count(),
            )
            for indices in key_to_indices.values()
        ]


class SameRelateClevrDataModule(pl.LightningDataModule):
    def __init__(self, datadir: str, batch_size: int):
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))
        self.datadir = datadir
        self.batch_size = batch_size

    def setup(self, stage=None):
        questions_base = pathlib.Path(self.datadir) / "custom-clevr" / "same-relate"
        scenes_base = pathlib.Path(self.datadir) / "clevr" / "CLEVR_v1.0" / "scenes"
        metadata_path = pathlib.Path(self.datadir) / "clevr" / "metadata.json"

        if stage in ("fit", "validate", "test", None):
            print("Loading same relate val dset")
            dataset = CustomClevr(
                questions_base / "same_relate_val_questions.json",
                scenes_base / "CLEVR_val_scenes.json",
                metadata_path,
            )
            self.val_split = dataset

        if stage in ("fit", None):
            print("Loading same relate train dset")
            dataset = CustomClevr(
                questions_base / "same_relate_train_questions.json",
                scenes_base / "CLEVR_train_scenes.json",
                metadata_path,
            )
            self.train_split = dataset

    def _get_dataloader(self, split):
        dataset = getattr(self, f"{split}_split")
        # dataset is now a torch.utils.data.Subset
        vocab = dataset.vocab

        def collate_fn(batch):
            graphs, questions, targets = utils.collate_nsmitems(batch)
            return (
                graphs,
                questions,
                vocab.concept_embeddings,
                vocab.property_embeddings,
                targets,
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

    def test_dataloader(self):
        # mapping between question_family_index and (relate_*, query_*)
        # 0: (same_size, query_color)
        # 1: (same_size, query_material)
        # 2: (same_size, query_shape)
        # 3: (same_color, query_color)
        # 4: (same_color, query_material)
        # 5: (same_color, query_shape)
        # 6: (same_material, query_size)
        # 7: (same_material, query_color)
        # 8: (same_material, query_shape)
        # 9: (same_shape, query_size)
        # 10: (same_shape, query_color)
        # 11: (same_shape, query_material)

        dataset = self.val_split
        vocab = dataset.vocab

        index_groups = [[] for _ in range(12)]
        for i, q in enumerate(dataset.questions):
            index_groups[q["question_family_index"]].append(i)

        def collate_fn(batch):
            graphs, questions, targets = utils.collate_nsmitems(batch)
            return (
                graphs,
                questions,
                vocab.concept_embeddings,
                vocab.property_embeddings,
                targets,
            )

        return [
            torch.utils.data.DataLoader(
                Subset(dataset, indices),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=os.cpu_count(),
            )
            for indices in index_groups
        ]


class SingleAndClevrDataModule(pl.LightningDataModule):
    def __init__(self, datadir: str, batch_size: int) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))
        self.datadir = datadir
        self.batch_size = batch_size

    def setup(self, stage=None):
        questions_base = pathlib.Path(self.datadir) / "custom-clevr" / "single-and"
        scenes_base = pathlib.Path(self.datadir) / "clevr" / "CLEVR_v1.0" / "scenes"
        metadata_path = pathlib.Path(self.datadir) / "clevr" / "metadata.json"

        if stage in ("fit", "validate", "test", None):
            print("Loading val dataset")
            dataset = CustomClevr(
                questions_base / "single_and_val_questions.json",
                scenes_base / "CLEVR_val_scenes.json",
                metadata_path,
            )

            self.val_split = dataset
        if stage in ("fit", None):
            print("Loading train dataset")
            dataset = CustomClevr(
                questions_base / "single_and_train_questions.json",
                scenes_base / "CLEVR_train_scenes.json",
                metadata_path,
            )
            self.train_split = dataset

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

    def test_dataloader(self):
        # NOTE: mapping of "question_family_index" to query_*
        # 0: query_size
        # 1: query_color
        # 2: query_material
        # 3: query_shape
        dataset = self.val_split
        vocab = dataset.vocab

        index_groups = [[] for _ in range(4)]
        for i, q in enumerate(dataset.questions):
            index_groups[q["question_family_index"]].append(i)

        def collate_fn(batch):
            graphs, questions, targets = utils.collate_nsmitems(batch)
            return (
                graphs,
                questions,
                vocab.concept_embeddings,
                vocab.property_embeddings,
                targets,
            )

        return [
            torch.utils.data.DataLoader(
                Subset(dataset, indices),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=os.cpu_count(),
            )
            for indices in index_groups
        ]


class ExistClevrDataModule(pl.LightningDataModule):
    def __init__(
        self, datadir: str, batch_size: int, train_nexamples: int = 10**5
    ) -> None:
        super().__init__()
        self.save_hyperparameters("batch_size", "train_nexamples")
        self.datadir = datadir
        self.batch_size = batch_size
        self.train_nexamples = train_nexamples

    def setup(self, stage):
        questions_base = pathlib.Path(self.datadir) / "custom-clevr" / "exist-only"
        scenes_base = pathlib.Path(self.datadir) / "clevr" / "CLEVR_v1.0" / "scenes"
        metadata_path = pathlib.Path(self.datadir) / "clevr" / "metadata.json"

        if stage in ("fit", "validate", "test", None):
            self.val_split = CustomClevr(
                questions_base / "exist_val_questions.json",
                scenes_base / "CLEVR_val_scenes.json",
                metadata_path,
                postprocess_fn=process_exist_qs,
            )
        if stage in ("fit", None):
            dataset = CustomClevr(
                questions_base / "exist_train_questions.json",
                scenes_base / "CLEVR_train_scenes.json",
                metadata_path,
                postprocess_fn=process_exist_qs,
            )
            self.train_split = Subset(
                dataset,
                torch.multinomial(
                    torch.ones(len(dataset)), num_samples=self.train_nexamples
                ),
            )

    def train_dataloader(self):
        def collate_fn(batch):
            graphs, questions, targets = utils.collate_nsmitems(batch)
            return (
                graphs,
                questions,
                self.train_split.vocab.concept_embeddings,
                self.train_split.vocab.property_embeddings,
                targets,
            )

        return torch.utils.data.DataLoader(
            self.train_split,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        def collate_fn(batch):
            graphs, questions, targets = utils.collate_nsmitems(batch)
            return (
                graphs,
                questions,
                self.val_split.vocab.concept_embeddings,
                self.val_split.vocab.property_embeddings,
                targets,
            )

        return torch.utils.data.DataLoader(
            Subset(
                self.val_split,
                torch.multinomial(
                    torch.ones(len(self.val_split)),
                    num_samples=int(0.1 * self.train_nexamples),
                ),
            ),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=os.cpu_count(),
        )

    def test_dataloader(self):
        # NOTE: on "question_family_index"
        # 0 is for 0 nhops
        # 1 is for 1 nhops
        # 2 is for 2 nhops
        # 3 is for 3 nhops
        # This sounds super dumb but it is not guaranteed :-/

        subset_indices = [[] for _ in range(4)]
        for i, q in enumerate(self.val_split.questions):
            subset_indices[q["question_family_index"]].append(i)

        def collate_fn(batch):
            graphs, questions, targets = utils.collate_nsmitems(batch)
            return (
                graphs,
                questions,
                self.val_split.vocab.concept_embeddings,
                self.val_split.vocab.property_embeddings,
                targets,
            )

        return [
            torch.utils.data.DataLoader(
                Subset(self.val_split, indices),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=os.cpu_count(),
            )
            for indices in subset_indices
        ]


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

        if stage in ("fit", "validate", "test", None):
            self.val_split = CustomClevr(
                questions_base / "count_val_questions.json",
                scenes_base / "CLEVR_val_scenes.json",
                metadata_path,
                postprocess_fn=functools.partial(
                    balance_counts, allowed_counts=list(range(0, 6))
                ),
            )
        if stage in ("fit", None):
            self.train_split = CustomClevr(
                questions_base / "count_train_questions.json",
                scenes_base / "CLEVR_train_scenes.json",
                metadata_path,
                postprocess_fn=functools.partial(
                    balance_counts, allowed_counts=list(range(0, 6))
                ),
            )

    def train_dataloader(self):
        dataset = self.train_split
        vocab = dataset.vocab

        def collate_fn(batch):
            graphs, questions, targets = utils.collate_nsmitems(batch)
            return (
                graphs,
                questions,
                vocab.concept_embeddings,
                vocab.property_embeddings,
                targets,
            )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        dataset = self.val_split
        vocab = dataset.vocab

        def collate_fn(batch):
            graphs, questions, targets = utils.collate_nsmitems(batch)
            return (
                graphs,
                questions,
                vocab.concept_embeddings,
                vocab.property_embeddings,
                targets,
            )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=os.cpu_count(),
        )

    def test_dataloader(self):
        # NOTE: mapping from "question_family_index" to nhops
        # 0: 0
        # 1: 1
        # 2: 2
        # 3: 3
        # This sounds dumb but it wasn't guaranteed

        dataset = self.val_split
        vocab = dataset.vocab

        counts = map(str, range(6))
        nhops = range(4)

        grouped_indices = collections.defaultdict(list)
        for i, question in enumerate(dataset.questions):
            grouped_indices[
                (question["question_family_index"], question["answer"])
            ].append(i)

        def collate_fn(batch):
            graphs, questions, targets = utils.collate_nsmitems(batch)
            return (
                graphs,
                questions,
                vocab.concept_embeddings,
                vocab.property_embeddings,
                targets,
            )

        dataloaders = []
        for key in itertools.product(nhops, counts):
            dataloaders.append(
                torch.utils.data.DataLoader(
                    Subset(dataset, indices=grouped_indices[key]),
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=os.cpu_count(),
                )
            )
        return dataloaders
