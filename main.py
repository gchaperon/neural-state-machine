from stanza.server import CoreNLPClient
import random
from tqdm import tqdm
import torch.utils.data as data
from nsm.datasets.gqa import (
    GQASceneGraphsOnlyDataset,
    batch_tag,
    SceneGraph,
    SceneGraphProcessor,
    batch_process,
    collate_gqa,
)
from pathlib import Path
from nsm.vocab import GloVe, get_concept_vocab
from nsm.logging import configure_logging
from nsm.datasets.utils import SequentialStrSampler, RandomStrSampler
from nsm.utils import split_batch, collate_graphs
import ijson
from itertools import islice
from functools import partial

configure_logging()


glove = GloVe()
concept_vocab = get_concept_vocab(Path("data"))
dset = GQASceneGraphsOnlyDataset(
    Path("data/GQA"),
    "train",
    glove,
    concept_vocab,
    Path("data/stanford-corenlp-4.2.0"),
    "cpu",
)
sampler = SequentialStrSampler(dset)
dloader = data.DataLoader(
    dset,
    batch_size=64,
    sampler=sampler,  # type: ignore [arg-type]
    num_workers=1,
    collate_fn=collate_gqa,
    pin_memory=True,
)
for batch in tqdm(dloader):
    pass
