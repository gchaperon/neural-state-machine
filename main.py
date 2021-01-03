from stanza.server import CoreNLPClient
import random
from tqdm import tqdm
from nsm.datasets.gqa import (
    GQASceneGraphsOnlyDataset,
    batch_tag,
    process_scene_graphs,
    SceneGraph,
    batch_process_scene_graphs,
)
from pathlib import Path
from nsm.vocab import GloVe, get_concept_vocab
from nsm.logging import configure_logging
import ijson
from itertools import islice
from functools import partial

configure_logging()


glove = GloVe()
concept_vocab = get_concept_vocab(Path("data"))
with open("data/GQA/sceneGraphs/train_sceneGraphs.json") as f:
    n = 1000
    scene_graphs = (
        SceneGraph(**val) for _, val in tqdm(islice(ijson.kvitems(f, ""), n), total=n)
    )
    process_fn = partial(process_scene_graphs, glove=glove, concept_vocab=concept_vocab)
    batch_size = 100
    for _ in batch_process_scene_graphs(scene_graphs, process_fn, batch_size):
        pass
