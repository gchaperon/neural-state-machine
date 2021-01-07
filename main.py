from stanza.server import CoreNLPClient
import random
from tqdm import tqdm
import torch
import torch.utils.data as data
from nsm.datasets.gqa import (
    GQASceneGraphsOnlyDataset,
    batch_tag,
    SceneGraph,
    SceneGraphProcessor,
    batch_process,
    collate_gqa,
)
from nsm.model import NSM
from pathlib import Path
from nsm.vocab import GloVe, get_concept_vocab
from nsm.logging import configure_logging
from nsm.datasets.utils import SequentialStrSampler, RandomStrSampler
from nsm.utils import split_batch, collate_graphs, forwardingpartial
import ijson
from itertools import islice
from functools import partial
import sys
import logging

configure_logging()

glove = GloVe()
concept_vocab = get_concept_vocab(Path("data"))
train_dataset, val_dataset = GQASceneGraphsOnlyDataset.splits(
    Path("data/GQA"),
    glove,
    concept_vocab,
    Path("data/stanford-corenlp-4.2.0"),
    "cpu",
)


def compute_acc(model, dataset, log_progress=False):
    loader = data.DataLoader(
        dataset,
        batch_size=64,
        sampler=SequentialStrSampler(dataset),
        num_workers=1,
        collate_fn=collate_gqa,
        pin_memory=True,
    )
    correct = 0
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()
        model.to(device)
        for batch in tqdm(loader, desc="Evaluating", disable=not log_progress):
            graph_batch, question_batch, target = [t.to(device) for t in batch]
            output = model(graph_batch, question_batch)
            correct += torch.sum(output.argmax(1) == target).item()

    return corret / len(dataset)


model = NSM(
    input_size=glove.dim,
    n_node_properties=len(concept_vocab.grouped_attrs) + 1,
    computation_steps=8,
    output_size=len(train_dataset.answer_vocab),
)


property_embeddings = torch.vstack(
    (
        concept_vocab.objects.vectors.mean(0),
        concept_vocab.attr_types.vectors,
        concept_vocab.relations.vectors.mean(0),
    )
)

acc = compute_acc(
    forwardingpartial(
        model,
        concept_vocabulary=concept_vocab.concepts.vectors,
        property_embeddings=property_embeddings,
    ),
    val_dataset,
    log_progress=True,
)
logging.info(f"accuracy: {acc}")
