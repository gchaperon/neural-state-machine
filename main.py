from stanza.server import CoreNLPClient
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.utils.tensorboard as tb

from nsm.datasets.gqa import (
    GQASceneGraphsOnlyDataset,
    SceneGraph,
    SceneGraphProcessor,
)
from nsm.datasets.utils import SequentialStrSampler, RandomStrSampler
from nsm.model import NSM
from pathlib import Path
from nsm.vocab import GloVe, get_concept_vocab
from nsm.logging import configure_logging
from nsm.datasets.utils import SequentialStrSampler, RandomStrSampler
from nsm.utils import (
    split_batch,
    collate_graphs,
    forwardingpartial,
    partial_module,
    collate_nsmitems,
)
import ijson
from itertools import islice
from functools import partial
import sys
import os
import logging

configure_logging()
logger =logging.getLogger()

glove = GloVe()
concept_vocab = get_concept_vocab(Path("data"))
train_set, val_set = GQASceneGraphsOnlyDataset.splits(
    Path("data/GQA"), glove, concept_vocab, Path("data/stanford-corenlp-4.2.0")
)

train_loader = data.DataLoader(
    train_set,
    batch_size=64,
    sampler=RandomStrSampler(train_set),  # type: ignore [arg-type]
    num_workers=min(4, len(os.sched_getaffinity(0))),
    collate_fn=collate_nsmitems,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
prop_embeds = concept_vocab.property_embeddings.to(device)
concept_embeds = concept_vocab.concepts.vectors.to(device)
model = NSM(
    input_size=glove.dim,
    n_node_properties=len(concept_vocab.grouped_attrs) + 1,
    computation_steps=8,
    output_size=len(train_set.answer_vocab),
).to(device)
# this should free some space
del glove

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

tb_writer = tb.SummaryWriter()
for epoch in tqdm(range(10)):
    for global_step, batch in enumerate(
        tqdm(train_loader), start=epoch * len(train_loader)
    ):
        graph_batch, question_batch, targets = (e.to(device) for e in batch)

        optimizer.zero_grad(set_to_none=True)

        out = model(graph_batch, question_batch, concept_embeds, prop_embeds)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()

        if global_step % 10 == 0:
            tb_writer.add_scalar("train_loss", loss.item(), global_step)

val_loader = data.DataLoader(
    val_set,
    batch_size=128,
    sampler=SequentialStrSampler(val_set),  # type: ignore [arg-type]
    num_workers=min(4, len(os.sched_getaffinity(0))),
    collate_fn=collate_nsmitems,
)
model.eval()
correct = 0
with torch.no_grad():
    for batch in tqdm(val_loader):
        graph_batch, question_batch, targets = (e.to(device) for e in batch)
        out = model(graph_batch, question_batch, concept_embeds, prop_embeds)
        correct += (out.argmax(1) == targets).sum().item()


logger.info(f"Val acc: {correct/len(val_set):.3f}")
