from stanza.server import CoreNLPClient
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.utils.tensorboard as tb
import glob
from nsm.datasets.gqa import (
    GQASceneGraphsOnlyDataset,
    SceneGraph,
    SceneGraphProcessor,
)
from nsm.datasets.utils import SortedStrSampler, SequentialStrSampler, RandomStrSampler
from nsm.model import NSM
from pathlib import Path
from nsm.vocab import GloVe, get_concept_vocab
from nsm.logging import configure_logging
from nsm.datasets.utils import SequentialStrSampler, RandomStrSampler
from nsm.config import get_config
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
import datetime as dt
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--skip-train", dest="train", action="store_false")
args = parser.parse_args()

configure_logging()
logger = logging.getLogger()
config = get_config("model_config.toml")


glove = GloVe(dim=config.glove_dim)
concept_vocab = get_concept_vocab(config.data_path)
train_set = GQASceneGraphsOnlyDataset(
    config.data_path / "GQA",
    "train",
    glove,
    concept_vocab,
    config.data_path / "stanford-corenlp-4.2.0",
)

train_loader = data.DataLoader(
    train_set,
    batch_size=config.batch_size,
    sampler=RandomStrSampler(train_set),  # type: ignore [arg-type]
    num_workers=min(6, len(os.sched_getaffinity(0))),
    collate_fn=collate_nsmitems,
)

# It is important to define the device before creating the optimizer,
# since moving models creates new tensor objects
device = "cuda" if torch.cuda.is_available() else "cpu"
prop_embeds = concept_vocab.property_embeddings.to(device)
concept_embeds = concept_vocab.concepts.vectors.to(device)
model = NSM(
    input_size=config.glove_dim,
    n_node_properties=len(concept_vocab.grouped_attrs) + 1,
    computation_steps=config.computation_steps,
    output_size=len(train_set.answer_vocab),
    dropout=config.dropout,
).to(device)
# this should free some space
del glove

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learn_rate)

checkpoints_path = config.data_path / "checkpoints"

if args.train:
    tb_writer = tb.SummaryWriter()
    for epoch in tqdm(range(config.epochs), desc="Epoch"):
        for global_step, batch in enumerate(
            tqdm(train_loader, desc="Batch"), start=epoch * len(train_loader)
        ):
            graph_batch, question_batch, targets = (e.to(device) for e in batch)

            optimizer.zero_grad(set_to_none=True)

            out = model(graph_batch, question_batch, concept_embeds, prop_embeds)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

            if global_step % 10 == 0:
                tb_writer.add_scalar("train_loss", loss.item(), global_step)

    model_path = (
        checkpoints_path / dt.datetime.now().isoformat(timespec="seconds") / "final.pt"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)


model.load_state_dict(
    torch.load(
        sorted(
            checkpoints_path.glob("*"), key=lambda p: dt.datetime.fromisoformat(p.name)
        )[-1]
        / "final.pt"
    )
)

train_eval_loader = data.DataLoader(
    train_set,
    batch_size=config.batch_size,
    sampler=SortedStrSampler(train_set),  # type: ignore [arg-type]
    num_workers=min(6, len(os.sched_getaffinity(0))),
    collate_fn=collate_nsmitems,
)
model.eval()
all_targets = []
predicted = []
with torch.no_grad():
    for batch in tqdm(train_eval_loader, desc="Evaluating"):
        graph_batch, question_batch, targets = (e.to(device) for e in batch)
        out = model(graph_batch, question_batch, concept_embeds, prop_embeds)
        all_targets += targets.tolist()
        predicted += out.argmax(1).tolist()

breakpoint()
with open(config.data_path / "all_targets.pkl", "wb") as f1, open(
    config.data_path / "predicted.pkl", "wb"
) as f2:
    pickle.dump(all_targets, f1)
    pickle.dump(predicted, f2)

logger.info(
    f"Train acc: {sum(t == p for t, p in zip(all_targets, predicted))/len(train_set):.2%}"
)
