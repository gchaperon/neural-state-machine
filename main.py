from stanza.server import CoreNLPClient
from operator import eq
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
from nsm.model import NSM
from pathlib import Path
from nsm.vocab import GloVe, get_concept_vocab
from nsm.logging import configure_logging
from nsm.config import get_config, get_args
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
import time
import argparse
import pickle

configure_logging()
logger = logging.getLogger(__name__)
    
args = get_args()
config = get_config(args)
logger.info(f"Config: {config}")

glove = GloVe(dim=config.glove_dim)
concept_vocab = get_concept_vocab(config.data_path, config.glove_dim)
# DATASET
gqa_train, gqa_val = GQASceneGraphsOnlyDataset.splits(
    config.data_path / "GQA",
    glove,
    concept_vocab,
    config.data_path / "stanford-corenlp-4.2.0",
)

train_set = data.Subset(
    dataset=gqa_train,
    indices=random.sample(
        range(len(gqa_train)), k=int(config.subset_size * len(gqa_train))
    ),
)
val_set = data.Subset(
    dataset=gqa_val,
    indices=random.sample(
        range(len(gqa_val)), k=int(config.subset_size * len(gqa_val))
    ),
)

train_loader = data.DataLoader(
    train_set,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=min(6, len(os.sched_getaffinity(0))),
    collate_fn=collate_nsmitems,
)
val_loader = data.DataLoader(
    val_set,
    batch_size=config.batch_size,
    shuffle=False,
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
    output_size=len(gqa_train.answer_vocab),
    dropout=config.dropout,
).to(device)
# this should free some space
del glove

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learn_rate)

checkpoints_path = config.data_path / "checkpoints"


def eval_acc(model: nn.Module, loader: data.DataLoader) -> float:
    previous_state = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            graph_batch, question_batch, targets = (e.to(device) for e in batch)
            out = model(graph_batch, question_batch, concept_embeds, prop_embeds)
            correct += (out.argmax(1) == targets).sum().item()
            total += len(targets)
    model.train(previous_state)
    return correct / total


if args.train:
    model.train()
    tb_log_dir = "runs/" + "__".join(
        [dt.datetime.now().isoformat(timespec="seconds")]
        + [f"{k}={v}" for k, v in vars(args).items() if v and k in config.__fields__]
    )
    tb_writer = tb.SummaryWriter(tb_log_dir)
    for epoch in tqdm(range(config.epochs), desc="Epoch"):
        for global_step, batch in enumerate(
            tqdm(train_loader, desc="Batch"), start=epoch * len(train_loader)
        ):
            graph_batch, question_batch, targets = (e.to(device) for e in batch)

            optimizer.zero_grad()

            out = model(graph_batch, question_batch, concept_embeds, prop_embeds)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

            if global_step % 20 == 0:
                tb_writer.add_scalar("train_loss", loss.item(), global_step)

        tb_writer.add_scalar("train_acc", eval_acc(model, train_loader), epoch)
        tb_writer.add_scalar("val_acc", eval_acc(model, val_loader), epoch)

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
    shuffle=False,
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

with open(config.data_path / "all_targets.pkl", "wb") as f1, open(
    config.data_path / "predicted.pkl", "wb"
) as f2:
    pickle.dump(all_targets, f1)
    pickle.dump(predicted, f2)

logger.info(f"Train acc: {sum(map(eq, all_targets, predicted))/len(train_set):.2%}")
