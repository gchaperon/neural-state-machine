import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import os
import torch.utils.tensorboard as tb
import torch.utils.data as data
import datetime as dt
from nsm.logging import configure_logging
from nsm.config import get_config, get_args
from nsm.datasets.random import RandomDataset
from nsm.utils import collate_nsmitems
from nsm.model import NSM
import logging

configure_logging()
logger = logging.getLogger(__name__)

args = get_args()
config = get_config(args)
logger.info(f"Config: {config}")


N_NODE_PROPERTIES = 1
ANSWER_VOCAB_SIZE = 10 ** 2
train_set = RandomDataset(
    size=10 ** 4,
    hidden_size=config.glove_dim,
    n_token_distribution=(8.0, 2),
    n_properties=N_NODE_PROPERTIES,
    node_distribution=(15, 5),
    density_distribution=(0.4, 0.2),
    answer_vocab_size=ANSWER_VOCAB_SIZE,
)
train_loader = data.DataLoader(
    train_set,
    shuffle=True,
    batch_size=config.batch_size,
    num_workers=min(6, len(os.sched_getaffinity(0))),
    collate_fn=collate_nsmitems,
)
val_loader = data.DataLoader(
    train_set,
    shuffle=False,
    batch_size=config.batch_size,
    num_workers=6,
    collate_fn=collate_nsmitems,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
prop_embeds = torch.rand(N_NODE_PROPERTIES + 1, config.glove_dim, device=device)
concept_embeds = torch.rand(1300 // 78, config.glove_dim, device=device)


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


model = NSM(
    input_size=config.glove_dim,
    n_node_properties=N_NODE_PROPERTIES,
    computation_steps=config.computation_steps,
    output_size=ANSWER_VOCAB_SIZE,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learn_rate)

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

    tb_writer.add_scalar("train_acc", eval_acc(model, val_loader), epoch)
