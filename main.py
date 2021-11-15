import os
from nsm.model import NSM, NSMLightningModule, instruction_model_types
import nsm.datasets.synthetic as syn
import nsm.datasets.clevr as clevr
import pytorch_lightning as pl
from tqdm import tqdm
import torch
import collections
import itertools
import argparse
from pprint import pprint
import logging
import random


LRS = [5e-5, 1e-4, 5e-4, 1e-3]
BSIZES = [16, 32, 64, 128]
ENCODED_SIZES = [45, 100, 200, 500]
INSTRUCTION_LOSS_SCALINGS = [10, 50, 100]
Args = collections.namedtuple(
    "Args", "learn_rate, batch_size, encoded_size, instruction_loss_scaling"
)


def main(args):
    logging.basicConfig(level=logging.INFO)

    print(args)
    datamodule = clevr.ClevrDataModule(
        datadir="data",
        batch_size=args.batch_size,
        cats=args.cats,
        prop_embeds_const=1.,
    )
    # most params obtained via inspection of dataset
    model = NSMLightningModule(
        input_size=45,
        n_node_properties=4,
        computation_steps=args.computation_steps,
        encoded_question_size=args.encoded_size,
        output_size=28,
        learn_rate=args.learn_rate,
        use_instruction_loss=False,
    )
    metric_to_track = "train_loss"
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else 0,
        max_epochs=200,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor=metric_to_track, patience=30, stopping_threshold=1e-3
            ),
            pl.callbacks.ModelCheckpoint(monitor=metric_to_track),
        ],
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoded-size", type=int, default=100)

    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learn-rate", type=float, required=True)
    parser.add_argument(
        "--cats",
        nargs="+",
        type=int,
        choices=clevr.ALL_CATS,
        required=True,
    )
    parser.add_argument("--computation-steps", type=int, required=True)

    args = parser.parse_args()
    main(args)
