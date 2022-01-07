import os
from nsm.model import (
    NSM,
    NSMLightningModule,
    instruction_model_types,
    NSMBaselineLightningModule,
)
import nsm.datasets.synthetic as syn
import nsm.datasets.clevr as clevr
import nsm.datasets.custom_clevr as cclevr
import pytorch_lightning as pl
from tqdm import tqdm
import torch
import collections
import itertools
import argparse
from pprint import pprint
import logging
import random


def main(args):
    logging.basicConfig(level=logging.INFO)

    print(args)
    datamodule = cclevr.ExistClevrDataModule(
        datadir="data",
        batch_size=args.batch_size,
    )
    # most params obtained via inspection of dataset
    model = NSMBaselineLightningModule(
        input_size=45,
        n_node_properties=4,
        # computation_steps=args.computation_steps,
        encoded_question_size=args.encoded_size,
        output_size=28,
        learn_rate=args.learn_rate,
        # use_instruction_loss=False,
    )
    metric_to_track = "train_loss"
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else 0,
        max_epochs=800,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor=metric_to_track, patience=50, stopping_threshold=1e-3
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
    # parser.add_argument("--computation-steps", type=int, required=True)

    args = parser.parse_args()
    main(args)
