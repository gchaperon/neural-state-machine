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
    datamodule = cclevr.SingleAndClevrDataModule(
        datadir="data",
        batch_size=args.batch_size,
        subset_ratio=args.subset_ratio,
        cats=args.single_and_cats,
    )
    # most params obtained via inspection of dataset

    if args.model_type == "NSM":
        model = NSMLightningModule(
            input_size=45,
            n_node_properties=4,
            computation_steps=args.computation_steps,
            encoded_question_size=args.encoded_size,
            output_size=28,
            learn_rate=args.learn_rate,
            # use_instruction_loss=False,
        )
    elif args.model_type == "NSMBaseline":
        print(f"NSMBaseline chosen, ignoring {args.computation_steps=}")
        model = NSMBaselineLightningModule(
            input_size=45,
            n_node_properties=4,
            encoded_question_size=args.encoded_size,
            output_size=28,
            learn_rate=args.learn_rate,
            # use_instruction_loss=False,
        )
    metric_to_track = "val_acc"
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else 0,
        max_epochs=500,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor=metric_to_track,
                patience=20,
                stopping_threshold=0.95,
                mode="max",
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
    parser.add_argument("--subset-ratio", type=float, required=True)
    parser.add_argument("--computation-steps", type=int, required=True)
    parser.add_argument("--model-type", required=True, choices=("NSM", "NSMBaseline"))
    parser.add_argument(
        "--single-and-cats",
        nargs="+",
        metavar="CAT",
        choices=cclevr.AND_CATS,
        default=cclevr.AND_CATS,
    )

    args = parser.parse_args()
    main(args)
