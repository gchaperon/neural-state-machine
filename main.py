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
    datamodule = clevr.ClevrGloveDataModule(
        "data",
        args.batch_size,
        args.glove_dim,
        nhops=args.nhops,
        question_type=args.question_type,
        prop_embed_method=args.prop_embed_method,
    )
    # most params obtained via inspection of dataset
    model = NSMLightningModule(
        input_size=args.glove_dim,
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
        max_epochs=1000,
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoded-size", type=int, default=100)
    parser.add_argument("--prop-embeds-const", type=float, default=5.0)
    parser.add_argument("--learn-rate", type=float, default=0.001)
    # parser.add_argument("--use-instruction-loss", action="store_true")

    parser.add_argument(
        "--nhops",
        nargs="+",
        type=int,
        choices=clevr.NHOPS_TO_CATS.keys(),
        default=list(clevr.NHOPS_TO_CATS.keys()),
    )
    parser.add_argument("--computation-steps", type=int, required=True)
    parser.add_argument(
        "--glove-dim", type=int, required=True, choices=(50, 100, 200, 300)
    )
    parser.add_argument(
        "--question-type", required=True, choices=("program", "question")
    )
    parser.add_argument("--prop-embed-method", required=True, choices=("embed", "mean"))

    args = parser.parse_args()
    assert args.computation_steps >= max(args.nhops) + 1
    main(args)
