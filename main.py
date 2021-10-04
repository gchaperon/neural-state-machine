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
Args = collections.namedtuple("Args", "learn_rate, batch_size, encoded_size, instruction_loss_scaling")


def main():
    for args in (
        Args(*t)
        for t in random.sample(
            list(itertools.product(LRS, BSIZES, ENCODED_SIZES, INSTRUCTION_LOSS_SCALINGS)), k=16
        )
    ):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        pl.seed_everything(seed=123, workers=True)
        logging.basicConfig(level=logging.INFO)

        print(args)
        datamodule = clevr.ClevrWInstructionsDataModule(
            "data", args.batch_size, nhops=[0]
        )
        # most params obtained via inspection of dataset
        model = NSMLightningModule(
            input_size=45,
            n_node_properties=4,
            computation_steps=0 + 1,  # lul
            encoded_question_size=args.encoded_size,
            output_size=28,
            learn_rate=args.learn_rate,
            instruction_loss_scaling=args.instruction_loss_scaling,
        )
        metric_to_track = "train_loss"
        trainer = pl.Trainer(
            gpus=-1 if torch.cuda.is_available() else 0,
            max_epochs=1000,
            callbacks=[
                pl.callbacks.EarlyStopping(monitor=metric_to_track, patience=100),
                pl.callbacks.ModelCheckpoint(monitor=metric_to_track),
            ],
        )
        trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
