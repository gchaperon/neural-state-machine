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



def main():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    pl.seed_everything(seed=123, workers=True)
    logging.basicConfig(level=logging.INFO)

    datamodule = clevr.ClevrWInstructionsDataModule(
        "data", batch_size=16, nhops=[0]
    )
    # most params obtained via inspection of dataset
    model = NSMLightningModule(
        input_size=45,
        n_node_properties=4,
        computation_steps=0 + 1,  # lul
        encoded_question_size=100,
        output_size=28,
        learn_rate=0.001,
    )
    metric_to_track = "train_loss"
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else 0,
        max_epochs=1000,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor=metric_to_track, patience=300),
            pl.callbacks.ModelCheckpoint(monitor=metric_to_track),
        ],
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
