from nsm.model import NSM, NSMLightningModule
import nsm.datasets.synthetic as syn
import nsm.datasets.clevr as clevr
import pytorch_lightning as pl
from tqdm import tqdm
import torch
from itertools import product
import argparse
from pprint import pprint
import logging


def main(args):
    pl.seed_everything(seed=123, workers=True)
    datamodule = clevr.ClevrNoImagesDataModule("data", batch_size=32)

    # most params obtained via inspection of dataset
    model = NSMLightningModule(
        input_size=45,
        n_node_properties=4,
        computation_steps=8,
        output_size=28,
        learn_rate=0.001,
    )
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else 0,
        max_epochs=100,
        callbacks=[pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss")],
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--q-len", type=int)
    # parser.add_argument("--n-unique", type=int)
    args = parser.parse_args()
    main(args)
