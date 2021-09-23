from nsm.model import InstructionsModelLightningModule
from nsm.datasets import ClevrWInstructionsDataModule
import pytorch_lightning as pl
import random
import torch
import argparse
import itertools
import logging

parser = argparse.ArgumentParser()
logging.basicConfig(level=logging.INFO)

learn_rates = [0.1, 0.01, 0.001, 0.0001]
batch_sizes = [16, 32, 64, 128]
encoded_sizes = [100, 200, 500, 1000]


def main(learn_rate, batch_size, encoded_size):
    datamodule = ClevrWInstructionsDataModule(
        "data", batch_size=batch_size, nhops=[0]
    )
    model = InstructionsModelLightningModule(
        embedding_size=45,
        n_instructions=0 + 2,
        encoded_question_size=encoded_size,
        learn_rate=learn_rate,
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


for args in random.sample(
    list(itertools.product(learn_rates, batch_sizes, encoded_sizes)),
    k=16,
):
    print(args)
    main(*args)
