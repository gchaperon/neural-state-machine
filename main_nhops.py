"""script to train model for nhop plots in theses"""
import nsm.datasets.clevr as clevr
from nsm.model import NSMLightningModule
import pytorch_lightning as pl
import torch

import argparse
import logging


def main(args):
    logging.basicConfig(level=logging.INFO)
    model = NSMLightningModule(
        input_size=45,
        n_node_properties=4,
        computation_steps=4,
        encoded_question_size=args.encoded_size,
        output_size=28,
        learn_rate=args.learn_rate,
    )
    metric_to_track = "train_loss"
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else 0,
        max_epochs=500,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor=metric_to_track,
                patience=20,
                mode="min",
            ),
            pl.callbacks.ModelCheckpoint(monitor=metric_to_track),
        ],
    )
    if args.train_cats:
        datamodule = clevr.ClevrDataModule(
            datadir="data",
            batch_size=args.batch_size,
            cats=args.train_cats,
            prop_embeds_const=1,
        )
        trainer.fit(model, datamodule=datamodule, ckpt_path=args.checkpoint)

    for cat in args.test_cats:
        datamodule = clevr.ClevrDataModule(
            datadir="data",
            batch_size=args.batch_size,
            cats=[cat],
            prop_embeds_const=1,
        )
        trainer.validate(model, datamodule=datamodule, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoded-size", type=int, default=100)

    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learn-rate", type=float, required=True)
    parser.add_argument("--train-cats", nargs="+", type=int)
    parser.add_argument("--test-cats", nargs="+", type=int, default=())
    parser.add_argument("--checkpoint")

    args = parser.parse_args()
    main(args)
