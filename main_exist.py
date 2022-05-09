import nsm.datasets.custom_clevr as cclevr
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
        computation_steps=args.computation_steps,
        encoded_question_size=args.encoded_size,
        output_size=28,
        learn_rate=args.learn_rate,
    )
    metric_to_track = "train_loss"
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else 0,
        max_epochs=1000,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor=metric_to_track,
                patience=200,
                mode="min",
            ),
            pl.callbacks.ModelCheckpoint(monitor=metric_to_track),
        ],
    )
    if args.train:
        datamodule = cclevr.ExistClevrDataModule(
            datadir="data",
            batch_size=args.batch_size,
        )
        trainer.fit(model, datamodule=datamodule, ckpt_path=args.checkpoint)

    if args.test:
        datamodule = cclevr.ExistClevrDataModule(
            datadir="data",
            batch_size=args.batch_size,
        )
        trainer.test(model, datamodule, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoded-size", type=int, default=100)

    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learn-rate", type=float, required=True)
    parser.add_argument("--computation-steps", type=int, default=4)
    parser.add_argument("--no-train", dest="train", action="store_false")
    parser.add_argument("--no-test", dest="test", action="store_false")
    parser.add_argument("--checkpoint")

    args = parser.parse_args()
    main(args)
