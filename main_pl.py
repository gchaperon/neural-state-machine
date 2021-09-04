from nsm.model import NSM, NSMLightningModule, instruction_model_types
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
    datamodule = clevr.ClevrNoImagesDataModule(
        "data", batch_size=args.batch_size, w_instructions=False, nhops=args.nhops
    )

    # most params obtained via inspection of dataset
    model = NSMLightningModule(
        input_size=45,
        n_node_properties=4,
        # computation_steps=args.steps,
        output_size=28,
        instruction_model_name=args.instruction_model_name,
        instruction_model_kwargs={
            "embedding_size": 45,
            "n_instructions": max(args.nhops) + 2,
            "encoded_question_size": args.encoded_question_size,
        },
        learn_rate=args.learn_rate,
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
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    # parser.add_argument("--q-len", type=int)
    # parser.add_argument("--steps", required=True, type=int)
    parser.add_argument(
        "--nhops",
        type=int,
        nargs="+",
        default=[],
        help="use questions in clevr that require these number of hops only",
    )
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--learn-rate", required=True, type=float)
    parser.add_argument("--encoded-question-size", required=True, type=int)
    parser.add_argument(
        "--instruction-model-name",
        required=True,
        choices=list(instruction_model_types.keys()),
    )

    args = parser.parse_args()
    main(args)
