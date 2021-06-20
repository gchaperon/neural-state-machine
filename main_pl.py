from nsm.model import NSM, NSMLightningModule
import nsm.datasets.synthetic as syn
import pytorch_lightning as pl
from tqdm import tqdm
import torch
from itertools import product
import argparse
from pprint import pprint


def main(args):
    pl.seed_everything(seed=123, workers=True)

    datamodules = [
        # small graphs, few concepts, low jump prob
        syn.SyntheticDataModule(
            syn.SyntheticDataset(
                vocab=syn.Concepts(
                    n_objects=10,
                    n_relations=5,
                ),
                n_nodes=10,
                n_edges=10,
                size=10_000,
                question_length=8,
                jump_prob=0.0,
            ),
            batch_size=4,
            split_ratio=0.9,
        ),
        # large graphs, few concepts, low jump prob, hopefully some paths repeat
        syn.SyntheticDataModule(
            syn.SyntheticDataset(
                vocab=syn.Concepts(
                    n_objects=5,
                    n_relations=3,
                ),
                n_nodes=100,
                n_edges=100,
                size=10_000,
                question_length=8,
                jump_prob=0.0,
            ),
            batch_size=4,
            split_ratio=0.9,
        ),
        # moderate jump prob
        syn.SyntheticDataModule(
            syn.SyntheticDataset(
                vocab=syn.Concepts(
                    n_objects=100,
                    n_relations=100,
                ),
                n_nodes=100,
                n_edges=100,
                size=10_000,
                question_length=8,
                jump_prob=0.5,
            ),
            batch_size=4,
            split_ratio=0.9,
        ),
        # always jumps
        syn.SyntheticDataModule(
            syn.SyntheticDataset(
                vocab=syn.Concepts(
                    n_objects=100,
                    n_relations=100,
                ),
                n_nodes=100,
                n_edges=100,
                size=10_000,
                question_length=8,
                jump_prob=1.0,
            ),
            batch_size=4,
            split_ratio=0.9,
        ),
    ]
    common_vocab = syn.Concepts(n_objects=100, n_relations=50)
    datamodules += [
        # small train graphs, large val graphs
        syn.SyntheticDataModule.from_splits(
            syn.SyntheticDataset(
                vocab=common_vocab,
                n_nodes=10,
                n_edges=10,
                size=9_000,
                jump_prob=0.0,
                question_length=8,
            ),
            syn.SyntheticDataset(
                vocab=common_vocab,
                n_nodes=100,
                n_edges=100,
                size=1_000,
                jump_prob=0.0,
                question_length=8,
            ),
            batch_size=4,
        ),
        # small train graphs and few instructions, large val graphs and many instructions
        # this is sadly nos possible for now :-/
        # syn.SyntheticDataModule.from_splits(
        #     syn.SyntheticDataset(
        #         vocab=common_vocab,
        #         n_nodes=10,
        #         n_edges=10,
        #         size=9_000,
        #         jump_prob=0.0,
        #         question_length=3,
        #     ),
        #     syn.SyntheticDataset(
        #         vocab=common_vocab,
        #         n_nodes=100,
        #         n_edges=100,
        #         size=1_000,
        #         jump_prob=0.0,
        #         question_length=10,
        #     ),
        #     batch_size=4,
        # ),
    ]

    # breakpoint()
    for i, datamodule in enumerate(datamodules[4:]):
        print(f"=========== Experimento {i} ===========")
        model = NSMLightningModule(
            NSM(
                input_size=len(datamodule.dataset.vocab),
                n_node_properties=1,
                computation_steps=datamodule.dataset.question_length - 1,
                output_size=len(datamodule.dataset.vocab.objects),
                instruction_model="dummy",
            ),
            learn_rate=0.001,
        )
        trainer = pl.Trainer(
            gpus=-1 if torch.cuda.is_available() else 0,
            max_epochs=100,
            callbacks=[pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss")],
        )
        trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-len", type=int)
    # parser.add_argument("--n-unique", type=int)
    args = parser.parse_args()
    main(args)
