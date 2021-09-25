from nsm.model import InstructionsModelLightningModule
from nsm.datasets import ClevrWInstructionsDataModule
import pytorch_lightning as pl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
args = parser.parse_args()

datamodule = ClevrWInstructionsDataModule("data", batch_size=1, nhops=[0])
model = InstructionsModelLightningModule.load_from_checkpoint(args.checkpoint)

datamodule.setup("validate")

for _, q_batch, vocab, *_ in datamodule.val_dataloader():
    model(vocab, q_batch)

