from nsm.model import NSMLightningModule
from nsm.datasets import ClevrWInstructionsDataModule
import logging
import argparse
import tqdm
from pathlib import Path

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--batch-size", required=True, type=int)
args = parser.parse_args()

datamodule = ClevrWInstructionsDataModule("data", args.batch_size, nhops=[0])
model = NSMLightningModule.load_from_checkpoint(args.checkpoint)

datamodule.setup("validate")

correct = []
for *inputs, targets, gold_instructions in tqdm.tqdm(datamodule.val_dataloader()):
    predictions, generated_instructions = model(*inputs)
    correct += predictions.argmax(1).eq(targets).tolist()

version = next(
    (p for p in Path(args.checkpoint).parts if p.startswith("version")), None
)
with open(f"val_results_{version}.txt", "w") as outfile:
    for val in correct:
        outfile.write(str(val) + "\n")
