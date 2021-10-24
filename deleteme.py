from nsm.model import InstructionsModelLightningModule, NSMLightningModule
from nsm.datasets import ClevrWInstructionsDataModule
import pytorch_lightning as pl
import pathlib
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
args = parser.parse_args()

datamodule = ClevrWInstructionsDataModule("data", batch_size=1, nhops=[0])
model = NSMLightningModule.load_from_checkpoint(args.checkpoint)

datamodule.setup("validate")


all_outs = []
all_tgts = []
for i, (*inputs, targets, gold_ins) in enumerate(tqdm.tqdm(datamodule.val_dataloader())):
    preds, gen_ins = model(*inputs)
    all_outs += preds.argmax(1).tolist()
    all_tgts += targets.tolist()
version = next(p for p in pathlib.Path(args.checkpoint).parts if p.startswith("version"))
with open(f"val_results_{version}.txt", "w") as out_f:
    for o, t in zip(all_outs, all_tgts):
        out_f.write(f"{o==t}\n")

