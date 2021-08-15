from nsm.model import NSMLightningModule
from nsm.datasets.clevr import ClevrNoImagesDataModule
import tqdm
from pprint import pp

print("print dev examples where NSM trained with my custom instruction fails")

PATH = "remote_logs/version_11/checkpoints/epoch=609-step=600239.ckpt"

model = NSMLightningModule.load_from_checkpoint(PATH)
datamodule = ClevrNoImagesDataModule("data", batch_size=128, w_instructions=True)
datamodule.setup("validate")

predicted = []
targets = []

for *input, target in tqdm.tqdm(datamodule.val_dataloader()):
    out = model(*input)
    targets += target.tolist()
    predicted += out.argmax(1).tolist()

wrong = [i for i, (p, t) in enumerate(zip(predicted, targets)) if p != t]
print(wrong)
breakpoint()
pass
