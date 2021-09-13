from nsm.model import InstructionsModelLightningModule
from nsm.datasets import ClevrWInstructionsDataModule
import pytorch_lightning as pl
import torch

datamodule = ClevrWInstructionsDataModule("data", batch_size=16, nhops=[0])
model = InstructionsModelLightningModule(
    embedding_size=45, n_instructions=0 + 2, encoded_question_size=45, learn_rate=0.1
)

trainer = pl.Trainer(gpus=-1 if torch.cuda.is_available() else 0, max_epochs=100)

trainer.fit(model, datamodule)
