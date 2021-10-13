from nsm.model import InstructionsModelLightningModule, NSMLightningModule
from nsm.datasets import ClevrWInstructionsDataModule
import pytorch_lightning as pl
import torch
import argparse

# these args are only for the NSM, the hparams of the instructions model are
# fix, and I used the best hparams i found when training the ins model alone
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--learn-rate",type=float, required=True)
args= parser.parse_args()


EMBEDDING_SIZE = 45
N_INSTRUCTIONS = 2
ENCODED_QUESTION_SIZE = 100
# Train instructions model only
datamodule = ClevrWInstructionsDataModule("data", batch_size=128, nhops=[0])
instructions_model = InstructionsModelLightningModule(
    embedding_size=EMBEDDING_SIZE,
    n_instructions=N_INSTRUCTIONS,
    encoded_question_size=ENCODED_QUESTION_SIZE,
    learn_rate=0.001,
)
trainer_instructions = pl.Trainer(
    gpus=-1 if torch.cuda.is_available() else 0,
    max_epochs=100,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=10),
        pl.callbacks.ModelCheckpoint(monitor="val_loss"),
    ],
)
trainer_instructions.fit(instructions_model, datamodule)


# Now train NSM, but load instructions model from previous step, freeze weights
# and set to eval (to disable dropout, not relevant now but to keep in mind)
datamodule = ClevrWInstructionsDataModule("data", batch_size=args.batch_size, nhops=[0])
nsm_model = NSMLightningModule(
    input_size=EMBEDDING_SIZE,
    n_node_properties=4,
    computation_steps=N_INSTRUCTIONS - 1,
    encoded_question_size=ENCODED_QUESTION_SIZE,
    output_size=28,
    learn_rate=args.learn_rate,
    use_instruction_loss=False,
)

nsm_model.nsm.instructions_model.load_state_dict(instructions_model.model.state_dict())
nsm_model.nsm.instructions_model.requires_grad_(False).eval()

trainer_nsm = pl.Trainer(
    gpus=-1 if torch.cuda.is_available() else 0,
    max_epochs=1000,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="train_loss", patience=100),
        pl.callbacks.ModelCheckpoint(monitor="train_loss"),
    ],
)
trainer_nsm.fit(nsm_model, datamodule)

