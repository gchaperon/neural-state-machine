from nsm.model import (
    InstructionsModelLightningModule,
    NSMLightningModule,
    DummyInstructionsModel,
)
from nsm.datasets import ClevrWInstructionsDataModule
import nsm.datasets.synthetic as syn
import pytorch_lightning as pl
import torch
import argparse


def main_freeze_instructions(args):
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
    datamodule = ClevrWInstructionsDataModule(
        "data", batch_size=args.batch_size, nhops=[0]
    )
    nsm_model = NSMLightningModule(
        input_size=EMBEDDING_SIZE,
        n_node_properties=4,
        computation_steps=N_INSTRUCTIONS - 1,
        encoded_question_size=ENCODED_QUESTION_SIZE,
        output_size=28,
        learn_rate=args.learn_rate,
        use_instruction_loss=False,
    )

    nsm_model.nsm.instructions_model.load_state_dict(
        instructions_model.model.state_dict()
    )
    if not args.fine_tune:
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


def main_freeze_automaton(args):
    # Train automaton with my instructions
    EMBEDDING_SIZE = 45
    N_INSTRUCTIONS = 2
    ENCODED_SIZE = 100

    class NSMDummyInstructions(NSMLightningModule):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            self.nsm.instructions_model = DummyInstructionsModel(
                embedding_size=kwargs["input_size"],
                n_instructions=kwargs["computation_steps"] + 1,
                encoded_question_size=kwargs["encoded_question_size"],
            )

        @staticmethod
        def _prep(batch):
            batch = list(batch)
            batch[1] = torch.nn.utils.rnn.pack_sequence(batch[5].unbind(0))
            return tuple(batch)

        def training_step(self, batch, batch_idx):
            return super().training_step(self._prep(batch), batch_idx)

        def validation_step(self, batch, batch_idx):
            return super().validation_step(self._prep(batch), batch_idx)

    datamodule = ClevrWInstructionsDataModule("data", batch_size=32, nhops=[0])
    nsm_dummy_ins = NSMDummyInstructions(
        input_size=EMBEDDING_SIZE,
        n_node_properties=4,
        computation_steps=N_INSTRUCTIONS - 1,
        encoded_question_size=ENCODED_SIZE,
        output_size=28,
        learn_rate=0.001,
        use_instruction_loss=False,
    )
    trainer_automaton = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else 0,
        max_epochs=10,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="train_loss", patience=3),
            pl.callbacks.ModelCheckpoint(monitor="train_loss"),
        ],
    )
    trainer_automaton.fit(nsm_dummy_ins, datamodule)

    # lead trained weight from the automaton to the nsm
    datamodule = ClevrWInstructionsDataModule(
        "data", batch_size=args.batch_size, nhops=[0]
    )
    nsm_model = NSMLightningModule(
        input_size=EMBEDDING_SIZE,
        n_node_properties=4,
        computation_steps=N_INSTRUCTIONS - 1,
        encoded_question_size=ENCODED_SIZE,
        output_size=28,
        learn_rate=args.learn_rate,
        use_instruction_loss=False,
    )
    nsm_model.nsm.nsm_cell.load_state_dict(nsm_dummy_ins.nsm.nsm_cell.state_dict())
    nsm_model.nsm.classifier.load_state_dict(nsm_dummy_ins.nsm.classifier.state_dict())
    if not args.fine_tune:
        nsm_model.nsm.nsm_cell.requires_grad_(False).eval()
        nsm_model.nsm.classifier.requires_grad_(False).eval()

    trainer_nsm = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else 0,
        max_epochs=1000,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="train_loss", patience=200),
            pl.callbacks.ModelCheckpoint(monitor="train_loss"),
        ],
    )
    trainer_nsm.fit(nsm_model, datamodule)


if __name__ == "__main__":

    # these args are only for the NSM, the hparams of the instructions model are
    # fix, and I used the best hparams i found when training the ins model alone
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        description="choose which part of the nsm you want to freeze!", required=True
    )

    parser_freeze_instructions = subparsers.add_parser("instructions")
    parser_freeze_instructions.add_argument("--batch-size", type=int, required=True)
    parser_freeze_instructions.add_argument("--learn-rate", type=float, required=True)
    parser_freeze_instructions.add_argument("--fine-tune", action="store_true")
    parser_freeze_instructions.set_defaults(func=main_freeze_instructions)

    # freezin automaton also freezes the answer classifier
    parser_freeze_automaton = subparsers.add_parser("automaton")
    parser_freeze_automaton.add_argument("--batch-size", type=int, required=True)
    parser_freeze_automaton.add_argument("--learn-rate", type=float, required=True)
    parser_freeze_automaton.add_argument("--fine-tune", action="store_true")
    parser_freeze_automaton.set_defaults(func=main_freeze_automaton)

    args = parser.parse_args()
    args.func(args)
