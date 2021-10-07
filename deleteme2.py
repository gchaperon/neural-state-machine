from nsm.model import Tagger
from nsm.datasets import ClevrWInstructionsDataModule

datamodule = ClevrWInstructionsDataModule("data", batch_size=1, nhops=[0])
tagger = Tagger(45)

datamodule.setup("validate")
_, q_batch, vocab, *_ = next(iter(datamodule.val_dataloader()))

out = tagger(vocab, q_batch)
breakpoint()
pass
