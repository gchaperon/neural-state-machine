from pathlib import Path
import argparse
from nsm.model import NSMLightningModule
from nsm.datasets.clevr import (
    THESE_TEMPLATES_SHOULD_BE_EASY_FOR_THE_NSM as CATS,
    ClevrNoImagesDataModule,
)
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import tqdm


Y_LABELS = ["three_hop", "one_hop", "two_hop", "zero_hop"]
X_LABELS = ["size", "shape", "color", "material"]
CAT_ORDERS = [
    X_LABELS,
    ["size", "color", "material", "shape"],
    ["size", "color", "material", "shape"],
    ["shape", "material", "color", "size"],
]


def fix_cat_grid(grid):
    out = np.zeros_like(grid)

    for i, row in enumerate(grid):
        for j, cat in enumerate(X_LABELS):
            out[i, j] = row[CAT_ORDERS[i].index(cat)]

    return out


def main(args):
    model = NSMLightningModule.load_from_checkpoint(args.checkpoint)
    datamodule = ClevrNoImagesDataModule("data", batch_size=128)
    datamodule.setup("fit")
    dataloader = datamodule.train_dataloader()
    question_cats = [q["question_family_index"] for q in datamodule.clevr_val.questions]
    cat_totals = Counter(question_cats)

    predictions = []
    targets = []
    for batch in tqdm.tqdm(dataloader, desc="Validating"):
        *first, target = batch
        predictions += model(*first).argmax(1).tolist()
        targets += target.tolist()

    cat_counts = defaultdict(int)
    for pred, target, cat in zip(predictions, targets, question_cats):
        cat_counts[cat] += pred == target

    cat_grid = fix_cat_grid(np.array(CATS).reshape(4, -1)).astype(float)
    for i in range(cat_grid.shape[0]):
        for j in range(cat_grid.shape[1]):
            cat = cat_grid[i, j]
            cat_grid[i, j] = cat_counts[cat] / cat_totals[cat]

    plt.imshow(cat_grid)
    # plot numbers
    for i, j in product(*map(range, cat_grid.shape)):
        plt.text(j, i, f"{cat_grid[i, j]:.2f}", ha="center", va="center", color="w")
    plt.xticks(ticks=np.arange(len(X_LABELS)), labels=X_LABELS)
    plt.yticks(ticks=np.arange(len(Y_LABELS)), labels=Y_LABELS)
    plt.colorbar()
    plt.title(Path(args.checkpoint).parts[-3])
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()
    main(args)
