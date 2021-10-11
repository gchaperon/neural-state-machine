from nsm.model import NSMLightningModule
import nsm.datasets.clevr as clevr
import argparse
import logging
import collections
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)

VERSIONS = [74, 75, 76, 77, 78, 82]
CHECKPOINTS = [
    next(Path(f"remote_logs/version_{version}/checkpoints").iterdir())
    for version in VERSIONS
]
models = [NSMLightningModule.load_from_checkpoint(ckpt) for ckpt in CHECKPOINTS]

datamodule = clevr.ClevrWInstructionsDataModule("data", batch_size=256, nhops=[0])
datamodule.setup("validate")


gold = []
results = collections.defaultdict(list)

for *inputs, targets, gold_instructins in tqdm.tqdm(
    datamodule.val_dataloader(), desc=f"Validating"
):
    gold += targets.tolist()
    for i, model in enumerate(models):
        predictions, _ = model(*inputs)
        results[f"version_{VERSIONS[i]}"] += predictions.argmax(1).tolist()

##################
# histogram for question categories, this code is shit, but had no time to
# make it better
cats = clevr.NHOPS_TO_CATS[0]
bin_heights = [collections.defaultdict(int) for _ in cats]
n_per_cat = [0] * len(cats)
for i, question in enumerate(datamodule.clevr_val.questions):
    cat = question["question_family_index"]
    n_per_cat[cats.index(cat)] += 1
    for version in VERSIONS:
        bin_heights[cats.index(cat)][f"version_{version}"] += (
            gold[i] == results[f"version_{version}"][i]
        )

fig, ax = plt.subplots(figsize=(15, 9))
x = np.arange(len(cats))
width = 0.75 / len(VERSIONS)
for i, v in enumerate(VERSIONS):
    label = f"version_{v}"
    hs = np.array([h[label] for h in bin_heights])
    rects = ax.bar(
        x + (i + 1 - len(VERSIONS) // 2 - 1 / 2) * width,
        hs,
        width,
        label=label,
    )
    ax.bar_label(
        rects, labels=[f"{acc:.2f}" for acc in hs / np.array(n_per_cat)], padding=2
    )

ax.set_title("acc per question cat for each model")
ax.set_xticks(x)
# found by hand
ax.set_xticklabels(["query_shape", "query_material", "query_color", "query_size"])
ax.legend()
plt.savefig("question_cat.png")


#########################
# histogram for each answer
fig, ax = plt.subplots(figsize=(15, 10))
answers = datamodule.clevr_val.vocab.answers[:15]
bin_heights = [[0] * len(answers) for _ in range(len(VERSIONS))]
n_per_answer = [0] * len(answers)
for i, q in enumerate(datamodule.clevr_val.questions):
    ans = q["answer"]
    idx = answers.index(ans)
    n_per_answer[idx] += 1
    for j, v in enumerate(VERSIONS):
        bin_heights[j][idx] += gold[i] == results[f"version_{v}"][i]

x = np.arange(len(answers))
for i, (hs, v) in enumerate(zip(bin_heights, VERSIONS)):
    label = f"version_{v}"
    rects = ax.bar(
        x + (i + 1 - len(VERSIONS) // 2 - 1 / 2) * width, hs, width, label=label
    )
    ax.bar_label(
        rects,
        labels=[f"{acc:.1f}" for acc in np.array(hs) / np.array(n_per_answer)],
        padding=2,
        fontsize="xx-small",
        # rotation="vertical",
    )

ax.set_title("acc per answer for each model")
ax.set_xticks(x)
ax.set_xticklabels(answers)
ax.legend()
plt.savefig("answers.png")

###################################
# histrogram for each question length (number of properties used to select
# the node which will be queried for some other property)
def n_filters(question):
    return len(
        [step for step in question["program"] if step["function"].startswith("filter")]
    )

q_lens = sorted({n_filters(q) for q in datamodule.clevr_val.questions})
bin_heights = [[0] * len(q_lens) for _ in VERSIONS]
n_per_len = [0] * len(q_lens)
for i, q in enumerate(datamodule.clevr_val.questions):
    q_len = n_filters(q)
    idx = q_lens.index(q_len)
    n_per_len[idx] += 1
    for j, v in enumerate(VERSIONS):
        bin_heights[j][idx] += gold[i] == results[f"version_{v}"][i]

fig, ax = plt.subplots(figsize=(15, 9))
x = np.arange(len(q_lens))
for i, (hs, v) in enumerate(zip(bin_heights, VERSIONS)):
    label = f"version_{v}"
    rects = ax.bar(x + (i + 1 - len(VERSIONS)//2 - 1/2) * width, hs, width, label=label)
    ax.bar_label(rects, labels = [f"{acc:.2f}" for acc in np.array(hs) / np.array(n_per_len)], padding=2,)

ax.set_xticks(x)
ax.set_xticklabels(q_lens)
ax.set_title("acc per number of filters in question")
ax.legend()
plt.savefig("n_filters.png")


