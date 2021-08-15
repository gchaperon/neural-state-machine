#! /usr/bin/env python
import pytorch_lightning as pl
import cmd
import argparse
import random
from pathlib import Path
import json
from pprint import pp
import readline
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import networkx as nx
from itertools import cycle
import numpy as np
from .syn_checkpoint import run as run_syn_checkpoint
from .clevr_checkpoint import run as run_clevr_checkpoint


def draw_sg(sg):
    node_labels = {}
    edges = []
    edge_labels = {}

    G = nx.MultiDiGraph()
    for o_key, obj in sg["objects"].items():
        G.add_node(o_key)
        node_labels[o_key] = obj["name"]
        for rel in obj["relations"]:
            edges.append((o_key, rel["object"]))
            edge_labels[(o_key, rel["object"])] = rel["name"]
    G.add_edges_from(edges)

    # pos = nx.spring_layout(G, k=1)
    try:
        pos = nx.planar_layout(G)
    except nx.NetworkXException:
        pos = nx.spring_layout(G, k=1)
    plt.subplot(122)
    nx.draw(
        G,
        pos=pos,
        # TODO: use actual attention values instead of random
        node_color=np.ones(len(G.nodes)) / len(G.nodes),
        cmap=plt.cm.Blues,
        labels=node_labels,
        node_size=400,
        font_size=10,
    )
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=8)
    ax = plt.gca()
    plt.colorbar(plt.cm.ScalarMappable(norm=ax.collections[0].norm, cmap=ax.collections[0].cmap))


def draw_image(image, sg):
    plt.subplot(121)
    plt.imshow(image)
    for obj, color in zip(sg["objects"].values(), cycle(mcolors.TABLEAU_COLORS.keys())):
        x, y = obj["x"], obj["y"]
        bbox = patches.Rectangle(
            (x, y), obj["w"], obj["h"], linewidth=1, edgecolor=color, facecolor="none"
        )
        plt.gca().add_patch(bbox)
        plt.text(
            x,
            y,
            obj["name"],
            color="white",
            backgroundcolor="black",
            size="xx-small",
            clip_on=True,
        )


class SceneGraphShell(cmd.Cmd):
    def __init__(self, datadir, *args, **kwargs):
        super(SceneGraphShell, self).__init__(*args, **kwargs)
        self.gqaroot = Path(datadir) / "GQA"
        self.imagesroot = self.gqaroot / "images"

    def preloop(self):
        sgsroot = self.gqaroot / "sceneGraphs"
        print(f"Loading all scene graphs in {sgsroot}")
        self.scenegraphs = {}
        for jsonpath in sgsroot.glob("*.json"):
            with open(jsonpath) as jsonfile:
                self.scenegraphs.update(json.load(jsonfile))
        self.scenegraph_keys = list(self.scenegraphs.keys())

        print("Loading train and val questions")
        self.questions = {}
        # TODO: change lines!
        # for split in ["train", "val"]:
        for split in ["val"]:
            with open(
                self.gqaroot / "questions" / f"{split}_balanced_questions.json"
            ) as jsonfile:
                self.questions.update(json.load(jsonfile))
        self.question_keys = list(self.questions.keys())

        print("Done!")

        plt.ion()
        # mpl.rcParams["figure.figsize"] = (18, 10)

    def precmd(self, line):
        "Clear figure before each command to prepare por the new graph to show"
        plt.clf()
        return line

    def do_random_sg(self, _):
        self.do_sg(random.choice(self.scenegraph_keys))

    def do_sg(self, key):
        try:
            sg = self.scenegraphs[key]
        except KeyError:
            print("Invalid key!")
            return
        plt.suptitle(f"id = {key}")
        draw_sg(sg)
        image = Image.open(self.imagesroot / f"{key}.jpg")
        draw_image(image, sg)
        pp(sg)

    def do_random_q(self, _):
        self.do_question(random.choice(self.question_keys))

    def do_question(self, key):
        try:
            question = self.questions[key]
        except KeyError:
            print("Invalid key!")
            return
        plt.suptitle(f"Q: {question['question']}\nR: {question['answer']}")
        image_key = question["imageId"]
        sg = self.scenegraphs[image_key]
        draw_sg(sg)
        draw_image(Image.open(self.imagesroot / f"{image_key}.jpg"), sg)
        pp(question)
        pp(sg)

    def do_EOF(self, _):
        print()
        return True

    def do_exit(self, _):
        return True

def run_scenegraphs(args):
    pl.seed_everything(seed=args.seed)
    SceneGraphShell(args.data_dir).cmdloop()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=123)
    subparsers = parser.add_subparsers()
    parser_scenegraphs = subparsers.add_parser("scenegraphs")
    parser_scenegraphs.set_defaults(func=run_scenegraphs)
    parser_scenegraphs.add_argument("--data-dir", default="./data")
    
    parser_checkpoint = subparsers.add_parser("checkpoint")
    checkpoint_subparsers = parser_checkpoint.add_subparsers()
    
    parser_checkpoint.add_argument("--path", dest="checkpoint_path", required=True)
    parser_syn = checkpoint_subparsers.add_parser("syn")
    parser_syn.set_defaults(func=run_syn_checkpoint)
    parser_syn.add_argument("--n-instructions", required=True, type=int)
    
    parser_clevr = checkpoint_subparsers.add_parser("clevr")
    parser_clevr.set_defaults(func=run_clevr_checkpoint)
    parser_clevr.add_argument("--max-examples", type=int, default=None)


    args = parser.parse_args()
    args.func(args)


