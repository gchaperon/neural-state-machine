import networkx as nx
from nsm.model import NSMLightningModule
from nsm.datasets.clevr import ClevrNoImagesDataModule
from nsm.utils import collate_nsmitems
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from itertools import islice, product
from collections import UserList, defaultdict
import cmd
import random
import logging
import textwrap
import argparse


class QuitCmd(cmd.Cmd):
    def do_EOF(self, _):
        """Quit"""
        print()
        return True

    def do_quit(self, _):
        """Quit"""
        return True

    def do_q(self, _):
        """Alias for quit"""
        return True


class DataShell(QuitCmd):
    prompt = "(data) "

    def __init__(self, model, dataset):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.hist = []

    def do_random(self, _):
        i = random.randrange(len(self.dataset))
        self.do_show(i)

    def do_last(self, _):
        if self.hist:
            self.do_show(self.hist[-1])
        else:
            print("error: none shown yet")

    def do_hist(self, arg):
        """Print arg last shown questions, if arg is empty show all"""
        n = int(arg or 0)
        for i in self.hist[-n:]:
            print(i)

    def do_show(self, arg):
        i = int(arg)
        if not 0 <= i < len(self.dataset):
            print("index out of range")
            return

        example = self.dataset.get_raw(i)

        instructions_hook = ExtractOutputHook()
        node_probs = ExtractProbsHook()

        handles = [
            module.register_forward_hook(hook)
            for module, hook in zip(
                [self.model.nsm.instructions_model, self.model.nsm.nsm_cell],
                [instructions_hook, node_probs],
            )
        ]

        *inputs, target = make_batch(example, self.dataset)
        output = self.model(*inputs)
        # breakpoint()
        instructions, encoded_questions = instructions_hook.value
        # breakpoint()
        GraphShell(
            index=i,
            example=example,
            predicted=self.dataset.vocab.answers[output[0].argmax().item()],
            instructions=instructions.detach().numpy()[0],
            node_probs=node_probs,
            vocab=self.dataset.vocab,
        ).cmdloop()
        self.hist.append(i)

    def postloop(self):
        plt.close(plt.figure(1))
        plt.close(plt.figure(2))


class GraphShell(QuitCmd):
    def __init__(self, index, example, predicted, instructions, node_probs, vocab):
        super().__init__()
        self.prompt = f"(question {index}) "
        graph, self.question, self.target = example
        self.graph = graph_to_networkx(graph)
        self.predicted = predicted
        self.instructions = instructions
        self.node_probs = node_probs
        self.vocab = vocab
        self.current = 0
        self._drawing_position = nx.circular_layout(self.graph)

        assert len(node_probs) == len(instructions) - 1

        self.question_figure = plt.figure(1, figsize=(13.6, 7.6))
        self.graph_figure = plt.figure(2)

    def preloop(self):
        self._draw()

    def postloop(self):
        self.question_figure.clf()
        self.graph_figure.clf()

    def _draw_graph(self):
        self.graph_figure.clf()
        ax = self.graph_figure.gca()

        node_labels = {n: "\n".join(e["attrs"]) for n, e in self.graph.nodes.items()}
        for i, val in enumerate(self.node_probs[self.current]):
            node_labels[i] = f"{val:.2f}\n" + node_labels[i]

        edge_labels = defaultdict(list)
        for u, v, attr in self.graph.edges(data="attr"):
            edge_labels[(u, v)].append(attr)

        nx.draw(
            self.graph,
            pos=self._drawing_position,
            node_color=self.node_probs[self.current],
            node_size=800,
            font_size=6,
            labels=node_labels,
            cmap=plt.cm.Oranges,
            ax=ax,
        )
        nx.draw_networkx_edge_labels(
            self.graph,
            pos=self._drawing_position,
            edge_labels=edge_labels,
            font_size=8,
            ax=ax,
        )

    def _draw_instructions(self):
        self.question_figure.clf()
        ax = self.question_figure.gca()

        instructions = np.log10(self.instructions)
        # instructions = np.copy(self.instructions)
        # instructions[-1] = instructions[-1] / 1e6
        im = ax.imshow(instructions)
        ax.add_patch(
            patches.Rectangle(
                xy=(-0.5, self.current - 0.5),
                width=self.instructions.shape[1],
                height=1,
                linewidth=2,
                edgecolor="r",
                fill=False,
            )
        )
        ax.set_xticks(np.arange(self.instructions.shape[1]))
        ax.set_xticklabels(
            self.vocab.everything,
            rotation=60,
            horizontalalignment="left",
            rotation_mode="anchor",
        )
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        for i, j in product(*map(range, instructions.shape)):
            val = instructions[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="w")

        ax.set_title(
            "\n".join(textwrap.wrap("  ".join(self.question)))
            + f"\nlabel: {self.target}\npredicted: {self.predicted}"
        )
        plt.colorbar(im, ax=ax, location="bottom")

    def _draw(self):
        self._draw_instructions()
        self._draw_graph()

    def do_next(self, arg):
        if self.current < len(self.node_probs) - 1:
            self.current += 1
            self._draw()
        else:
            print("Out of bounds")

    def do_prev(self, arg):
        if self.current > 0:
            self.current -= 1
            self._draw()
        else:
            print("Out of bounds")


def make_batch(example, dataset):
    """Batch of a single example to feed to model"""
    vocab = dataset.vocab
    embedded = (
        vocab.embed(example[0]),
        # dataset.instructions_from_question(example[1]),
        vocab.embed(example[1]),
        vocab.answers.index(example[2]),
    )
    graphs, questions, targets = collate_nsmitems([embedded])
    return (
        graphs,
        questions,
        vocab.concept_embeddings,
        vocab.property_embeddings,
        targets,
    )


def graph_to_networkx(graph):
    G = nx.MultiDiGraph()
    for i, node in enumerate(graph.node_attrs):
        G.add_node(i, attrs=node)

    for edge, attr in zip(graph.edge_indices, graph.edge_attrs):
        G.add_edge(*edge, attr=attr)
    return G


class ExtractProbsHook(list):
    def __call__(self, module, input, output):
        self.append(output.detach().numpy())


class ExtractOutputHook:
    def __init__(self):
        self.value = None

    def __call__(self, module, input, output):
        self.value = output


def run(args):
    """ Recordar cambiar los nhops y el w_instructions del dataset, 
    y el metodo de embedding en make_batch
    tambien cambiar la normalizacion de la ultima instruccion"""
    logging.basicConfig(level=logging.INFO)
    pl.seed_everything(args.seed)
    model = NSMLightningModule.load_from_checkpoint(args.checkpoint_path)
    # breakpoint()
    datamodule = ClevrNoImagesDataModule("data", batch_size=1, w_instructions=False, nhops=[0])
    # TODO: quiza cambiar esto a "validate"? por ahora solo estoy probando overfitear
    # asi que tiene sentido solo revisar ejemplos de train
    datamodule.setup("validate")
    dataset = datamodule.clevr_val
    example = dataset.get_raw(random.randrange(len(dataset)))

    # attn_hook = ExtractAttnHook()
    # node_probs = ExtractProbsHook()

    # handle1 = model.nsm.instructions_model.softmax.register_forward_hook(attn_hook)
    # handle2 = model.nsm.nsm_cell.register_forward_hook(node_probs)
    # out = model(*make_batch(example, dataset.vocab)[:-1])
    # handle1.remove()
    # handle2.remove()
    # q_attns = attn_hook.value.squeeze(0)  # remove first dim (batch dim)

    plt.ion()
    DataShell(model, dataset).cmdloop()
    # GraphShell(example, q_attns, node_probs).cmdloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--path", dest="checkpoint_path")
    args = parser.parse_args()

    run(args)
