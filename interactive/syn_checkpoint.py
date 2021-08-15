import numpy as np
import pytorch_lightning as pl
from nsm.model import NSM
from nsm.datasets import synthetic
from nsm.utils import split_batch, Graph
import torch
from torch.nn.utils.rnn import pad_packed_sequence
import cmd
from functools import partial
import typing as tp
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from collections import defaultdict, UserList


def _removeprefix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix) :]
    else:
        return s


def load_model(path: str, n_instructions: int) -> NSM:
    checkpoint = torch.load(path)
    state_dict = checkpoint["state_dict"]
    state_dict = {
        _removeprefix(key, "nsm."): value for key, value in state_dict.items()
    }
    n_node_properties, input_size, _ = state_dict[
        "nsm_cell.weight_node_properties"
    ].shape
    output_size = state_dict["classifier.fc_layers.1.weight"].shape[0]
    model = NSM(
        input_size=input_size,
        n_node_properties=n_node_properties,
        computation_steps=n_instructions - 1,
        output_size=output_size,
        instruction_model="dummy",
    )
    model.load_state_dict(state_dict, strict=True)
    return model


def dataloader_for_model(model):

    n_concepts = model.classifier.fc_layers[1].weight.size(0)
    n_relations = int(model.classifier.fc_layers[1].weight.size(1) / 2 - n_concepts)
    question_length = model.instructions_model.n_instructions
    datamodule = synthetic.SyntheticDataModule(
        synthetic.SyntheticDataset(
            synthetic.Concepts(
                n_concepts,
                n_relations,
            ),
            n_nodes=10,
            n_edges=15,
            size=100,
            jump_prob=0.0,
            question_length=question_length,
        ),
        batch_size=1,
        split_ratio=0.0,
    )
    return datamodule.val_dataloader()


def graph_to_networkx(graph: Graph):
    G = nx.MultiDiGraph()
    for node, attrs in enumerate(graph.node_attrs.argmax(2).tolist()):
        G.add_node(node, value=attrs[0])
    for edge, attr in zip(
        graph.edge_indices.T.tolist(), graph.edge_attrs.argmax(1).tolist()
    ):
        G.add_edge(*edge, value=attr)

    return G


class ExtractOutputHook(UserList):
    def __call__(self, module, input, output):
        self.data.append(output.tolist())


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


class GraphShell(QuitCmd):
    prompt = "(graph) "

    def __init__(
        self, graph: Graph, weights: tp.List[list], instructions: torch.Tensor
    ):
        super().__init__()
        self.graph = graph_to_networkx(graph)
        self.weights = weights
        self.instructions = instructions.squeeze().argmax(1).tolist()[:-1]
        self.current = 0
        self._drawing_position = nx.spring_layout(self.graph, k=1)

    def preloop(self):
        print(self.instructions)
        self._draw()

    def postloop(self):
        plt.clf()

    def _draw(self):
        title = f"Instruction (index, value): ({self.current}, {self.instructions[self.current]})"
        # print(title)
        plt.clf()
        plt.title(title)
        node_labels = {n: e["value"] for n, e in self.graph.nodes.items()}
        edge_labels = defaultdict(list)
        for u, v, attr in self.graph.edges(data="value"):
            edge_labels[(u, v)].append(attr)

        nx.draw(
            self.graph,
            pos=self._drawing_position,
            node_color=self.weights[self.current],
            node_size=400,
            font_size=10,
            labels=node_labels,
            cmap=plt.cm.Oranges,
        )
        nx.draw_networkx_edge_labels(
            self.graph, pos=self._drawing_position, edge_labels=edge_labels, font_size=8
        )
        ax = plt.gca()
        plt.colorbar(
            plt.cm.ScalarMappable(
                norm=ax.collections[0].norm, cmap=ax.collections[0].cmap
            )
        )

    def do_next(self, _):
        if self.current == len(self.instructions) - 1:
            print("No more instructions!")
            return
        self.current = self.current + 1
        self._draw()

    def do_prev(self, _):
        if self.current == 0:
            print("No previous instructions!")
            return
        self.current = self.current - 1
        self._draw()

    do_n = do_next
    do_p = do_prev


class DataShell(QuitCmd):
    prompt = "(data) "

    def __init__(self, dataloader, model):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.iter = iter(dataloader)

    def do_next(self, _):
        example = next(self.iter)

        probs = ExtractOutputHook()
        handle = self.model.nsm_cell.register_forward_hook(probs)
        self.model(*example[:-1])
        handle.remove()

        batch, instructions, _, __, target = example
        graph = split_batch(batch)[0]
        instructions = pad_packed_sequence(instructions)[0].squeeze()
        GraphShell(graph, probs, instructions).cmdloop()


def run(args):
    pl.seed_everything(seed=args.seed)
    mpl.rcParams["figure.figsize"] = (18, 10)
    model = load_model(args.checkpoint_path, args.n_instructions)
    dataloader = dataloader_for_model(model)

    plt.ion()
    DataShell(dataloader, model).cmdloop()
