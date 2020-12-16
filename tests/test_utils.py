import unittest
import random
from itertools import islice
import torch
import warnings
from nsm.utils import (
    Graph,
    infinite_graphs,
    is_connected,
    collate_graphs,
    matmul_memcapped,
    Batch,
)

warnings.filterwarnings("ignore", category=UserWarning)


class InfiniteGraphs(unittest.TestCase):
    def setUp(self) -> None:
        self.hidden_size = 300
        self.n_properties = 77
        self.node_distribution = (16.4, 8.2)
        self.density_distribution = (0.2, 0.25)
        self.graph_gen = infinite_graphs(
            self.hidden_size,
            self.n_properties,
            self.node_distribution,
            self.density_distribution,
        )

    def test_return_type(self) -> None:
        g = next(self.graph_gen)
        self.assertIsInstance(g, Graph)

    def test_graph_tensors_ndims(self) -> None:
        g = next(self.graph_gen)
        self.assertEqual(len(g.node_attrs.size()), 3)
        self.assertEqual(len(g.edge_indices.size()), 2)
        self.assertEqual(len(g.edge_attrs.size()), 2)

    def test_nodes_shape(self) -> None:
        g = next(self.graph_gen)
        _, n_properties, hidden_size = g.node_attrs.size()
        self.assertEqual(n_properties, self.n_properties)
        self.assertEqual(hidden_size, self.hidden_size)

    def test_edge_indices_shape(self) -> None:
        g = next(self.graph_gen)
        should_be_2, *rest = g.edge_indices.size()
        self.assertEqual(should_be_2, 2)

    def test_edge_attrs_shape(self) -> None:
        g = next(self.graph_gen)
        _, hidden_size = g.edge_attrs.size()
        self.assertEqual(hidden_size, self.hidden_size)

    def test_edge_tensors_dim_match(self) -> None:
        g = next(self.graph_gen)
        self.assertEqual(g.edge_indices.size(1), g.edge_attrs.size(0))

    def test_valid_index(self) -> None:
        g = next(self.graph_gen)
        n_nodes = g.node_attrs.size(0)
        self.assertTrue(g.edge_indices.lt(n_nodes).all())

    def test_generate_many(self) -> None:
        n = 20
        for _ in islice(self.graph_gen, n):
            pass


class CheckConnectivity(unittest.TestCase):
    def test_no_edges_connected(self) -> None:
        edges: list = []
        n_nodes = 1
        self.assertTrue(is_connected(edges, n_nodes))

    def test_no_edges_disconnected(self) -> None:
        edges: list = []
        n_nodes = 2
        self.assertFalse(is_connected(edges, n_nodes))

    def test_simple_connected(self) -> None:
        n_nodes = 4
        edges = [(0, 1), (1, 2), (2, 3)]
        self.assertTrue(is_connected(edges, n_nodes))

    def test_simple_disconnected_1(self) -> None:
        edges = [(0, 1), (1, 2), (2, 3)]
        n_nodes = 5
        self.assertFalse(is_connected(edges, n_nodes))

    def test_simple_disconnected_2(self) -> None:
        edges = [(0, 1), (1, 2), (3, 4)]
        n_nodes = 5
        self.assertFalse(is_connected(edges, n_nodes))

    def test_raises_not_enough_nodes(self) -> None:
        edges = [(0, 1), (1, 2), (3, 4)]
        n_nodes = 4
        with self.assertRaises(ValueError):
            self.assertFalse(is_connected(edges, n_nodes))

    def test_non_integer_values_connected(self) -> None:
        edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
        n_nodes = 4
        self.assertTrue(is_connected(edges, n_nodes))

    def test_connected(self) -> None:
        edges = [
            ("a", "b"),
            ("b", "c"),
            ("c", "e"),
            ("e", "d"),
            ("d", "b"),
            ("e", "f"),
        ]
        n_nodes = 6
        self.assertTrue(is_connected(edges, n_nodes))

    def test_disconnected(self) -> None:
        edges = [
            ("a", "g"),
            ("g", "b"),
            ("g", "c"),
            ("d", "h"),
            ("e", "h"),
            ("h", "f"),
        ]
        n_nodes = 8
        self.assertFalse(is_connected(edges, n_nodes))


class CollateGraphs(unittest.TestCase):
    def setUp(self) -> None:
        self.hidden_size = 300
        self.n_properties = 77
        self.graph_gen = infinite_graphs(
            self.hidden_size,
            self.n_properties,
            node_distribution=(16.4, 8.2),
            density_distribution=(0.2, 0.4),
        )

    def test_output_type(self) -> None:
        batch = collate_graphs(list(islice(self.graph_gen, 8)))
        self.assertIsInstance(batch, Batch)

    def test_output_shapes(self) -> None:
        batch_size = 8
        graphs = list(islice(self.graph_gen, batch_size))
        batch = collate_graphs(graphs)
        with self.subTest("node_attrs"):
            total_nodes = sum(g.node_attrs.size(0) for g in graphs)
            self.assertEqual(
                batch.node_attrs.size(),
                (total_nodes, self.n_properties, self.hidden_size),
            )
        with self.subTest("edges"):
            total_edges = sum(g.edge_attrs.size(0) for g in graphs)
            with self.subTest("edge_attrs"):
                self.assertEqual(
                    batch.edge_attrs.size(), (total_edges, self.hidden_size)
                )
            with self.subTest("edge_indices"):
                self.assertEqual(batch.edge_indices.size(), (2, total_edges))
        with self.subTest("nodes_per_graph"):
            self.assertEqual(batch.nodes_per_graph.size(0), batch_size)
        # with self.subTest("edges_per_graph"):
        #    self.assertEqual(batch.edges_per_graph.size(0), batch_size)

    def test_nodes_per_graph(self) -> None:
        graphs = list(islice(self.graph_gen, 8))
        nodes_per_graph = [g.node_attrs.size(0) for g in graphs]
        batch = collate_graphs(graphs)
        self.assertEqual(batch.nodes_per_graph.tolist(), nodes_per_graph)
        self.assertEqual(batch.node_attrs.size(0), sum(nodes_per_graph))

    def test_edge_indices_shift_edge_attrs(self) -> None:
        graphs = list(islice(self.graph_gen, 8))
        batch = collate_graphs(graphs)
        for _ in range(5):
            # Get 5 random nodes and test that their adjacency list match
            g_idx = random.randrange(len(graphs))
            graph = graphs[g_idx]
            node_idx = random.randrange(graph.node_attrs.size(0))
            # breakpoint()
            correct_edge_attrs = graph.edge_attrs[
                graph.edge_indices[0] == node_idx
            ]
            new_node_idx = (
                sum(batch.nodes_per_graph.tolist()[:g_idx]) + node_idx
            )
            new_edge_attrs = batch.edge_attrs[
                batch.edge_indices[0] == new_node_idx
            ]
            # breakpoint()
            self.assertEqual(new_edge_attrs.size(), correct_edge_attrs.size())
            self.assertTrue(new_edge_attrs.eq(correct_edge_attrs).all())

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_collate_cuda(self) -> None:
        graphs = list(islice(self.graph_gen, 8))
        batch = collate_graphs(graphs, device="cuda")
        for key, val in vars(batch).items():
            with self.subTest(attr=key):
                self.assertTrue(val.is_cuda)


class BatchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        graph_gen = infinite_graphs(
            300,
            77,
            node_distribution=(16.4, 8.2),
            density_distribution=(0.2, 0.4),
        )
        self.batch = collate_graphs(list(islice(graph_gen, 8)))

    def test_node_indices_shape(self) -> None:
        self.assertEqual(
            self.batch.node_indices.size(), (self.batch.node_attrs.size(0),)
        )

    def test_sparse_coo_indices(self) -> None:
        self.assertEqual(
            self.batch.sparse_coo_indices.size(),
            (2, self.batch.node_attrs.size(0)),
        )

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_properties_cuda(self) -> None:
        batch = Batch(*map(torch.Tensor.cuda, vars(self.batch).values()))
        props = ["node_indices", "sparse_coo_indices"]
        for prop in props:
            with self.subTest(property=prop):
                self.assertTrue(getattr(batch, prop).is_cuda)

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_to(self) -> None:
        batch = self.batch.to("cuda")
        self.assertTrue(all(t.is_cuda for t in vars(batch).values()))


class MatmulMemcapped(unittest.TestCase):
    def setUp(self) -> None:
        n_props = 10
        hidden = 50
        n_nodes = 100
        self.param = torch.rand(n_props, hidden, hidden)
        self.nodes = torch.rand(n_nodes, n_props, hidden, 1)

    def test_output_matches_needs_chunking(self) -> None:
        out = matmul_memcapped(self.param, self.nodes, memory_cap=10 ** 6)
        self.assertTrue(self.param.matmul(self.nodes).eq(out).all())

    def test_output_matches_no_chunking(self) -> None:
        out = matmul_memcapped(self.param, self.nodes, memory_cap=10 ** 9)
        self.assertTrue(self.param.matmul(self.nodes).eq(out).all())
