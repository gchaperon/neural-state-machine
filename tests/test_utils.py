import unittest
from itertools import islice

from nsm.utils import Graph, infinite_graphs, is_connected


class InfiniteGraphs(unittest.TestCase):
    def setUp(self) -> None:
        self.hidden_size = 300
        self.n_properties = 77
        self.node_distribution = (16.4, 8.2)
        self.density_distribution = (50.6, 56.2)
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
        self.assertEqual(len(g.edge_index.size()), 2)
        self.assertEqual(len(g.edge_attrs.size()), 2)

    def test_nodes_shape(self) -> None:
        g = next(self.graph_gen)
        _, n_properties, hidden_size = g.node_attrs.size()
        self.assertEqual(n_properties, self.n_properties)
        self.assertEqual(hidden_size, self.hidden_size)

    def test_edge_index_shape(self) -> None:
        g = next(self.graph_gen)
        should_be_2, *rest = g.edge_index.size()
        self.assertEqual(should_be_2, 2)

    def test_edge_attrs_shape(self) -> None:
        g = next(self.graph_gen)
        _, hidden_size = g.edge_attrs.size()
        self.assertEqual(hidden_size, self.hidden_size)

    def test_edge_tensors_dim_match(self) -> None:
        g = next(self.graph_gen)
        self.assertEqual(g.edge_index.size(1), g.edge_attrs.size(0))

    def test_valid_index(self) -> None:
        g = next(self.graph_gen)
        n_nodes = g.node_attrs.size(0)
        self.assertTrue(g.edge_index.lt(n_nodes).all())

    def test_generate_many(self) -> None:
        n = 10
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
