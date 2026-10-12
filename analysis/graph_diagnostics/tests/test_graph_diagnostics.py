from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


clustering = load_module("publication_directed_clustering", "directed_clustering_participant_average.py")
sequential = load_module("publication_sequential_attack", "sequential_routing_attack.py")


def write_graphml(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="edge_weight" for="edge" attr.name="weight" attr.type="double"/>
  <graph edgedefault="directed">
    <node id="a"/><node id="b"/><node id="c"/>
    <edge source="a" target="b"><data key="edge_weight">0.4</data></edge>
    <edge source="a" target="b"><data key="edge_weight">0.2</data></edge>
    <edge source="b" target="c"><data key="edge_weight">1.0</data></edge>
    <edge source="c" target="c"><data key="edge_weight">0.1</data></edge>
  </graph>
</graphml>
""",
        encoding="utf-8",
    )


class GraphDiagnosticsTests(unittest.TestCase):
    def test_clustering_parser_preserves_data_children_and_deduplicates(self):
        with self.subTest("temporary GraphML"):
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                graph = Path(directory) / "test.graphml"
                write_graphml(graph)
                strengths = clustering.parse_active_edge_strengths(graph, active_tol=1e-12)
        self.assertEqual(set(strengths), {("a", "b")})
        self.assertAlmostEqual(strengths[("a", "b")], 0.8)

    def test_sequential_parser_is_self_contained_and_reads_weight_alias(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            graph = Path(directory) / "test.graphml"
            write_graphml(graph)
            nodes, adjacency, edge_weights = sequential.parse_active_graphml(graph, active_tol=1e-12)
        self.assertEqual(nodes, ["a", "b", "c"])
        # Sequential routing retains self-loops exactly as the recovered production parser did.
        self.assertEqual(adjacency["a"], ["b"])
        self.assertAlmostEqual(edge_weights[("a", "b")], 0.2)
        self.assertAlmostEqual(edge_weights[("c", "c")], 0.1)

    def test_complete_bidirected_triangle_has_unit_clustering(self):
        strengths = {
            (u, v): 1.0
            for u in ("a", "b", "c")
            for v in ("a", "b", "c")
            if u != v
        }
        result = clustering.exact_directed_clustering(strengths)
        self.assertAlmostEqual(result["directed_clustering"], 1.0)
        self.assertAlmostEqual(result["weighted_directed_clustering"], 1.0)

    def test_weighted_coefficient_uses_raw_strength_without_max_normalization(self):
        strengths = {
            (u, v): 0.125
            for u in ("a", "b", "c")
            for v in ("a", "b", "c")
            if u != v
        }
        result = clustering.exact_directed_clustering(strengths)
        self.assertAlmostEqual(result["directed_clustering"], 1.0)
        self.assertAlmostEqual(result["weighted_directed_clustering"], 0.125)

    def test_directed_three_cycle_has_half_clustering(self):
        strengths = {("a", "b"): 1.0, ("b", "c"): 1.0, ("c", "a"): 1.0}
        result = clustering.exact_directed_clustering(strengths)
        self.assertAlmostEqual(result["directed_clustering"], 0.5)
        self.assertAlmostEqual(result["weighted_directed_clustering"], 0.5)

    def test_attack_edge_uses_baseline_frequency_then_lexicographic_tie_break(self):
        path = ["a", "b", "c", "d"]
        counts = sequential.Counter({("a", "b"): 3, ("b", "c"): 7, ("c", "d"): 7})
        self.assertEqual(sequential.choose_routing_attack_edge(path, counts), ("c", "d"))


if __name__ == "__main__":
    unittest.main()
