import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import edge_concentration_weight_top001 as edge_concentration
import make_all_path_backbone_latex as latex_tables
import outdegree_hist_active_participant_mean_cap6_excl0 as outdegree
import path_backbone_longrange_diagnostics as path_backbone
import weight_density_curves_participant_average_clip099 as weight_density


GRAPHML = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d1" for="edge" attr.name="weight" attr.type="double"/>
  <graph edgedefault="directed">
    <node id="a"/><node id="b"/><node id="c"/>
    <edge source="a" target="b"><data key="d1">0.2</data></edge>
    <edge source="a" target="c"><data key="d1">1.0</data></edge>
    <edge source="b" target="c"><data key="d1">0.9</data></edge>
  </graph>
</graphml>
"""


class GraphTableFigureGeneratorTests(unittest.TestCase):
    def graph_path(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "sample.graphml"
        path.write_text(GRAPHML)
        return path

    def test_shared_parser_preserves_edge_data(self):
        nodes, adjacency, edge_w = path_backbone.parse_graphml_active(
            self.graph_path(), 1e-12
        )
        self.assertEqual(nodes, ["a", "b", "c"])
        self.assertEqual(adjacency["a"], ["b"])
        self.assertEqual(edge_w, {("a", "b"): 0.2, ("b", "c"): 0.9})

    def test_outdegree_uses_shared_active_edges(self):
        hist, vertices, edges = outdegree.active_outdegree_histogram(
            self.graph_path(), 1e-12
        )
        self.assertEqual((vertices, edges), (3, 2))
        self.assertEqual(hist["1"], 2)
        self.assertEqual(sum(hist.values()), 2)

    def test_weight_density_is_normalized_or_zero(self):
        values = weight_density.clipped_weights(self.graph_path(), 1e-12, 0.99)
        self.assertTrue(np.allclose(np.sort(values), [0.2, 0.9]))
        bins = np.linspace(0.0, 0.99, 200)
        nonempty = weight_density.density(values, bins)
        empty = weight_density.density(np.array([]), bins)
        self.assertAlmostEqual(float(nonempty.sum() * (bins[1] - bins[0])), 1.0)
        self.assertTrue(np.array_equal(empty, np.zeros(199)))

    def test_path_summary_records_valid_and_total_units(self):
        sample = pd.DataFrame(
            [
                {"sample_id": "a", "participant_id": "p1", "cond": "cd", "mode": "minimum_hop", "path_set": "all", "n_paths": 4, "path_length_q50": 2.0},
                {"sample_id": "b", "participant_id": "p2", "cond": "uc", "mode": "minimum_hop", "path_set": "all", "n_paths": 0, "path_length_q50": np.nan},
            ]
        )
        summary = path_backbone.condition_summary(sample, pd.DataFrame())
        overall = summary[summary["cond"].eq("overall")].iloc[0]
        self.assertEqual(int(overall["n_total"]), 2)
        self.assertEqual(int(overall["n_valid"]), 1)
        self.assertEqual(latex_tables.fmt_n(overall), "1/2")

    def test_edge_generator_imports_repository_path_helpers(self):
        self.assertIs(edge_concentration.pb.parse_graphml_active, path_backbone.parse_graphml_active)


if __name__ == "__main__":
    unittest.main()
