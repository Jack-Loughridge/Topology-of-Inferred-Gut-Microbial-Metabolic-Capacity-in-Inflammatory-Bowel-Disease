from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "canonicalize_graph_inputs.py"
SPEC = importlib.util.spec_from_file_location("canonicalize_graph_inputs", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_name_rules() -> None:
    assert MODULE.first_two_tokens("M_Escherichia_coli_MP021552_7", remove_model_prefix=True) == "Escherichia_coli"
    assert MODULE.selected_abundance_column("k__Bacteria|g__Foo|s__Foo_bar_CAG_1") == (
        True,
        "s",
        "Foo_bar",
    )
    assert MODULE.selected_abundance_column("k__Bacteria|g__Family_unclassified") == (
        True,
        "g",
        "Family_unclassified",
    )
    assert MODULE.selected_abundance_column("k__Bacteria|g__Bacteroides") == (
        False,
        "g",
        "",
    )


def test_species_canonicalization_selection_aggregation_and_order(tmp_path: Path) -> None:
    source_path = tmp_path / "source.xlsx"
    output_path = tmp_path / "canonical.xlsx"
    source = pd.DataFrame(
        {
            "External ID": ["sample-b", "sample-a"],
            "k__Bacteria": [100.0, 100.0],
            "k__Bacteria|g__Alpha|s__Alpha_beta_strain_1": [1.0, 2.0],
            "k__Bacteria|g__Alpha|s__Alpha_beta_strain_2": [3.0, 4.0],
            "k__Bacteria|g__Unresolved_group": [5.0, 6.0],
            "k__Bacteria|g__Single": [50.0, 60.0],
            "k__Bacteria|f__Family_unclassified": [70.0, 80.0],
            "k__Bacteria|g__Candidatus_Stoquefichus": [7.0, 8.0],
            "k__Bacteria|g__Candidatus_Stoquefichus|s__Candidatus_Stoquefichus_strain": [9.0, 10.0],
            "k__Bacteria|g__Gamma|s__Gamma_delta": [11.0, 12.0],
        }
    )
    source.to_excel(source_path, index=False)

    record = MODULE.canonicalize_species(source_path, output_path)
    observed = pd.read_excel(output_path)
    assert list(observed.columns) == [
        "External ID",
        "Alpha_beta",
        "Candidatus_Stoquefichus",
        "Gamma_delta",
        "Unresolved_group",
    ]
    assert observed["External ID"].tolist() == ["sample-b", "sample-a"]
    np.testing.assert_allclose(observed["Alpha_beta"], [4.0, 6.0])
    np.testing.assert_allclose(observed["Candidatus_Stoquefichus"], [16.0, 18.0])
    np.testing.assert_allclose(observed["Gamma_delta"], [11.0, 12.0])
    np.testing.assert_allclose(observed["Unresolved_group"], [5.0, 6.0])
    assert record["counts"]["selected_source_columns"] == 6
    assert record["counts"]["output_columns"] == 5


def test_reaction_canonicalization_changes_only_catalysts(tmp_path: Path) -> None:
    source_path = tmp_path / "source.parquet"
    output_path = tmp_path / "canonical.parquet"
    source = pd.DataFrame(
        {
            "model_id": ["M_Escherichia_coli_A", "M_Bacillus_cereus_B", "M_Escherichia_coli_C"],
            "species_file": ["a.xml", "b.xml", "c.xml"],
            "reaction_id": ["r1", "r2", "r3"],
            "inputs": ["A", "B", "C"],
            "outputs": ["D", "E", "F"],
            "catalysts": ["M_Escherichia_coli_A", "M_Bacillus_cereus_B", "M_Escherichia_coli_C"],
        }
    )
    source.to_parquet(source_path, index=False)

    record = MODULE.canonicalize_reactions(source_path, output_path, batch_size=2)
    observed = pd.read_parquet(output_path)
    pd.testing.assert_frame_equal(
        observed.drop(columns="catalysts"), source.drop(columns="catalysts")
    )
    assert observed["catalysts"].tolist() == [
        "Escherichia_coli",
        "Bacillus_cereus",
        "Escherichia_coli",
    ]
    assert record["counts"] == {
        "rows": 3,
        "columns": 6,
        "unique_model_ids": 3,
        "unique_canonical_catalysts": 2,
    }


def test_reaction_source_invariant_is_enforced(tmp_path: Path) -> None:
    source_path = tmp_path / "bad.parquet"
    output_path = tmp_path / "canonical.parquet"
    pd.DataFrame(
        {
            "model_id": ["M_Foo_bar_A"],
            "species_file": ["a.xml"],
            "reaction_id": ["r1"],
            "inputs": ["A"],
            "outputs": ["B"],
            "catalysts": ["different"],
        }
    ).to_parquet(source_path, index=False)
    with pytest.raises(ValueError, match="catalysts differs from model_id"):
        MODULE.canonicalize_reactions(source_path, output_path)


def test_production_hash_guard_rejects_unknown_input(tmp_path: Path) -> None:
    source_path = tmp_path / "source.xlsx"
    output_path = tmp_path / "canonical.xlsx"
    pd.DataFrame({"External ID": ["x"], "k__Bacteria|g__Foo|s__Foo_bar": [1.0]}).to_excel(
        source_path, index=False
    )
    with pytest.raises(ValueError, match="input SHA-256 mismatch"):
        MODULE.canonicalize_species(
            source_path,
            output_path,
            require_production_hash=True,
        )
