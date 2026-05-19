"""Regression tests for leak-safe feature engineering.

The most important test in this suite is
:func:`test_neighbor_pop_uses_only_train_nodes`: it constructs a graph where
the test node's neighbour-popularity feature would change if test-node
popularity ever leaked into the computation. If that test fails, the leakage
that originally produced R²≈1.0 is back, and the regression suite halts.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.features import (
    build_feature_table,
    compute_neighbor_popularity,
    compute_structural_features,
    split_nodes,
)


@pytest.fixture
def small_graph() -> nx.Graph:
    # A-B-C-D chain plus isolated E
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    g.add_node("E")
    nx.set_node_attributes(
        g,
        {"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0, "E": 99.0},
        name="popularity",
    )
    nx.set_node_attributes(
        g,
        {n: "pop" for n in g.nodes()},
        name="top_genre",
    )
    return g


def test_split_nodes_deterministic():
    nodes = [f"n{i}" for i in range(20)]
    a_train, a_test = split_nodes(nodes, test_size=0.25, random_state=42)
    b_train, b_test = split_nodes(nodes, test_size=0.25, random_state=42)
    assert a_train == b_train
    assert a_test == b_test


def test_split_nodes_disjoint_and_complete():
    nodes = [f"n{i}" for i in range(20)]
    train, test = split_nodes(nodes, test_size=0.25, random_state=42)
    assert set(train).isdisjoint(set(test))
    assert set(train) | set(test) == set(nodes)
    assert len(test) == 5


def test_neighbor_pop_isolated_node_returns_nan(small_graph):
    train = ["A", "B", "C", "D"]  # all connected nodes
    feats = compute_neighbor_popularity(small_graph, train_nodes=train)
    assert math.isnan(feats["E"])  # E has no neighbours at all


def test_neighbor_pop_uses_only_train_nodes(small_graph):
    """The leakage regression test.

    Holding out node C, we check that C's avg_neighbor_popularity is computed
    over its train neighbours (B and D — both held in train). Then we mutate
    C's own popularity and recompute: C's neighbour feature must NOT change,
    because C itself is in test and its popularity must not leak.
    """
    train = ["A", "B", "D"]  # held out C
    feats_before = compute_neighbor_popularity(small_graph, train_nodes=train)
    # B and D are both train nodes; their mean popularity = (20 + 40) / 2 = 30.
    assert feats_before["C"] == pytest.approx(30.0)

    # Mutate C's own popularity. If leakage exists, feats_after['C'] will move.
    g2 = small_graph.copy()
    nx.set_node_attributes(g2, {"C": 999.0}, name="popularity")
    feats_after = compute_neighbor_popularity(g2, train_nodes=train)
    assert feats_after["C"] == pytest.approx(30.0), (
        "Leakage detected: test-node popularity is flowing into its own "
        "avg_neighbor_popularity feature."
    )


def test_neighbor_pop_excludes_test_neighbours(small_graph):
    """A train node whose only neighbour is in test must get NaN."""
    train = ["A", "C", "E"]  # B is in test; A's only neighbour B is excluded
    feats = compute_neighbor_popularity(small_graph, train_nodes=train)
    assert math.isnan(feats["A"])

    # C has neighbours B (test) and D (test), so C also gets NaN here
    assert math.isnan(feats["C"])


def test_structural_features_shape(small_graph):
    feats = compute_structural_features(small_graph, betweenness_k=None)
    expected_cols = {
        "degree",
        "weighted_degree",
        "pagerank",
        "clustering",
        "core_number",
        "betweenness",
        "eigenvector",
    }
    assert expected_cols.issubset(set(feats.columns))
    assert set(feats.index) == set(small_graph.nodes())
    assert feats.loc["E", "degree"] == 0
    assert feats.loc["E", "eigenvector"] == 0.0  # isolated node


def test_build_feature_table_end_to_end(small_graph):
    train = ["A", "B", "D"]
    audio = []
    table = build_feature_table(
        small_graph, train_nodes=train, audio_features=audio, betweenness_k=None
    )
    assert "popularity" in table.columns
    assert "avg_neighbor_popularity" in table.columns
    assert "pagerank" in table.columns
    assert set(table.index) == set(small_graph.nodes())
    # Sanity: C is held out, its avg_neighbor_pop must equal mean(B, D) = 30
    assert table.loc["C", "avg_neighbor_popularity"] == pytest.approx(30.0)
    # E is isolated -> NaN neighbour feature
    assert np.isnan(table.loc["E", "avg_neighbor_popularity"])
