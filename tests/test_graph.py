"""Regression tests for src.graph."""

from __future__ import annotations

import networkx as nx
import pandas as pd

from src.graph import attach_node_attributes, build_collab_graph, network_stats


def _toy_tracks() -> pd.DataFrame:
    # T1: A,B share a track (one collab)
    # T2: A,B share another track (weight grows to 2)
    # T3: A,C single collab
    # T4: D solo (isolated)
    # T5: duplicate (A,T1) row that previously caused self-loops
    return pd.DataFrame(
        [
            ("T1", "A", "pop"),
            ("T1", "B", "pop"),
            ("T2", "A", "pop"),
            ("T2", "B", "rock"),
            ("T3", "A", "rock"),
            ("T3", "C", "rock"),
            ("T4", "D", "jazz"),
            ("T1", "A", "rock"),  # duplicate -> would have caused A-A self-loop
        ],
        columns=["track_id", "artist_name", "genre"],
    )


def test_build_graph_no_self_loops():
    tracks = _toy_tracks()
    g = build_collab_graph(tracks)
    assert nx.number_of_selfloops(g) == 0
    # All four artists present even though D is isolated
    assert set(g.nodes()) == {"A", "B", "C", "D"}


def test_build_graph_edge_weights():
    tracks = _toy_tracks()
    g = build_collab_graph(tracks)
    assert g.has_edge("A", "B")
    assert g["A"]["B"]["weight"] == 2  # T1 and T2
    assert g.has_edge("A", "C")
    assert g["A"]["C"]["weight"] == 1
    assert not g.has_edge("B", "C")


def test_build_graph_min_collabs_filter():
    tracks = _toy_tracks()
    g = build_collab_graph(tracks, min_collabs=2)
    assert g.has_edge("A", "B")  # weight 2 survives
    assert not g.has_edge("A", "C")  # weight 1 dropped


def test_attach_node_attributes_roundtrip():
    tracks = _toy_tracks()
    g = build_collab_graph(tracks)
    artists = pd.DataFrame(
        {"popularity": [50.0, 60.0, 70.0, 80.0], "top_genre": ["pop"] * 4},
        index=pd.Index(["A", "B", "C", "D"], name="artist_name"),
    )
    g = attach_node_attributes(g, artists)
    assert g.nodes["A"]["popularity"] == 50.0
    assert g.nodes["D"]["top_genre"] == "pop"


def test_network_stats_shape():
    tracks = _toy_tracks()
    g = build_collab_graph(tracks)
    stats = network_stats(g)
    assert stats["num_nodes"] == 4
    assert stats["num_edges"] == 2
    assert stats["self_loops"] == 0
    assert stats["num_isolated_nodes"] == 1  # D
    assert stats["num_connected_components"] >= 2
    assert 0.0 <= stats["density"] <= 1.0
