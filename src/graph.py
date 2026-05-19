"""Collaboration graph construction.

The previous notebook iterated every (track_id, artist_name) row and called
``G.add_edge(artists[i], artists[j])`` over the resulting per-track list. When
the same artist appeared multiple times on one track — which happens whenever a
track is listed under more than one genre — that produced self-loops and
inflated edge weights. Self-loops in turn made each node its own neighbour,
which contaminated downstream popularity features.

This module fixes both issues by deduplicating (track_id, artist_name) pairs
*before* enumerating co-artist pairs and by refusing to write self-loops at all.
"""

from __future__ import annotations

import logging
from collections import Counter
from itertools import combinations
from typing import Any

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def build_collab_graph(
    tracks: pd.DataFrame,
    min_collabs: int = 1,
) -> nx.Graph:
    """Build an undirected weighted artist collaboration graph.

    Each track contributes one undirected edge for every distinct pair of
    artists that appear together. Edge weight equals the number of shared
    tracks. Nodes are artist names.

    Parameters
    ----------
    tracks
        Long-form tracks DataFrame produced by :func:`src.data.load_tracks`.
        Must contain ``track_id`` and ``artist_name`` columns.
    min_collabs
        Drop edges whose weight (collaboration count) is below this threshold.
        Defaults to 1, i.e. keep every collaboration.

    Returns
    -------
    A ``networkx.Graph`` whose nodes carry no attributes (popularity, genre,
    audio features are attached separately via :func:`attach_node_attributes`).

    Notes
    -----
    The deduplication step on (track_id, artist_name) is the regression fix for
    the silent self-loop / inflated-weight bug in the original notebook.
    """
    required = {"track_id", "artist_name"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"tracks DataFrame missing columns: {missing}")

    deduped = tracks[["track_id", "artist_name"]].drop_duplicates()
    n_dropped = len(tracks) - len(deduped)
    if n_dropped:
        logger.info(
            "Dropped %d duplicate (track_id, artist_name) rows before edge "
            "construction (these would have created self-loops).",
            n_dropped,
        )

    weights: Counter[tuple[str, str]] = Counter()
    for _, group in deduped.groupby("track_id", sort=False):
        artists = sorted(set(group["artist_name"]))
        if len(artists) < 2:
            continue
        for a, b in combinations(artists, 2):
            weights[(a, b)] += 1

    g = nx.Graph()
    g.add_nodes_from(deduped["artist_name"].unique())
    for (a, b), w in weights.items():
        if a == b:
            continue
        if w < min_collabs:
            continue
        g.add_edge(a, b, weight=w)

    self_loops = list(nx.selfloop_edges(g))
    if self_loops:
        g.remove_edges_from(self_loops)
        logger.warning(
            "Removed %d self-loop edges after construction (should be zero).",
            len(self_loops),
        )

    logger.info(
        "Built collaboration graph: %d nodes, %d edges (min_collabs=%d).",
        g.number_of_nodes(),
        g.number_of_edges(),
        min_collabs,
    )
    return g


def attach_node_attributes(g: nx.Graph, artists: pd.DataFrame) -> nx.Graph:
    """Annotate graph nodes with per-artist DataFrame columns.

    Modifies ``g`` in place and also returns it.

    Parameters
    ----------
    g
        Graph produced by :func:`build_collab_graph`.
    artists
        DataFrame indexed by ``artist_name`` (as returned by
        :func:`src.data.aggregate_artists`). Every column is attached as a node
        attribute on matching nodes; missing artists are left untouched.
    """
    if artists.index.name != "artist_name":
        raise ValueError("artists DataFrame must be indexed by artist_name")

    for col in artists.columns:
        values = artists[col].to_dict()
        nx.set_node_attributes(g, values, name=col)
    return g


def network_stats(g: nx.Graph) -> dict[str, Any]:
    """Compute summary statistics about the collaboration graph.

    Returns a JSON-serialisable dictionary covering size, density, component
    structure, degree summary, and an explicit ``self_loops`` count. The latter
    is reported so the regression fix is auditable in the saved report.
    """
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    self_loops = sum(1 for _ in nx.selfloop_edges(g))

    degrees = [d for _, d in g.degree()]
    if degrees:
        avg_degree = float(sum(degrees) / len(degrees))
        max_degree = int(max(degrees))
    else:
        avg_degree = 0.0
        max_degree = 0

    components = list(nx.connected_components(g))
    n_components = len(components)
    if components:
        largest = max(components, key=len)
        largest_cc_size = len(largest)
        largest_cc_fraction = float(largest_cc_size / n_nodes) if n_nodes else 0.0
    else:
        largest_cc_size = 0
        largest_cc_fraction = 0.0

    density = float(nx.density(g)) if n_nodes > 1 else 0.0
    isolated_nodes = sum(1 for _, d in g.degree() if d == 0)

    return {
        "num_nodes": int(n_nodes),
        "num_edges": int(n_edges),
        "self_loops": int(self_loops),
        "density": density,
        "avg_degree": avg_degree,
        "max_degree": max_degree,
        "num_isolated_nodes": int(isolated_nodes),
        "num_connected_components": int(n_components),
        "largest_cc_size": int(largest_cc_size),
        "largest_cc_fraction": largest_cc_fraction,
    }
