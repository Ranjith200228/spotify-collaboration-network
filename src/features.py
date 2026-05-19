"""Leak-safe graph feature engineering.

The original notebook computed each artist's ``avg_neighbor_popularity`` over
*every* neighbour — including neighbours that later ended up in the held-out
test split. With train/test rows drawn from the same population that meant the
feature literally averaged future targets, and a trivial regression hit
R² ≈ 1.0. This module fixes the leak by enforcing a strict invariant:

    Neighbour-aware features may only reference TRAIN-node popularity.

Concretely, :func:`compute_neighbor_popularity` accepts ``train_nodes`` and
restricts its lookup to that set. Pure structural features (PageRank,
clustering, core number, betweenness, eigenvector centrality) are computed on
the full graph because they describe topology, not target values.
"""

from __future__ import annotations

import logging
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_nodes(
    nodes: Iterable[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[str], list[str]]:
    """Deterministically split nodes into train and test sets.

    Sorting before splitting guarantees the same partition across runs even if
    the upstream graph iteration order changes — important for reproducibility
    in CI and for the regression test in :mod:`tests.test_features`.
    """
    nodes_sorted = sorted(nodes)
    if not nodes_sorted:
        return [], []
    train, test = train_test_split(
        nodes_sorted, test_size=test_size, random_state=random_state, shuffle=True
    )
    return list(train), list(test)


def compute_neighbor_popularity(
    g: nx.Graph,
    train_nodes: Iterable[str],
    popularity_attr: str = "popularity",
) -> pd.Series:
    """Mean popularity of each node's neighbours, restricted to train nodes.

    Returns a Series indexed by every node in ``g``. The value at node ``v`` is
    the unweighted mean of ``popularity_attr`` over ``v``'s neighbours that are
    members of ``train_nodes``. Nodes with no qualifying neighbours (isolates,
    or test nodes surrounded only by other test nodes) get ``NaN`` — never zero,
    because zero would silently bias downstream regressions.

    Parameters
    ----------
    g
        Annotated graph from :func:`src.graph.attach_node_attributes`.
    train_nodes
        Iterable of node names that are in the training split. Only these
        nodes' popularities may flow into the feature.
    popularity_attr
        Node attribute name carrying the regression target.
    """
    train_set = set(train_nodes)
    pop_map: dict[str, float] = nx.get_node_attributes(g, popularity_attr)

    values: dict[str, float] = {}
    for node in g.nodes():
        train_neighbours = [
            nb for nb in g.neighbors(node) if nb in train_set and nb in pop_map
        ]
        if not train_neighbours:
            values[node] = np.nan
            continue
        values[node] = float(
            np.mean([pop_map[nb] for nb in train_neighbours])
        )
    return pd.Series(values, name="avg_neighbor_popularity")


def compute_structural_features(
    g: nx.Graph,
    betweenness_k: int | None = 500,
    eigenvector_max_iter: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute per-node structural features.

    Returned columns:

    - ``degree`` — node degree (collaboration count)
    - ``weighted_degree`` — sum of incident edge weights
    - ``pagerank`` — PageRank score
    - ``clustering`` — local clustering coefficient
    - ``core_number`` — k-core membership
    - ``betweenness`` — betweenness centrality (sampled if ``betweenness_k`` set)
    - ``eigenvector`` — eigenvector centrality on the largest connected
      component; nodes outside the largest CC receive 0.

    Notes
    -----
    Betweenness on a 30 K-node graph takes minutes; we use ``k``-sample
    approximation by default. Eigenvector centrality can fail to converge on
    disconnected graphs, so we restrict it to the largest component and zero-
    fill the rest.
    """
    nodes = list(g.nodes())

    logger.info("Computing PageRank")
    pagerank = nx.pagerank(g, weight="weight")

    logger.info("Computing clustering")
    clustering = nx.clustering(g, weight="weight")

    logger.info("Computing core number")
    g_for_core = g
    if nx.number_of_selfloops(g) > 0:
        g_for_core = g.copy()
        g_for_core.remove_edges_from(nx.selfloop_edges(g_for_core))
    core_number = nx.core_number(g_for_core)

    logger.info("Computing betweenness (k=%s)", betweenness_k)
    if betweenness_k is None or betweenness_k >= len(nodes):
        betweenness = nx.betweenness_centrality(g, weight="weight")
    else:
        betweenness = nx.betweenness_centrality(
            g, k=betweenness_k, seed=random_state, weight="weight"
        )

    logger.info("Computing eigenvector centrality on largest CC")
    eigenvector: dict[str, float] = {n: 0.0 for n in nodes}
    if g.number_of_edges() > 0:
        largest_cc = max(nx.connected_components(g), key=len)
        sub = g.subgraph(largest_cc)
        try:
            sub_eig = nx.eigenvector_centrality_numpy(sub, weight="weight")
        except (nx.NetworkXException, np.linalg.LinAlgError) as exc:
            logger.warning(
                "eigenvector_centrality_numpy failed (%s); falling back to "
                "power iteration",
                exc,
            )
            sub_eig = nx.eigenvector_centrality(
                sub, max_iter=eigenvector_max_iter, weight="weight"
            )
        eigenvector.update(sub_eig)

    weighted_degree = dict(g.degree(weight="weight"))
    degree = dict(g.degree())

    return pd.DataFrame(
        {
            "degree": pd.Series(degree),
            "weighted_degree": pd.Series(weighted_degree),
            "pagerank": pd.Series(pagerank),
            "clustering": pd.Series(clustering),
            "core_number": pd.Series(core_number),
            "betweenness": pd.Series(betweenness),
            "eigenvector": pd.Series(eigenvector),
        }
    ).reindex(nodes)


def build_feature_table(
    g: nx.Graph,
    train_nodes: Iterable[str],
    audio_features: Iterable[str],
    betweenness_k: int | None = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Assemble the full per-node feature table.

    Combines:

    - the target ``popularity`` (read from node attributes)
    - ``top_genre`` (categorical)
    - audio feature means (from node attributes)
    - structural features from :func:`compute_structural_features`
    - leak-safe ``avg_neighbor_popularity`` from
      :func:`compute_neighbor_popularity`

    The returned DataFrame is indexed by node name. It is safe to pass to a
    downstream model — the train/test split has already been baked into the
    neighbour feature.
    """
    nodes = list(g.nodes())

    pop = pd.Series(nx.get_node_attributes(g, "popularity"), name="popularity")
    top_genre = pd.Series(nx.get_node_attributes(g, "top_genre"), name="top_genre")

    audio_cols: dict[str, pd.Series] = {}
    for feat in audio_features:
        audio_cols[feat] = pd.Series(nx.get_node_attributes(g, feat), name=feat)

    structural = compute_structural_features(
        g, betweenness_k=betweenness_k, random_state=random_state
    )
    neighbour = compute_neighbor_popularity(g, train_nodes=train_nodes)

    parts = [pop, top_genre]
    parts.extend(audio_cols.values())
    parts.append(structural)
    parts.append(neighbour)

    table = pd.concat(parts, axis=1).reindex(nodes)
    table.index.name = "artist_name"
    return table
