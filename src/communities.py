"""Community detection on the collaboration graph.

Uses Louvain to partition artists into communities and compares those
communities against the ground-truth genre labels via two standard external
clustering metrics:

- Adjusted Rand Index (ARI) — corrects for chance, range roughly [-0.5, 1.0].
  ARI > 0 means the partitions agree more than random; ARI ≈ 0 is random.
- Normalized Mutual Information (NMI) — information-theoretic agreement,
  range [0, 1].

Together they answer research question #1: does the collaboration graph
encode genre structure? If both metrics are well above 0, the graph topology
alone recovers genres an external annotator labelled — strong evidence that
collaboration is genre-clustered.
"""

from __future__ import annotations

import logging
from typing import Any

import community as community_louvain
import networkx as nx
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

logger = logging.getLogger(__name__)


def detect_communities(
    g: nx.Graph,
    random_state: int = 42,
    resolution: float = 1.0,
) -> dict[str, int]:
    """Partition nodes into Louvain communities.

    Parameters
    ----------
    g
        Collaboration graph from :func:`src.graph.build_collab_graph`.
    random_state
        Seed for the underlying randomized greedy refinement, so partitions
        are reproducible across runs.
    resolution
        Louvain resolution parameter; higher values favour smaller communities.

    Returns
    -------
    Mapping ``node -> community_id``. Isolated nodes get their own singleton
    community ids — Louvain handles this natively.
    """
    if g.number_of_edges() == 0:
        # No edges -> every node is its own community.
        return {n: i for i, n in enumerate(g.nodes())}

    partition = community_louvain.best_partition(
        g, random_state=random_state, resolution=resolution, weight="weight"
    )
    logger.info(
        "Louvain detected %d communities over %d nodes",
        len(set(partition.values())),
        len(partition),
    )
    return partition


def evaluate_community_genre_agreement(
    partition: dict[str, int],
    genres: dict[str, str],
) -> dict[str, Any]:
    """Compare a community partition against genre labels.

    Both inputs are dicts keyed by node name. Only nodes present in BOTH
    mappings are scored (an artist with missing genre is dropped from the
    comparison rather than silently labelled). The function returns the
    cluster-quality metrics plus a top-genres-per-community breakdown that is
    useful for sanity-checking the partitions.
    """
    common = sorted(set(partition.keys()) & set(genres.keys()))
    if not common:
        raise ValueError("No nodes shared between partition and genres dicts")

    labels_cluster = [partition[n] for n in common]
    labels_genre = [genres[n] for n in common]

    ari = float(adjusted_rand_score(labels_genre, labels_cluster))
    nmi = float(normalized_mutual_info_score(labels_genre, labels_cluster))

    # Community size distribution
    df = pd.DataFrame(
        {"node": common, "community": labels_cluster, "genre": labels_genre}
    )
    sizes = (
        df.groupby("community").size().sort_values(ascending=False)
    )
    largest = sizes.head(10).to_dict()

    # Dominant genre per top community (only top 10 to keep report compact)
    top_communities = sizes.head(10).index.tolist()
    dominant_genre: dict[int, dict[str, Any]] = {}
    for c in top_communities:
        sub = df[df["community"] == c]
        if sub.empty:
            continue
        counts = sub["genre"].value_counts()
        top_genre = counts.index[0]
        purity = float(counts.iloc[0] / counts.sum())
        dominant_genre[int(c)] = {
            "size": int(len(sub)),
            "top_genre": str(top_genre),
            "purity": purity,
        }

    return {
        "ari": ari,
        "nmi": nmi,
        "num_communities": int(len(set(labels_cluster))),
        "num_evaluated_nodes": int(len(common)),
        "largest_communities": {int(k): int(v) for k, v in largest.items()},
        "top_community_genres": dominant_genre,
    }


def compute_modularity(g: nx.Graph, partition: dict[str, int]) -> float:
    """Modularity of a partition on the weighted graph.

    Returned in the standard [-0.5, 1.0] Louvain range; values near 0.4+ are
    considered strong evidence of modular structure for collaboration networks.
    """
    if g.number_of_edges() == 0:
        return 0.0
    return float(
        community_louvain.modularity(partition, g, weight="weight")
    )


def community_metrics(
    g: nx.Graph,
    artists: pd.DataFrame,
    genre_column: str = "top_genre",
    random_state: int = 42,
) -> dict[str, Any]:
    """Top-level entry point used by the analysis script.

    Combines partitioning, modularity, and the genre-agreement report into a
    single JSON-serialisable dictionary suitable for
    ``reports/community_metrics.json``.
    """
    partition = detect_communities(g, random_state=random_state)
    modularity = compute_modularity(g, partition)

    if genre_column not in artists.columns:
        raise ValueError(
            f"artists DataFrame has no '{genre_column}' column for evaluation"
        )

    genres = artists[genre_column].dropna().astype(str).to_dict()
    agreement = evaluate_community_genre_agreement(partition, genres)

    return {
        "modularity": modularity,
        **agreement,
    }
