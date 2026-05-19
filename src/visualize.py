"""Publication-quality visualization of the collaboration graph.

Renders ``docs/network_hero.png``: the top-PageRank artists of the largest
connected component, coloured by Louvain community. The intent is a single
hero image for the README — informative at a glance, legible at thumbnail
size, and reproducible from the same dataset+seed used by ``run_analysis.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def _top_pagerank_subgraph(
    g: nx.Graph, top_n: int, random_state: int
) -> tuple[nx.Graph, dict[str, float]]:
    """Return a subgraph of the top-``top_n`` PageRank nodes from the largest CC.

    Restricting to the largest connected component (rather than the whole
    graph) keeps the visualization dense and connected — most of the graph's
    nodes are singletons or tiny components that would just appear as
    disconnected dots.
    """
    if g.number_of_edges() == 0:
        raise ValueError("Graph has no edges; nothing to visualize.")

    largest_cc = max(nx.connected_components(g), key=len)
    sub = g.subgraph(largest_cc).copy()
    pagerank = nx.pagerank(sub, weight="weight")

    top = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_n]
    keep = [n for n, _ in top]
    return sub.subgraph(keep).copy(), pagerank


def render_network_hero(
    g: nx.Graph,
    output_path: str | Path = "docs/network_hero.png",
    top_n: int = 200,
    figsize: tuple[float, float] = (14, 10),
    random_state: int = 42,
    dpi: int = 200,
    show_labels_top: int = 25,
) -> Path:
    """Render and save the hero image.

    Parameters
    ----------
    g
        Annotated collaboration graph from :func:`src.graph.build_collab_graph`
        with node attributes already attached.
    output_path
        Destination PNG path. Parent dirs are created as needed.
    top_n
        Number of top-PageRank artists in the largest CC to include.
    show_labels_top
        How many of those artists get text labels (to keep the figure legible).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sub, pagerank = _top_pagerank_subgraph(g, top_n=top_n, random_state=random_state)
    partition = community_louvain.best_partition(
        sub, random_state=random_state, weight="weight"
    )

    pos = nx.spring_layout(sub, seed=random_state, weight="weight", k=0.4)

    pr_values = np.array([pagerank[n] for n in sub.nodes()])
    if pr_values.max() > 0:
        sizes = 40 + 600 * (pr_values / pr_values.max())
    else:
        sizes = np.full_like(pr_values, 100)

    communities = [partition[n] for n in sub.nodes()]
    n_communities = len(set(communities))
    cmap = plt.get_cmap("tab20" if n_communities <= 20 else "viridis")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("#0e0e10")
    fig.patch.set_facecolor("#0e0e10")

    nx.draw_networkx_edges(
        sub, pos, ax=ax, alpha=0.25, edge_color="#999999", width=0.6
    )
    nx.draw_networkx_nodes(
        sub,
        pos,
        ax=ax,
        node_size=sizes,
        node_color=communities,
        cmap=cmap,
        edgecolors="white",
        linewidths=0.4,
    )

    top_labelled = sorted(pagerank, key=lambda n: pagerank[n], reverse=True)[
        :show_labels_top
    ]
    labels = {n: n for n in sub.nodes() if n in set(top_labelled)}
    nx.draw_networkx_labels(
        sub,
        pos,
        labels=labels,
        ax=ax,
        font_size=8,
        font_color="#ffffff",
        font_weight="bold",
    )

    ax.set_title(
        f"Spotify Artist Collaboration Network — top {top_n} by PageRank, "
        f"coloured by Louvain community ({n_communities} communities)",
        color="white",
        fontsize=12,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.info("Wrote %s", output_path)
    return output_path
