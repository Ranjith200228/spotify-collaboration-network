"""End-to-end analysis pipeline.

Runs every phase against the CSV in ``data/raw/`` and writes three JSON
reports to ``reports/``:

- ``network_stats.json``      — graph size, density, components, self-loops
- ``community_metrics.json``  — Louvain modularity + genre-agreement metrics
- ``model_ablation.json``     — five-model RMSE/MAE/R² comparison + lift

Invoke from the repo root:

.. code-block:: shell

    python scripts/run_analysis.py
    # or with explicit CSV path
    python scripts/run_analysis.py --data data/raw/SpotifyFeatures.csv

Outputs feed ``README.md`` directly — never hand-edit those numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Make `src` importable when this script is run as a file from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.communities import community_metrics  # noqa: E402
from src.data import AUDIO_FEATURES, aggregate_artists, load_tracks  # noqa: E402
from src.features import build_feature_table, split_nodes  # noqa: E402
from src.graph import (  # noqa: E402
    attach_node_attributes,
    build_collab_graph,
    network_stats,
)
from src.models import run_ablation  # noqa: E402

logger = logging.getLogger("run_analysis")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to the tracks CSV. Defaults to probing data/raw/.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "raw",
        help="Directory to search for the CSV when --data is omitted.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports",
        help="Where to write the JSON reports.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of nodes held out for evaluation.",
    )
    parser.add_argument(
        "--betweenness-k",
        type=int,
        default=500,
        help="Sample size for betweenness centrality approximation. "
        "Use a negative number to disable sampling (exact betweenness).",
    )
    parser.add_argument(
        "--min-collabs",
        type=int,
        default=1,
        help="Minimum edge weight (collaboration count) to retain.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for split, Louvain, and gradient boosting.",
    )
    return parser.parse_args(argv)


def _dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Wrote %s", path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    args = parse_args(argv)
    started = time.time()

    logger.info("== 1/8 loading tracks ==")
    tracks = load_tracks(path=args.data, data_dir=args.data_dir)
    logger.info("Loaded %d track-artist rows", len(tracks))

    logger.info("== 2/8 aggregating per-artist ==")
    artists = aggregate_artists(tracks)
    logger.info("Aggregated %d artists", len(artists))

    logger.info("== 3/8 building graph ==")
    graph = build_collab_graph(tracks, min_collabs=args.min_collabs)
    graph = attach_node_attributes(graph, artists)

    logger.info("== 4/8 computing network stats ==")
    stats = network_stats(graph)
    _dump_json(args.reports_dir / "network_stats.json", stats)

    logger.info("== 5/8 splitting nodes ==")
    train_nodes, test_nodes = split_nodes(
        graph.nodes(), test_size=args.test_size, random_state=args.random_state
    )
    logger.info(
        "Train=%d, Test=%d (test_size=%.2f)",
        len(train_nodes),
        len(test_nodes),
        args.test_size,
    )

    logger.info("== 6/8 computing leak-safe features ==")
    betweenness_k = args.betweenness_k if args.betweenness_k > 0 else None
    table = build_feature_table(
        graph,
        train_nodes=train_nodes,
        audio_features=AUDIO_FEATURES,
        betweenness_k=betweenness_k,
        random_state=args.random_state,
    )
    logger.info("Feature table shape: %s", table.shape)

    logger.info("== 7/8 community detection ==")
    comm_metrics = community_metrics(
        graph, artists, random_state=args.random_state
    )
    _dump_json(args.reports_dir / "community_metrics.json", comm_metrics)

    logger.info("== 8/8 model ablation ==")
    ablation = run_ablation(
        table,
        train_nodes=train_nodes,
        test_nodes=test_nodes,
        random_state=args.random_state,
    )

    # Attach split metadata so the report is fully reproducible from JSON alone
    ablation["dataset"] = {
        "source_csv": str(
            args.data if args.data else (args.data_dir / "(probed)")
        ),
        "num_track_rows": int(len(tracks)),
        "num_artists": int(len(artists)),
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
    }
    _dump_json(args.reports_dir / "model_ablation.json", ablation)

    elapsed = time.time() - started
    logger.info("Done in %.1fs", elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
