"""Lightweight smoke tests for src.models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import FEATURE_GROUPS, run_ablation


def _synthetic_table(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    audio = rng.normal(size=(n, 3))
    graph = rng.normal(size=(n, 4))
    # Target depends on a couple of features + noise
    pop = (
        20
        + 5 * audio[:, 0]
        + 2 * graph[:, 0]
        + rng.normal(scale=5.0, size=n)
    )
    return pd.DataFrame(
        {
            "popularity": pop,
            "top_genre": rng.choice(["pop", "rock", "jazz"], size=n),
            "acousticness": audio[:, 0],
            "danceability": audio[:, 1],
            "energy": audio[:, 2],
            "degree": graph[:, 0],
            "pagerank": graph[:, 1],
            "clustering": graph[:, 2],
            "avg_neighbor_popularity": graph[:, 3],
        },
        index=[f"a{i}" for i in range(n)],
    )


def test_feature_groups_disjoint():
    audio = set(FEATURE_GROUPS["audio_only"])
    graph = set(FEATURE_GROUPS["graph_only"])
    assert audio.isdisjoint(graph)


def test_run_ablation_returns_expected_models():
    table = _synthetic_table()
    train = table.index[:150].tolist()
    test = table.index[150:].tolist()
    results = run_ablation(table, train_nodes=train, test_nodes=test, random_state=0)
    assert {"baseline_mean", "genre_only", "audio_only", "graph_only", "combined"}.issubset(
        results["models"].keys()
    )
    for name, metrics in results["models"].items():
        assert {"mae", "rmse", "r2"}.issubset(metrics.keys())
        assert metrics["rmse"] > 0


def test_lift_graph_over_audio_present():
    table = _synthetic_table()
    train = table.index[:150].tolist()
    test = table.index[150:].tolist()
    results = run_ablation(table, train_nodes=train, test_nodes=test, random_state=0)
    assert "lift_graph_over_audio" in results
    assert isinstance(results["lift_graph_over_audio"], dict)
    assert "rmse_delta" in results["lift_graph_over_audio"]
