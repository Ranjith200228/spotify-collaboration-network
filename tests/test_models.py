"""Lightweight smoke tests for src.models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import FEATURE_GROUPS, run_ablation


def _synthetic_table(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    audio_cols = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence",
    ]
    graph_cols = [
        "degree",
        "weighted_degree",
        "pagerank",
        "clustering",
        "core_number",
        "betweenness",
        "eigenvector",
        "avg_neighbor_popularity",
    ]
    audio = rng.normal(size=(n, len(audio_cols)))
    graph = rng.normal(size=(n, len(graph_cols)))
    pop = (
        20
        + 5 * audio[:, 0]
        + 2 * graph[:, 0]
        + rng.normal(scale=5.0, size=n)
    )
    data: dict[str, np.ndarray] = {
        "popularity": pop,
        "top_genre": rng.choice(["pop", "rock", "jazz"], size=n),
    }
    for i, c in enumerate(audio_cols):
        data[c] = audio[:, i]
    for i, c in enumerate(graph_cols):
        data[c] = graph[:, i]
    return pd.DataFrame(data, index=[f"a{i}" for i in range(n)])


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
    for _name, metrics in results["models"].items():
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
