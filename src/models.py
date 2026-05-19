"""Feature-ablation experiments for popularity prediction.

Answers research question #2: do graph-position features improve popularity
prediction beyond audio features? We train five regressors on the same
train/test split, varying only the feature set:

- ``baseline_mean``    — always predicts the train mean (DummyRegressor)
- ``genre_only``       — one-hot of the artist's dominant genre
- ``audio_only``       — the nine standard Spotify audio features
- ``graph_only``       — structural + leak-safe neighbour-popularity features
- ``combined``         — genre + audio + graph

Only ``graph_only`` and ``combined`` depend on the leak-safe
``avg_neighbor_popularity`` feature; if any of those R² scores creeps toward
1.0 it is a leakage signal that should trip the regression test in
``tests/test_features.py`` before getting this far.

The headline metric is ``lift_graph_over_audio``: how much RMSE drops when we
move from ``audio_only`` to ``combined``. That's the answer to the research
question in a single number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

#: The feature groups used by each ablation. Membership matters — adding a
#: graph feature here is what would cause leakage if the upstream computation
#: is buggy, so changes must be reviewed against the regression test.
FEATURE_GROUPS: dict[str, list[str]] = {
    "audio_only": [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence",
    ],
    "graph_only": [
        "degree",
        "weighted_degree",
        "pagerank",
        "clustering",
        "core_number",
        "betweenness",
        "eigenvector",
        "avg_neighbor_popularity",
    ],
    "genre_only": ["top_genre"],
}


@dataclass
class ModelResult:
    name: str
    mae: float
    rmse: float
    r2: float
    n_train: int
    n_test: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


def _materialize_features(
    table: pd.DataFrame,
    feature_names: list[str],
    train_nodes: list[str],
    test_nodes: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Return X_train, X_test, y_train, y_test for the requested feature set.

    Categorical columns (e.g. ``top_genre``) are one-hot encoded with the
    encoder fit on the training rows only — no leakage of test-only categories
    into the schema. Numeric NaNs are median-imputed using the train fold.
    """
    if "popularity" not in table.columns:
        raise ValueError("table missing 'popularity' column")

    cat_cols = [c for c in feature_names if c == "top_genre"]
    num_cols = [c for c in feature_names if c != "top_genre"]

    train = table.loc[train_nodes]
    test = table.loc[test_nodes]

    parts_train: list[np.ndarray] = []
    parts_test: list[np.ndarray] = []

    if num_cols:
        imputer = SimpleImputer(strategy="median")
        x_num_train = imputer.fit_transform(train[num_cols])
        x_num_test = imputer.transform(test[num_cols])
        parts_train.append(x_num_train)
        parts_test.append(x_num_test)

    if cat_cols:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # sklearn < 1.2
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        x_cat_train = encoder.fit_transform(train[cat_cols].astype(str))
        x_cat_test = encoder.transform(test[cat_cols].astype(str))
        parts_train.append(x_cat_train)
        parts_test.append(x_cat_test)

    x_train = np.hstack(parts_train) if parts_train else np.zeros((len(train), 0))
    x_test = np.hstack(parts_test) if parts_test else np.zeros((len(test), 0))

    return x_train, x_test, train["popularity"], test["popularity"]


def _score(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return mae, rmse, r2


def _train_one(
    name: str,
    feature_names: list[str] | None,
    table: pd.DataFrame,
    train_nodes: list[str],
    test_nodes: list[str],
    random_state: int,
) -> ModelResult:
    """Train one ablation row.

    ``feature_names=None`` (used for ``baseline_mean``) skips feature assembly
    entirely and fits a DummyRegressor on the target — no risk of features
    leaking into the trivial baseline.
    """
    train = table.loc[train_nodes]
    test = table.loc[test_nodes]
    y_train = train["popularity"]
    y_test = test["popularity"]

    if feature_names is None:
        model = DummyRegressor(strategy="mean")
        x_train = np.zeros((len(train), 1))
        x_test = np.zeros((len(test), 1))
    else:
        x_train, x_test, y_train, y_test = _materialize_features(
            table, feature_names, train_nodes, test_nodes
        )
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, random_state=random_state
        )

    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    mae, rmse, r2 = _score(y_test, preds)
    logger.info("%s -> MAE=%.3f RMSE=%.3f R2=%.4f", name, mae, rmse, r2)
    return ModelResult(
        name=name,
        mae=mae,
        rmse=rmse,
        r2=r2,
        n_train=len(train_nodes),
        n_test=len(test_nodes),
    )


def run_ablation(
    table: pd.DataFrame,
    train_nodes: list[str],
    test_nodes: list[str],
    random_state: int = 42,
) -> dict[str, Any]:
    """Run all five ablations and assemble the summary report.

    Parameters
    ----------
    table
        Per-node feature table from :func:`src.features.build_feature_table`.
        Must contain ``popularity`` plus every feature listed in
        :data:`FEATURE_GROUPS`.
    train_nodes, test_nodes
        Node lists from :func:`src.features.split_nodes`. Must be disjoint.

    Returns
    -------
    Dictionary with one entry per model plus ``lift_graph_over_audio`` summary
    measuring RMSE/MAE improvement of ``combined`` over ``audio_only``.
    """
    train_set = set(train_nodes)
    test_set = set(test_nodes)
    if train_set & test_set:
        raise ValueError("train_nodes and test_nodes must be disjoint")

    # Drop nodes whose popularity is NaN — happens for artists with no track
    # popularity values. We keep the split decision deterministic by filtering
    # AFTER the split was made.
    pop = table["popularity"]
    keep_mask = pop.notna()
    train_nodes = [n for n in train_nodes if keep_mask.get(n, False)]
    test_nodes = [n for n in test_nodes if keep_mask.get(n, False)]

    combined_features = (
        FEATURE_GROUPS["genre_only"]
        + FEATURE_GROUPS["audio_only"]
        + FEATURE_GROUPS["graph_only"]
    )

    runs: dict[str, ModelResult] = {}
    runs["baseline_mean"] = _train_one(
        "baseline_mean", None, table, train_nodes, test_nodes, random_state
    )
    runs["genre_only"] = _train_one(
        "genre_only",
        FEATURE_GROUPS["genre_only"],
        table,
        train_nodes,
        test_nodes,
        random_state,
    )
    runs["audio_only"] = _train_one(
        "audio_only",
        FEATURE_GROUPS["audio_only"],
        table,
        train_nodes,
        test_nodes,
        random_state,
    )
    runs["graph_only"] = _train_one(
        "graph_only",
        FEATURE_GROUPS["graph_only"],
        table,
        train_nodes,
        test_nodes,
        random_state,
    )
    runs["combined"] = _train_one(
        "combined", combined_features, table, train_nodes, test_nodes, random_state
    )

    audio = runs["audio_only"]
    combined = runs["combined"]
    lift = {
        # R² lift (the spec's canonical headline: positive = improvement)
        "r2_audio": audio.r2,
        "r2_combined": combined.r2,
        "r2_delta": float(combined.r2 - audio.r2),
        # RMSE lift (positive = improvement)
        "rmse_audio": audio.rmse,
        "rmse_combined": combined.rmse,
        "rmse_delta": float(audio.rmse - combined.rmse),
        "rmse_relative_pct": (
            float(100.0 * (audio.rmse - combined.rmse) / audio.rmse)
            if audio.rmse
            else 0.0
        ),
        # MAE lift (positive = improvement)
        "mae_audio": audio.mae,
        "mae_combined": combined.mae,
        "mae_delta": float(audio.mae - combined.mae),
    }

    return {
        "models": {name: r.as_dict() for name, r in runs.items()},
        "lift_graph_over_audio": lift,
    }
