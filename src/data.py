"""Dataset loading and per-artist aggregation.

Supports two real-world Spotify CSV schemas and normalizes them to a single
long-form DataFrame with one row per (track_id, artist_name, genre) record:

1. Kaggle "Ultimate Spotify Tracks DB" (SpotifyFeatures.csv) — already long-form;
   columns include ``artist_name`` and ``genre``.
2. HuggingFace ``maharshipandya/spotify-tracks-dataset`` — short-form; uses an
   ``artists`` column (semicolon-separated collaborators) and ``track_genre``.

The normalization choice is deliberate: graph construction wants explicit
(track, artist) edges, so we always expand multi-artist rows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Audio features carried through the pipeline. Both supported schemas expose
#: this exact subset, so they are safe to use as ML features.
AUDIO_FEATURES: tuple[str, ...] = (
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
)

#: Filenames (relative to ``data/raw/``) the loader will probe, in order.
DEFAULT_CANDIDATES: tuple[str, ...] = (
    "SpotifyFeatures.csv",
    "hf_spotify_tracks.csv",
    "dataset.csv",
)


def _resolve_csv(path: str | Path | None, data_dir: str | Path) -> Path:
    """Pick the CSV to read.

    If ``path`` is given, use it. Otherwise probe ``DEFAULT_CANDIDATES`` inside
    ``data_dir`` and return the first existing file. Raises ``FileNotFoundError``
    if nothing is found, listing the locations searched.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found at {p}")
        return p

    data_dir = Path(data_dir)
    tried: list[Path] = []
    for name in DEFAULT_CANDIDATES:
        candidate = data_dir / name
        tried.append(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No Spotify CSV found. Searched: " + ", ".join(str(t) for t in tried)
    )


def _normalize_hf_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the HuggingFace short-form schema to long-form.

    The HF dataset stores collaborators as a single ``artists`` cell containing
    semicolon-separated names. We explode that into multiple rows.
    """
    df = df.copy()
    df = df.rename(columns={"artists": "artist_name", "track_genre": "genre"})
    df["artist_name"] = (
        df["artist_name"]
        .fillna("")
        .astype(str)
        .str.split(";")
    )
    df = df.explode("artist_name", ignore_index=True)
    df["artist_name"] = df["artist_name"].str.strip()
    return df


def load_tracks(
    path: str | Path | None = None,
    data_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Load and normalize a Spotify tracks CSV into long form.

    Parameters
    ----------
    path
        Explicit CSV path. If ``None``, probes ``data_dir`` for known filenames.
    data_dir
        Directory to probe when ``path`` is not given. Defaults to ``data/raw``.

    Returns
    -------
    DataFrame with columns ``track_id``, ``artist_name``, ``genre``,
    ``popularity``, and every audio feature in :data:`AUDIO_FEATURES`. Rows with
    missing identifiers or non-numeric popularity are dropped.

    Notes
    -----
    Popularity is coerced with ``errors="coerce"``; the original notebook
    silently kept malformed strings, this version drops them and logs the count.
    """
    csv_path = _resolve_csv(path, data_dir)
    logger.info("Loading tracks from %s", csv_path)
    df = pd.read_csv(csv_path)

    if "artists" in df.columns and "track_genre" in df.columns:
        df = _normalize_hf_schema(df)
    elif "artist_name" not in df.columns or "genre" not in df.columns:
        raise ValueError(
            f"Unsupported CSV schema. Got columns: {list(df.columns)}"
        )

    required = ["track_id", "artist_name", "genre", "popularity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    df["track_id"] = df["track_id"].astype(str).str.strip()
    df["artist_name"] = df["artist_name"].astype(str).str.strip()
    df["genre"] = df["genre"].astype(str).str.strip()

    before = len(df)
    df = df[
        df["track_id"].ne("")
        & df["artist_name"].ne("")
        & df["popularity"].notna()
    ]
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with invalid id/artist/popularity", dropped)

    available_audio = [f for f in AUDIO_FEATURES if f in df.columns]
    keep_cols = required + available_audio
    return df.loc[:, keep_cols].reset_index(drop=True)


def aggregate_artists(
    tracks: pd.DataFrame,
    audio_features: Iterable[str] = AUDIO_FEATURES,
) -> pd.DataFrame:
    """Aggregate long-form tracks into per-artist statistics.

    Parameters
    ----------
    tracks
        Output of :func:`load_tracks`.
    audio_features
        Audio feature columns to average per artist. Missing columns are
        skipped silently — the loader already pruned them.

    Returns
    -------
    DataFrame indexed by ``artist_name`` with columns:

    - ``popularity`` — mean track popularity (the regression target)
    - ``num_tracks`` — count of distinct tracks the artist appears on
    - ``top_genre`` — modal genre across the artist's tracks
    - one column per available audio feature (mean across the artist's tracks)
    """
    if tracks.empty:
        raise ValueError("Cannot aggregate empty tracks DataFrame")

    audio_cols = [f for f in audio_features if f in tracks.columns]

    grouped = tracks.groupby("artist_name", sort=False)

    agg: dict[str, tuple[str, str] | tuple[str, object]] = {
        "popularity": ("popularity", "mean"),
        "num_tracks": ("track_id", "nunique"),
    }
    for col in audio_cols:
        agg[col] = (col, "mean")

    artists = grouped.agg(**agg)

    top_genre = (
        grouped["genre"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        .rename("top_genre")
    )
    artists = artists.join(top_genre)

    return artists
