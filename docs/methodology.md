# Methodology

This document explains the experimental design, the original-repo leakage bug
that motivated the rewrite, and the leak-safe procedure now used by
`scripts/run_analysis.py`. Every number quoted in the README is generated from
the JSON reports produced by that script — never hand-edited.

## 1. Research questions

1. **Does the artist collaboration graph encode genre structure?**
   We answer this by partitioning the graph with Louvain and comparing the
   resulting communities to the dominant genre per artist using ARI and NMI.
2. **Do graph-position features improve popularity prediction beyond audio
   features?**
   We answer this with a five-way feature ablation on a single train/test
   split, and report the RMSE/MAE/R² lift of the combined model over the
   audio-only baseline.
3. **What are the structural properties of the collaboration network?**
   We report node/edge counts, density, degree distribution summary,
   connected-component sizes, and isolation rate.

## 2. Dataset

The pipeline accepts either of two real-world Spotify CSV schemas:

- **Kaggle “Ultimate Spotify Tracks DB” (`SpotifyFeatures.csv`)** — long form,
  one row per `(track, artist, genre)`. This is the schema the original
  notebook in this repo used.
- **HuggingFace `maharshipandya/spotify-tracks-dataset`** — short form, one
  row per `(track, genre)` with collaborators packed into a single
  `;`-separated `artists` cell.

`src/data.py` auto-detects the format from the column names and normalises
both into the long form. The downloaded file in `data/raw/` for the metrics
reported here is the HuggingFace dataset (the Kaggle file is not available on
a public mirror so we ship the HuggingFace fallback that `scripts/fetch_data.ps1`
uses); the schemas are otherwise interchangeable.

## 3. Graph construction — and the original leakage bug

### The bug

The original notebook built collaboration edges with roughly:

```python
for _, row in track_groups.iterrows():
    artists = row["artist_name"]  # list of artist names on this track_id
    for i in range(len(artists)):
        for j in range(i + 1, len(artists)):
            G.add_edge(artists[i], artists[j])
```

This looks fine, but the input list is built by grouping rows on `track_id`
and pulling the `artist_name` column from each row — and the dataset has
multiple rows per `track_id` whenever a track is filed under more than one
genre. The same artist therefore appears multiple times in the per-track
list, and `add_edge(artists[i], artists[j])` produces **self-loops** for
duplicate entries. Self-loops in `networkx.Graph` make a node its own
neighbour, which silently contaminates every neighbour-based feature
downstream.

### The fix

`src/graph.py:build_collab_graph` deduplicates `(track_id, artist_name)`
pairs **before** enumerating co-artist pairs, refuses to write self-loops
during edge addition, and also calls `remove_edges_from(selfloop_edges(g))`
as a belt-and-braces step. The resulting `reports/network_stats.json`
reports `"self_loops": 0`, which is an audit trail any reviewer can check.

On the dataset shipped here, the dedup step removed **34,868** duplicate
rows (~22% of the long-form rows) that would have caused self-loops in the
original.

## 4. Leak-safe features — and the bigger original leak

### The bug

The original notebook computed `avg_neighbor_popularity` like this:

```python
for node in G.nodes():
    neighbors = list(G.neighbors(node))
    avg_neighbor_pop = sum(
        G.nodes[n].get("popularity", 0) for n in neighbors
    ) / (len(neighbors) or 1)
```

…and then used `[num_collaborations, avg_neighbor_popularity]` as features
in a `train_test_split` regression. Two compounding leaks:

1. Self-loops (see §3) meant `n in neighbors` could equal `node` itself, so
   the target literally fed into its own feature.
2. Even without self-loops, the test-node popularities flow into the
   neighbour computation, because `neighbors` is unfiltered. After the
   split, the test target's value is reachable via any train neighbour, and
   the feature matrix already encodes the held-out labels.

The original notebook reported **R² = 1.00**, which is the visible
fingerprint of the leak — popularity prediction from two graph features
should not have a perfect fit.

### The fix

`src/features.py:compute_neighbor_popularity` accepts an explicit
`train_nodes` set and ignores every neighbour outside it. Concretely:

```python
train_neighbours = [
    nb for nb in g.neighbors(node)
    if nb in train_set and nb in pop_map
]
if not train_neighbours:
    values[node] = np.nan
```

Nodes with no qualifying neighbours get `NaN`, never `0` — the imputer in
`src/models.py` fills `NaN` with the train-fold median, which is a
documented choice rather than a silent zero-bias.

The invariant is locked in by
`tests/test_features.py::test_neighbor_pop_uses_only_train_nodes`, which
mutates a held-out node's popularity and asserts its
`avg_neighbor_popularity` does not change. If the leak is ever
reintroduced, this regression test fails and CI halts.

### Pure-structural features

PageRank, weighted/unweighted degree, clustering, core number, sampled
betweenness, and eigenvector centrality describe **topology** — they do not
reference node attributes — so they are computed on the full graph. They
cannot leak the target by construction.

## 5. Train/test split

`split_nodes(nodes, test_size=0.2, random_state=42)` sorts the node list
deterministically before splitting, so the partition is reproducible across
runs even if upstream NetworkX iteration order changes. The split is performed
**before** any neighbour-popularity computation; that ordering is the entire
point of the rewrite.

## 6. Models

`run_ablation` trains five models on the same split and reports MAE, RMSE,
and R² for each:

| Model           | Features                                                |
| --------------- | ------------------------------------------------------- |
| `baseline_mean` | `DummyRegressor(strategy="mean")` — no features         |
| `genre_only`    | One-hot of `top_genre`                                  |
| `audio_only`    | 9 standard Spotify audio features                       |
| `graph_only`    | 8 structural + leak-safe `avg_neighbor_popularity`      |
| `combined`      | `genre_only` ∪ `audio_only` ∪ `graph_only`              |

All non-baseline models use `GradientBoostingRegressor(n_estimators=200,
max_depth=3)`. Categorical features are one-hot encoded with the encoder
fit on the train fold only (so test-only categories silently collapse).
Numeric NaNs are median-imputed using the train fold only.

## 7. Reproducing the reports

```bash
python -m venv .venv
.venv/Scripts/activate            # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Place SpotifyFeatures.csv at data/raw/, or run the HuggingFace fallback:
powershell -ExecutionPolicy Bypass -File scripts/fetch_data.ps1
python scripts/run_analysis.py
pytest -v
```

The script writes three JSONs under `reports/` and the test suite asserts
the leakage regression is still caught.
