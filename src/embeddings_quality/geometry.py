"""Exact geometry with bounded distance-matrix memory."""

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances_chunked

if TYPE_CHECKING:
    from .audit import AuditConfig


def geometry(
    X: NDArray[np.float64],
    y: NDArray[np.intp],
    classes: NDArray[np.str_],
    counts: NDArray[np.intp],
    groups: NDArray[np.str_] | None,
    config: "AuditConfig",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n, n_classes = len(X), len(classes)
    values = {f"purity@{k}": np.full(n, np.nan) for k in config.ks}
    values.update({f"lift@{k}": np.full(n, np.nan) for k in config.ks})
    values.update(
        {
            "neighbor_baseline": np.full(n, np.nan),
            "eligible_neighbors": np.zeros(n, dtype=int),
            "positive_neighbors_used": np.zeros(n, dtype=int),
            "d_positive": np.full(n, np.nan),
            "d_negative": np.full(n, np.nan),
            "margin": np.full(n, np.nan),
            "silhouette": np.full(n, np.nan),
        }
    )
    by_class = [np.flatnonzero(y == c) for c in range(n_classes)]
    start = 0
    for distances in pairwise_distances_chunked(
        X, metric=config.metric, working_memory=config.working_memory_mb
    ):
        for row, original_distances in enumerate(distances):
            i = start + row
            # Silhouette uses all samples, and excludes only the anchor itself.
            original_distances[i] = 0.0
            if n_classes < n and counts[y[i]] > 1:
                means = np.array([original_distances[idx].sum() for idx in by_class])
                a = means[y[i]] / (counts[y[i]] - 1)
                means /= counts
                means[y[i]] = np.inf
                b = means.min()
                values["silhouette"][i] = (b - a) / max(a, b) if max(a, b) else 0.0
            elif n_classes < n:
                values["silhouette"][i] = 0.0  # sklearn singleton convention

            eligible = np.ones(n, dtype=bool)
            eligible[i] = False
            if groups is not None and config.exclude_same_group:
                eligible &= groups != groups[i]
            total = int(eligible.sum())
            values["eligible_neighbors"][i] = total
            if not total:
                continue
            baseline = np.sum(eligible & (y == y[i])) / total
            values["neighbor_baseline"][i] = baseline
            d = original_distances.copy()
            d[~eligible] = np.inf
            # Stable index tie break: self is excluded even among duplicate vectors.
            ordered = np.argsort(d, kind="stable")[: min(max(config.ks), total)]
            for k in config.ks:
                if k <= total:
                    purity = np.mean(y[ordered[:k]] == y[i])
                    values[f"purity@{k}"][i] = purity
                    if baseline > 0:
                        values[f"lift@{k}"][i] = purity / baseline
            positives = d[eligible & (y == y[i])]
            negatives = d[eligible & (y != y[i])]
            used = min(config.margin_k, len(positives))
            values["positive_neighbors_used"][i] = used
            if used:
                nearest = np.partition(positives, used - 1)[:used]
                values["d_positive"][i] = np.median(nearest)
            if len(negatives):
                values["d_negative"][i] = negatives.min()
            values["margin"][i] = values["d_negative"][i] - values["d_positive"][i]
        start += len(distances)

    samples = pd.DataFrame(values)
    records = []
    columns = (
        [f"purity@{k}" for k in config.ks]
        + [f"lift@{k}" for k in config.ks]
        + ["silhouette", "margin"]
    )
    for c, label in enumerate(classes):
        subset = samples.loc[y == c]
        record: dict[str, Any] = {
            "label": label,
            "count": int(counts[c]),
            "prevalence": counts[c] / n,
        }
        for column in columns:
            finite = subset[column].dropna()
            record[f"{column}_mean"] = finite.mean()
            record[f"{column}_n"] = len(finite)
            for q in (0.1, 0.5, 0.9):
                record[f"{column}_p{int(q * 100)}"] = finite.quantile(q)
        margins = subset.margin.dropna()
        record["negative_margin_fraction"] = (margins < 0).mean()
        records.append(record)
    return samples, pd.DataFrame(records).set_index("label")
