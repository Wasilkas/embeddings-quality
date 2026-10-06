"""Public API: all diagnostics retain the original number of coordinates."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral
from time import perf_counter

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

from ._logging import log_stage
from .geometry import geometry
from .probes import linear_probe, nuisance_probes, pairwise_probes
from .report import EmbeddingQualityReport


@dataclass(frozen=True)
class AuditConfig:
    ks: tuple[int, ...] = (5, 10, 20, 50)
    metric: str = "cosine"
    normalize: bool | None = None  # auto: cosine=True, euclidean/manhattan=False
    margin_k: int = 5
    cv_folds: int = 5
    probe_k: int = 10
    linear_c: float = 1.0
    max_iter: int = 2000
    random_state: int = 42
    working_memory_mb: float = 128
    exclude_same_group: bool = True

    def __post_init__(self) -> None:
        if self.metric not in ("cosine", "euclidean", "manhattan"):
            raise ValueError("metric must be 'cosine', 'euclidean', or 'manhattan'.")
        if not self.ks or len(set(self.ks)) != len(self.ks):
            raise ValueError("ks must be a nonempty sequence of distinct positive integers.")
        for name, value in [("k", k) for k in self.ks] + [
            ("margin_k", self.margin_k),
            ("cv_folds", self.cv_folds),
            ("probe_k", self.probe_k),
            ("max_iter", self.max_iter),
        ]:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2.")
        for name in ("linear_c", "working_memory_mb"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.normalize is not None and not isinstance(self.normalize, bool):
            raise ValueError("normalize must be True, False, or None.")
        if not isinstance(self.exclude_same_group, bool):
            raise ValueError("exclude_same_group must be boolean.")
        if not isinstance(self.random_state, Integral) or not 0 <= self.random_state < 2**32:
            raise ValueError("random_state must be an integer in [0, 2**32).")


def _strings(values: ArrayLike, n: int, name: str) -> NDArray[np.str_]:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != n:
        raise ValueError(f"{name} must have shape (n_samples,).")
    if pd.isna(array).any():
        raise ValueError(f"{name} contains missing values.")
    if array.dtype.kind in "fc" and not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values.")
    return array.astype(str)


def evaluate_embeddings(
    embeddings: ArrayLike,
    labels: ArrayLike,
    *,
    sample_ids: ArrayLike | None = None,
    groups: ArrayLike | None = None,
    metadata: Mapping[str, ArrayLike] | None = None,
    config: AuditConfig | None = None,
) -> EmbeddingQualityReport:
    """Evaluate fixed SSL features against single-label annotations.

    Parameters
    ----------
    embeddings : array-like, (N, D)
        Frozen features; no PCA, UMAP, or t-SNE is applied.
    labels : array-like, (N,)
        Class labels (reported as strings).
    sample_ids : array-like, (N,), optional
        Unique Qdrant point IDs or object IDs, aligned with the input rows.
    groups : array-like, (N,), optional
        Source image/coil/batch IDs. Used in every probe's CV and, by default,
        to exclude same-group neighbors from purity and distance margins.
    metadata : mapping of name to aligned 1D array, optional
        Numeric targets use RF regression, string/category targets use linear
        classification. Cast numeric camera IDs to strings for classification.
    config : AuditConfig, optional

    Progress is emitted at INFO through Loguru. Enable library logs with
    ``loguru.logger.enable("embeddings_quality")`` in the calling application.
    """
    started = perf_counter()
    logger.info("Embedding analysis: started; validating inputs and preprocessing")
    config = config or AuditConfig()
    X = np.asarray(embeddings, dtype=np.float64)
    if X.ndim != 2 or min(X.shape) < 1 or len(X) < 2:
        raise ValueError("embeddings must have shape (N, D), N >= 2, D >= 1.")
    if not np.isfinite(X).all():
        raise ValueError("embeddings contains NaN or infinity.")
    normalized = config.normalize if config.normalize is not None else config.metric == "cosine"
    if normalized or config.metric == "cosine":
        norms = np.linalg.norm(X, axis=1)
        if np.any(norms == 0) or not np.isfinite(norms).all():
            raise ValueError("Cosine/normalization requires nonzero vectors with finite norms.")
    if normalized:
        X = normalize(X, norm="l2")
    label_strings = _strings(labels, len(X), "labels")
    classes, y, counts = np.unique(label_strings, return_inverse=True, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Separability requires at least two classes.")
    ids = (
        np.arange(len(X)).astype(str)
        if sample_ids is None
        else _strings(sample_ids, len(X), "sample_ids")
    )
    if len(np.unique(ids)) != len(X):
        raise ValueError("sample_ids must be unique.")
    group_strings = None if groups is None else _strings(groups, len(X), "groups")
    logger.info(
        "Input ready: {} samples, {} dimensions, {} classes; metric={}, "
        "L2 normalized={}, grouped CV={} ({:.2f}s)",
        len(X),
        X.shape[1],
        len(classes),
        config.metric,
        normalized,
        group_strings is not None,
        perf_counter() - started,
    )
    notices = []
    if groups is None:
        notices.append("No groups supplied: probe CV assumes independent samples.")
    if np.min(counts) < 2:
        notices.append(
            "Singleton classes: positive distances are undefined and some probes are skipped."
        )
    with log_stage(logger, "Stage 1/4: geometry (purity, silhouette, distance margins)"):
        samples, class_table = geometry(X, y, classes, counts, group_strings, config)
    samples.insert(0, "sample_id", ids)
    samples.insert(1, "label", label_strings)
    if group_strings is not None:
        samples.insert(2, "group", group_strings)
    for k in config.ks:
        undefined = int(samples[f"purity@{k}"].isna().sum())
        if undefined:
            notices.append(
                f"purity@{k} is undefined for {undefined} samples: fewer than k eligible neighbors."
            )
    if samples.silhouette.isna().all():
        notices.append("Silhouette is undefined when every sample is its own class.")
    with log_stage(logger, "Stage 2/4: out-of-fold probes (linear, kNN, dummy)"):
        probe, predicted, probabilities, splits = linear_probe(X, y, group_strings, config, notices)
    confusion = pd.DataFrame(np.nan, index=classes, columns=classes)
    if predicted is not None and probabilities is not None:
        samples["linear_oof_prediction"] = classes[predicted]
        samples["linear_oof_label_probability"] = probabilities[np.arange(len(X)), y]
        samples["linear_oof_confidence"] = probabilities.max(axis=1)
        confusion = pd.DataFrame(confusion_matrix(y, predicted), index=classes, columns=classes)
        probe["per_class_recall"] = {
            str(c): float(np.mean(predicted[y == index] == index))
            for index, c in enumerate(classes)
        }
    with log_stage(logger, "Stage 3/4: pairwise class probes"):
        auc, balanced_accuracy, pairs = pairwise_probes(
            X, y, classes, group_strings, config, notices
        )
    with log_stage(logger, "Stage 4/4: nuisance metadata probes"):
        nuisance = nuisance_probes(
            X, metadata if metadata is not None else {}, group_strings, splits, config, notices
        )
    summary = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": len(classes),
        "classes": classes.tolist(),
        "config": asdict(config),
        "preprocessing": {"l2_normalized": normalized, "dimensionality_reduction": None},
        "geometry": {
            "silhouette_mean": samples.silhouette.mean(),
            "silhouette_macro_mean": class_table["silhouette_mean"].mean(),
            "purity_mean": {str(k): samples[f"purity@{k}"].mean() for k in config.ks},
            "purity_macro_mean": {
                str(k): class_table[f"purity@{k}_mean"].mean() for k in config.ks
            },
            "negative_margin_fraction": (samples.margin.dropna() < 0).mean(),
            "silhouette_reference": "all samples; anchor excluded, same-group samples retained",
            "neighborhood_reference": "other groups"
            if groups is not None and config.exclude_same_group
            else "all other samples",
        },
        "linear_probe": probe,
        "pairwise_probes": pairs,
        "nuisance_probes": nuisance,
        "notices": list(dict.fromkeys(notices)),
    }
    report = EmbeddingQualityReport(
        summary, samples, class_table, auc, balanced_accuracy, confusion
    )
    logger.info("Embedding analysis: completed in {:.2f}s", perf_counter() - started)
    return report
