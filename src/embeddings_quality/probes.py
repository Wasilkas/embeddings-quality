"""Out-of-fold probes with explicit failure states for infeasible splits."""

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import ArrayLike, NDArray
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from ._logging import log_stage

if TYPE_CHECKING:
    from .audit import AuditConfig

Splits = list[tuple[NDArray[np.intp], NDArray[np.intp]]]


def make_splits(
    y: NDArray[np.intp],
    groups: NDArray[np.str_] | None,
    config: "AuditConfig",
) -> tuple[Splits | None, str | None]:
    counts = np.bincount(y)
    possible = int(np.min(counts))
    if groups is not None:
        possible = min(possible, min(len(np.unique(groups[y == c])) for c in np.unique(y)))
    folds = min(config.cv_folds, possible)
    if folds < 2:
        return None, "Each class needs at least two samples and, with groups, two distinct groups."
    splitter = (
        StratifiedKFold(folds, shuffle=True, random_state=config.random_state)
        if groups is None
        else StratifiedGroupKFold(folds, shuffle=True, random_state=config.random_state)
    )
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    all_classes = set(np.unique(y))
    for train, test in splits:
        if set(y[train]) != all_classes or set(y[test]) != all_classes:
            return None, "A group-aware fold lacks a class; change groups/folds or add data."
        if groups is not None and set(groups[train]) & set(groups[test]):
            raise RuntimeError("Group leakage in CV splitter")
    return splits, None


def fit_linear(
    X: NDArray[np.float64],
    y: NDArray[np.intp],
    config: "AuditConfig",
    notices: list[str],
) -> LogisticRegression:
    model = LogisticRegression(
        C=config.linear_c,
        class_weight="balanced",
        max_iter=config.max_iter,
        solver="lbfgs",
        random_state=config.random_state,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    for warning in captured:
        if issubclass(warning.category, ConvergenceWarning):
            notices.append(
                "Linear probe did not converge; increase max_iter before interpreting it."
            )
        else:
            warnings.warn(warning.message, warning.category, stacklevel=2)
    return model


def classification_scores(y: NDArray[np.intp], pred: NDArray[np.intp]) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def linear_probe(
    X: NDArray[np.float64],
    y: NDArray[np.intp],
    groups: NDArray[np.str_] | None,
    config: "AuditConfig",
    notices: list[str],
) -> tuple[dict[str, Any], NDArray[np.intp] | None, NDArray[np.float64] | None, Splits | None]:
    splits, reason = make_splits(y, groups, config)
    if splits is None:
        logger.info("OOF probes skipped: {}", reason)
        return {"status": "skipped", "reason": reason}, None, None, None
    logger.info(
        "OOF probes: {} folds, {} CV",
        len(splits),
        "group-aware" if groups is not None else "stratified",
    )
    n_classes = len(np.unique(y))
    probabilities = np.zeros((len(y), n_classes))
    predictions = {name: np.empty(len(y), dtype=int) for name in ("linear", "knn", "dummy")}
    fold_scores = []
    for fold, (train, test) in enumerate(splits):
        context = f"OOF fold {fold + 1}/{len(splits)} (train={len(train)}, test={len(test)})"
        with log_stage(logger, f"{context}: linear fit and prediction"):
            model = fit_linear(X[train], y[train], config, notices)
            probabilities[test] = model.predict_proba(X[test])
            predictions["linear"][test] = model.predict(X[test])
        with log_stage(logger, f"{context}: kNN fit and prediction"):
            knn = KNeighborsClassifier(
                n_neighbors=min(config.probe_k, len(train)),
                metric=config.metric,
                algorithm="brute",
            ).fit(X[train], y[train])
            predictions["knn"][test] = knn.predict(X[test])
        dummy = DummyClassifier(strategy="prior").fit(X[train], y[train])
        predictions["dummy"][test] = dummy.predict(X[test])
        scores: dict[str, Any] = {
            "fold": fold,
            "train_size": len(train),
            "test_size": len(test),
            "knn_k": min(config.probe_k, len(train)),
        }
        for name, pred in predictions.items():
            scores[name] = classification_scores(y[test], pred[test])
        fold_scores.append(scores)
        logger.debug("{}: scores={}", context, scores)
    summary = {
        "status": "ok",
        "splitter": "stratified_group" if groups is not None else "stratified",
        "n_folds": len(splits),
        "folds": fold_scores,
        "oof": {name: classification_scores(y, pred) for name, pred in predictions.items()},
    }
    return summary, predictions["linear"], probabilities, splits


def pairwise_probes(
    X: NDArray[np.float64],
    y: NDArray[np.intp],
    classes: NDArray[np.str_],
    groups: NDArray[np.str_] | None,
    config: "AuditConfig",
    notices: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    auc = pd.DataFrame(np.nan, index=classes, columns=classes)
    accuracy = auc.copy()
    records: list[dict[str, Any]] = []
    total_pairs = len(classes) * (len(classes) - 1) // 2
    logger.info("Pairwise probes: {} class pairs", total_pairs)
    for left in range(len(classes)):
        for right in range(left + 1, len(classes)):
            context = (
                f"Pair {len(records) + 1}/{total_pairs}: "
                f"{str(classes[left])!r} vs {str(classes[right])!r}"
            )
            logger.info("{}: preparing CV", context)
            selected = (y == left) | (y == right)
            binary_y = (y[selected] == right).astype(int)
            pair_X = X[selected]
            pair_groups = None if groups is None else groups[selected]
            splits, reason = make_splits(binary_y, pair_groups, config)
            record: dict[str, Any] = {"class_a": classes[left], "class_b": classes[right]}
            if splits is None:
                record.update(status="skipped", reason=reason)
                logger.info("{}: skipped ({})", context, reason)
            else:
                scores = []
                for fold, (train, test) in enumerate(splits, start=1):
                    with log_stage(logger, f"{context}, fold {fold}/{len(splits)}"):
                        model = fit_linear(pair_X[train], binary_y[train], config, notices)
                        scores.append(
                            {
                                "roc_auc": float(
                                    roc_auc_score(
                                        binary_y[test], model.predict_proba(pair_X[test])[:, 1]
                                    )
                                ),
                                "balanced_accuracy": float(
                                    balanced_accuracy_score(
                                        binary_y[test], model.predict(pair_X[test])
                                    )
                                ),
                            }
                        )
                record.update(status="ok", n_folds=len(splits), folds=scores)
                for key in ("roc_auc", "balanced_accuracy"):
                    values = [s[key] for s in scores]
                    record[key] = float(np.mean(values))
                    record[f"{key}_std"] = float(np.std(values, ddof=1))
                auc.iloc[left, right] = auc.iloc[right, left] = record["roc_auc"]
                accuracy.iloc[left, right] = accuracy.iloc[right, left] = record[
                    "balanced_accuracy"
                ]
                logger.info(
                    "{}: completed, mean ROC-AUC={:.4f}, balanced accuracy={:.4f}",
                    context,
                    record["roc_auc"],
                    record["balanced_accuracy"],
                )
            records.append(record)
    return auc, accuracy, records


def nuisance_probes(
    X: NDArray[np.float64],
    metadata: Mapping[str, ArrayLike],
    groups: NDArray[np.str_] | None,
    main_splits: Splits | None,
    config: "AuditConfig",
    notices: list[str],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    if not metadata:
        logger.info("Nuisance probes skipped: no metadata supplied")
    for column_index, (name, column) in enumerate(metadata.items(), start=1):
        context = f"Metadata {column_index}/{len(metadata)}: {name!r}"
        logger.info("{}: validating target", context)
        values = np.asarray(column)
        if values.ndim != 1 or len(values) != len(X):
            raise ValueError(f"Metadata {name!r} must have shape (n_samples,).")
        if pd.isna(values).any():
            results[name] = {"status": "skipped", "reason": "Missing metadata values."}
            logger.info("{}: skipped ({})", context, results[name]["reason"])
            continue
        numeric = pd.api.types.is_numeric_dtype(values.dtype) and values.dtype.kind != "b"
        if numeric:
            if not np.isfinite(values).all() or np.unique(values).size < 2:
                results[name] = {"status": "skipped", "reason": "Nonfinite or constant target."}
                logger.info("{}: skipped ({})", context, results[name]["reason"])
                continue
            if main_splits is None:
                results[name] = {"status": "skipped", "reason": "Main CV is infeasible."}
                logger.info("{}: skipped ({})", context, results[name]["reason"])
                continue
            predictions, baseline = np.empty(len(X)), np.empty(len(X))
            scores = []
            for fold, (train, test) in enumerate(main_splits, start=1):
                with log_stage(logger, f"{context}, regression fold {fold}/{len(main_splits)}"):
                    model = RandomForestRegressor(
                        n_estimators=100,
                        min_samples_leaf=2,
                        n_jobs=1,
                        random_state=config.random_state,
                    ).fit(X[train], values[train])
                    predictions[test] = model.predict(X[test])
                dummy = DummyRegressor().fit(X[train], values[train])
                baseline[test] = dummy.predict(X[test])
                scores.append(
                    {
                        "r2": float(r2_score(values[test], predictions[test]))
                        if len(test) > 1
                        else None,
                        "mae": float(mean_absolute_error(values[test], predictions[test])),
                    }
                )
            results[name] = {
                "status": "ok",
                "kind": "regression",
                "folds": scores,
                "oof_r2": float(r2_score(values, predictions)),
                "oof_mae": float(mean_absolute_error(values, predictions)),
                "baseline_oof_r2": float(r2_score(values, baseline)),
                "baseline_oof_mae": float(mean_absolute_error(values, baseline)),
            }
        else:
            _, target = np.unique(values.astype(str), return_inverse=True)
            if len(np.unique(target)) < 2:
                results[name] = {"status": "skipped", "reason": "Constant target."}
                logger.info("{}: skipped ({})", context, results[name]["reason"])
                continue
            with log_stage(logger, f"{context}: classification"):
                result, _, _, _ = linear_probe(X, target, groups, config, notices)
            results[name] = {"kind": "classification", **result}
        logger.info("{}: {}", context, results[name]["status"])
    return results
