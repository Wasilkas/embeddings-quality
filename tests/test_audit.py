import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import pairwise_distances, silhouette_samples
from sklearn.preprocessing import normalize

from embeddings_quality import AuditConfig, evaluate_embeddings
from embeddings_quality.cli import main
from embeddings_quality.probes import make_splits


@pytest.fixture
def separated():
    rng = np.random.default_rng(3)
    y = np.repeat(np.arange(3), 45)
    X = np.eye(3, 32)[y] * 5 + rng.normal(0, 0.15, (len(y), 32))
    return X, y


def test_separated_original_dimensions(separated):
    X, y = separated
    report = evaluate_embeddings(X, y, config=AuditConfig(ks=(5, 10), cv_folds=3))
    assert report.summary["n_features"] == 32
    assert report.summary["preprocessing"]["dimensionality_reduction"] is None
    assert report.samples["purity@5"].min() == 1
    assert report.samples.margin.min() > 0
    assert report.summary["geometry"]["silhouette_mean"] > 0.95
    assert report.summary["linear_probe"]["oof"]["linear"]["balanced_accuracy"] == 1
    assert report.summary["linear_probe"]["oof"]["dummy"]["balanced_accuracy"] == pytest.approx(
        1 / 3
    )
    np.testing.assert_allclose(report.pairwise_auc.to_numpy()[~np.eye(3, dtype=bool)], 1)


@pytest.mark.parametrize(
    "metric,normalize_vectors",
    [("cosine", None), ("euclidean", None), ("manhattan", None), ("manhattan", True)],
)
def test_geometry_matches_reference(metric, normalize_vectors):
    rng = np.random.default_rng(10)
    X = rng.normal(size=(16, 7))
    X[1] = X[0]  # duplicates: never count the anchor as its own neighbor
    y = np.array([0] * 7 + [1] * 8 + [2])
    config = AuditConfig(
        ks=(1, 3),
        metric=metric,
        normalize=normalize_vectors,
        margin_k=3,
        working_memory_mb=0.001,
    )
    report = evaluate_embeddings(X, y, config=config)
    normalized = metric == "cosine" if normalize_vectors is None else normalize_vectors
    assert report.summary["preprocessing"]["l2_normalized"] is normalized
    reference_X = normalize(X) if normalized else X
    np.testing.assert_allclose(
        report.samples.silhouette, silhouette_samples(reference_X, y, metric=metric), atol=1e-12
    )
    distances = pairwise_distances(reference_X, metric=metric)
    for i in range(len(y)):
        neighbors = sorted((j for j in range(len(y)) if j != i), key=lambda j: (distances[i, j], j))
        assert report.samples.loc[i, "purity@3"] == pytest.approx(np.mean(y[neighbors[:3]] == y[i]))
        positive = sorted(distances[i, j] for j in range(len(y)) if i != j and y[j] == y[i])
        negative = min(distances[i, j] for j in range(len(y)) if y[j] != y[i])
        if positive:
            assert report.samples.loc[i, "margin"] == pytest.approx(
                negative - np.median(positive[:3])
            )
        else:
            assert np.isnan(report.samples.loc[i, "margin"])
        baseline = (np.sum(y == y[i]) - 1) / (len(y) - 1)
        assert report.samples.loc[i, "neighbor_baseline"] == pytest.approx(baseline)


def test_manhattan_knn_oof_and_cli(tmp_path):
    rng = np.random.default_rng(71)
    X = rng.normal(size=(30, 4))
    X[0] = 0  # L1 accepts zero vectors when normalization is disabled.
    y = np.tile([0, 1], 15)
    config = AuditConfig(metric="manhattan", ks=(1,), probe_k=1, cv_folds=3)
    report = evaluate_embeddings(X, y, config=config)
    splits, reason = make_splits(y, None, config)
    assert reason is None
    correct_by_class = {label: [] for label in np.unique(y)}
    for train, test in splits:
        distances = np.abs(X[test, None, :] - X[None, train, :]).sum(axis=2)
        predicted = y[train[distances.argmin(axis=1)]]
        for label in correct_by_class:
            correct_by_class[label].extend((predicted[y[test] == label] == label).tolist())
    expected = np.mean([np.mean(correct) for correct in correct_by_class.values()])
    assert report.summary["linear_probe"]["oof"]["knn"]["balanced_accuracy"] == pytest.approx(
        expected
    )
    np.savez(tmp_path / "data.npz", embeddings=X, labels=y)
    assert (
        main(
            [
                str(tmp_path / "data.npz"),
                "--output",
                str(tmp_path / "report"),
                "--metric",
                "manhattan",
                "--ks",
                "1",
                "--probe-k",
                "1",
                "--cv-folds",
                "3",
                "--log-level",
                "WARNING",
            ]
        )
        == 0
    )
    saved = json.loads((tmp_path / "report" / "summary.json").read_text())
    assert saved["config"]["metric"] == "manhattan"
    assert saved["preprocessing"]["l2_normalized"] is False
    assert saved["linear_probe"]["oof"]["knn"]["balanced_accuracy"] == pytest.approx(expected)


def test_group_neighborhood_baselines_and_no_leakage(separated):
    X, y = separated
    groups = np.tile(np.arange(15), 9)
    config = AuditConfig(ks=(1, 5), cv_folds=3)
    report = evaluate_embeddings(X, y, groups=groups, config=config)
    assert report.summary["linear_probe"]["splitter"] == "stratified_group"
    splits, reason = make_splits(y, groups, config)
    assert reason is None
    for train, test in splits:
        assert not set(groups[train]) & set(groups[test])
    for i in range(len(X)):
        eligible = groups != groups[i]
        assert report.samples.loc[i, "eligible_neighbors"] == eligible.sum()
        assert report.samples.loc[i, "neighbor_baseline"] == pytest.approx(
            np.mean(y[eligible] == y[i])
        )
    # Descriptive silhouette does not change when groups are supplied.
    np.testing.assert_allclose(
        report.samples.silhouette, silhouette_samples(normalize(X), y, metric="cosine")
    )


def test_random_labels_have_chance_probe():
    rng = np.random.default_rng(27)
    X = rng.normal(size=(360, 24))
    labels = np.tile(np.arange(3), 120)
    rng.shuffle(labels)
    report = evaluate_embeddings(X, labels, config=AuditConfig(ks=(10,), cv_folds=3))
    score = report.summary["linear_probe"]["oof"]["linear"]["balanced_accuracy"]
    assert 0.24 < score < 0.43
    assert 0.8 < report.classes["lift@10_mean"].mean() < 1.2
    assert abs(report.samples.silhouette.mean()) < 0.04
    aucs = report.pairwise_auc.to_numpy()[~np.eye(3, dtype=bool)]
    assert np.all((aucs > 0.38) & (aucs < 0.62))


def test_repeated_crops_inflate_random_cv():
    rng = np.random.default_rng(51)
    anchors = rng.normal(size=(60, 128))
    labels = np.repeat(np.arange(60) % 2, 4)
    X = np.repeat(anchors, 4, axis=0)
    groups = np.repeat(np.arange(60), 4)
    config = AuditConfig(ks=(1,), cv_folds=3, linear_c=10)
    naive = evaluate_embeddings(X, labels, config=config)
    grouped = evaluate_embeddings(X, labels, groups=groups, config=config)
    naive_score = naive.summary["linear_probe"]["oof"]["linear"]["balanced_accuracy"]
    grouped_score = grouped.summary["linear_probe"]["oof"]["linear"]["balanced_accuracy"]
    assert naive_score > 0.95
    assert grouped_score < 0.7
    assert naive_score - grouped_score > 0.25
    assert naive.samples["purity@1"].mean() == 1
    assert grouped.samples["purity@1"].mean() < 0.7


def test_rare_classes_large_k_and_strict_json(tmp_path):
    X = np.array([[1, 0], [1, 0.1], [0, 1]], dtype=float)
    report = evaluate_embeddings(X, ["a", "a", "b"], config=AuditConfig(ks=(1, 10)))
    assert report.summary["linear_probe"]["status"] == "skipped"
    assert report.samples["purity@10"].isna().all()
    assert np.isnan(report.samples.loc[2, "margin"])
    report.save(tmp_path)
    raw = (tmp_path / "summary.json").read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    assert json.loads(raw)["geometry"]["purity_mean"]["10"] is None
    assert len(pd.read_csv(tmp_path / "samples.csv")) == 3


def test_one_group_per_class_skips_probes(separated):
    X, y = separated
    report = evaluate_embeddings(X, y, groups=y, config=AuditConfig(ks=(1,)))
    assert report.summary["linear_probe"]["status"] == "skipped"
    assert report.pairwise_auc.isna().all().all()
    assert report.samples.d_positive.isna().all()


def test_all_same_group_has_no_neighbors(separated):
    X, y = separated
    report = evaluate_embeddings(X, y, groups=np.zeros(len(y)), config=AuditConfig(ks=(1,)))
    assert report.samples["purity@1"].isna().all()
    assert report.samples.neighbor_baseline.isna().all()
    assert report.samples.eligible_neighbors.max() == 0


def test_every_sample_its_own_class():
    report = evaluate_embeddings(np.eye(3), ["a", "b", "c"], config=AuditConfig(ks=(1,)))
    assert report.samples.silhouette.isna().all()
    assert report.samples.margin.isna().all()


def test_nuisance_probes():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(180, 4))
    y = (X[:, 0] > 0).astype(int)
    metadata = {
        "area": X[:, 1],
        "camera": np.where(X[:, 2] > 0, "A", "B"),
        "constant": np.ones(len(X)),
        "missing": [np.nan] * len(X),
    }
    report = evaluate_embeddings(
        X, y, metadata=metadata, config=AuditConfig(ks=(5,), metric="euclidean", cv_folds=3)
    )
    results = report.summary["nuisance_probes"]
    assert results["area"]["oof_r2"] > 0.9
    assert results["area"]["baseline_oof_r2"] < 0.05
    assert results["camera"]["oof"]["linear"]["balanced_accuracy"] > 0.9
    assert results["constant"]["status"] == "skipped"
    assert results["missing"]["status"] == "skipped"


@pytest.mark.parametrize(
    "X,y,kwargs,match",
    [
        ([[0, 0], [1, 0]], ["a", "b"], {}, "nonzero"),
        ([[1, np.nan], [0, 1]], ["a", "b"], {}, "NaN"),
        ([1, 2], ["a", "b"], {}, "shape"),
        ([[1, 0], [0, 1]], ["a"], {}, "shape"),
        ([[1, 0], [0, 1]], ["a", "a"], {}, "two classes"),
        ([[1, 0], [0, 1]], ["a", None], {}, "missing"),
        ([[1, 0], [0, 1]], ["a", "b"], {"groups": [None, "b"]}, "missing"),
        ([[1, 0], [0, 1]], ["a", "b"], {"sample_ids": ["x", "x"]}, "unique"),
        ([[1, 0], [0, 1]], ["a", "b"], {"metadata": {"area": [1]}}, "shape"),
    ],
)
def test_invalid_inputs(X, y, kwargs, match):
    with pytest.raises(ValueError, match=match):
        evaluate_embeddings(X, y, config=AuditConfig(ks=(1,)), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ks": ()},
        {"ks": (1, 1)},
        {"ks": (0,)},
        {"ks": (1.5,)},
        {"cv_folds": 1},
        {"metric": "precomputed"},
        {"linear_c": 0},
        {"working_memory_mb": float("nan")},
        {"normalize": "yes"},
        {"random_state": -1},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        AuditConfig(**kwargs)


def test_cli_metadata_id_join_and_plot(tmp_path, separated, capsys):
    X, y = separated
    ids = np.array([f"q-{i}" for i in range(len(X))])
    np.savez(tmp_path / "data.npz", embeddings=X, labels=y, sample_ids=ids)
    # Reverse CSV order to verify explicit ID matching rather than row matching.
    pd.DataFrame({"sample_id": ids, "camera": y.astype(str)}).iloc[::-1].to_csv(
        tmp_path / "metadata.csv", index=False
    )
    main(
        [
            str(tmp_path / "data.npz"),
            "--output",
            str(tmp_path / "report"),
            "--metadata",
            str(tmp_path / "metadata.csv"),
            "--categorical-metadata",
            "camera",
            "--ks",
            "5",
            "--plots",
        ]
    )
    assert "Report saved" in capsys.readouterr().out
    assert (tmp_path / "report" / "pairwise_separability.svg").exists()
    summary = json.loads((tmp_path / "report" / "summary.json").read_text())
    assert summary["nuisance_probes"]["camera"]["kind"] == "classification"
    assert summary["nuisance_probes"]["camera"]["oof"]["linear"]["balanced_accuracy"] == 1


def test_cli_rejects_pickle_and_missing_keys(tmp_path):
    for name, values in (
        ("pickle", {"embeddings": np.eye(2), "labels": np.array(["a", "b"], dtype=object)}),
        ("keys", {"X": np.eye(2)}),
    ):
        np.savez(tmp_path / f"{name}.npz", **values)
        with pytest.raises(SystemExit) as exc:
            main([str(tmp_path / f"{name}.npz"), "--output", str(tmp_path / "report")])
        assert exc.value.code == 2


def test_convergence_warning_is_reported(separated):
    X, y = separated
    report = evaluate_embeddings(X, y, config=AuditConfig(ks=(1,), max_iter=1))
    assert any("did not converge" in notice for notice in report.summary["notices"])
