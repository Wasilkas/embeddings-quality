"""Smoke demo: 512-dimensional features, known separation vs shuffled labels."""

import argparse
from pathlib import Path

import numpy as np

from embeddings_quality import AuditConfig, evaluate_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/demo"))
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    labels = np.repeat(["scratch", "crack", "pitting"], 60)
    centers = np.eye(3, 512) * 8
    X = centers[np.repeat(np.arange(3), 60)] + rng.normal(0, 0.15, (180, 512))
    ids = np.array([f"sample-{i}" for i in range(len(labels))])
    config = AuditConfig(ks=(5, 10, 20, 50), cv_folds=3)
    for name, target in (("separated", labels), ("shuffled_labels", rng.permutation(labels))):
        report = evaluate_embeddings(X, target, sample_ids=ids, config=config)
        report.save(args.output / name, plots=True)
        score = report.summary["linear_probe"]["oof"]["linear"]["balanced_accuracy"]
        print(f"{name}: linear OOF balanced accuracy={score:.3f}")
    np.savez_compressed(args.output / "separated.npz", embeddings=X, labels=labels, sample_ids=ids)


if __name__ == "__main__":
    main()
