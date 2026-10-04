"""CLI for aligned arrays saved as NPZ; pickle loading is disabled."""

import argparse
import sys

import numpy as np
import pandas as pd

from .audit import AuditConfig, evaluate_embeddings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check SSL embedding separability in original dimensions."
    )
    parser.add_argument("input", help="NPZ with embeddings, labels; optional sample_ids, groups")
    parser.add_argument("--output", required=True, help="Report directory")
    parser.add_argument(
        "--metadata", help="CSV with sample_id and nuisance columns; joined by sample_id"
    )
    parser.add_argument(
        "--categorical-metadata",
        nargs="+",
        default=[],
        help="CSV columns to treat as categories, including numeric IDs",
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20, 50])
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    normalization = parser.add_mutually_exclusive_group()
    normalization.add_argument("--normalize", dest="normalize", action="store_true")
    normalization.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--margin-k", type=int, default=5)
    parser.add_argument("--probe-k", type=int, default=10)
    parser.add_argument("--linear-c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--working-memory-mb", type=float, default=128)
    parser.add_argument("--include-same-group-neighbors", action="store_true")
    parser.add_argument(
        "--plots", action="store_true", help="Export SVG heatmap (requires [plots])"
    )
    args = parser.parse_args(argv)
    try:
        with np.load(args.input, allow_pickle=False) as data:
            if "embeddings" not in data or "labels" not in data:
                raise ValueError("NPZ must contain embeddings and labels arrays.")
            X, labels = data["embeddings"], data["labels"]
            ids = data["sample_ids"] if "sample_ids" in data else None
            groups = data["groups"] if "groups" in data else None
        metadata = None
        if args.categorical_metadata and not args.metadata:
            raise ValueError("--categorical-metadata requires --metadata.")
        if args.metadata:
            if ids is None:
                raise ValueError("Metadata CSV requires sample_ids in NPZ.")
            frame = pd.read_csv(args.metadata, dtype={"sample_id": str})
            if (
                "sample_id" not in frame
                or frame.sample_id.isna().any()
                or frame.sample_id.duplicated().any()
            ):
                raise ValueError("Metadata needs a unique, nonmissing sample_id column.")
            frame = frame.set_index("sample_id")
            string_ids = ids.astype(str)
            if not set(string_ids).issubset(frame.index):
                raise ValueError("Metadata CSV is missing NPZ sample IDs.")
            for column in args.categorical_metadata:
                if column not in frame.columns:
                    raise ValueError(f"Categorical metadata column {column!r} is missing.")
                frame[column] = frame[column].astype("string")
            metadata = frame.loc[string_ids].to_dict("list")
        config = AuditConfig(
            ks=tuple(args.ks),
            metric=args.metric,
            normalize=args.normalize,
            cv_folds=args.cv_folds,
            margin_k=args.margin_k,
            probe_k=args.probe_k,
            linear_c=args.linear_c,
            max_iter=args.max_iter,
            random_state=args.seed,
            working_memory_mb=args.working_memory_mb,
            exclude_same_group=not args.include_same_group_neighbors,
        )
        report = evaluate_embeddings(
            X, labels, sample_ids=ids, groups=groups, metadata=metadata, config=config
        )
        report.save(args.output, plots=args.plots)
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    print(f"Report saved to {args.output}")
    probe = report.summary["linear_probe"]
    if probe["status"] == "ok":
        print(f"Linear OOF balanced accuracy: {probe['oof']['linear']['balanced_accuracy']:.4f}")
    else:
        print(f"Linear probe skipped: {probe['reason']}")
    for notice in report.summary["notices"]:
        print(f"Note: {notice}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
