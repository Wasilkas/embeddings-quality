"""Portable artifacts, preserving unavailable estimates as null/empty cells."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass
class EmbeddingQualityReport:
    summary: dict[str, Any]
    samples: pd.DataFrame
    classes: pd.DataFrame
    pairwise_auc: pd.DataFrame
    pairwise_balanced_accuracy: pd.DataFrame
    confusion: pd.DataFrame

    def review_queue(self, limit: int | None = None) -> pd.DataFrame:
        """Lowest geometric margins first; no automatic relabeling/deletion."""
        k = min(self.summary["config"]["ks"])
        ordered: pd.DataFrame = self.samples.sort_values(
            ["margin", f"purity@{k}"], ascending=True, na_position="last", kind="stable"
        )
        return ordered if limit is None else ordered.head(limit)

    def to_dict(self) -> dict[str, Any]:
        return {key: _json_value(value) for key, value in self.summary.items()}

    def save(self, directory: str | Path, *, plots: bool = False) -> Path:
        """Write JSON, CSVs, interpretation notes, optionally an SVG heatmap."""
        directory = Path(directory)
        # Check optional dependency before writing any artifacts.
        if plots:
            try:
                import matplotlib  # noqa: F401
            except ImportError as exc:
                raise ImportError("Install embeddings-quality[plots] to export heatmaps.") from exc
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "summary.json").write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for name, table, indexed in (
            ("samples", self.samples, False),
            ("classes", self.classes, True),
            ("pairwise_auc", self.pairwise_auc, True),
            ("pairwise_balanced_accuracy", self.pairwise_balanced_accuracy, True),
            ("linear_confusion", self.confusion, True),
            ("review_queue", self.review_queue(), False),
        ):
            table.to_csv(directory / f"{name}.csv", index=indexed)
        (directory / "REPORT.md").write_text(self._markdown(), encoding="utf-8")
        if plots:
            self._heatmap(directory / "pairwise_separability.svg")
        return directory

    def _markdown(self) -> str:
        s = self.to_dict()
        probe = s["linear_probe"]
        lines = [
            "# Embedding Quality Report",
            "",
            f"Samples: {s['n_samples']}; dimensions: {s['n_features']}; classes: {s['n_classes']}.",
            f"Metric: {s['config']['metric']}; L2 normalization: {s['preprocessing']['l2_normalized']}.",
            "All metrics use the original coordinates; no dimensionality reduction.",
            "",
            f"Silhouette (sample mean): {s['geometry']['silhouette_mean']}.",
            f"Silhouette (class macro mean): {s['geometry']['silhouette_macro_mean']}.",
            f"Neighborhood reference: {s['geometry']['neighborhood_reference']}.",
            "Silhouette includes all samples regardless of group.",
            "",
        ]
        if probe["status"] == "ok":
            lines += ["| OOF probe | Balanced accuracy | Macro F1 |", "|---|---:|---:|"]
            for name, scores in probe["oof"].items():
                lines.append(
                    f"| {name} | {scores['balanced_accuracy']:.4f} | {scores['macro_f1']:.4f} |"
                )
            lines += ["", f"CV: {probe['splitter']}, {probe['n_folds']} folds.", ""]
        else:
            lines += [f"Linear probe skipped: {probe['reason']}", ""]
        lines += [
            "## Interpretation",
            "",
            "- High held-out linear scores support linear separability under this split policy.",
            "- Low linear scores do not rule out nonlinear separability; compare the kNN probe.",
            "- Binary ROC-AUC near 0.5 is chance ranking; near 1 means strong held-out ranking.",
            "- Pairwise matrices contain mean fold scores; fold standard deviation is not a confidence interval.",
            "- Purity should be compared to its eligible-neighbor class baseline; lift near 1 is baseline.",
            "- Negative margin means a different class is closer than the median of the nearest same-class samples.",
            "- Silhouette measures class compactness and overlap. Multimodal classes can have low silhouette despite useful features.",
            "- Predictable nuisance metadata shows encoded information, not proof that it causes classification performance.",
            "- Missing estimates are null/empty cells. Review counts and notices before comparing reports.",
            "- review_queue.csv sorts by margin then purity; it is a queue for manual inspection.",
            "",
            "## Artifacts",
            "",
            "See classes.csv for per-class means/quantiles and valid counts, samples.csv for per-object geometry and OOF scores,",
            "pairwise_auc.csv and pairwise_balanced_accuracy.csv for separability, and summary.json for fold/nuisance results.",
            "",
            "## Notices",
            "",
        ]
        lines += [f"- {notice}" for notice in s["notices"]] or ["None."]
        return "\n".join(lines) + "\n"

    def _heatmap(self, path: str | Path) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        n = len(self.pairwise_auc)
        figure = Figure(figsize=(max(6, 0.7 * n), max(5, 0.65 * n)), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 2)
        for axis, table, title in (
            (axes[0], self.pairwise_auc, "Linear CV ROC-AUC"),
            (axes[1], self.pairwise_balanced_accuracy, "Linear CV balanced accuracy"),
        ):
            data = table.to_numpy(dtype=float)
            mesh = axis.imshow(np.ma.masked_invalid(data), vmin=0, vmax=1, cmap="viridis")
            axis.set_xticks(range(n), table.columns, rotation=60, ha="right")
            axis.set_yticks(range(n), table.index)
            axis.set_title(title)
            if n <= 20:
                for i in range(n):
                    for j in range(n):
                        if np.isfinite(data[i, j]):
                            axis.text(
                                j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8
                            )
            figure.colorbar(mesh, ax=axis, shrink=0.7)
        figure.savefig(path, format="svg")
