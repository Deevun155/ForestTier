# Visualize baseline vs RF predictions against actuals.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    csv_path = Path(__file__).resolve().parent.parent / "test_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"song_id", "actual", "baseline_pred", "rf_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions CSV: {sorted(missing)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    def annotate_isolated(ax, actual, predicted, labels, top_n=10, k=5):
        if len(actual) < k + 1:
            return
        x_vals = actual.to_numpy()
        y_vals = predicted.to_numpy()
        x_norm = (x_vals - x_vals.mean()) / (x_vals.std() or 1.0)
        y_norm = (y_vals - y_vals.mean()) / (y_vals.std() or 1.0)
        points = np.column_stack([x_norm, y_norm])
        dists = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        knn = np.sort(dists, axis=1)[:, :k]
        isolation = knn.mean(axis=1)
        isolated_idx = np.argsort(isolation)[-top_n:]
        for idx in isolated_idx:
            ax.annotate(
                labels.iloc[idx],
                (x_vals[idx], y_vals[idx]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )

    axes[0].scatter(df["actual"], df["baseline_pred"], s=12, alpha=0.7)
    axes[0].plot([df["actual"].min(), df["actual"].max()],
                 [df["actual"].min(), df["actual"].max()],
                 color="red", linestyle="--", linewidth=1)
    axes[0].set_title("Baseline vs Actual")
    axes[0].set_xlabel("Actual difficulty")
    axes[0].set_ylabel("Predicted difficulty")

    axes[1].scatter(df["actual"], df["rf_pred"], s=12, alpha=0.7)
    axes[1].plot([df["actual"].min(), df["actual"].max()],
                 [df["actual"].min(), df["actual"].max()],
                 color="red", linestyle="--", linewidth=1)
    axes[1].set_title("RandomForest vs Actual")
    axes[1].set_xlabel("Actual difficulty")

    annotate_isolated(axes[0], df["actual"], df["baseline_pred"], df["song_id"])
    annotate_isolated(axes[1], df["actual"], df["rf_pred"], df["song_id"])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
