# Visualize baseline vs RF predictions against actuals.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    csv_path = Path(__file__).resolve().parent.parent / "test_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"actual", "baseline_pred", "rf_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions CSV: {sorted(missing)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

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

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
