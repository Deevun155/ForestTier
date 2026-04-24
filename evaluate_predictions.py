import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import joblib
from scipy import stats
from sklearn.metrics import mean_absolute_error


LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


def evaluate_predictions(csv_path: Path, model_path: Path | None) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"actual", "baseline_pred", "rf_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions CSV: {sorted(missing)}")

    baseline_errors = (df["actual"] - df["baseline_pred"]).abs()
    rf_errors = (df["actual"] - df["rf_pred"]).abs()

    baseline_mae = mean_absolute_error(df["actual"], df["baseline_pred"])
    rf_mae = mean_absolute_error(df["actual"], df["rf_pred"])

    t_stat, t_p = stats.ttest_rel(baseline_errors, rf_errors)
    w_stat, w_p = stats.wilcoxon(baseline_errors, rf_errors)

    LOGGER.info("Rows: %s", len(df))
    LOGGER.info("Baseline MAE: %.4f", baseline_mae)
    LOGGER.info("RandomForest MAE: %.4f", rf_mae)
    LOGGER.info("Paired t-test: t=%.4f p=%.6f", t_stat, t_p)
    LOGGER.info("Wilcoxon: statistic=%.4f p=%.6f", w_stat, w_p)

    if model_path:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        model = joblib.load(model_path)
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model does not expose feature_importances_")

        importances = model.feature_importances_
        feature_names = [
            "total_active_time",
            "avg_nps",
            "peak_nps",
            "avg_strums_per_sec",
            "peak_strums_per_sec",
            "avg_fret_changes_per_sec",
            "peak_fret_changes_per_sec",
        ]

        order = importances.argsort()[::-1]
        plt.figure(figsize=(8, 4))
        plt.bar(
            [feature_names[idx] for idx in order],
            importances[order],
        )
        plt.title("RandomForest Feature Importance")
        plt.ylabel("Importance")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions CSV without retraining.")
    parser.add_argument("--csv", default="test_predictions.csv", help="Path to predictions CSV")
    parser.add_argument("--model", default="rf_model.joblib", help="Path to trained model for feature importance")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    model_path = Path(args.model) if args.model else None
    evaluate_predictions(Path(args.csv), model_path)


if __name__ == "__main__":
    main()
