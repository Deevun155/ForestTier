import argparse
import logging
from pathlib import Path

import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error


LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


def evaluate_predictions(csv_path: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions CSV without retraining.")
    parser.add_argument("--csv", default="test_predictions.csv", help="Path to predictions CSV")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    evaluate_predictions(Path(args.csv))


if __name__ == "__main__":
    main()
