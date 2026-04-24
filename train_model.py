import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


LOGGER = logging.getLogger(__name__)


FEATURE_COLUMNS = [
	"total_active_time",
	"avg_nps",
	"peak_nps",
	"avg_strums_per_sec",
	"peak_strums_per_sec",
	"avg_fret_changes_per_sec",
	"peak_fret_changes_per_sec",
]


def _configure_logging(verbose: bool) -> None:
	logging.basicConfig(
		level=logging.DEBUG if verbose else logging.INFO,
		format="%(levelname)s %(message)s",
	)


def _load_dataset(csv_path: Path, sample_fraction: float, random_state: int) -> pd.DataFrame:
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV not found at {csv_path}")

	df = pd.read_csv(csv_path)
	if sample_fraction < 1.0:
		df = df.sample(frac=sample_fraction, random_state=random_state)
	return df


def train_models(
	df: pd.DataFrame,
	baseline_feature: str,
	model_features: list[str],
	random_state: int,
	output_predictions: Path | None,
	) -> None:
	missing = [col for col in model_features + [baseline_feature, "difficulty"] if col not in df.columns]
	if missing:
		raise ValueError(f"Missing columns in dataset: {missing}")
	if output_predictions and "song_id" not in df.columns:
		raise ValueError("song_id column is required to export predictions")

	X_baseline = df[[baseline_feature]]
	X_full = df[model_features]
	y = df["difficulty"]
	ids = df["song_id"] if "song_id" in df.columns else None

	split_items = [X_full, y]
	if ids is not None:
		split_items.append(ids)
	train_test = train_test_split(
		*split_items,
		test_size=0.2,
		random_state=random_state,
	)
	X_train_full = train_test[0]
	X_test_full = train_test[1]
	y_train = train_test[2]
	y_test = train_test[3]
	ids_test = train_test[5] if ids is not None else None
	X_train_base = X_train_full[[baseline_feature]]
	X_test_base = X_test_full[[baseline_feature]]

	baseline = LinearRegression()
	baseline.fit(X_train_base, y_train)
	baseline_pred = baseline.predict(X_test_base)
	baseline_mae = mean_absolute_error(y_test, baseline_pred)

	rf = RandomForestRegressor(
		n_estimators=300,
		random_state=random_state,
		n_jobs=-1,
	)
	rf.fit(X_train_full, y_train)
	rf_pred = rf.predict(X_test_full)
	rf_mae = mean_absolute_error(y_test, rf_pred)

	LOGGER.info("Rows: %s", len(df))
	LOGGER.info("Baseline feature: %s", baseline_feature)
	LOGGER.info("Baseline MAE: %.4f", baseline_mae)
	LOGGER.info("RandomForest MAE: %.4f", rf_mae)

	if output_predictions:
		predictions = pd.DataFrame(
			{
				"song_id": ids_test,
				"actual": y_test,
				"baseline_pred": baseline_pred,
				"rf_pred": rf_pred,
			}
		)
		predictions.to_csv(output_predictions, index=False)
		LOGGER.info("Wrote predictions to %s", output_predictions)


def main() -> None:
	parser = argparse.ArgumentParser(description="Train baseline and RF models on RB3 dataset.")
	parser.add_argument("--csv", default="hmx_dataset.csv", help="Path to dataset CSV")
	parser.add_argument("--sample-fraction", type=float, default=0.25, help="Fraction of rows to use")
	parser.add_argument("--random-state", type=int, default=42, help="Random seed")
	parser.add_argument("--output-predictions", help="Write test predictions to CSV")
	parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
	args = parser.parse_args()

	if not 0.0 < args.sample_fraction <= 1.0:
		raise ValueError("sample-fraction must be in (0, 1]")

	_configure_logging(args.verbose)
	data = _load_dataset(Path(args.csv), args.sample_fraction, args.random_state)
	predictions_path = Path(args.output_predictions) if args.output_predictions else None
	train_models(
		df=data,
		baseline_feature="peak_fret_changes_per_sec",
		model_features=FEATURE_COLUMNS,
		random_state=args.random_state,
		output_predictions=predictions_path,
	)


if __name__ == "__main__":
	main()
