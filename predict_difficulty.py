import argparse
import logging
from pathlib import Path

import pandas as pd
import joblib

from parsers.midi_parser import extract_midi_features


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


def _load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict RB3 guitar difficulty from a MIDI file.")
    parser.add_argument("--model", default="rf_model.joblib", help="Path to trained model")
    parser.add_argument("--midi", required=True, help="Path to the .mid file")
    parser.add_argument("--hopo", type=int, default=170, help="HOPO threshold in ticks (default: 170)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    model = _load_model(Path(args.model))
    features = extract_midi_features(args.midi, hopo_threshold=args.hopo)
    X_pred = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    prediction = model.predict(X_pred)[0]
    LOGGER.info("Predicted difficulty: %.2f", prediction)


if __name__ == "__main__":
    main()
