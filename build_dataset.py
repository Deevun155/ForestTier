import argparse
import logging
from pathlib import Path

import pandas as pd

from parsers.dta_parser import get_guitar_difficulty, get_hopo_threshold
from parsers.midi_parser import extract_midi_features


LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
	logging.basicConfig(
		level=logging.DEBUG if verbose else logging.INFO,
		format="%(levelname)s %(message)s",
	)


def build_dataset(data_dir: Path, output_csv: Path) -> None:
	dataset = []
	total_folders = 0
	processed = 0
	missing_files = 0
	no_guitar = 0
	errors = 0

	LOGGER.info("Starting feature extraction pipeline in %s", data_dir)

	for song_folder in sorted(data_dir.iterdir()):
		if not song_folder.is_dir():
			continue

		total_folders += 1
		song_id = song_folder.name
		dta_file = song_folder / f"{song_id}.dta"
		midi_file = song_folder / f"{song_id}.mid"

		if not dta_file.exists() or not midi_file.exists():
			missing_files += 1
			LOGGER.warning("Skipping %s (missing .dta or .mid)", song_id)
			continue

		try:
			difficulty = get_guitar_difficulty(dta_file)
			if difficulty is None:
				no_guitar += 1
				LOGGER.info("Skipping %s (no guitar difficulty)", song_id)
				continue

			hopo_threshold = get_hopo_threshold(dta_file)
			features = extract_midi_features(str(midi_file), hopo_threshold=hopo_threshold)

			row = {
				"song_id": song_id,
				"difficulty": difficulty,
				**features,
			}
			dataset.append(row)
			processed += 1

			LOGGER.info(
				"Processed %s (difficulty=%s, hopo=%s)",
				song_id,
				difficulty,
				hopo_threshold,
			)
		except Exception as exc:
			errors += 1
			LOGGER.exception("Failed processing %s: %s", song_id, exc)

	if not dataset:
		LOGGER.warning("No rows produced. Check the data directory and parsing logic.")
		return

	df = pd.DataFrame(dataset)
	df.to_csv(output_csv, index=False)

	LOGGER.info(
		"Extraction complete. rows=%s total_folders=%s processed=%s missing=%s no_guitar=%s errors=%s output=%s",
		len(dataset),
		total_folders,
		processed,
		missing_files,
		no_guitar,
		errors,
		output_csv,
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Build RB3 guitar difficulty dataset.")
	parser.add_argument("--data-dir", default="data_hmx", help="Folder containing song subdirectories")
	parser.add_argument("--output", default="hmx_dataset.csv", help="Output CSV path")
	parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
	args = parser.parse_args()

	_configure_logging(args.verbose)
	build_dataset(Path(args.data_dir), Path(args.output))


if __name__ == "__main__":
	main()
