#A program to organise the data from the organisation spat out by Nautilus into a neater format that's easier to parse

import shutil
from pathlib import Path

def reorganize_rock_band_data(source_dir, target_dir):
    # Set up our paths using pathlib for clean cross-platform compatibility
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    dta_folder = source_path / "dta_files"
    midi_folder = source_path / "midi_files"

    # Safety check
    if not dta_folder.exists() or not midi_folder.exists():
        print("Error: Could not find 'dta_files' or 'midi_files' in the source directory.")
        return

    # Create the base 'data' directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)

    success_count = 0
    missing_midi_count = 0

    print("Starting file reorganization...")

    # Iterate through every .dta file in the dta_files folder
    for dta_file in dta_folder.glob("*.dta"):
        # .stem extracts just the filename without the extension (e.g., '21guns')
        song_id = dta_file.stem 
        
        # Define the target folder for this specific song (data/21guns/)
        song_dest_folder = target_path / song_id
        song_dest_folder.mkdir(parents=True, exist_ok=True)
        
        # Define exact file paths
        expected_midi = midi_folder / song_id / f"{song_id}.mid"
        dest_dta = song_dest_folder / f"{song_id}.dta"
        dest_midi = song_dest_folder / f"{song_id}.mid"

        # 1. Copy the .dta file
        shutil.copy2(dta_file, dest_dta)
        
        # 2. Check for the MIDI file and copy it
        if expected_midi.exists():
            shutil.copy2(expected_midi, dest_midi)
            success_count += 1
        else:
            print(f"Warning: Missing MIDI file for {song_id}. Expected at: {expected_midi}")
            missing_midi_count += 1

    print("-" * 30)
    print("Reorganization Complete!")
    print(f"Successfully paired {success_count} songs.")
    if missing_midi_count > 0:
        print(f"Failed to find MIDI files for {missing_midi_count} songs.")

# Run the function
# Assumes the script is running in the parent directory of 'CONs'
reorganize_rock_band_data("CONs", "data")