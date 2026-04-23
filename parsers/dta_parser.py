import os
import re
import pandas as pd
import mido
from pathlib import Path

def get_guitar_difficulty(dta_path):
    """
    Reads the songs.dta file and extracts the integer difficulty for 5-lane guitar.
    """
    try:
        with open(dta_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            
            # Regex to find the pattern: (guitar 234)
            # The \d+ captures the 1-605 integer
            match = re.search(r'\(guitar\s+(\d+)\)', content, re.IGNORECASE)
            
            if match:
                return int(match.group(1))
            else:
                return None # Song might not have a guitar part charted
    except Exception as e:
        print(f"Error reading {dta_path}: {e}")
        return None

def extract_midi_features(midi_path):
    """
    Parses the MIDI file to extract our physical charting metrics.
    Currently a stub - we will build the math for this next!
    """
    # TODO: Load MIDI with mido
    # TODO: Calculate Active Play Time
    # TODO: Calculate True Average NPS
    # TODO: Calculate Peak NPS (3-second window)
    
    # Returning dummy data for now to test the pipeline
    return {
        "true_avg_nps": 0.0,
        "peak_nps": 0.0,
        "strums_per_sec": 0.0
    }

def build_dataset(data_dir):
    data_path = Path(data_dir)
    dataset = []

    print("Starting feature extraction pipeline...")

    # Iterate through every song folder
    for song_folder in data_path.iterdir():
        if song_folder.is_dir():
            song_id = song_folder.name
            dta_file = song_folder / f"{song_id}.dta"
            midi_file = song_folder / f"{song_id}.mid"

            # Ensure both files exist
            if dta_file.exists() and midi_file.exists():
                
                # 1. Get the Label (The Difficulty)
                difficulty = get_guitar_difficulty(dta_file)
                
                # If there's no guitar part, skip this song
                if difficulty is None:
                    continue
                
                # 2. Get the Features (The Metrics)
                features = extract_midi_features(midi_file)
                
                # 3. Combine them into a single row of data
                row = {
                    "song_id": song_id,
                    "difficulty": difficulty,
                    **features # This unpacks the dictionary from extract_midi_features
                }
                dataset.append(row)

    # Convert the list of dictionaries into a pandas DataFrame and save to CSV
    df = pd.DataFrame(dataset)
    df.to_csv("rock_band_guitar_dataset.csv", index=False)
    print(f"Extraction complete! Saved {len(dataset)} songs to rock_band_guitar_dataset.csv")

# Run the pipeline
build_dataset("data")