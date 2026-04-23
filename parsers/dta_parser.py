import re

DEFAULT_HOPO_THRESHOLD = 170

def get_guitar_difficulty(dta_path):
    """
    Reads the songs.dta file and extracts the integer difficulty for 5-lane guitar.
    """
    try:
        with open(dta_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            
            # Regex to find the pattern: (guitar 234)
            # The \d+ captures the 1-605 integer
            match = re.search(r"\(\s*'?guitar'?\s+(\d+)\s*\)", content, re.IGNORECASE)
            
            if match:
                difficulty = int(match.group(1))
                return None if difficulty == 0 else difficulty
            else:
                return None # Song might not have a guitar part charted
    except Exception as e:
        print(f"Error reading {dta_path}: {e}")
        return None

def get_hopo_threshold(dta_path):
    """
    Reads the songs.dta file and extracts the hopo threshold if present.
    Defaults to 170 when missing.
    """
    try:
        with open(dta_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()

            match = re.search(r"\(\s*'?hopo_threshold'?\s+(\d+)\s*\)", content, re.IGNORECASE)
            if match:
                return int(match.group(1))
            return DEFAULT_HOPO_THRESHOLD
    except Exception as e:
        print(f"Error reading {dta_path}: {e}")
        return DEFAULT_HOPO_THRESHOLD

