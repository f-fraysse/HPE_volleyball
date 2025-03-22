from pathlib import Path

# Get the root directory (assuming this file is inside scripts/)
ROOT_DIR = Path(__file__).resolve().parents[1]

# Key directories
MODEL_DIR   = ROOT_DIR / 'models'
DATA_DIR    = ROOT_DIR / 'data'
OUTPUT_H5_DIR = ROOT_DIR / 'output' / 'h5'      # HDF5 files (poses, bboxes, scores)
OUTPUT_VIDEO_DIR  = ROOT_DIR / 'output' / 'video'   # Processed video output

# Optional: list of output dirs to auto-create
OUTPUT_FOLDERS = [MODEL_DIR, DATA_DIR, OUTPUT_H5_DIR, OUTPUT_VIDEO_DIR]

def ensure_output_dirs():
    """Create output folders if they don't exist."""
    for folder in OUTPUT_FOLDERS:
        folder.mkdir(parents=True, exist_ok=True)