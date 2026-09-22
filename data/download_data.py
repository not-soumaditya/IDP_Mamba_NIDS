"""
Download UNSW-NB15 dataset using kagglehub.

Usage:
    python data/download_data.py

Expects Kaggle API credentials at ~/.kaggle/kaggle.json
If not configured, prints clear setup instructions.
"""

import os
import sys
import shutil
from pathlib import Path

# Resolve project root (one level up from data/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download_dataset():
    """Download UNSW-NB15 dataset from Kaggle via kagglehub."""
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)

    print("Downloading UNSW-NB15 dataset from Kaggle...")
    try:
        dataset_path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
        print(f"Dataset downloaded to: {dataset_path}")
    except Exception as e:
        print(f"ERROR downloading dataset: {e}")
        print()
        print("Possible fixes:")
        print("  - Check your internet connection")
        print("  - Set up Kaggle credentials:")
        print("    1. Go to https://www.kaggle.com/settings")
        print("    2. Scroll to 'API' section and click 'Create New Token'")
        print("    3. This downloads a kaggle.json file")
        print(f"    4. Place it at: {Path.home() / '.kaggle' / 'kaggle.json'}")
        print("  - Try: pip install --upgrade kagglehub")
        sys.exit(1)

    # Create raw directory
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Copy the relevant CSV files to data/raw/
    dataset_path = Path(dataset_path)
    target_files = [
        "UNSW_NB15_training-set.csv",
        "UNSW_NB15_testing-set.csv",
    ]

    found_files = []
    # Search for target files recursively
    for target in target_files:
        matches = list(dataset_path.rglob(target))
        if matches:
            src = matches[0]
            dst = RAW_DIR / target
            shutil.copy2(src, dst)
            print(f"  Copied: {target} ({dst.stat().st_size / 1e6:.1f} MB)")
            found_files.append(target)
        else:
            print(f"  WARNING: {target} not found in downloaded dataset")

    if not found_files:
        # List what we actually got, to help debug
        print("\nFiles found in download directory:")
        for f in sorted(dataset_path.rglob("*.csv")):
            print(f"  {f.name}")
        print("\nTrying to copy all CSV files instead...")
        for f in dataset_path.rglob("*.csv"):
            dst = RAW_DIR / f.name
            shutil.copy2(f, dst)
            print(f"  Copied: {f.name}")

    print(f"\nRaw data saved to: {RAW_DIR}")
    print("Next step: run data cleaning (model/dataset.py or data/clean.py)")
    return RAW_DIR


if __name__ == "__main__":
    download_dataset()
