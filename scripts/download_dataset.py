from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fashion_pipeline import download_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Fashion-MNIST dataset into the project.")
    parser.add_argument("--force", action="store_true", help="Re-download even if the dataset already exists.")
    args = parser.parse_args()

    path = download_dataset(force=args.force)
    print(f"Dataset saved to: {path}")


if __name__ == "__main__":
    main()
